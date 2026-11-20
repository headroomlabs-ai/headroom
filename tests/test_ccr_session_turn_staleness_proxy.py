"""Exercise conversation age through the real Anthropic forwarding path."""

import copy
from contextlib import contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.cache.backends import SQLiteBackend
from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.ccr.context_tracker import reset_context_tracker
from headroom.proxy.server import ProxyConfig, create_app


@contextmanager
def _client():
    config = ProxyConfig(
        optimize=False,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        image_optimize=False,
        ccr_context_tracking=True,
        ccr_proactive_expansion=True,
        ccr_max_turn_distance=3,
        ccr_inject_tool=True,
        ccr_inject_system_instructions=False,
    )
    forwarded = []
    with TestClient(
        create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 1234)
    ) as client:

        async def upstream(method, url, headers, body, **kwargs):
            forwarded.append(copy.deepcopy(body))
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        client.app.state.proxy._retry_request = upstream
        yield client, forwarded


def _request(client, workspace, key, query, session=None):
    headers = {"x-api-key": "test-key"}
    if session is not None:
        headers["x-headroom-session-id"] = session
    response = client.post(
        "/v1/messages",
        headers=headers,
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "system": f"cwd: {workspace}",
            "messages": [
                {"role": "user", "content": f"Earlier tool result: <<ccr:{key},json,1KB>>"},
                {"role": "user", "content": query},
            ],
        },
    )
    assert response.status_code == 200, response.text


@pytest.fixture
def stored_events(tmp_path):
    reset_compression_store()
    reset_context_tracker()
    database = tmp_path / "compression.sqlite"
    store = get_compression_store(backend=SQLiteBackend(db_path=database))
    original = '{"authentication_middleware":"SESSION_A_CONTEXT_SENTINEL"}'
    key = store.store(
        original, "authentication middleware", query_context="authentication middleware"
    )
    other = store.store('{"deployment":"session B data"}', "deployment")
    yield database, store, key, other, original
    reset_context_tracker()
    reset_compression_store()


@pytest.mark.parametrize("same_workspace", [True, False])
@pytest.mark.parametrize("explicit_session", [True, False])
def test_other_conversation_requests_do_not_age_original_context(
    tmp_path, stored_events, same_workspace, explicit_session
):
    _, _, key, other, _ = stored_events
    other_workspace = tmp_path if same_workspace else tmp_path / "other-workspace"
    session_a = "session-a" if explicit_session else None
    session_b = "session-b" if explicit_session else None
    with _client() as (client, forwarded):
        _request(client, tmp_path, key, "Describe unrelated deployment work", session_a)
        for _ in range(4):
            _request(client, other_workspace, other, "Describe deployment", session_b)
        _request(client, tmp_path, key, "authentication middleware", session_a)
        assert "SESSION_A_CONTEXT_SENTINEL" in str(forwarded[-1])


def test_turn_expiry_survives_reopening_proxy_and_sqlite_store(
    tmp_path, stored_events, monkeypatch
):
    database, _, key, _, original = stored_events
    with _client() as (client, forwarded):
        for _ in range(4):
            _request(client, tmp_path, key, "Describe unrelated deployment work", "session-a")
        _request(client, tmp_path, key, "authentication middleware", "session-a")
        assert "SESSION_A_CONTEXT_SENTINEL" not in str(forwarded[-1])

    reset_context_tracker()
    # reset_compression_store is a destructive test helper, not process loss.
    # Drop only the singleton reference; reopening must retain the actual rows.
    monkeypatch.setattr("headroom.cache.compression_store._compression_store", None)
    reopened = get_compression_store(backend=SQLiteBackend(db_path=database))
    assert reopened.retrieve(key).original_content == original
    with _client() as (client, forwarded):
        _request(client, tmp_path, key, "authentication middleware", "session-a")
        assert "SESSION_A_CONTEXT_SENTINEL" not in str(forwarded[-1])


def test_same_hash_in_same_workspace_retains_each_sessions_age(tmp_path, stored_events):
    _, _, key, _, _ = stored_events
    with _client() as (client, forwarded):
        for _ in range(4):
            _request(client, tmp_path, key, "Describe unrelated deployment work.", "session-a")
        _request(client, tmp_path, key, "Describe unrelated deployment work.", "session-b")
        _request(client, tmp_path, key, "authentication middleware", "session-a")
        assert "SESSION_A_CONTEXT_SENTINEL" not in str(forwarded[-1])
        _request(client, tmp_path, key, "authentication middleware", "session-b")
        assert "SESSION_A_CONTEXT_SENTINEL" in str(forwarded[-1])


@pytest.mark.parametrize("unrelated_turns, eligible", [(4, False), (1, True)])
def test_trimmed_initial_user_message_does_not_revive_expired_fallback_context(
    tmp_path, stored_events, unrelated_turns, eligible
):
    _, _, key, _, _ = stored_events
    with _client() as (client, forwarded):

        def request(query, *, keep_origin=True):
            messages = [
                {"role": "user", "content": f"Earlier tool result: <<ccr:{key},json,1KB>>"},
                {"role": "user", "content": query},
            ]
            if keep_origin:
                messages.insert(
                    0, {"role": "user", "content": "Original conversation introduction"}
                )
            response = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 16,
                    "system": f"cwd: {tmp_path}",
                    "messages": messages,
                },
            )
            assert response.status_code == 200, response.text

        for _ in range(unrelated_turns):
            request("Describe unrelated deployment work.")
        request("authentication middleware")
        assert ("SESSION_A_CONTEXT_SENTINEL" in str(forwarded[-1])) is eligible
        request("authentication middleware", keep_origin=False)
        assert ("SESSION_A_CONTEXT_SENTINEL" in str(forwarded[-1])) is eligible


def test_same_timestamp_recreated_payload_is_fresh_through_proxy(
    tmp_path, stored_events, monkeypatch
):
    _, store, key, _, original = stored_events
    timestamp = store.get_metadata(key)["created_at"]
    old_event = store.get_metadata(key)["event_id"]
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: timestamp)
    with _client() as (client, forwarded):
        for _ in range(4):
            _request(client, tmp_path, key, "Describe unrelated deployment work.", "session-a")
        _request(client, tmp_path, key, "authentication middleware", "session-a")
        assert "SESSION_A_CONTEXT_SENTINEL" not in str(forwarded[-1])
        assert store._backend.delete(key)
        assert (
            store.store(
                original, "authentication middleware", query_context="authentication middleware"
            )
            == key
        )
        assert store.get_metadata(key)["created_at"] == timestamp
        assert store.get_metadata(key)["event_id"] != old_event
        _request(client, tmp_path, key, "authentication middleware", "session-a")
        assert "SESSION_A_CONTEXT_SENTINEL" in str(forwarded[-1])

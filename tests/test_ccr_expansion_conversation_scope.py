"""Regression for #1174: proactive expansion must stay inside one conversation.

A Claude Code lead and its agent-team teammates run in the same directory, so
they resolve the same CCR workspace and share the proxy's process-global
``ContextTracker``. When a teammate's incoming message shared a few keywords
with tool output the lead had compressed, the proxy appended the lead's full
original output to the teammate's message. The teammate had never run that
tool and reported the block as garbage context.

These tests drive the real ``/v1/messages`` handler in token mode (where the
append is live) and inspect the body forwarded upstream.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.proxy.server import ProxyConfig, create_app

_EXPANSION_TAG = "<headroom_proactive_expansion>"
_LEAD_OUTPUT = "\n".join(
    f"src/auth/middleware_{i}.py session token login handler" for i in range(40)
)
_LEAD_COMPRESSED = "\n".join(_LEAD_OUTPUT.splitlines()[:4])
_LEAD_QUERY = "list the auth middleware files"
_TEAMMATE_QUERY = (
    '<teammate-message teammate_id="team-lead">Review the auth middleware '
    "session handling and report back.</teammate-message>"
)


@pytest.fixture(autouse=True)
def _fresh_store():
    reset_compression_store()
    yield
    reset_compression_store()


def _client() -> TestClient:
    config = ProxyConfig(
        optimize=False,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_handle_responses=False,
        ccr_context_tracking=True,
        ccr_proactive_expansion=True,
        image_optimize=False,
    )
    return TestClient(create_app(config))


def _track_lead_compression(proxy, cwd: str) -> str:  # noqa: ANN001
    """Record a compression the way the handler does after the lead's turn."""
    hash_key = get_compression_store().store(
        _LEAD_OUTPUT,
        _LEAD_COMPRESSED,
        original_item_count=40,
        compressed_item_count=4,
        tool_name="Bash",
        query_context=_LEAD_QUERY,
    )
    workspace_key, _ = proxy._resolve_ccr_workspace(
        SimpleNamespace(headers={"x-headroom-cwd": cwd}), {}
    )
    assert workspace_key
    proxy.ccr_context_tracker.track_compression(
        hash_key=hash_key,
        turn_number=1,
        tool_name="Bash",
        original_count=40,
        compressed_count=4,
        workspace_key=workspace_key,
        query_context=_LEAD_QUERY,
        sample_content=_LEAD_COMPRESSED,
    )
    return hash_key


def _forwarded_last_user_text(client: TestClient, cwd: str, messages: list[dict]) -> str:
    proxy = client.app.state.proxy
    captured: dict = {}

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "id": "msg_scope",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    proxy._retry_request = _fake_retry
    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "x-headroom-cwd": cwd,
        },
        json={"model": "claude-sonnet-4-6", "max_tokens": 16, "messages": messages},
    )
    assert response.status_code == 200, response.text
    content = captured["body"]["messages"][-1]["content"]
    return content if isinstance(content, str) else content[0]["text"]


def test_teammate_does_not_receive_lead_tool_output(tmp_path) -> None:  # noqa: ANN001
    cwd = str(tmp_path)
    with _client() as client:
        _track_lead_compression(client.app.state.proxy, cwd)

        forwarded = _forwarded_last_user_text(
            client, cwd, [{"role": "user", "content": _TEAMMATE_QUERY}]
        )

    assert _EXPANSION_TAG not in forwarded
    assert forwarded == _TEAMMATE_QUERY


def test_conversation_holding_the_marker_still_expands(tmp_path) -> None:  # noqa: ANN001
    cwd = str(tmp_path)
    with _client() as client:
        hash_key = _track_lead_compression(client.app.state.proxy, cwd)

        forwarded = _forwarded_last_user_text(
            client,
            cwd,
            [
                {"role": "user", "content": _LEAD_QUERY},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": f"[40 items compressed to 4. Retrieve more: hash={hash_key}]",
                        }
                    ],
                },
                {"role": "assistant", "content": "Listed them."},
                {"role": "user", "content": _TEAMMATE_QUERY},
            ],
        )

    assert _EXPANSION_TAG in forwarded
    # Line 39 is only in the original, never in the compressed preview.
    assert "src/auth/middleware_39.py" in forwarded

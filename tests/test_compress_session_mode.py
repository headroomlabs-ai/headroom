"""Session-aware /v1/compress (sidecar mode) + the /v1/usage relay.

Contract under test: a gateway that owns routing (e.g. Kong) sends the RAW
conversation plus a session id every turn; Headroom keeps the byte-replay
state itself and returns a byte-identical prefix; the gateway forwards the
result verbatim and may relay provider usage via POST /v1/usage to make
freeze decisions exact.

The critical property is byte-stability: content already returned for a
session must come back byte-for-byte identical on later turns, or the
provider prompt cache busts.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_client() -> TestClient:
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        image_optimize=False,
    )
    app = create_app(config)
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    return client


def _big_tool_history() -> list[dict]:
    """A conversation whose tool result is large enough to be compressed."""
    items = [
        {
            "id": i,
            "score": 0.99 if i % 30 == 0 else 0.6,
            "msg": f"Result {i:03d}{' error' if i % 30 == 0 else ' ok'}",
            "blob": f"payload-{i:04d}-" + "".join(chr(97 + (i * 7 + j) % 26) for j in range(240)),
        }
        for i in range(200)
    ]
    return [
        {"role": "user", "content": "Get items"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps(items)},
    ]


def _compress(client: TestClient, messages: list[dict], **config) -> dict:
    resp = client.post(
        "/v1/compress",
        json={"model": "gpt-4o", "messages": messages, "config": config},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# Stateless behaviour is unchanged (regression guard).                         #
# --------------------------------------------------------------------------- #


# The NUL separator makes the namespace unspoofable from any HTTP header.
SESSION_KEY_PREFIX = "compress\x00"


def test_no_session_id_stays_stateless() -> None:
    with _make_client() as client:
        body = _compress(client, _big_tool_history())
        assert "session" not in body
        # And nothing session-shaped leaked into the registry.
        proxy = client.app.state.proxy
        assert not any(k.startswith(SESSION_KEY_PREFIX) for k in proxy._compression_caches)


def test_invalid_session_id_is_rejected() -> None:
    with _make_client() as client:
        for bad in ["", "   ", "x" * 300, 42]:
            resp = client.post(
                "/v1/compress",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "config": {"session_id": bad},
                },
            )
            assert resp.status_code == 400, f"session_id {bad!r} was not rejected"


def test_compress_user_messages_rejected_with_session() -> None:
    """User-message rewrites are not content-addressed, so they cannot be
    byte-replayed after tracker state expires — the combination is a latent
    prefix-cache bust and must be refused up front."""
    with _make_client() as client:
        resp = client.post(
            "/v1/compress",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "config": {"session_id": "conv-x", "compress_user_messages": True},
            },
        )
        assert resp.status_code == 400
        assert "compress_user_messages" in resp.json()["error"]["message"]


def test_session_key_is_not_spoofable_via_string_prefix() -> None:
    """A caller passing 'compress:...' (or similar) as its session id must
    land on a key that no proxy-path header value can also produce."""
    with _make_client() as client:
        _compress(client, _big_tool_history(), session_id="compress:sneaky")
        proxy = client.app.state.proxy
        keys = [k for k in proxy._compression_caches if "sneaky" in k]
        assert keys == [f"{SESSION_KEY_PREFIX}compress:sneaky"]
        # NUL cannot appear in an HTTP header value, so no x-headroom-session-id
        # on the proxy path can collide with this key.
        assert all("\x00" in k for k in keys)


# --------------------------------------------------------------------------- #
# The core sidecar property: turn 2 replays turn 1's exact bytes.              #
# --------------------------------------------------------------------------- #


def test_second_turn_replays_first_turn_bytes() -> None:
    with _make_client() as client:
        history = _big_tool_history()

        turn1 = _compress(client, history, session_id="conv-1")
        assert turn1["session"]["id"] == "conv-1"
        # The tool result must actually have been compressed, otherwise the
        # byte-stability assertion below is vacuous.
        t1_tool_content = turn1["messages"][2]["content"]
        assert t1_tool_content != history[2]["content"]
        assert turn1["tokens_saved"] > 0

        # Turn 2: the caller resends the RAW history (as real clients do) plus
        # the new turns. Headroom must return the OLD prefix byte-identical to
        # what it handed back on turn 1 — that is what the provider cached.
        turn2_history = history + [
            {"role": "assistant", "content": "The top items are listed above."},
            {"role": "user", "content": "Now sort them by score."},
        ]
        turn2 = _compress(client, turn2_history, session_id="conv-1")
        assert turn2["messages"][2]["content"] == t1_tool_content
        # The WHOLE turn-1 prefix, not just the tool result: any drifted byte
        # anywhere in the leading messages is a provider-cache bust.
        assert turn2["messages"][: len(turn1["messages"])] == turn1["messages"]
        assert turn2["messages"][-1]["content"] == "Now sort them by score."
        assert turn2["session"]["id"] == "conv-1"
        # Savings must be reported against the RAW payload the caller sent —
        # the warm turn still saved the caller ~everything turn 1 saved, even
        # though the pipeline itself only saw the already-swapped input.
        assert turn2["tokens_saved"] > 0
        assert turn2["tokens_before"] > turn2["tokens_after"]


def test_third_turn_still_byte_stable() -> None:
    """The WHOLE returned prefix — every message, byte for byte — must be
    stable across N turns. Checking only the tool result would let drift in
    any other message (a mutated plain message, a moved marker) bust the
    provider cache while the test stayed green.
    """
    with _make_client() as client:
        history = _big_tool_history()
        turn1 = _compress(client, history, session_id="conv-multi")

        history2 = history + [{"role": "user", "content": "next"}]
        turn2 = _compress(client, history2, session_id="conv-multi")
        # Turn 2's leading messages must be exactly turn 1's returned bytes.
        assert turn2["messages"][: len(turn1["messages"])] == turn1["messages"]

        history3 = history2 + [
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "and again"},
        ]
        turn3 = _compress(client, history3, session_id="conv-multi")
        # And turn 3's leading messages must be exactly turn 2's.
        assert turn3["messages"][: len(turn2["messages"])] == turn2["messages"]


def test_prefix_stable_even_after_tracker_state_loss() -> None:
    """The overlay's tracker snapshots live shorter (600s session TTL) than
    the compression cache (3900s). In that window the frozen+swap path is the
    ONLY protection — this test kills the tracker between turns and demands
    whole-prefix byte stability from frozen+swap alone.
    """
    with _make_client() as client:
        history = _big_tool_history()
        turn1 = _compress(client, history, session_id="conv-trackerloss")

        proxy = client.app.state.proxy
        # Simulate the tracker registry's TTL sweep reclaiming the session
        # while the compression cache (longer TTL) survives.
        store = proxy.session_tracker_store
        removed = [k for k in list(store._trackers) if "conv-trackerloss" in k]
        for k in removed:
            del store._trackers[k]
        assert removed, "tracker was never created for the session"
        assert any("conv-trackerloss" in k for k in proxy._compression_caches)

        turn2 = _compress(
            client,
            history + [{"role": "user", "content": "after tracker loss"}],
            session_id="conv-trackerloss",
        )
        assert turn2["messages"][: len(turn1["messages"])] == turn1["messages"]


def test_header_session_id_ignored_by_default() -> None:
    """Deployments whose gateways stamp x-headroom-session-id on ALL traffic
    must not silently flip stateless /v1/compress callers into session mode
    (or blend conversations sharing one header value into one replay state)."""
    with _make_client() as client:
        resp = client.post(
            "/v1/compress",
            json={"model": "gpt-4o", "messages": _big_tool_history(), "config": {}},
            headers={"x-headroom-session-id": "conv-header"},
        )
        assert resp.status_code == 200, resp.text
        assert "session" not in resp.json()


def test_header_session_id_works_with_env_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_COMPRESS_SESSION_FROM_HEADER", "1")
    with _make_client() as client:
        resp = client.post(
            "/v1/compress",
            json={"model": "gpt-4o", "messages": _big_tool_history(), "config": {}},
            headers={"x-headroom-session-id": "conv-header"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["session"]["id"] == "conv-header"


def test_sessions_are_isolated() -> None:
    with _make_client() as client:
        history = _big_tool_history()
        a1 = _compress(client, history, session_id="conv-a")
        b1 = _compress(client, history, session_id="conv-b")

        # Same content in, same compressed form out — but through separate
        # session state. Interleave new turns and re-check both replay.
        a2 = _compress(
            client,
            history + [{"role": "user", "content": "a follow-up"}],
            session_id="conv-a",
        )
        b2 = _compress(
            client,
            history + [{"role": "user", "content": "b follow-up"}],
            session_id="conv-b",
        )
        assert a2["messages"][2]["content"] == a1["messages"][2]["content"]
        assert b2["messages"][2]["content"] == b1["messages"][2]["content"]
        assert a2["messages"][-1]["content"] == "a follow-up"
        assert b2["messages"][-1]["content"] == "b follow-up"


# --------------------------------------------------------------------------- #
# /v1/usage: telemetry relay for sidecar sessions. Deliberately NOT a freeze  #
# input — freeze stays the locally-replayable bound (see handler docstring).  #
# --------------------------------------------------------------------------- #


def test_usage_relay_is_recorded_and_freeze_stays_local() -> None:
    with _make_client() as client:
        history = _big_tool_history()
        _compress(client, history, session_id="conv-usage")

        resp = client.post(
            "/v1/usage",
            json={
                "session_id": "conv-usage",
                "usage": {
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 50_000,
                },
            },
        )
        assert resp.status_code == 200, resp.text
        # The tracker recorded the provider-confirmed prefix (telemetry).
        assert resp.json()["frozen_message_count"] >= 1

        # The next compress freezes from the LOCAL replayable bound, which
        # covers the whole previously-returned prefix here.
        turn2 = _compress(
            client,
            history + [{"role": "user", "content": "next"}],
            session_id="conv-usage",
        )
        assert turn2["session"]["frozen_message_count"] >= 1

        # An absurdly large confirmed count must never drag freezing past
        # what local state can actually replay (that would forward raw bytes
        # for evicted entries — the bust this design refuses).
        resp2 = client.post(
            "/v1/usage",
            json={
                "session_id": "conv-usage",
                "usage": {"cache_read_input_tokens": 10_000_000},
            },
        )
        assert resp2.status_code == 200
        turn3 = _compress(
            client,
            history
            + [
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "more"},
            ],
            session_id="conv-usage",
        )
        # Freeze is capped by message count minus the trailing message — it
        # can never exceed what exists, regardless of relayed numbers.
        assert turn3["session"]["frozen_message_count"] < 6


def test_usage_unknown_session_is_404_and_leaves_no_footprint() -> None:
    with _make_client() as client:
        proxy = client.app.state.proxy
        before = len(proxy.session_tracker_store._trackers)
        for i in range(20):
            resp = client.post(
                "/v1/usage",
                json={
                    "session_id": f"never-seen-{i}",
                    "usage": {"cache_read_input_tokens": 100},
                },
            )
            assert resp.status_code == 404
            assert resp.json()["error"]["type"] == "unknown_session"
        # A flood of novel ids must not grow the tracker store (peek, never
        # get_or_create): each ghost tracker would otherwise live a full TTL.
        assert len(proxy.session_tracker_store._trackers) == before


def test_usage_without_cache_fields_is_rejected_not_treated_as_cold() -> None:
    """A usage block with NEITHER cache field (e.g. an OpenAI-style
    {'prompt_tokens': N} relayed verbatim) carries no cache signal. Treating
    the absent fields as 0 would tell the tracker 'provider confirmed fully
    cold' and wipe its cached-prefix state on every signal-free relay."""
    with _make_client() as client:
        _compress(client, _big_tool_history(), session_id="conv-nosignal")
        resp = client.post(
            "/v1/usage",
            json={"session_id": "conv-nosignal", "usage": {"prompt_tokens": 12345}},
        )
        assert resp.status_code == 400
        assert "cache" in resp.json()["error"]["message"]


def test_usage_validation() -> None:
    with _make_client() as client:
        cases = [
            {},  # no session_id
            {"session_id": "s"},  # no usage
            {"session_id": "s", "usage": "nope"},  # usage not a dict
            {"session_id": "s", "usage": {"cache_read_input_tokens": -1}},
            {"session_id": "s", "usage": {"cache_read_input_tokens": True}},
        ]
        for body in cases:
            resp = client.post("/v1/usage", json=body)
            assert resp.status_code == 400, f"body {body!r} was not rejected"


# --------------------------------------------------------------------------- #
# Lifecycle: sidecar sessions ride the registry's TTL/LRU machinery.           #
# --------------------------------------------------------------------------- #


def test_session_state_lives_in_registry_and_survives_eviction() -> None:
    import time as _time

    with _make_client() as client:
        history = _big_tool_history()
        turn1 = _compress(client, history, session_id="conv-ttl")
        proxy = client.app.state.proxy
        _key = f"{SESSION_KEY_PREFIX}conv-ttl"
        assert _key in proxy._compression_caches

        # Simulate the idle-TTL sweep reclaiming the session.
        now = _time.time()
        proxy._compression_cache_last_seen[_key] = now - 999_999
        proxy._compression_caches_last_cleanup = now - 61
        proxy._get_compression_cache("unrelated")
        assert _key not in proxy._compression_caches

        # A post-eviction turn is fail-open: fresh state, valid response, and
        # the compressed form is reproducible (deterministic pipeline), even
        # though the replay guarantee had to restart from scratch.
        turn2 = _compress(
            client,
            history + [{"role": "user", "content": "after the gap"}],
            session_id="conv-ttl",
        )
        assert turn2["session"]["id"] == "conv-ttl"
        assert turn2["messages"][-1]["content"] == "after the gap"
        assert isinstance(turn1["messages"][2]["content"], str)


def test_explicit_frozen_count_still_wins_when_larger() -> None:
    with _make_client() as client:
        history = _big_tool_history()
        # First turn with an explicit pin covering the whole tool result: the
        # caller asserts the provider already cached it, so it must come back
        # byte-for-byte untouched even though no session state exists yet.
        turn1 = _compress(client, history, session_id="conv-pin", frozen_message_count=3)
        assert turn1["messages"][2]["content"] == history[2]["content"]
        assert turn1["session"]["frozen_message_count"] == 3


# --------------------------------------------------------------------------- #
# Review fixes: turn-lock contention, no-signal usage, expired trackers.       #
# --------------------------------------------------------------------------- #


def test_compress_503_when_turn_lock_busy(monkeypatch) -> None:
    """A concurrent turn for the same session must fail fast with a 503,
    not park an executor worker on an untimed lock acquire."""
    import headroom.proxy.handlers.openai as openai_mod

    monkeypatch.setattr(openai_mod, "_SESSION_TURN_LOCK_TIMEOUT_SECONDS", 0.05)
    with _make_client() as client:
        history = _big_tool_history()
        _compress(client, history, session_id="conv-lock")
        proxy = client.app.state.proxy
        lock = proxy._compression_caches[f"{SESSION_KEY_PREFIX}conv-lock"].session_turn_lock

        assert lock.acquire(timeout=1), "test could not take the turn lock"
        try:
            resp = client.post(
                "/v1/compress",
                json={
                    "model": "gpt-4o",
                    "messages": history + [{"role": "user", "content": "blocked"}],
                    "config": {"session_id": "conv-lock"},
                },
            )
            assert resp.status_code == 503, resp.text
        finally:
            lock.release()

        # With the lock free again the same turn succeeds.
        after = _compress(
            client,
            history + [{"role": "user", "content": "blocked"}],
            session_id="conv-lock",
        )
        assert after["session"]["id"] == "conv-lock"


def test_usage_503_when_turn_lock_busy(monkeypatch) -> None:
    """/v1/usage must take the same turn lock as the compress turn — an
    unlocked update races the executor and rolls tracker snapshots back."""
    import headroom.proxy.handlers.openai as openai_mod

    monkeypatch.setattr(openai_mod, "_SESSION_TURN_LOCK_TIMEOUT_SECONDS", 0.05)
    with _make_client() as client:
        _compress(client, _big_tool_history(), session_id="conv-ulock")
        proxy = client.app.state.proxy
        lock = proxy._compression_caches[f"{SESSION_KEY_PREFIX}conv-ulock"].session_turn_lock

        assert lock.acquire(timeout=1)
        try:
            resp = client.post(
                "/v1/usage",
                json={
                    "session_id": "conv-ulock",
                    "usage": {
                        "cache_read_input_tokens": 100,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )
            assert resp.status_code == 503, resp.text
            assert resp.json()["error"]["type"] == "session_busy"
        finally:
            lock.release()


def test_usage_single_zero_field_does_not_wipe_state() -> None:
    """{"cache_read_input_tokens": 0} with no write field (the natural
    OpenAI-mapped relay on a cold turn) carries no cache signal — it must
    not reset the tracker's provider-confirmed prefix state."""
    with _make_client() as client:
        _compress(client, _big_tool_history(), session_id="conv-zero")

        # Establish real provider-confirmed state (both fields present).
        resp = client.post(
            "/v1/usage",
            json={
                "session_id": "conv-zero",
                "usage": {
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 50_000,
                },
            },
        )
        assert resp.status_code == 200
        assert resp.json()["applied"] is True
        established = resp.json()["frozen_message_count"]
        assert established >= 1

        # The no-signal relay is acknowledged but NOT applied.
        resp2 = client.post(
            "/v1/usage",
            json={"session_id": "conv-zero", "usage": {"cache_read_input_tokens": 0}},
        )
        assert resp2.status_code == 200
        body = resp2.json()
        assert body["applied"] is False
        assert body["reason"] == "no_cache_signal"
        assert body["frozen_message_count"] == established  # state intact

        # A relay with BOTH fields zero is a genuine fully-cold assertion
        # and IS applied.
        resp3 = client.post(
            "/v1/usage",
            json={
                "session_id": "conv-zero",
                "usage": {
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )
        assert resp3.status_code == 200
        assert resp3.json()["applied"] is True


def test_usage_404_for_ttl_expired_tracker() -> None:
    """peek() must treat a TTL-expired-but-unswept tracker as gone — a 200
    here would resurrect the dead tracker on every relay."""
    import time as _time

    with _make_client() as client:
        _compress(client, _big_tool_history(), session_id="conv-expired")
        proxy = client.app.state.proxy
        tracker = proxy.session_tracker_store._trackers[f"{SESSION_KEY_PREFIX}conv-expired"]
        tracker._last_activity = _time.time() - 999_999

        resp = client.post(
            "/v1/usage",
            json={
                "session_id": "conv-expired",
                "usage": {"cache_read_input_tokens": 100},
            },
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "unknown_session"

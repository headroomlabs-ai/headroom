"""Regression coverage for PrefixCacheTracker wiring on Bedrock backend paths.

Both Bedrock-routed branches of ``handle_anthropic_messages``
(non-streaming in ``anthropic.py``, streaming ``_stream_response_bedrock``
in ``streaming.py``) used to return before ever calling
``prefix_tracker.update_from_response()``. Only the direct-Anthropic-API
branch called it. Practical effect: on any ``--backend bedrock --mode
cache`` deployment, ``PrefixCacheTracker`` state stayed permanently at
turn 0 for the life of a session — ``get_frozen_message_count()`` always
returned 0, ``extract_cache_stable_delta()`` always saw no previous turn,
and cache mode fell back to full unmodified passthrough on every single
turn instead of freezing the already-cached prefix and compressing only
the new suffix.

These tests drive two turns through the real proxy (with a mocked
Bedrock-shaped backend) and inspect the real ``PrefixCacheTracker`` the
proxy keeps in ``session_tracker_store`` — not a fake — to pin that the
tracker's turn counter and last-forwarded/-original messages actually
advance after a Bedrock call, for both the non-streaming and the
streaming code path.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.backends.base import BackendResponse, StreamEvent  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_anthropic_backend(body: dict[str, Any]) -> MagicMock:
    """Mock backend whose ``send_message`` returns an Anthropic-shaped body."""

    async def fake_send(body_: dict, headers: dict) -> BackendResponse:
        return BackendResponse(body=body, status_code=200)

    mock = MagicMock()
    mock.name = "bedrock"
    mock.send_message = fake_send
    mock.map_model_id = MagicMock(return_value="claude-3-5-sonnet-20241022")
    mock.supports_model = MagicMock(return_value=True)
    return mock


def _make_bedrock_streaming_backend(events: list[StreamEvent]) -> MagicMock:
    """Mock backend that yields Anthropic ``StreamEvent`` objects."""

    async def fake_stream(body: dict, headers: dict) -> AsyncIterator[StreamEvent]:
        for evt in events:
            yield evt

    mock = MagicMock()
    mock.name = "bedrock"
    mock.stream_message = fake_stream
    mock.map_model_id = MagicMock(return_value="claude-3-5-sonnet-20241022")
    mock.supports_model = MagicMock(return_value=True)
    return mock


def _sse_data(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _cache_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="anthropic",
        mode="cache",
    )


def _anthropic_body(cache_read: int, cache_write: int) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 50,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        },
    }


# =============================================================================
# Non-streaming Bedrock path (anthropic.py)
# =============================================================================


def test_bedrock_nonstreaming_advances_prefix_tracker_turn() -> None:
    """A non-streaming Bedrock request must call ``update_from_response``.

    Before the fix, the Bedrock non-streaming branch returned its
    ``JSONResponse`` without ever touching ``prefix_tracker`` — the
    tracker stayed at ``_turn_number == 0`` and ``_last_original_messages
    == []`` no matter how many turns went through. After the fix, one
    turn through this path must leave the tracker recording turn 1 and
    the sent + assistant messages as its "last" snapshot.
    """
    config = _cache_config()
    backend = _make_anthropic_backend(_anthropic_body(cache_read=500, cache_write=200))

    with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
        app = create_app(config)
        proxy = app.state.proxy
        with TestClient(app) as client:
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 64,
                },
                headers={
                    "x-api-key": "sk-ant-test",
                    "anthropic-version": "2023-06-01",
                    "x-headroom-session-id": "bedrock-nonstream-session",
                },
            )
            assert resp.status_code == 200, resp.text[:200]

    tracker = proxy.session_tracker_store.get_or_create("bedrock-nonstream-session", "anthropic")
    assert tracker._turn_number == 1, (
        "prefix tracker never advanced past turn 0 — update_from_response() "
        "was not called on the Bedrock non-streaming path"
    )
    assert tracker.get_last_original_messages(), (
        "tracker recorded no 'last turn' messages — the Bedrock non-streaming "
        "branch is not feeding it the sent + assistant messages"
    )
    # cache_read=500 + cache_write=200 = 700 total_cached, above the default
    # min_cached_tokens=1024 threshold is NOT met here, but the turn/messages
    # advancing (asserted above) is the actual regression signal — frozen
    # count only matters once the session crosses the threshold, which is
    # covered by test_cross_turn_cache_safety.py and test_cache/test_prefix_tracker.py.


def test_bedrock_nonstreaming_second_turn_sees_frozen_prefix() -> None:
    """Two Bedrock non-streaming turns: turn 2 must see turn 1 as its frozen prefix.

    This is the concrete consequence of the tracker actually updating:
    once cache_read+cache_write clears ``min_cached_tokens``, turn 2's
    ``get_frozen_message_count()`` must be nonzero and its
    ``get_last_original_messages()`` must equal turn 1's full message
    history (user + assistant) — the input the freeze/delta-compression
    path needs to detect an append-only turn. Before the fix this was
    always 0 / [] regardless of turn count.
    """
    config = _cache_config()
    # 1200 total cached tokens clears the default min_cached_tokens=1024.
    backend = _make_anthropic_backend(_anthropic_body(cache_read=1000, cache_write=200))

    with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
        app = create_app(config)
        proxy = app.state.proxy
        with TestClient(app) as client:
            turn1 = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 64,
                },
                headers={
                    "x-api-key": "sk-ant-test",
                    "anthropic-version": "2023-06-01",
                    "x-headroom-session-id": "bedrock-nonstream-2turn",
                },
            )
            assert turn1.status_code == 200, turn1.text[:200]

            turn2 = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                        {"role": "user", "content": "and again"},
                    ],
                    "max_tokens": 64,
                },
                headers={
                    "x-api-key": "sk-ant-test",
                    "anthropic-version": "2023-06-01",
                    "x-headroom-session-id": "bedrock-nonstream-2turn",
                },
            )
            assert turn2.status_code == 200, turn2.text[:200]

    tracker = proxy.session_tracker_store.get_or_create("bedrock-nonstream-2turn", "anthropic")
    assert tracker._turn_number == 2
    assert tracker.get_frozen_message_count() > 0, (
        "frozen_message_count stayed 0 on turn 2 despite a cache hit on turn 1 "
        "— PrefixCacheTracker never saw turn 1's response"
    )


# =============================================================================
# Streaming Bedrock path (streaming.py, _stream_response_bedrock)
# =============================================================================


def test_bedrock_streaming_advances_prefix_tracker_turn() -> None:
    """A streaming Bedrock request must also call ``update_from_response``.

    Mirrors the non-streaming test above for ``_stream_response_bedrock``.
    Before the fix, this function had no ``prefix_tracker`` parameter at
    all — the tracker was never even threaded in, let alone updated.
    """
    config = _cache_config()

    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "model": "claude-3-5-sonnet-20241022",
            "role": "assistant",
            "type": "message",
            "content": [],
            "usage": {
                "input_tokens": 1000,
                "cache_read_input_tokens": 500,
                "cache_creation_input_tokens": 200,
            },
        },
    }
    block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    block_delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hi"},
    }
    block_stop = {"type": "content_block_stop", "index": 0}
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 50},
    }
    message_stop = {"type": "message_stop"}

    events = [
        StreamEvent(event_type=e["type"], data=e, raw_sse=_sse_data(e["type"], e))
        for e in [message_start, block_start, block_delta, block_stop, message_delta, message_stop]
    ]
    backend = _make_bedrock_streaming_backend(events)

    with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
        app = create_app(config)
        proxy = app.state.proxy
        with TestClient(app) as client:
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 64,
                    "stream": True,
                },
                headers={
                    "x-api-key": "sk-ant-test",
                    "anthropic-version": "2023-06-01",
                    "x-headroom-session-id": "bedrock-stream-session",
                },
            )
            assert resp.status_code == 200, resp.text[:200]
            assert "message_stop" in resp.text

    tracker = proxy.session_tracker_store.get_or_create("bedrock-stream-session", "anthropic")
    assert tracker._turn_number == 1, (
        "prefix tracker never advanced past turn 0 on the Bedrock streaming "
        "path — update_from_response() was not called from "
        "_stream_response_bedrock"
    )
    assert tracker.get_last_original_messages(), (
        "tracker recorded no 'last turn' messages on the streaming path — "
        "the reconstructed assistant message from the SSE stream never "
        "reached the tracker"
    )

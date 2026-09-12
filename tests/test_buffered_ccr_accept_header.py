"""A buffered CCR turn must not ask for SSE it no longer wants (#3078).

Server-side retrieval flips a ``stream: true`` turn to ``stream: false`` so the
whole reply is in hand before answering. The body was rewritten; the client's
``Accept: text/event-stream`` was not, so the request that went on the wire
contradicted itself — "answer as JSON" in the body, "I only accept SSE" in the
headers.

Anthropic tolerates that, which is why it never showed up against the first-party
API. GitHub Copilot's Anthropic-compatible gateway does not, and answers with a
generic ``api_error``. That produced the reported shape exactly: the first call
of an OpenCode session succeeds (no marker yet, so no buffering), and the next
one — the first to carry a redeemable marker, and so the first to be flipped —
fails.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.cache.backends import InMemoryBackend  # noqa: E402
from headroom.cache.compression_store import (  # noqa: E402
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.tool_injection import create_ccr_tool_definition  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _store():
    reset_compression_store()
    get_compression_store(backend=InMemoryBackend())
    try:
        yield
    finally:
        reset_compression_store()


def _drive(*, with_marker: bool, accept: str | None) -> dict[str, object]:
    """Run one streaming turn; report the body `stream` and headers sent upstream."""
    marker = get_compression_store().store(
        original=json.dumps({"earlier": "tool output"}),
        compressed="{}",
        original_item_count=400,
    )
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        ccr_inject_tool=True,
        ccr_handle_responses=True,
        ccr_event_streaming=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    seen: dict[str, object] = {}
    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            sent = json.loads(body) if isinstance(body, (str, bytes)) else body
            seen["stream"] = sent.get("stream")
            seen["headers"] = dict(headers or {})
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry  # type: ignore[assignment]

        # A turn that is *not* flipped never reaches `_retry_request` — the plain
        # streaming path has its own upstream call — so capture that one too.
        async def _fake_stream(url, headers, body, *args, **kwargs):  # noqa: ANN001
            from fastapi.responses import StreamingResponse

            sent = json.loads(body) if isinstance(body, (str, bytes)) else body
            seen["stream"] = sent.get("stream")
            seen["headers"] = dict(headers or {})

            async def _gen():
                yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

            return StreamingResponse(_gen(), media_type="text/event-stream")

        proxy._stream_response = _fake_stream  # type: ignore[assignment]
        headers = {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}
        if accept is not None:
            headers["accept"] = accept
        content = f"go <<ccr:{marker}>>" if with_marker else "go"
        client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "stream": True,
                "tools": [create_ccr_tool_definition("anthropic")],
                "messages": [{"role": "user", "content": content}],
            },
            headers=headers,
        )
    return seen


def _accepts(headers: dict) -> list[str]:
    return [v for k, v in headers.items() if k.lower() == "accept"]


def test_buffered_turn_asks_for_json() -> None:
    seen = _drive(with_marker=True, accept="text/event-stream")

    # Precondition: this turn really was flipped to buffered.
    assert seen["stream"] is False
    assert _accepts(seen["headers"]) == ["application/json"]  # type: ignore[arg-type]


def test_buffered_turn_leaves_exactly_one_accept_header() -> None:
    """Replaced, never appended — two Accept values is its own bug."""
    seen = _drive(with_marker=True, accept="text/event-stream")

    assert len(_accepts(seen["headers"])) == 1  # type: ignore[arg-type]


def test_accept_header_is_replaced_regardless_of_casing() -> None:
    """Header names are case-insensitive; the SSE value must not survive."""
    seen = _drive(with_marker=True, accept="TEXT/EVENT-STREAM")

    values = _accepts(seen["headers"])  # type: ignore[arg-type]
    assert values == ["application/json"]
    assert not any("event-stream" in v.lower() for v in values)


def test_buffered_turn_without_a_client_accept_still_asks_for_json() -> None:
    seen = _drive(with_marker=True, accept=None)

    assert seen["stream"] is False
    assert _accepts(seen["headers"]) == ["application/json"]  # type: ignore[arg-type]


def test_a_streaming_turn_keeps_its_sse_accept() -> None:
    """No marker means no flip, so nothing about the request should change."""
    seen = _drive(with_marker=False, accept="text/event-stream")

    assert seen["stream"] is not False
    assert _accepts(seen["headers"]) == ["text/event-stream"]  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The same flip exists on the OpenAI Responses path, which serves Copilot
# --------------------------------------------------------------------------- #
def _drive_responses(*, accept: str) -> dict[str, object]:
    """Run one streaming /v1/responses turn and report what went upstream."""
    from headroom.ccr import CCR_TOOL_NAME

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        ccr_inject_tool=True,
        ccr_handle_responses=True,
        ccr_event_streaming=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    seen: dict[str, object] = {}
    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            sent = json.loads(body) if isinstance(body, (str, bytes)) else body
            seen["stream"] = sent.get("stream")
            seen["headers"] = dict(headers or {})
            return httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "object": "response",
                    "model": "gpt-4o",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
                },
            )

        proxy._retry_request = _fake_retry  # type: ignore[assignment]
        client.post(
            "/v1/responses",
            json={
                "model": "gpt-4o",
                "stream": True,
                # Responses tool defs are flat, not nested under "function".
                "tools": [{"type": "function", "name": CCR_TOOL_NAME}],
                "input": "go",
            },
            headers={
                "authorization": "Bearer test-key",
                "accept": accept,
                "content-type": "application/json",
            },
        )
    return seen


def test_responses_buffered_turn_asks_for_json() -> None:
    """This handler serves GitHub Copilot, the gateway that rejects the mismatch."""
    seen = _drive_responses(accept="text/event-stream")

    assert seen.get("stream") is False, "precondition: the turn must be buffered"
    assert _accepts(seen["headers"]) == ["application/json"]  # type: ignore[arg-type]
    assert len(_accepts(seen["headers"])) == 1  # type: ignore[arg-type]

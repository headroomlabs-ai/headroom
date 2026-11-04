"""SSE ping passthrough in the Bedrock streaming path (issue #902).

When Headroom routes through a Bedrock (LiteLLM/AnyLLM) backend the
translation layer emits only Anthropic-semantic events — it never produces
SSE-level ping keepalives.  Claude Code uses ping events to keep a turn in
the interruptible / steering-armed state; without them, mid-turn interjections
are silently dropped.

Fix: ``_stream_response_bedrock.generate()`` now emits one synthetic
``event: ping\\ndata: {}\\n\\n`` before the first ``message_start`` so the
downstream client sees the same ping-then-content cadence as a real Anthropic
stream.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.backends.base import StreamEvent  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _ev(event_type: str, data: dict[str, Any]) -> StreamEvent:
    return StreamEvent(event_type=event_type, data=data)


def _minimal_events() -> list[StreamEvent]:
    return [
        _ev(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "model": "claude-3-5-sonnet-20241022",
                    "role": "assistant",
                    "type": "message",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        ),
        _ev(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _ev(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
        ),
        _ev("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _ev(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
        ),
        _ev("message_stop", {"type": "message_stop"}),
    ]


def _make_bedrock_backend(events: list[StreamEvent]) -> MagicMock:
    async def fake_stream(body: dict, headers: dict) -> AsyncIterator[StreamEvent]:
        for evt in events:
            yield evt

    mock = MagicMock()
    mock.name = "bedrock"
    mock.stream_message = fake_stream
    mock.map_model_id = MagicMock(return_value="claude-3-5-sonnet-20241022")
    mock.supports_model = MagicMock(return_value=True)
    return mock


def _run_bedrock_stream(events: list[StreamEvent]) -> str:
    """Return the full SSE response body from a Bedrock-backend streaming request."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="anthropic",
    )
    backend = _make_bedrock_backend(events)
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
        app = create_app(config)
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
                },
            )
            assert resp.status_code == 200, resp.text[:200]
            return resp.text


def test_bedrock_stream_emits_ping_before_message_start() -> None:
    """Bedrock path must emit a ping event before message_start (issue #902)."""
    body = _run_bedrock_stream(_minimal_events())

    ping_idx = body.find("event: ping")
    start_idx = body.find("event: message_start")

    assert ping_idx != -1, "No ping event found in Bedrock stream response"
    assert start_idx != -1, "No message_start event found in Bedrock stream response"
    assert ping_idx < start_idx, (
        f"ping (offset {ping_idx}) must appear before message_start (offset {start_idx})"
    )


def test_bedrock_stream_ping_has_empty_data() -> None:
    """Ping event must carry data: {} to match real Anthropic wire format."""
    body = _run_bedrock_stream(_minimal_events())

    ping_start = body.find("event: ping")
    assert ping_start != -1, "No ping event found"

    # The next ~30 bytes after 'event: ping' should contain 'data: {}'
    ping_block = body[ping_start : ping_start + 40]
    assert "data: {}" in ping_block, f"Ping block must contain 'data: {{}}', got: {ping_block!r}"


def test_bedrock_stream_contains_message_stop() -> None:
    """Smoke test: full event sequence still reaches the client alongside ping."""
    body = _run_bedrock_stream(_minimal_events())
    assert "event: message_stop" in body
    assert "event: message_start" in body

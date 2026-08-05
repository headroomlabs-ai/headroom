"""Bedrock/LiteLLM streaming must report real input tokens on message_start.

Regression coverage for issue #1132.

LiteLLM/Bedrock streaming never surfaces prompt tokens during a stream — it
emits ``message_start`` with ``usage.input_tokens=0`` and only reports
``output_tokens`` (at the end, in ``message_delta``). Anthropic clients such as
Claude Code read ``usage.input_tokens`` from the ``message_start`` SSE event to
emit OTel/cost metrics, so every Headroom+Bedrock streaming request reported ~0
input tokens — underreporting token usage by ~99%.

``StreamingMixin._stream_response_bedrock`` now backfills ``input_tokens`` on
``message_start`` with the count Headroom actually sent upstream
(``optimized_tokens``) when the backend left it unset/zero, while preserving any
non-zero value the backend genuinely reported.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.backends.base import StreamEvent  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_bedrock_backend(events: list[StreamEvent]) -> MagicMock:
    """Mock backend yielding Anthropic ``StreamEvent``s (no ``raw_sse``).

    Mirrors ``LiteLLMBackend.stream_message``: it constructs each event from a
    ``data`` dict and never sets ``raw_sse``, so the handler re-serializes
    ``event.data`` — the exact path that carries the #1132 bug.
    """

    async def fake_stream(body: dict, headers: dict) -> AsyncIterator[StreamEvent]:
        for evt in events:
            yield evt

    mock = MagicMock()
    mock.name = "bedrock"
    mock.stream_message = fake_stream
    mock.map_model_id = MagicMock(return_value="claude-3-5-sonnet-20241022")
    mock.supports_model = MagicMock(return_value=True)
    return mock


def _bedrock_events(input_tokens: int) -> list[StreamEvent]:
    """Build a minimal Anthropic streaming sequence as LiteLLM emits it."""
    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "model": "claude-3-5-sonnet-20241022",
            "role": "assistant",
            "type": "message",
            "content": [],
            # LiteLLM hardcodes this to 0 — the bug under test.
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
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
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 50},
    }
    message_stop = {"type": "message_stop"}

    # raw_sse=None mirrors LiteLLMBackend.stream_message (data-only events).
    return [
        StreamEvent(event_type=e["type"], data=e)
        for e in [
            message_start,
            block_start,
            block_delta,
            block_stop,
            message_delta,
            message_stop,
        ]
    ]


def _message_start_input_tokens(sse_body: str) -> int:
    """Extract ``message.usage.input_tokens`` from the message_start SSE event."""
    for block in sse_body.split("\n\n"):
        if "message_start" not in block:
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
                return payload["message"]["usage"]["input_tokens"]
    raise AssertionError(f"No message_start event with usage found in SSE:\n{sse_body[:500]}")


def _post_stream(backend: MagicMock) -> str:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="anthropic",
    )
    with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
        app = create_app(config)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "hello there general"}],
                    "max_tokens": 64,
                    "stream": True,
                },
                headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            )
            assert resp.status_code == 200, resp.text[:200]
            return resp.text


def test_bedrock_streaming_backfills_input_tokens_on_message_start() -> None:
    """When the backend reports input_tokens=0, the client must see a real count."""
    body = _post_stream(_make_bedrock_backend(_bedrock_events(input_tokens=0)))
    client_input_tokens = _message_start_input_tokens(body)
    assert client_input_tokens > 0, (
        "message_start.usage.input_tokens reached the client as "
        f"{client_input_tokens}; expected the upstream-sent token count (#1132)."
    )


def test_bedrock_streaming_preserves_nonzero_upstream_input_tokens() -> None:
    """A genuine non-zero input_tokens from the backend must pass through untouched."""
    upstream_input_tokens = 777
    body = _post_stream(_make_bedrock_backend(_bedrock_events(input_tokens=upstream_input_tokens)))
    assert _message_start_input_tokens(body) == upstream_input_tokens

"""Regression test: LiteLLMBackend.stream_message must emit real input_tokens.

Issue #1132: When Headroom proxies Bedrock (via LiteLLM) in streaming mode,
`message_start` always carries `input_tokens: 0`. Downstream clients (Claude
Code, OTel dashboards) read `input_tokens` from `message_start`, so they
always report zero input tokens.

Root cause: `stream_message()` emits `message_start` with hardcoded
`{"input_tokens": 0, "output_tokens": 0}` before the stream starts.
Real token counts arrive only in the final LiteLLM chunk (when
`stream_options.include_usage=True` is set).

Fix: buffer all LiteLLM chunks, extract usage from the final chunk, then
emit `message_start` with real `input_tokens` / cache token fields.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeUsage:
    """Minimal litellm.Usage stand-in with only the attrs we set."""

    def __init__(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        if cache_read_input_tokens is not None:
            self.cache_read_input_tokens = cache_read_input_tokens
        if cache_creation_input_tokens is not None:
            self.cache_creation_input_tokens = cache_creation_input_tokens


def _text_chunk(text: str, finish_reason: str | None = None) -> MagicMock:
    """Build a minimal streaming chunk with text content."""
    delta = MagicMock()
    delta.content = text
    delta.tool_calls = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _usage_chunk(
    prompt_tokens: int,
    completion_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> MagicMock:
    """Build a final usage-only chunk (emitted by LiteLLM with include_usage=True)."""
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = _FakeUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cache_read_input_tokens=cache_read if cache_read else None,
        cache_creation_input_tokens=cache_creation if cache_creation else None,
    )
    return chunk


def _make_backend() -> LiteLLMBackend:
    with patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}):
        return LiteLLMBackend(provider="bedrock", region="us-east-1")


async def _collect_events(backend: LiteLLMBackend, body: dict[str, Any]) -> list[dict]:
    """Run stream_message and collect all StreamEvent.data dicts."""
    events = []
    async for event in backend.stream_message(body, {"x-api-key": "test"}):
        events.append({"type": event.event_type, "data": event.data})
    return events


def _body() -> dict[str, Any]:
    return {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
    }


def _fake_acompletion_stream(chunks: list[MagicMock]):
    """Return an AsyncMock that yields chunks as an async iterator."""

    async def _gen(*args, **kwargs):
        for c in chunks:
            yield c

    mock = MagicMock()
    mock.__aiter__ = _gen
    mock.__anext__ = MagicMock()

    # acompletion returns the async iterator directly when stream=True
    async def _acompletion(*args, **kwargs):
        return _gen()

    return _acompletion


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_message_emits_real_input_tokens_in_message_start() -> None:
    """message_start must carry real input_tokens from the final usage chunk.

    Before the fix, message_start always had input_tokens=0 regardless of
    what LiteLLM reported. This caused Claude Code / OTel to see 0 input
    tokens for every Bedrock streaming request.
    """
    chunks = [
        _text_chunk("Hello"),
        _text_chunk(" world", finish_reason="stop"),
        _usage_chunk(prompt_tokens=1000, completion_tokens=5),
    ]

    backend = _make_backend()
    with patch("headroom.backends.litellm.acompletion", new=_fake_acompletion_stream(chunks)):
        events = await _collect_events(backend, _body())

    msg_start = next(e for e in events if e["type"] == "message_start")
    usage = msg_start["data"]["message"]["usage"]
    assert usage["input_tokens"] == 1000, (
        f"Expected input_tokens=1000 in message_start, got {usage['input_tokens']}. "
        "message_start must carry real tokens from the LiteLLM final usage chunk."
    )


@pytest.mark.asyncio
async def test_stream_message_emits_real_output_tokens_in_message_delta() -> None:
    """message_delta must carry real output_tokens from the final usage chunk."""
    chunks = [
        _text_chunk("Hello"),
        _text_chunk(" world", finish_reason="stop"),
        _usage_chunk(prompt_tokens=100, completion_tokens=42),
    ]

    backend = _make_backend()
    with patch("headroom.backends.litellm.acompletion", new=_fake_acompletion_stream(chunks)):
        events = await _collect_events(backend, _body())

    msg_delta = next(e for e in events if e["type"] == "message_delta")
    output_tokens = msg_delta["data"]["usage"]["output_tokens"]
    assert output_tokens == 42, f"Expected output_tokens=42 in message_delta, got {output_tokens}."


@pytest.mark.asyncio
async def test_stream_message_emits_cache_tokens_in_message_start() -> None:
    """message_start must carry cache_read and cache_creation tokens from Bedrock usage."""
    chunks = [
        _text_chunk("Hi"),
        _text_chunk("!", finish_reason="stop"),
        _usage_chunk(
            prompt_tokens=2000,
            completion_tokens=10,
            cache_read=1500,
            cache_creation=300,
        ),
    ]

    backend = _make_backend()
    with patch("headroom.backends.litellm.acompletion", new=_fake_acompletion_stream(chunks)):
        events = await _collect_events(backend, _body())

    msg_start = next(e for e in events if e["type"] == "message_start")
    usage = msg_start["data"]["message"]["usage"]
    assert usage.get("cache_read_input_tokens") == 1500, (
        f"Expected cache_read_input_tokens=1500, got {usage.get('cache_read_input_tokens')}."
    )
    assert usage.get("cache_creation_input_tokens") == 300, (
        f"Expected cache_creation_input_tokens=300, got {usage.get('cache_creation_input_tokens')}."
    )


@pytest.mark.asyncio
async def test_stream_message_graceful_when_no_usage_chunk() -> None:
    """Graceful fallback: if no usage chunk is present, input_tokens stays 0.

    Some LiteLLM backends / older configs don't emit a usage chunk.
    The proxy must not crash; input_tokens=0 is acceptable fallback.
    """
    chunks = [
        _text_chunk("Hello"),
        _text_chunk("!", finish_reason="stop"),
    ]

    backend = _make_backend()
    with patch("headroom.backends.litellm.acompletion", new=_fake_acompletion_stream(chunks)):
        events = await _collect_events(backend, _body())

    msg_start = next(e for e in events if e["type"] == "message_start")
    usage = msg_start["data"]["message"]["usage"]
    # Should not crash; 0 is acceptable when backend doesn't provide usage
    assert isinstance(usage["input_tokens"], int)


@pytest.mark.asyncio
async def test_stream_message_event_order() -> None:
    """SSE events must arrive in Anthropic protocol order.

    Order: message_start → content_block_start → content_block_delta(s)
           → content_block_stop → message_delta → message_stop
    """
    chunks = [
        _text_chunk("Hello"),
        _text_chunk("!", finish_reason="stop"),
        _usage_chunk(prompt_tokens=50, completion_tokens=2),
    ]

    backend = _make_backend()
    with patch("headroom.backends.litellm.acompletion", new=_fake_acompletion_stream(chunks)):
        events = await _collect_events(backend, _body())

    types = [e["type"] for e in events]
    assert types[0] == "message_start", f"First event must be message_start, got {types}"
    assert types[-1] == "message_stop", f"Last event must be message_stop, got {types}"
    assert "message_delta" in types

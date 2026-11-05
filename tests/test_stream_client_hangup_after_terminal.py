"""A client that hangs up after the terminal event still got a complete stream.

Codex CLI closes the connection as soon as it reads ``response.completed``,
before the upstream EOF ends the proxy's read loop. The outcome was decided
only after that loop, so every such turn recorded as a 502 failure with no
usage and no cache metrics, although the client got a complete 200 response.
Success needs the terminal event the client was handed, not one only parsed
from upstream: the memory filter holds ``message_stop`` until after the stream.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from headroom.proxy.server import HeadroomProxy


def _event(name: str, payload: dict[str, Any]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()


START = _event("response.created", {"type": "response.created", "response": {"id": "r1"}})
COMPLETED = _event(
    "response.completed",
    {
        "type": "response.completed",
        "response": {
            "id": "r1",
            "status": "completed",
            "usage": {
                "input_tokens": 1_000,
                "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 0},
                "output_tokens": 5,
            },
        },
    },
)

# A whole Anthropic message in one upstream chunk: the parser sees message_stop
# before the client gets the first frame the memory filter splits it into.
MESSAGE = b"".join(
    [
        _event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "m1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "usage": {"input_tokens": 1_000, "output_tokens": 1},
                },
            },
        ),
        _event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        ),
        _event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
        ),
        _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
        ),
        _event("message_stop", {"type": "message_stop"}),
    ]
)
MEMORY_STREAM = {
    "url": "https://api.anthropic.com/v1/messages",
    "headers": {"x-api-key": "sk-test"},
    "body": {"model": "claude-test", "max_tokens": 100, "stream": True, "messages": []},
    "provider": "anthropic",
    "model": "claude-test",
    "memory_user_id": "user-1",
    "server_memory_tool_names": frozenset({"memory_save"}),
}


def _proxy(*chunks: bytes, upstream_eof: asyncio.Event | None = None) -> HeadroomProxy:
    proxy = object.__new__(HeadroomProxy)
    proxy.http_client = MagicMock(spec=httpx.AsyncClient)
    proxy._config = MagicMock()
    proxy._config.ccr_inject_tool = False
    proxy._config.retry_max_attempts = 1
    proxy.config = proxy._config
    proxy._record_request_outcome = AsyncMock()

    response = MagicMock()
    response.headers = httpx.Headers({"content-type": "text/event-stream"})
    response.status_code = 200

    async def aiter_bytes():
        for chunk in chunks or (START, COMPLETED):
            yield chunk
        if upstream_eof is not None:
            await upstream_eof.wait()  # the upstream EOF the client never waits for

    response.aiter_bytes = aiter_bytes
    response.aclose = AsyncMock()
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(return_value=response)
    return proxy


async def _stream(proxy: HeadroomProxy, **overrides: Any):
    arguments: dict[str, Any] = {
        "url": "https://chatgpt.com/backend-api/codex/responses",
        "headers": {"authorization": "Bearer test"},
        "body": {"model": "gpt-5.4", "stream": True, "input": "Reply with: ok"},
        "provider": "openai",
        "model": "gpt-5.4",
        "request_id": "test-hangup",
        "original_tokens": 10,
        "optimized_tokens": 10,
        "tokens_saved": 0,
        "transforms_applied": [],
        "tags": {},
        "optimization_latency": 0.0,
    }
    if overrides.get("memory_user_id"):
        proxy.memory_handler = MagicMock()
    response = await proxy._stream_response(**{**arguments, **overrides})
    return response.body_iterator


@pytest.mark.parametrize("hangup", ["closed_at_terminal_yield", "cancelled_awaiting_eof"])
async def test_client_hangup_after_terminal_event_records_success_with_usage(hangup):
    proxy = _proxy(upstream_eof=asyncio.Event())
    body = await _stream(proxy)
    seen = b""
    if hangup == "closed_at_terminal_yield":
        # The generator is suspended at the yield of the terminal chunk.
        while b"response.completed" not in seen:
            seen += await body.__anext__()
        await body.aclose()
    else:
        # The generator read past the terminal chunk and waits for EOF.
        async def read_all() -> None:
            nonlocal seen
            async for chunk in body:
                seen += chunk

        reader = asyncio.create_task(read_all())
        while b"response.completed" not in seen:
            await asyncio.sleep(0)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader

    proxy._record_request_outcome.assert_awaited_once()
    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.status_code == 200
    assert outcome.provider_input_tokens == 1_000
    assert outcome.cache_read_tokens == 600
    assert outcome.output_tokens == 5


async def test_client_hangup_before_terminal_event_is_still_a_failure():
    proxy = _proxy(upstream_eof=asyncio.Event())
    body = await _stream(proxy)
    assert b"response.created" in await body.__anext__()
    await body.aclose()

    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.status_code == 502


@pytest.mark.parametrize("frames_read", [1, 2, 3, 4, 5])
async def test_memory_filtered_hangup_before_the_client_gets_message_stop_is_a_failure(
    frames_read,
):
    """Frames 1-4 come from the one upstream chunk; frame 5 is the held message_delta."""
    proxy = _proxy(MESSAGE)
    body = await _stream(proxy, **MEMORY_STREAM)
    frames = [await body.__anext__() for _ in range(frames_read)]
    assert b"message_start" in frames[0]
    assert not any(b"message_stop" in frame for frame in frames)
    await body.aclose()

    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.status_code == 502


async def test_memory_filtered_hangup_after_message_stop_is_forwarded_records_success():
    proxy = _proxy(MESSAGE)
    body = await _stream(proxy, **MEMORY_STREAM)
    while b"message_stop" not in await body.__anext__():
        pass
    await body.aclose()

    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.status_code == 200
    assert outcome.provider_input_tokens == 1_000
    assert outcome.output_tokens == 5


async def test_malformed_event_after_a_valid_one_still_forwards_the_chunk():
    """Invalid UTF-8 ends the stream with an error, after the upstream bytes."""
    valid = _event("response.output_text.delta", {"type": "response.output_text.delta"})
    malformed = b'event: response.output_text.delta\ndata: {"delta": "\xff"}\n\n'
    proxy = _proxy(START, valid + malformed)
    body = await _stream(proxy)
    frames = [frame async for frame in body]

    assert frames[:2] == [START, valid + malformed]
    assert b"error" in frames[2]
    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.status_code == 502

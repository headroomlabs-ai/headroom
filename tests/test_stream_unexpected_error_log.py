"""An unexpected mid-stream error must name itself in the proxy log.

httpx transport errors raised while reading the upstream body (``ReadError``,
``RemoteProtocolError`` and friends) often carry an empty message. The
catch-all in ``_stream_response``'s generator logged only ``str(e)``, so the
operator saw a bare ``Unexpected streaming error:`` with nothing after it and
no way to tell a dropped connection from a bug.

Exception messages, including those of chained causes, can quote request or
response content, so they reach the log only with ``HEADROOM_DEBUG_DUMP=full``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from headroom.proxy.server import HeadroomProxy

MESSAGE_START = (
    "event: message_start\ndata: "
    + json.dumps(
        {
            "type": "message_start",
            "message": {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        }
    )
    + "\n\n"
).encode()


def _proxy(mid_stream_error: Exception | Callable[[], None]) -> HeadroomProxy:
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
        yield MESSAGE_START
        if callable(mid_stream_error):
            mid_stream_error()
        raise mid_stream_error

    response.aiter_bytes = aiter_bytes
    response.aclose = AsyncMock()
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(return_value=response)
    return proxy


async def _drain(proxy: HeadroomProxy) -> bytes:
    arguments: dict[str, Any] = {
        "url": "https://api.anthropic.com/v1/messages",
        "headers": {"x-api-key": "test"},
        "body": {"model": "claude-test", "stream": True, "messages": []},
        "provider": "anthropic",
        "model": "claude-test",
        "request_id": "test-unexpected",
        "original_tokens": 10,
        "optimized_tokens": 10,
        "tokens_saved": 0,
        "transforms_applied": [],
        "tags": {},
        "optimization_latency": 0.0,
    }
    response = await proxy._stream_response(**arguments)
    seen = b""
    async for chunk in response.body_iterator:
        seen += chunk if isinstance(chunk, bytes) else chunk.encode()
    return seen


def _unexpected_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if "Unexpected streaming error" in r.getMessage()
    ]


async def test_mid_stream_error_with_empty_message_logs_its_type(caplog):
    proxy = _proxy(httpx.ReadError(""))

    with caplog.at_level(logging.ERROR, logger="headroom.proxy"):
        body = await _drain(proxy)

    assert b"event: error" in body  # the client still gets a clean SSE error
    lines = _unexpected_lines(caplog)
    assert len(lines) == 1
    assert "ReadError" in lines[0]


@pytest.fixture
def _no_content_logging(monkeypatch):
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)


@pytest.mark.usefixtures("_no_content_logging")
async def test_mid_stream_error_message_stays_out_of_the_log_by_default(caplog):
    proxy = _proxy(RuntimeError("parser exploded"))

    with caplog.at_level(logging.ERROR, logger="headroom.proxy"):
        await _drain(proxy)

    lines = _unexpected_lines(caplog)
    assert len(lines) == 1
    assert "RuntimeError" in lines[0]
    assert "parser exploded" not in lines[0]


async def test_mid_stream_error_message_is_logged_with_full_debug_dump(caplog, monkeypatch):
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")
    proxy = _proxy(RuntimeError("parser exploded"))

    with caplog.at_level(logging.ERROR, logger="headroom.proxy"):
        await _drain(proxy)

    lines = _unexpected_lines(caplog)
    assert len(lines) == 1
    assert "parser exploded" in lines[0]


@pytest.mark.usefixtures("_no_content_logging")
async def test_chained_inner_message_stays_out_of_every_log_record(caplog):
    """A cause's message must not leak through a traceback, at any level."""
    secret = "synthetic-private-inner-exception-message"

    def _raise_chained() -> None:
        raise RuntimeError("parser exploded") from ValueError(secret)

    proxy = _proxy(_raise_chained)

    with caplog.at_level(logging.DEBUG), caplog.at_level(logging.DEBUG, logger="headroom"):
        body = await _drain(proxy)

    assert b"event: error" in body
    assert secret not in caplog.text
    assert "parser exploded" not in caplog.text
    lines = _unexpected_lines(caplog)
    assert len(lines) == 1
    assert "RuntimeError" in lines[0]
    assert "caused by ValueError" in lines[0]

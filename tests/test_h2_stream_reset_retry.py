"""HTTP/2 stream-reset resilience (issue #1639).

Under concurrent load a single upstream HTTP/2 stream reset poisons the shared
h2 connection and surfaces as `RemoteProtocolError` / `LocalProtocolError` on
every in-flight request. Those are transport errors, so the proxy must retry
them (dropping the bad connection and re-sending on a fresh one) instead of
collapsing to a 502. These tests drive the real `_retry_request` and
`_stream_response` paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from headroom.proxy.server import HeadroomProxy


def _mock_proxy():
    proxy = object.__new__(HeadroomProxy)
    proxy.http_client = MagicMock(spec=httpx.AsyncClient)
    proxy._config = MagicMock()
    proxy._config.memory_enabled = False
    proxy._config.ccr_inject_tool = False
    proxy._config.retry_enabled = True
    proxy._config.retry_max_attempts = 2
    proxy._config.retry_base_delay_ms = 0
    proxy._config.retry_max_delay_ms = 0
    proxy.config = proxy._config
    proxy.memory_handler = None
    proxy._parse_sse_usage_from_buffer = MagicMock(return_value=None)
    proxy._finalize_stream_response = AsyncMock(return_value=None)
    return proxy


def _good_stream_response(chunks):
    resp = AsyncMock()
    resp.headers = httpx.Headers({"content-type": "text/event-stream"})
    resp.status_code = 200

    async def aiter_bytes():
        for chunk in chunks:
            yield chunk

    resp.aiter_bytes = aiter_bytes
    resp.aclose = AsyncMock()
    return resp


async def _run_stream(proxy, session_key="k"):
    return await proxy._stream_response(
        url="https://api.anthropic.com/v1/messages",
        headers={"x-api-key": "sk-test"},
        body={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        request_id="test-1639",
        original_tokens=10,
        optimized_tokens=10,
        tokens_saved=0,
        transforms_applied=[],
        tags={},
        optimization_latency=0.0,
        session_key=session_key,
    )


@pytest.mark.asyncio
async def test_retry_request_retries_remote_protocol_error():
    proxy = _mock_proxy()
    good = MagicMock()
    good.status_code = 200
    good.request = MagicMock()
    proxy.http_client.post = AsyncMock(
        side_effect=[httpx.RemoteProtocolError("<StreamReset stream_id:35>"), good]
    )

    result = await proxy._retry_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": "sk-test"},
        {"model": "claude-sonnet-4-20250514", "messages": []},
    )

    assert result is good
    assert proxy.http_client.post.await_count == 2


@pytest.mark.asyncio
async def test_retry_request_reraises_after_exhaustion():
    proxy = _mock_proxy()
    proxy.http_client.post = AsyncMock(side_effect=httpx.RemoteProtocolError("reset"))

    with pytest.raises(httpx.RemoteProtocolError):
        await proxy._retry_request(
            "POST",
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": "sk-test"},
            {"model": "claude-sonnet-4-20250514", "messages": []},
        )
    assert proxy.http_client.post.await_count == 2


@pytest.mark.asyncio
async def test_stream_retries_h2_stream_reset_then_succeeds():
    proxy = _mock_proxy()
    good = _good_stream_response(
        [
            b'event: message_start\ndata: {"type":"message_start"}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
    )
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(
        side_effect=[httpx.RemoteProtocolError("<StreamReset stream_id:35>"), good]
    )

    result = await _run_stream(proxy)
    body = b"".join([chunk async for chunk in result.body_iterator])

    assert proxy.http_client.send.await_count == 2
    assert b"message_start" in body
    assert b"connection_error" not in body


@pytest.mark.asyncio
async def test_stream_reset_exhaustion_yields_sse_error_not_crash():
    proxy = _mock_proxy()
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(side_effect=httpx.RemoteProtocolError("reset"))

    result = await _run_stream(proxy)
    body = b"".join([chunk async for chunk in result.body_iterator])

    assert proxy.http_client.send.await_count == 2
    assert b"event: error" in body
    assert b"connection_error" in body

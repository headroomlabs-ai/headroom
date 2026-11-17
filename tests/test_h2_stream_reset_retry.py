"""HTTP/2 stream-reset resilience (issue #1639).

Under concurrent load a single upstream HTTP/2 stream reset poisons the shared
h2 connection and surfaces as `RemoteProtocolError` / `LocalProtocolError` on
every in-flight request. Those are transport errors, so the proxy must retry
them (dropping the bad connection and re-sending on a fresh one) instead of
collapsing to a 502. These tests drive the real `_retry_request` and
`_stream_response` paths.

A corrupt TLS record surfaces as a raw `ssl.SSLError`, not an
`httpx.TransportError`. It is retried only when it hit while the request was
still being written. After the write, upstream may already have accepted and
billed the POST, so the error propagates and the POST is sent once.
"""

from __future__ import annotations

import ssl
from unittest.mock import AsyncMock, MagicMock

import httpcore
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
    proxy.metrics = MagicMock()
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


@pytest.mark.asyncio
async def test_stream_reset_exhaustion_status_is_not_200():
    """The synthesized error response must not claim success.

    A 200 here is indistinguishable, to every Anthropic/OpenAI SDK, from a
    successful stream that produced no events — the client reports "empty or
    malformed response (HTTP 200)" and cannot retry, because 200 is not a
    retryable status. This asserts the status line specifically: the body
    assertions above passed for the entire time the status was 200.
    """
    proxy = _mock_proxy()
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(side_effect=httpx.RemoteProtocolError("reset"))

    result = await _run_stream(proxy)

    assert result.status_code == 502
    proxy.metrics.record_upstream_connection_error.assert_called_once()


@pytest.mark.asyncio
async def test_successful_stream_still_returns_200():
    """Guard the happy path against the 502 change above."""
    proxy = _mock_proxy()
    good = _good_stream_response([b'event: message_stop\ndata: {"type":"message_stop"}\n\n'])
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(return_value=good)

    result = await _run_stream(proxy)

    assert result.status_code == 200
    proxy.metrics.record_upstream_connection_error.assert_not_called()


def _bad_record() -> ssl.SSLError:
    # What httpx raises on a connection whose next TLS record is corrupt.
    return ssl.SSLError(1, "[SSL] record layer failure (_ssl.c:2580)")


_SSE_STOP = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
_SSE_OK = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: %d\r\n\r\n%s" % (
    len(_SSE_STOP),
    _SSE_STOP,
)


class _Upstream(httpcore.AsyncNetworkBackend):
    """Real httpcore over fake sockets: one failure mode per connection.

    ``"write"``: the TLS error hits while the request is written (never sent).
    ``"read"``: upstream receives the whole POST, then the TLS error hits.
    ``None``: a healthy connection that answers with an SSE stream.
    """

    def __init__(self, *modes: str | None) -> None:
        self.modes = list(modes)
        self.received = b""

    @property
    def posts(self) -> int:
        return self.received.count(b"POST /v1/messages")

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        return _UpstreamStream(self, self.modes.pop(0))


class _UpstreamStream(httpcore.AsyncNetworkStream):
    def __init__(self, upstream: _Upstream, mode: str | None) -> None:
        self.upstream, self.mode, self.replied = upstream, mode, False

    async def write(self, buffer, timeout=None):
        if self.mode == "write":
            raise _bad_record()
        self.upstream.received += buffer

    async def read(self, max_bytes, timeout=None):
        if self.mode == "read":
            raise _bad_record()
        reply, self.replied = (b"" if self.replied else _SSE_OK), True
        return reply

    async def aclose(self):
        pass

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        return self


def _proxy_over(upstream: _Upstream):
    proxy = _mock_proxy()
    transport = httpx.AsyncHTTPTransport()
    # Same hook upstream_pinning uses: swap the pool's network backend.
    transport._pool._network_backend = upstream
    proxy.http_client = httpx.AsyncClient(transport=transport)
    return proxy


async def _retry_post(proxy):
    return await proxy._retry_request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": "sk-test"},
        {"model": "claude-sonnet-4-20250514", "messages": []},
    )


async def test_retry_request_does_not_replay_a_post_upstream_received():
    upstream = _Upstream("read", None)

    with pytest.raises(ssl.SSLError):
        await _retry_post(_proxy_over(upstream))

    assert upstream.posts == 1


async def test_retry_request_retries_ssl_error_while_writing_the_request():
    upstream = _Upstream("write", None)

    result = await _retry_post(_proxy_over(upstream))

    assert result.status_code == 200
    assert upstream.posts == 1
    assert upstream.modes == []


async def test_stream_does_not_replay_a_post_upstream_received():
    upstream = _Upstream("read", None)

    result = await _run_stream(_proxy_over(upstream))
    body = b"".join([chunk async for chunk in result.body_iterator])

    assert upstream.posts == 1
    assert result.status_code == 502
    assert b"upstream_tls_error" in body


async def test_stream_retries_ssl_error_while_writing_the_request():
    upstream = _Upstream("write", None)

    result = await _run_stream(_proxy_over(upstream))
    body = b"".join([chunk async for chunk in result.body_iterator])

    assert result.status_code == 200
    assert _SSE_STOP in body
    assert upstream.posts == 1
    assert upstream.modes == []


async def test_ssl_error_without_write_evidence_is_not_retried():
    # The review's harness: the transport accepts the POST, then raises. It
    # emits no httpcore trace events, so nothing shows the POST went unsent.
    seen: list[httpx.Request] = []

    def accept_then_fail(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise _bad_record()

    proxy = _mock_proxy()
    proxy.http_client = httpx.AsyncClient(transport=httpx.MockTransport(accept_then_fail))

    with pytest.raises(ssl.SSLError):
        await _retry_post(proxy)
    assert len(seen) == 1

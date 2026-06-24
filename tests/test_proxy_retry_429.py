"""Upstream 429 rate-limit retry + Retry-After honoring (fixes #1221).

Both the non-streaming (``server.py:_retry_request``) and streaming
(``streaming.py:_stream_response``) forwarders must retry an upstream 429 with
backoff instead of passing it straight back to the client, since a parallel
agent fan-out that exceeds the per-minute limit otherwise aborts every run.
"""

from __future__ import annotations

import asyncio

import httpx

from headroom.proxy.server import ProxyConfig, create_app


class _RateLimitTransport(httpx.AsyncBaseTransport):
    """Returns ``fail_status`` for the first ``fail_times`` calls, then 200.

    Records ``calls`` so a test can assert whether a retry happened.
    """

    def __init__(
        self,
        *,
        fail_status: int = 429,
        fail_times: int = 1,
        retry_after: str | None = None,
        sse: bool = False,
    ) -> None:
        self.fail_status = fail_status
        self.fail_times = fail_times
        self.retry_after = retry_after
        self.sse = sse
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        async for _ in request.stream:  # drain the request body
            pass
        if self.calls <= self.fail_times:
            headers = {"retry-after": self.retry_after} if self.retry_after is not None else {}
            return httpx.Response(
                self.fail_status,
                headers=headers,
                json={"type": "error", "error": {"type": "rate_limit_error"}},
            )
        if self.sse:
            body = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )


def _proxy_with(transport: _RateLimitTransport, *, max_attempts: int = 3):
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        retry_enabled=True,
        retry_max_attempts=max_attempts,
        retry_base_delay_ms=1,
        retry_max_delay_ms=5000,
    )
    proxy = create_app(config).state.proxy
    proxy.http_client = httpx.AsyncClient(transport=transport)
    return proxy


# --- non-streaming: _retry_request ---------------------------------------


def test_retry_request_retries_429_then_succeeds() -> None:
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after="0")
    proxy = _proxy_with(transport)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 200
    assert transport.calls == 2  # one 429 + one success — the retry happened


def test_retry_request_returns_429_verbatim_on_exhaustion() -> None:
    # Always 429: must return the 429 to the client, NOT raise / convert to 5xx.
    transport = _RateLimitTransport(fail_status=429, fail_times=99, retry_after="0")
    proxy = _proxy_with(transport, max_attempts=3)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 429
    assert transport.calls == 3  # exhausted all attempts


def test_retry_request_honors_retry_after(monkeypatch) -> None:
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("headroom.proxy.server.asyncio.sleep", _fake_sleep)
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after="2")
    proxy = _proxy_with(transport)
    asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    # Retry-After: 2s honored (not the ~1-10ms jittered exponential backoff).
    assert slept and abs(slept[0] - 2.0) < 0.01


def test_retry_request_does_not_retry_other_4xx() -> None:
    transport = _RateLimitTransport(fail_status=400, fail_times=99)
    proxy = _proxy_with(transport)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 400
    assert transport.calls == 1  # 4xx (non-429) still short-circuits — no retry


def test_retry_request_still_retries_5xx() -> None:
    transport = _RateLimitTransport(fail_status=503, fail_times=1)
    proxy = _proxy_with(transport)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 200
    assert transport.calls == 2  # 5xx retry path unchanged


# --- streaming: _stream_response -----------------------------------------


def test_stream_response_retries_429() -> None:
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after="0", sse=True)
    proxy = _proxy_with(transport)
    asyncio.run(
        proxy._stream_response(
            "https://up/v1/messages",
            {},
            {"messages": []},
            "anthropic",
            "claude-3",
            "r1",
            0,
            0,
            0,
            [],
            {},
            0.0,
        )
    )
    assert transport.calls == 2  # streaming 429 retried, not forwarded raw

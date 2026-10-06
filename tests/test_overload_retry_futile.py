"""A 429/529 must not be retried when the retry cannot succeed.

Anthropic answers an exhausted subscription or a hard rate limit with a 429 whose
Retry-After points minutes or hours ahead, and marks non-retryable errors with
``x-should-retry: false``. Retrying those after the capped backoff only holds the
client for ``retry_max_attempts * retry_max_delay_ms`` before returning the same
429, and every proxy layer chained in front multiplies that wait.
"""

import asyncio
import types

import httpx

from headroom.proxy.helpers import overload_retry_is_futile
from headroom.proxy.server import HeadroomProxy


def _resp(status, headers=None):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(
        status_code=status, request=req, headers=headers or {}, content=b'{"type":"error"}'
    )


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    async def post(self, url, content=None, headers=None):
        self.calls += 1
        return self._response


def _call(response, *, retry_max_delay_ms=30_000):
    proxy = HeadroomProxy.__new__(HeadroomProxy)
    client = _FakeClient(response)
    proxy.http_client = client
    proxy.config = types.SimpleNamespace(
        retry_enabled=True,
        retry_max_attempts=3,
        retry_base_delay_ms=1,
        retry_max_delay_ms=retry_max_delay_ms,
    )
    out = asyncio.run(
        proxy._retry_request("POST", "https://api.anthropic.com/v1/messages", {}, {}, stream=False)
    )
    return out, client.calls


def test_retry_after_beyond_cap_is_futile():
    assert overload_retry_is_futile(_resp(429, {"retry-after": "3600"}), 30_000)


def test_retry_after_within_cap_is_not_futile():
    assert not overload_retry_is_futile(_resp(429, {"retry-after": "5"}), 30_000)


def test_should_retry_false_is_futile():
    assert overload_retry_is_futile(_resp(429, {"x-should-retry": "false"}), 30_000)


def test_no_hint_is_not_futile():
    assert not overload_retry_is_futile(_resp(529), 30_000)


def test_long_retry_after_429_returned_without_retry():
    out, calls = _call(_resp(429, {"retry-after": "3600"}))
    assert out.status_code == 429
    assert out.headers["retry-after"] == "3600"
    assert calls == 1


def test_should_retry_false_429_returned_without_retry():
    out, calls = _call(_resp(429, {"x-should-retry": "false"}))
    assert out.status_code == 429
    assert calls == 1


def test_short_retry_after_429_still_retried():
    out, calls = _call(_resp(429, {"retry-after": "0"}))
    assert out.status_code == 429
    assert calls == 3

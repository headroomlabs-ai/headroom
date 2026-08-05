"""Retry-exhaustion should preserve the upstream 5xx status (500/502/503/504)
instead of collapsing every exhausted 5xx into a generic 502.

When upstream keeps returning a server error, the proxy used to retry, then
mask the final failure as a 502, which hides the real status from the client.
It should now surface the real 5xx so the client can apply its own backoff.

The 429/529 overload statuses take a separate path (RETRYABLE_OVERLOAD_STATUSES,
returned verbatim by #1495's branch) and never reach the exhaustion handler
exercised here, so these tests use non-overload 5xx codes to hit it.
"""

import asyncio
import types

import httpx
import pytest

from headroom.proxy.server import HeadroomProxy


def _make_proxy(http_client, *, retry_max_attempts=3, retry_enabled=True):
    proxy = HeadroomProxy.__new__(HeadroomProxy)
    proxy.http_client = http_client
    proxy.config = types.SimpleNamespace(
        retry_enabled=retry_enabled,
        retry_max_attempts=retry_max_attempts,
        retry_base_delay_ms=1,
        retry_max_delay_ms=2,
    )
    return proxy


class _FakeClient:
    def __init__(self, *, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = 0

    async def post(self, url, content=None, headers=None):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._response


def _resp(status, body=b'{"type":"error"}'):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status_code=status, request=req, content=body)


def _call(proxy):
    return asyncio.run(
        proxy._retry_request(
            "POST",
            "https://api.anthropic.com/v1/messages",
            {},
            {},
            stream=False,
        )
    )


def test_exhausted_500_preserves_status():
    client = _FakeClient(response=_resp(500))
    out = _call(_make_proxy(client, retry_max_attempts=3))
    assert out.status_code == 500  # not collapsed to 502
    assert client.calls == 3  # retried up to the cap


def test_exhausted_503_preserves_status():
    client = _FakeClient(response=_resp(503))
    out = _call(_make_proxy(client, retry_max_attempts=2))
    assert out.status_code == 503
    assert client.calls == 2


def test_exhausted_5xx_preserves_body():
    client = _FakeClient(
        response=_resp(503, body=b'{"type":"error","error":{"type":"overloaded_error"}}')
    )
    out = _call(_make_proxy(client, retry_max_attempts=2))
    assert out.status_code == 503
    assert out.json()["error"]["type"] == "overloaded_error"  # body survives, not just status


def test_retry_disabled_returns_5xx_on_first_attempt():
    # Use a non-overload 5xx (503) so this exercises the HTTPStatusError
    # exhaustion branch rather than the 429/529 overload fast path.
    client = _FakeClient(response=_resp(503))
    out = _call(_make_proxy(client, retry_enabled=False))
    assert out.status_code == 503
    assert client.calls == 1  # no retry, but status still preserved instead of raised


def test_4xx_returned_without_retry():
    client = _FakeClient(response=_resp(400))
    out = _call(_make_proxy(client))
    assert out.status_code == 400
    assert client.calls == 1  # client errors are not retried


def test_connect_error_still_raises():
    client = _FakeClient(exc=httpx.ConnectError("boom"))
    with pytest.raises(httpx.ConnectError):
        _call(_make_proxy(client))


def test_success_passes_through():
    client = _FakeClient(response=_resp(200))
    out = _call(_make_proxy(client))
    assert out.status_code == 200
    assert client.calls == 1

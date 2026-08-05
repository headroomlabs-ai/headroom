"""Regression tests for GH #1112.

The Headroom proxy returned an opaque HTTP 502 when an OpenAI-compatible
upstream closed a pooled keep-alive connection mid-response, surfacing
``httpx.RemoteProtocolError`` ("peer closed connection without sending
complete message body (incomplete chunked read)"). The same upstream answers
a direct ``curl`` with 200 because curl opens a fresh connection per call;
Headroom reuses pooled connections, so the first request on a stale connection
fails even though the upstream is healthy.

The fix adds :func:`headroom.proxy.helpers.request_with_transient_retry`,
which retries the buffered request once on a fresh connection, and wires it
into ``OpenAIHandlerMixin.handle_passthrough`` with a clean 502 fallback when
the protocol error persists.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.proxy.helpers import request_with_transient_retry

_INCOMPLETE_CHUNKED = (
    "peer closed connection without sending complete message body (incomplete chunked read)"
)


def _ok_response() -> httpx.Response:
    request = httpx.Request("GET", "https://api.openai.com/v1/models")
    return httpx.Response(
        200,
        request=request,
        headers={"content-type": "application/json"},
        json={"object": "list", "data": []},
    )


# ---------------------------------------------------------------------------
# Unit tests for the retry helper
# ---------------------------------------------------------------------------


def test_helper_returns_response_without_retry_on_success() -> None:
    ok = _ok_response()
    client = SimpleNamespace(request=AsyncMock(return_value=ok))

    result = asyncio.run(
        request_with_transient_retry(client, method="GET", url="https://up/v1/models")
    )

    assert result is ok
    assert client.request.await_count == 1


def test_helper_recovers_after_one_remote_protocol_error() -> None:
    ok = _ok_response()
    client = SimpleNamespace(
        request=AsyncMock(side_effect=[httpx.RemoteProtocolError(_INCOMPLETE_CHUNKED), ok])
    )

    result = asyncio.run(
        request_with_transient_retry(client, method="GET", url="https://up/v1/models")
    )

    assert result is ok
    # one initial attempt + one retry on a fresh connection
    assert client.request.await_count == 2


def test_helper_reraises_persistent_remote_protocol_error() -> None:
    client = SimpleNamespace(
        request=AsyncMock(side_effect=httpx.RemoteProtocolError(_INCOMPLETE_CHUNKED))
    )

    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(request_with_transient_retry(client, method="GET", url="https://up/v1/models"))

    # default max_retries=1 → exactly 2 attempts before giving up
    assert client.request.await_count == 2


def test_helper_does_not_retry_other_errors() -> None:
    client = SimpleNamespace(
        request=AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    )

    with pytest.raises(httpx.ConnectError):
        asyncio.run(request_with_transient_retry(client, method="GET", url="https://up/v1/models"))

    # ConnectError is not a transient keep-alive close; no retry
    assert client.request.await_count == 1


def test_helper_respects_max_retries() -> None:
    ok = _ok_response()
    client = SimpleNamespace(
        request=AsyncMock(
            side_effect=[
                httpx.RemoteProtocolError(_INCOMPLETE_CHUNKED),
                httpx.RemoteProtocolError(_INCOMPLETE_CHUNKED),
                ok,
            ]
        )
    )

    result = asyncio.run(
        request_with_transient_retry(
            client, method="GET", url="https://up/v1/models", max_retries=2
        )
    )

    assert result is ok
    assert client.request.await_count == 3


# ---------------------------------------------------------------------------
# Handler-level tests: the exact path from the issue traceback
# (proxy_routes.list_models → OpenAIHandlerMixin.handle_passthrough)
# ---------------------------------------------------------------------------


class _PassthroughModelsRequest:
    method = "GET"
    headers: dict[str, str] = {}
    url = SimpleNamespace(path="/v1/models", query="")

    async def body(self) -> bytes:
        return b""


class _FlakyThenOkClient:
    """Raises RemoteProtocolError on the first request, then succeeds —
    a stale pooled keep-alive connection followed by a fresh one."""

    def __init__(self) -> None:
        self.calls = 0

    async def request(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls += 1
        if self.calls == 1:
            raise httpx.RemoteProtocolError(_INCOMPLETE_CHUNKED)
        request = httpx.Request(kwargs["method"], kwargs["url"])
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            json={"object": "list", "data": []},
        )


class _AlwaysProtocolErrorClient:
    def __init__(self) -> None:
        self.calls = 0

    async def request(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls += 1
        raise httpx.RemoteProtocolError(_INCOMPLETE_CHUNKED)


def test_passthrough_recovers_from_incomplete_chunked_read() -> None:
    handler = object.__new__(OpenAIHandlerMixin)
    client = _FlakyThenOkClient()
    handler.http_client = client
    handler.http_client_h1 = client

    response = asyncio.run(
        handler.handle_passthrough(_PassthroughModelsRequest(), "https://api.openai.com")
    )

    assert response.status_code == 200
    assert client.calls == 2
    assert json.loads(response.body) == {"object": "list", "data": []}


def test_passthrough_returns_clean_502_on_persistent_protocol_error() -> None:
    handler = object.__new__(OpenAIHandlerMixin)
    client = _AlwaysProtocolErrorClient()
    handler.http_client = client
    handler.http_client_h1 = client

    response = asyncio.run(
        handler.handle_passthrough(_PassthroughModelsRequest(), "https://api.openai.com")
    )

    assert response.status_code == 502
    payload = json.loads(response.body)
    assert payload["error"]["type"] == "upstream_protocol_error"
    assert "complete response" in payload["error"]["message"]
    # initial attempt + one retry, then the clean 502
    assert client.calls == 2

"""Provider clients must not forward the client's hop-by-hop headers.

Handlers copy the inbound request headers, so a client that sent
`Connection: close` (curl, or a local proxy that wants one request per
connection) had it forwarded to the provider, which then closed the upstream
connection after every response. The keep-alive pool never reused a
connection: each request paid a fresh DNS lookup, TCP connect and TLS
handshake, and was exposed to every DNS or handshake hiccup on the way.
"""

from __future__ import annotations

import httpx

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import _provider_httpx_client_options


async def _sent_headers(headers: dict[str, str], content: bytes = b"{}") -> httpx.Headers:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200)

    _http2, kwargs = _provider_httpx_client_options(ProxyConfig(), verify=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs) as client:
        await client.post("https://api.example.com/v1/messages", headers=headers, content=content)
    return seen[0]


async def test_connection_close_and_other_hop_by_hop_headers_stay_local() -> None:
    sent = await _sent_headers(
        {
            "Connection": "close",
            "Keep-Alive": "timeout=5",
            "Proxy-Connection": "keep-alive",
            "TE": "trailers",
            "Trailer": "x-checksum",
            "Upgrade": "websocket",
            "x-api-key": "k",
        }
    )

    for name in ("connection", "keep-alive", "proxy-connection", "te", "trailer", "upgrade"):
        assert name not in sent, name
    assert sent["x-api-key"] == "k"


async def test_headers_named_by_connection_are_dropped_but_framing_is_kept() -> None:
    sent = await _sent_headers(
        {"Connection": "x-session-hint, content-length", "x-session-hint": "1"},
        content=b'{"a":1}',
    )

    assert "x-session-hint" not in sent
    assert sent["content-length"] == "7"
    assert sent["host"] == "api.example.com"


async def test_ordinary_requests_are_unchanged() -> None:
    sent = await _sent_headers({"anthropic-version": "2023-06-01", "x-api-key": "k"})

    assert sent["anthropic-version"] == "2023-06-01"
    assert sent["x-api-key"] == "k"

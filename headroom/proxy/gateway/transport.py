"""Gateway-owned TLS and connection pinning, independent of legacy TLS settings."""

from __future__ import annotations

import logging
import ssl
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import httpx
import websockets

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.egress import AuthorizedDestination
from headroom.proxy.gateway.errors import GatewayEgressDenied

_SENSITIVE_TRANSPORT = ContextVar("gateway_sensitive_transport", default=False)


class _TransportLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _SENSITIVE_TRANSPORT.get()


@contextmanager
def private_transport() -> Iterator[None]:
    # Filters are task-local in effect; legacy traffic retains its logging behavior.
    for name in (
        "httpx",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.connection",
        "httpcore.proxy",
        "httpcore.socks",
    ):
        logger = logging.getLogger(name)
        if not any(isinstance(item, _TransportLogFilter) for item in logger.filters):
            logger.addFilter(_TransportLogFilter())
    token = _SENSITIVE_TRANSPORT.set(True)
    try:
        yield
    finally:
        _SENSITIVE_TRANSPORT.reset(token)


class _PrivateStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        iterator = self._stream.__aiter__()
        while True:
            with private_transport():
                try:
                    chunk = await anext(iterator)
                except StopAsyncIteration:
                    return
            yield chunk

    async def aclose(self) -> None:
        with private_transport():
            await self._stream.aclose()


def tls_context(snapshot: GatewayConfigSnapshot) -> ssl.SSLContext:
    try:
        context = ssl.create_default_context(cafile=snapshot.transport.ca_bundle)
    except (OSError, ValueError):
        raise ValueError("Invalid gateway CA bundle") from None
    if context.cert_store_stats()["x509_ca"] == 0:
        raise ValueError("Gateway CA bundle contains no CA certificates")
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


class PinnedHTTPTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        destination = request.extensions.get("gateway_destination")
        if not isinstance(destination, AuthorizedDestination) or (
            request.url.host != destination.hostname
            or (request.url.port or 443) != destination.port
            or str(request.url) != str(httpx.URL(destination.url))
        ):
            raise GatewayEgressDenied(
                status_code=502,
                code="gateway_egress_denied",
                message="Upstream destination is not authorized for this credential",
            )
        original_url = request.url
        request.url = request.url.copy_with(host=destination.addresses[0])
        request.extensions["sni_hostname"] = destination.hostname
        try:
            with private_transport():
                response = await super().handle_async_request(request)
            if isinstance(response.stream, httpx.AsyncByteStream):
                response.stream = _PrivateStream(response.stream)
            return response
        finally:
            request.url = original_url


def http_client(snapshot: GatewayConfigSnapshot) -> httpx.AsyncClient:
    # Avoid pooling different hostname/TLS audiences that share a pinned IP.
    transport = PinnedHTTPTransport(
        verify=tls_context(snapshot),
        trust_env=False,
        limits=httpx.Limits(max_keepalive_connections=0),
    )
    return httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False)


def websocket_connection(
    destination: AuthorizedDestination, headers: dict[str, str], context: ssl.SSLContext
) -> Any:
    connector: Any = websockets.connect(
        "wss://" + destination.url.removeprefix("https://"),
        additional_headers=headers,
        max_size=1_048_576,
        ping_interval=20,
        ssl=context,
        server_hostname=destination.hostname,
        host=destination.addresses[0],
        port=destination.port,
        proxy=None,
        logger=logging.Logger("headroom.gateway.websocket", level=logging.CRITICAL + 1),
    )
    # websockets follows redirects by default, including on authenticated handshakes.
    connector.process_redirect = lambda exception: exception
    return connector

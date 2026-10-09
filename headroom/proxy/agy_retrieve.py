"""In-process hypercorn PLAIN-HTTP retrieve server for agy.

The proxy compresses tool_result payloads and emits ``[Retrieve more:
hash=…]`` markers.  For agy those markers are produced on the decrypted
stream inside the HTTPS dispatch server (:mod:`headroom.proxy.agy_dispatch`).
To resolve a marker the agent runs the ``headroom mcp serve`` stdio child,
which calls the proxy's retrieve HTTP endpoint via ``HEADROOM_PROXY_URL``.

The dispatch server is HTTPS with a Cloud-Code-SNI leaf only, so a stdio
retrieve child cannot reach it over loopback.  This module stands up a
SECOND loopback listener — PLAIN HTTP, no TLS — serving the same FastAPI
app on an ephemeral port for the session.  The compression/marker cache is
a process-global singleton (:func:`headroom.cache.compression_store.get_compression_store`),
so this second ``create_app()`` shares the exact cache the dispatch server
populates: a marker minted on the HTTPS side resolves over plain HTTP here.

Why plain HTTP is safe: the listener binds ``127.0.0.1`` only, serves the
retrieve endpoints to a stdio child in the *same* trust boundary, and never
carries upstream credentials (it only reads the in-memory marker cache).

The hypercorn plumbing (lifespan, TCPServer, socket options, lifecycle) is
:class:`headroom.proxy.agy_dispatch.AgyDispatchServer`'s — this listener is
that same server in its ``plain_http`` mode: no SSL context, no CA touched,
A separate outer guard requires a loopback peer and Host and exposes only
retrieval GET/POST routes.
"""

from __future__ import annotations

from typing import Any

from headroom.proxy.agy_dispatch import AgyDispatchServer
from headroom.proxy.loopback_guard import is_loopback_host, is_loopback_host_header


def make_retrieve_guard(app: Any) -> Any:
    """Expose only retrieval operations to loopback peers and authorities."""

    async def guarded(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope.get("type") == "http":
            client = scope.get("client")
            peer = client[0] if isinstance(client, (tuple, list)) and client else None
            hosts = [value for name, value in scope.get("headers", ()) if name.lower() == b"host"]
            browser_cross_origin = any(
                name.lower() == b"origin"
                or (
                    name.lower() == b"sec-fetch-site"
                    and value.lower() not in {b"none", b"same-origin"}
                )
                for name, value in scope.get("headers", ())
            )
            local = (
                is_loopback_host(peer)
                and len(hosts) == 1
                and is_loopback_host_header(hosts[0].decode("latin-1"))
                and not browser_cross_origin
            )
            path = scope.get("path", "")
            method = scope.get("method", "")
            suffix = path.removeprefix("/v1/retrieve/")
            retrieve_get = (
                method == "GET"
                and path.startswith("/v1/retrieve/")
                and bool(suffix)
                and "/" not in suffix
            )
            retrieve_post = method == "POST" and path in {"/v1/retrieve", "/v1/retrieve/tool_call"}
            if not local or not (retrieve_get or retrieve_post):
                body = b"Not Found"
                await send(
                    {
                        "type": "http.response.start",
                        "status": 404,
                        "headers": [
                            (b"content-type", b"text/plain"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await app(scope, receive, send)

    return guarded


class AgyRetrieveServer(AgyDispatchServer):
    """PLAIN-HTTP loopback listener serving the headroom FastAPI app.

    Serves the process-global compression cache via ``create_app()`` so
    ``GET /v1/retrieve/{hash}`` resolves markers the HTTPS dispatch server
    populated.

    Usage::

        server = AgyRetrieveServer()
        await server.start()
        # server.address → ("127.0.0.1", <ephemeral-port>)
        await server.stop()

    Or as an async context manager::

        async with AgyRetrieveServer() as srv:
            host, port = srv.address
    """

    def __init__(self, port: int = 0) -> None:
        super().__init__(port=port, plain_http=True)

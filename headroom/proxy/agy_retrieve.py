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

Lifecycle mirrors :class:`headroom.proxy.agy_dispatch.AgyDispatchServer`
(hypercorn lifespan + ``asyncio.start_server``), minus all TLS machinery.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import Any

logger = logging.getLogger("headroom.proxy.agy_retrieve")

_BIND_HOST = "127.0.0.1"


class AgyRetrieveServer:
    """In-process hypercorn PLAIN-HTTP server serving the headroom FastAPI app.

    Binds on loopback only (no TLS).  Serves the process-global compression
    cache via ``create_app()`` so ``GET /v1/retrieve/{hash}`` resolves markers
    the HTTPS dispatch server populated.  Hypercorn handles http/1.1 + lifespan.

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
        self._port = port

        self._server: asyncio.Server | None = None
        self._lifespan_task: asyncio.Task[None] | None = None
        self._lifespan: Any | None = None  # hypercorn.asyncio.run.Lifespan
        self._context: Any | None = None  # hypercorn.asyncio.run.WorkerContext
        self._app_wrapper: Any | None = None
        self._config: Any | None = None
        self._lifespan_state: dict[str, Any] = {}

    async def start(self) -> None:
        """Start the hypercorn server; binds loopback PLAIN HTTP on an ephemeral port."""
        from hypercorn.asyncio import wrap_app
        from hypercorn.asyncio.run import Lifespan, TCPServer, WorkerContext
        from hypercorn.config import Config

        # Build minimal hypercorn Config (no TLS — plain HTTP loopback).
        config = Config()
        config.bind = [f"{_BIND_HOST}:{self._port}"]
        config.accesslog = "-"  # suppress hypercorn access log noise in tests
        config.errorlog = "-"
        config.loglevel = "WARNING"
        self._config = config

        # Import and build the FastAPI app. create_app() wires the retrieve
        # routes against the process-global compression store, so this second
        # app instance shares the cache the dispatch server populates.
        from headroom.proxy.server import create_app

        app = create_app()
        # wrap_app accepts the ASGI callable directly; ignore the narrow stub type.
        app_wrapper = wrap_app(app, config.wsgi_max_body_size, mode="asgi")  # type: ignore[arg-type]
        self._app_wrapper = app_wrapper

        # Run hypercorn lifespan (startup/shutdown events).
        loop = asyncio.get_event_loop()
        lifespan_state: dict[str, Any] = {}
        self._lifespan_state = lifespan_state
        lifespan = Lifespan(app_wrapper, config, loop, lifespan_state)
        self._lifespan = lifespan
        self._lifespan_task = loop.create_task(lifespan.handle_lifespan())
        await lifespan.wait_for_startup()
        if self._lifespan_task.done():
            exc = self._lifespan_task.exception()
            if exc is not None:
                raise exc

        worker_context = WorkerContext(max_requests=None)
        self._context = worker_context

        # Bind a plain TCP socket on loopback. No SSL context is supplied to
        # asyncio.start_server, so the listener speaks plain HTTP.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_REUSEADDR means fast TIME_WAIT reuse on POSIX, but on Windows it
        # lets a second process bind this same loopback port and intercept the
        # decrypted retrieve traffic. Restrict to POSIX; on Windows enforce
        # exclusive use so a duplicate bind fails loudly.
        if os.name == "posix":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        elif hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((_BIND_HOST, self._port))

        async def _connection_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            await TCPServer(
                app_wrapper,
                loop,
                config,
                worker_context,
                lifespan_state,
                reader,
                writer,
            )

        self._server = await asyncio.start_server(
            _connection_handler,
            sock=sock,
        )
        addr = self._server.sockets[0].getsockname()
        logger.info("event=retrieve_started address=%s:%d", addr[0], addr[1])

    async def stop(self) -> None:
        """Gracefully shut down the server and hypercorn lifespan."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._lifespan is not None:
            try:
                await self._lifespan.wait_for_shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._lifespan = None

        if self._lifespan_task is not None:
            self._lifespan_task.cancel()
            try:
                await self._lifespan_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._lifespan_task = None

        logger.info("event=retrieve_stopped")

    @property
    def address(self) -> tuple[str, int]:
        """Return ``(host, port)`` the server is bound to. Requires :meth:`start`."""
        if self._server is None:
            raise RuntimeError("AgyRetrieveServer not started")
        sock = self._server.sockets[0]
        host, port = sock.getsockname()[:2]
        return host, port

    async def __aenter__(self) -> AgyRetrieveServer:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

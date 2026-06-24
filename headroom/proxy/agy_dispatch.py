"""In-process hypercorn HTTPS dispatch server for agy MITM transport.

Serves the existing headroom FastAPI app on a loopback HTTPS port so that
the agy CONNECT terminator can byte-splice accepted client connections straight
to this server — no second upstream TLS dial, no logic duplication.

Architecture (ADR 0001 §"Dispatch via hypercorn"):
  agy → CONNECT terminator (T8) → byte-splice → this server (TLS) → FastAPI app
                                                  ↑ mints leaf per SNI via _LeafCache

Security invariants:
  - Binds 127.0.0.1 only (loopback guard).
  - Leaf private keys are loaded from anonymous memory (memfd) on Linux and
    never touch the filesystem; on platforms without memfd, a 0600 temp file
    is written and unlinked immediately after load (perms asserted).
  - ALPN offers ["h2", "http/1.1"] matching the terminator leaf context.

Header handling: the Gemini handler strips the inbound ``accept-encoding``
header so the upstream returns a compressible (plain) body; agy UA and other
client headers are forwarded unchanged. The handler recompresses for the
upstream connection where applicable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import ssl
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509 import Certificate

from headroom.proxy.agy_ca import ensure_root_ca, load_cert_chain_in_memory
from headroom.proxy.agy_terminator import DEFAULT_ALLOWLIST, _LeafCache

logger = logging.getLogger("headroom.proxy.agy_dispatch")

_BIND_HOST = "127.0.0.1"
_PLACEHOLDER_HOST = "headroom.internal"

# ---------------------------------------------------------------------------
# ASGI helpers
# ---------------------------------------------------------------------------


async def _send_421(send: Any) -> None:
    """Send a minimal HTTP 421 Misdirected Request response."""
    body = b"Misdirected Request"
    await send(
        {
            "type": "http.response.start",
            "status": 421,
            "headers": [
                (b"content-type", b"text/plain"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


def make_host_guard(app: Any, allowlist: frozenset[str]) -> Any:
    """Wrap an ASGI *app* with a post-handshake Host/authority allowlist guard.

    Mandatory defense-in-depth for the no-SNI / placeholder path (where the
    TLS SNI guard may not fire). Hypercorn normalizes the HTTP/2 ``:authority``
    pseudo-header into a ``host`` header, so reading ``host`` covers h2 and
    http/1.1 uniformly. Module-level (not a closure) so it is unit-testable
    with synthetic ASGI scopes.
    """

    async def _host_guard_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") in ("http", "websocket"):
            # Enforce exactly ONE Host header — multiple Host headers are a
            # request-smuggling vector (guard validates one, backend may route
            # on another). RFC 7230 §5.4 requires rejecting them.
            host_values = [
                value for name, value in scope.get("headers", ()) if name.lower() == b"host"
            ]
            if len(host_values) != 1:
                logger.warning("event=host_refused host_count=%d", len(host_values))
                await _send_421(send)
                return
            host_str = host_values[0].decode("latin-1")
            if not host_str:
                logger.warning("event=host_refused host=%r", host_str)
                await _send_421(send)
                return
            # Normalize: strip a single trailing :port; lowercase (RFC 6066/7230).
            normalized = host_str.lower()
            if ":" in normalized:
                left, _, right = normalized.rpartition(":")
                if right.isdigit():
                    normalized = left
            if normalized not in allowlist:
                logger.warning("event=host_refused host=%s", host_str)
                await _send_421(send)
                return
        await app(scope, receive, send)

    return _host_guard_app


# ---------------------------------------------------------------------------
# SNI-capable SSL context builder
# ---------------------------------------------------------------------------


def _build_sni_ssl_context(
    leaf_cache: _LeafCache,
    ca_key: RSAPrivateKey,
    ca_cert: Certificate,
    allowlist: frozenset[str],
) -> ssl.SSLContext:
    """Return a server SSLContext whose SNI callback mints leaf certs on demand.

    The initial certfile/keyfile uses a wildcard placeholder cert so that
    ssl.SSLContext accepts the load_cert_chain call; the SNI callback replaces
    it per-connection before the handshake completes.

    ALPN: ["h2", "http/1.1"] — required for HTTP/2 negotiation.
    """
    # Mint a placeholder leaf for the initial load_cert_chain (SNI callback
    # guards against it before the handshake completes — placeholder never
    # served to real clients because SNI guard rejects non-allowlisted names).
    init_cert_pem, init_key_pem = leaf_cache.get_or_mint(_PLACEHOLDER_HOST, ca_key, ca_cert)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.set_alpn_protocols(["h2", "http/1.1"])

    # Load the placeholder cert chain (required before SNI callback fires).
    load_cert_chain_in_memory(ctx, init_cert_pem, init_key_pem)

    def _sni_callback(
        ssl_obj: ssl.SSLObject,
        server_name: str | None,
        ctx_in: ssl.SSLContext,  # noqa: ARG001
    ) -> int | None:
        """Guard SNI then mint or reuse a leaf cert for *server_name* and swap it in-place."""
        # Case-insensitive per RFC 6066; lowercase once so the membership check
        # AND the cache key match the (lowercase) allowlist and the Host guard.
        host = server_name.lower() if server_name is not None else None
        if host is None or host not in allowlist:
            logger.warning("event=sni_refused host=%s", server_name)
            return ssl.ALERT_DESCRIPTION_UNRECOGNIZED_NAME

        cert_pem, key_pem = leaf_cache.get_or_mint(host, ca_key, ca_cert)

        new_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        new_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        new_ctx.set_alpn_protocols(["h2", "http/1.1"])
        load_cert_chain_in_memory(new_ctx, cert_pem, key_pem)

        ssl_obj.context = new_ctx  # type: ignore[assignment]
        return None

    ctx.set_servername_callback(_sni_callback)  # type: ignore[arg-type]
    return ctx


# ---------------------------------------------------------------------------
# AgyDispatchServer
# ---------------------------------------------------------------------------


class AgyDispatchServer:
    """In-process hypercorn HTTPS server serving the headroom FastAPI app.

    Binds on loopback only; TLS via SNI callback (mints leaf per hostname
    from the headroom root CA).  Hypercorn handles h2/http1.1 + lifespan.

    Usage::

        server = AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert)
        await server.start()
        # server.address → ("127.0.0.1", <ephemeral-port>)
        await server.stop()

    Or as an async context manager::

        async with AgyDispatchServer(ca_key=ca_key, ca_cert=ca_cert) as srv:
            host, port = srv.address
    """

    def __init__(
        self,
        ca_key: RSAPrivateKey | None = None,
        ca_cert: Certificate | None = None,
        base_dir: Path | None = None,
        port: int = 0,
        allowlist: frozenset[str] | None = None,
    ) -> None:
        self._ca_key_init = ca_key
        self._ca_cert_init = ca_cert
        self._base_dir = base_dir
        self._port = port
        self._allowlist: frozenset[str] = allowlist if allowlist is not None else DEFAULT_ALLOWLIST

        self._server: asyncio.Server | None = None
        self._lifespan_task: asyncio.Task[None] | None = None
        self._lifespan: Any | None = None  # hypercorn.asyncio.run.Lifespan
        self._context: Any | None = None  # hypercorn.asyncio.run.WorkerContext
        self._app_wrapper: Any | None = None
        self._config: Any | None = None
        self._lifespan_state: dict[str, Any] = {}
        self._leaf_cache: _LeafCache | None = None

    async def start(self) -> None:
        """Start the hypercorn server; binds loopback HTTPS on an ephemeral port."""
        from hypercorn.asyncio import wrap_app
        from hypercorn.asyncio.run import Lifespan, TCPServer, WorkerContext
        from hypercorn.config import Config

        # Resolve CA.
        if self._ca_key_init is not None and self._ca_cert_init is not None:
            ca_key = self._ca_key_init
            ca_cert = self._ca_cert_init
        else:
            ca_key, ca_cert, _, _ = ensure_root_ca(base_dir=self._base_dir)

        self._leaf_cache = _LeafCache(max_size=len(self._allowlist) + 1)
        ssl_ctx = _build_sni_ssl_context(self._leaf_cache, ca_key, ca_cert, self._allowlist)

        # Build minimal hypercorn Config (no certfile/keyfile — we supply ssl directly).
        config = Config()
        config.bind = [f"{_BIND_HOST}:{self._port}"]
        config.accesslog = "-"  # suppress hypercorn access log noise in tests
        config.errorlog = "-"
        config.loglevel = "WARNING"
        self._config = config

        # Import and build the FastAPI app.
        from headroom.proxy.server import create_app

        app = make_host_guard(create_app(), self._allowlist)

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

        # Bind a plain TCP socket on loopback then wrap with our SSL context.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_REUSEADDR means fast TIME_WAIT reuse on POSIX, but on Windows it
        # lets a second process bind this same loopback port and intercept the
        # decrypted MITM traffic. Restrict to POSIX; on Windows enforce
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
            ssl=ssl_ctx,
            ssl_handshake_timeout=config.ssl_handshake_timeout,
        )
        addr = self._server.sockets[0].getsockname()
        logger.info("event=dispatch_started address=%s:%d", addr[0], addr[1])

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

        logger.info("event=dispatch_stopped")

    @property
    def address(self) -> tuple[str, int]:
        """Return ``(host, port)`` the server is bound to. Requires :meth:`start`."""
        if self._server is None:
            raise RuntimeError("AgyDispatchServer not started")
        sock = self._server.sockets[0]
        host, port = sock.getsockname()[:2]
        return host, port

    async def __aenter__(self) -> AgyDispatchServer:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

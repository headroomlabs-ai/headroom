"""Streamable HTTP transport helpers for the Headroom MCP server."""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from headroom.proxy.loopback_guard import is_loopback_host

logger = logging.getLogger(__name__)

#: Extra ``Host`` values (``name:port``, comma-separated) the HTTP transport
#: accepts, for a server reached through a hostname or a reverse proxy.
ALLOWED_HOSTS_ENV = "HEADROOM_MCP_ALLOWED_HOSTS"
_LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")
_WILDCARD_BINDS = ("0.0.0.0", "::", "")


def _normalize_path(path: str) -> str:
    normalized = path.strip()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if normalized != "/":
        normalized = normalized.rstrip("/")
    return normalized


class StreamableHTTPASGIApp:
    """Delegate ASGI requests to the SDK session manager."""

    def __init__(self, session_manager: StreamableHTTPSessionManager):
        self.session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.session_manager.handle_request(scope, receive, send)


def transport_security_settings(host: str, port: int) -> TransportSecuritySettings:
    """Host/Origin allowlist for a server bound to ``host:port``.

    The SDK checks neither header unless it is given settings. Without the
    Host check, a page whose hostname is re-pointed at 127.0.0.1 can reach
    this server from the user's browser. A loopback or wildcard bind accepts
    the loopback names; any other bind accepts its own address. More names
    come from ``HEADROOM_MCP_ALLOWED_HOSTS``.
    """
    if host in _WILDCARD_BINDS or is_loopback_host(host):
        names = list(_LOOPBACK_NAMES)
    else:
        names = [f"[{host}]" if ":" in host else host]
    allowed_hosts = [f"{name}:{port}" for name in names]
    extra = [entry.strip() for entry in os.environ.get(ALLOWED_HOSTS_ENV, "").split(",")]
    allowed_hosts.extend(entry for entry in extra if entry)
    if host in _WILDCARD_BINDS and not any(extra):
        logger.warning(
            "MCP HTTP server bound to %s accepts only loopback Host headers; "
            "set %s=<name:port,...> for the names remote clients use",
            host or "all interfaces",
            ALLOWED_HOSTS_ENV,
        )
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=[
            f"{scheme}://{allowed}" for allowed in allowed_hosts for scheme in ("http", "https")
        ],
    )


def create_streamable_http_session_manager(
    server: Any, *, host: str = "127.0.0.1", port: int = 8788
) -> StreamableHTTPSessionManager:
    """Build the SDK session manager for an existing MCP server bound to ``host:port``."""
    return StreamableHTTPSessionManager(
        app=server.server,
        security_settings=transport_security_settings(host, port),
    )


def create_streamable_http_app(
    session_manager: StreamableHTTPSessionManager,
    *,
    path: str,
    debug: bool = False,
) -> Starlette:
    """Build the Starlette app that exposes the MCP server over HTTP."""
    streamable_http_app = StreamableHTTPASGIApp(session_manager)
    return Starlette(
        debug=debug,
        routes=[
            Route(
                _normalize_path(path),
                endpoint=streamable_http_app,
                methods=["GET", "POST", "DELETE", "OPTIONS"],
            )
        ],
        lifespan=lambda app: session_manager.run(),
    )


async def serve_streamable_http(
    server: Any,
    *,
    host: str,
    port: int,
    path: str,
    debug: bool = False,
) -> None:
    """Serve the MCP server over Streamable HTTP with uvicorn."""
    import uvicorn

    session_manager = create_streamable_http_session_manager(server, host=host, port=port)
    starlette_app = create_streamable_http_app(session_manager, path=path, debug=debug)
    config = uvicorn.Config(
        starlette_app,
        host=host,
        port=port,
        log_level="debug" if debug else "warning",
    )
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()

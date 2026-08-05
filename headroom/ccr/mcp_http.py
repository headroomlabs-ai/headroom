"""Streamable HTTP transport helpers for the Headroom MCP server."""

from __future__ import annotations

from typing import Any

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.types import Receive, Scope, Send


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


def create_streamable_http_session_manager(server: Any) -> StreamableHTTPSessionManager:
    """Build the SDK session manager for an existing MCP server."""
    return StreamableHTTPSessionManager(app=server.server)


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

    session_manager = create_streamable_http_session_manager(server)
    starlette_app = create_streamable_http_app(session_manager, path=path, debug=debug)
    config = uvicorn.Config(
        starlette_app,
        host=host,
        port=port,
        log_level="debug" if debug else "warning",
    )
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()

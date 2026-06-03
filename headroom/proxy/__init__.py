"""Headroom Proxy Server.

A transparent proxy that sits between LLM clients (Claude Code, Cursor, etc.)
and LLM APIs (Anthropic, OpenAI), applying Headroom optimizations.

Usage:
    # Start the proxy
    python -m headroom.proxy.server

    # Use with Claude Code
    ANTHROPIC_BASE_URL=http://localhost:8787 claude

    # Use with Cursor (if using Anthropic)
    Set base URL in Cursor settings to http://localhost:8787
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .server import create_app, run_server

__all__ = ["create_app", "run_server"]


def __getattr__(name: str) -> Any:
    """Lazily expose the server entry points (PEP 562).

    ``headroom.proxy.server`` pulls in ``fastapi`` and the rest of the proxy
    stack, which only ship with the ``[proxy]`` extra. Importing it eagerly
    here broke ``headroom --help`` (and every other CLI command) for base
    installs without that extra, because ``headroom.cli.proxy`` only needs the
    dependency-free ``headroom.proxy.modes`` module. Deferring the import keeps
    ``from headroom.proxy import create_app`` working while no longer requiring
    ``fastapi`` just to import this package. See issue #441.
    """
    if name in ("create_app", "run_server"):
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

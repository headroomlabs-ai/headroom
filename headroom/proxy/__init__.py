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

from typing import Any
from importlib import import_module

_LAZY_EXPORTS = {
    "create_app": ("headroom.proxy.server", "create_app"),
    "run_server": ("headroom.proxy.server", "run_server"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        val = getattr(import_module(module_name), attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS.keys()))


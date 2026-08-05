"""Operator configuration policy for proxy tool injection."""

from __future__ import annotations

import os

from headroom.proxy.tool_injection_policy import (
    TOOL_INJECTION_STICKY_DEFAULT,
    TOOL_INJECTION_STICKY_ENV,
    TOOL_TRACKER_MAX_SESSIONS_DEFAULT,
    TOOL_TRACKER_MAX_SESSIONS_ENV,
    ToolInjectionStickyMode,
    resolve_tool_injection_sticky_mode,
    resolve_tool_tracker_max_sessions,
)

__all__ = [
    "TOOL_INJECTION_STICKY_DEFAULT",
    "TOOL_INJECTION_STICKY_ENV",
    "TOOL_TRACKER_MAX_SESSIONS_DEFAULT",
    "TOOL_TRACKER_MAX_SESSIONS_ENV",
    "ToolInjectionStickyMode",
    "get_tool_injection_sticky_mode",
    "get_tool_tracker_max_sessions",
]


def get_tool_injection_sticky_mode() -> ToolInjectionStickyMode:
    """Return the active memory-tool stickiness mode."""

    return resolve_tool_injection_sticky_mode(os.environ.get(TOOL_INJECTION_STICKY_ENV))


def get_tool_tracker_max_sessions() -> int:
    """Return the LRU bound for memory tool session tracking."""

    return resolve_tool_tracker_max_sessions(os.environ.get(TOOL_TRACKER_MAX_SESSIONS_ENV))

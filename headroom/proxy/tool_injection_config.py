"""Operator configuration policy for proxy tool injection."""

from __future__ import annotations

import os

from headroom.proxy.tool_injection_policy import (
    CCR_TOOL_INJECTION_DEFAULT,
    CCR_TOOL_INJECTION_ENV,
    TOOL_INJECTION_STICKY_DEFAULT,
    TOOL_INJECTION_STICKY_ENV,
    TOOL_TRACKER_MAX_SESSIONS_DEFAULT,
    TOOL_TRACKER_MAX_SESSIONS_ENV,
    CcrToolInjectionMode,
    ToolInjectionStickyMode,
    resolve_ccr_tool_injection_mode,
    resolve_tool_injection_sticky_mode,
    resolve_tool_tracker_max_sessions,
)

__all__ = [
    "CCR_TOOL_INJECTION_DEFAULT",
    "CCR_TOOL_INJECTION_ENV",
    "CcrToolInjectionMode",
    "TOOL_INJECTION_STICKY_DEFAULT",
    "TOOL_INJECTION_STICKY_ENV",
    "TOOL_TRACKER_MAX_SESSIONS_DEFAULT",
    "TOOL_TRACKER_MAX_SESSIONS_ENV",
    "ToolInjectionStickyMode",
    "get_ccr_tool_injection_mode",
    "get_tool_injection_sticky_mode",
    "get_tool_tracker_max_sessions",
]


def get_ccr_tool_injection_mode() -> CcrToolInjectionMode:
    """Return when the CCR retrieval tool enters the tools array."""

    return resolve_ccr_tool_injection_mode(os.environ.get(CCR_TOOL_INJECTION_ENV))


def get_tool_injection_sticky_mode() -> ToolInjectionStickyMode:
    """Return the active memory-tool stickiness mode."""

    return resolve_tool_injection_sticky_mode(os.environ.get(TOOL_INJECTION_STICKY_ENV))


def get_tool_tracker_max_sessions() -> int:
    """Return the LRU bound for memory tool session tracking."""

    return resolve_tool_tracker_max_sessions(os.environ.get(TOOL_TRACKER_MAX_SESSIONS_ENV))

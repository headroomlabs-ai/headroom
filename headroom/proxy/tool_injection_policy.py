"""Policy helpers for memory tool injection stickiness configuration."""

from __future__ import annotations

from typing import Literal, cast

TOOL_INJECTION_STICKY_ENV = "HEADROOM_TOOL_INJECTION_STICKY"
ToolInjectionStickyMode = Literal["enabled", "disabled"]
TOOL_INJECTION_STICKY_DEFAULT: ToolInjectionStickyMode = "enabled"

TOOL_TRACKER_MAX_SESSIONS_ENV = "HEADROOM_TOOL_TRACKER_MAX_SESSIONS"
TOOL_TRACKER_MAX_SESSIONS_DEFAULT = 1000


def resolve_tool_injection_sticky_mode(raw: str | None) -> ToolInjectionStickyMode:
    """Resolve memory-tool injection stickiness mode from an environment value."""

    normalized = (raw or "").strip().lower()
    if not normalized:
        return TOOL_INJECTION_STICKY_DEFAULT
    if normalized in ("enabled", "disabled"):
        return cast(ToolInjectionStickyMode, normalized)
    raise ValueError(
        f"Invalid {TOOL_INJECTION_STICKY_ENV}={normalized!r}; expected 'enabled' or 'disabled'"
    )


def resolve_tool_tracker_max_sessions(raw: str | None) -> int:
    """Resolve the positive LRU session bound for tool injection tracking."""

    normalized = (raw or "").strip()
    if not normalized:
        return TOOL_TRACKER_MAX_SESSIONS_DEFAULT
    try:
        value = int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {TOOL_TRACKER_MAX_SESSIONS_ENV}={normalized!r}; expected positive int"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"Invalid {TOOL_TRACKER_MAX_SESSIONS_ENV}={normalized!r}; expected positive int"
        )
    return value

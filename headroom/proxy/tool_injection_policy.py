"""Policy helpers for proxy tool injection configuration."""

from __future__ import annotations

from typing import Literal, cast

TOOL_INJECTION_STICKY_ENV = "HEADROOM_TOOL_INJECTION_STICKY"
ToolInjectionStickyMode = Literal["enabled", "disabled"]
TOOL_INJECTION_STICKY_DEFAULT: ToolInjectionStickyMode = "enabled"

CCR_TOOL_INJECTION_ENV = "HEADROOM_CCR_TOOL_INJECTION"
CcrToolInjectionMode = Literal["eager", "lazy"]
#: Eager by default. ``tools`` is the head of Anthropic's cache key, so adding
#: ``headroom_retrieve`` at the first compression — typically many turns in,
#: against a fully warm prefix — invalidates the entire prefix behind it. The
#: definition is ~119 tokens; paying for it in the session's first (unavoidable)
#: cache write is orders of magnitude cheaper than a mid-session rewrite.
CCR_TOOL_INJECTION_DEFAULT: CcrToolInjectionMode = "eager"

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


def resolve_ccr_tool_injection_mode(raw: str | None) -> CcrToolInjectionMode:
    """Resolve when the CCR retrieval tool enters the tools array.

    ``eager`` injects from the session's first request, so the tools array is
    byte-stable for the life of the session. ``lazy`` restores the historical
    behaviour of waiting for the first compression, which keeps the tool out of
    conversations that never compress at the cost of one full prefix
    invalidation in those that do.
    """

    normalized = (raw or "").strip().lower()
    if not normalized:
        return CCR_TOOL_INJECTION_DEFAULT
    if normalized in ("eager", "lazy"):
        return cast(CcrToolInjectionMode, normalized)
    raise ValueError(f"Invalid {CCR_TOOL_INJECTION_ENV}={normalized!r}; expected 'eager' or 'lazy'")


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

"""Policy helpers for beta-header stickiness configuration."""

from __future__ import annotations

from typing import Literal, cast

BETA_HEADER_STICKY_ENV = "HEADROOM_BETA_HEADER_STICKY"
BetaHeaderStickyMode = Literal["enabled", "disabled"]
BETA_HEADER_STICKY_DEFAULT: BetaHeaderStickyMode = "enabled"

BETA_TRACKER_MAX_SESSIONS_ENV = "HEADROOM_BETA_TRACKER_MAX_SESSIONS"
BETA_TRACKER_MAX_SESSIONS_DEFAULT = 1000


def resolve_beta_header_sticky_mode(raw: str | None) -> BetaHeaderStickyMode:
    """Resolve beta-header stickiness mode from an environment value."""

    normalized = (raw or "").strip().lower()
    if not normalized:
        return BETA_HEADER_STICKY_DEFAULT
    if normalized in ("enabled", "disabled"):
        return cast(BetaHeaderStickyMode, normalized)
    raise ValueError(
        f"Invalid {BETA_HEADER_STICKY_ENV}={normalized!r}; expected 'enabled' or 'disabled'"
    )


def resolve_beta_tracker_max_sessions(raw: str | None) -> int:
    """Resolve the positive LRU session bound for beta-header tracking."""

    normalized = (raw or "").strip()
    if not normalized:
        return BETA_TRACKER_MAX_SESSIONS_DEFAULT
    try:
        value = int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {BETA_TRACKER_MAX_SESSIONS_ENV}={normalized!r}; expected positive int"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"Invalid {BETA_TRACKER_MAX_SESSIONS_ENV}={normalized!r}; expected positive int"
        )
    return value

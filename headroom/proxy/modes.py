"""Proxy run mode helpers.

Canonical modes:
- token: prioritize compression (history may be rewritten for max savings)
- cache: prioritize provider prefix cache stability (freeze prior turns)
"""

from __future__ import annotations

import logging

from headroom.proxy.proxy_mode_policy import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    normalize_proxy_mode_decision,
)

logger = logging.getLogger("headroom.proxy")


def normalize_proxy_mode(mode: str | None, *, default: str = PROXY_MODE_TOKEN) -> str:
    """Normalize a user-provided proxy mode to canonical token/cache values."""
    decision = normalize_proxy_mode_decision(mode, default=default)
    if decision.unknown:
        logger.warning("Unknown HEADROOM_MODE '%s', falling back to '%s'", mode, default)
    elif decision.alias_used:
        logger.info("HEADROOM_MODE alias '%s' normalized to '%s'", mode, decision.normalized)
    return decision.normalized


def is_token_mode(mode: str | None) -> bool:
    """Return True when mode resolves to token mode."""
    return normalize_proxy_mode(mode) == PROXY_MODE_TOKEN


def is_cache_mode(mode: str | None) -> bool:
    """Return True when mode resolves to cache mode."""
    return normalize_proxy_mode(mode) == PROXY_MODE_CACHE


def read_maturation_inert_reason(
    mode: str | None, *, quiesce_turns: int, max_hold_turns: int
) -> str | None:
    """Why read maturation can never fire under ``mode``, else None.

    Cache mode freezes every turn but the latest, and maturation skips frozen
    messages. A Read in the latest turn has had no quiet turns yet, so it is
    held unless a threshold is 0; by the time it could quiesce it is frozen.
    """
    if is_cache_mode(mode) and quiesce_turns > 0 and max_hold_turns > 0:
        return "cache mode freezes a Read before it can quiesce"
    return None

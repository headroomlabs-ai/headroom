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

"""Pure proxy mode normalization policy."""

from __future__ import annotations

from dataclasses import dataclass

PROXY_MODE_TOKEN = "token"
PROXY_MODE_CACHE = "cache"

MODE_ALIASES = {
    "token": PROXY_MODE_TOKEN,
    "token_mode": PROXY_MODE_TOKEN,
    "token_savings": PROXY_MODE_TOKEN,
    "token_headroom": PROXY_MODE_TOKEN,
    "cache": PROXY_MODE_CACHE,
    "cache_mode": PROXY_MODE_CACHE,
    "cost_savings": PROXY_MODE_CACHE,
}


@dataclass(frozen=True)
class ProxyModeDecision:
    """Result of normalizing a user-provided proxy mode."""

    raw: str | None
    key: str
    normalized: str
    used_default: bool = False
    unknown: bool = False
    alias_used: bool = False


def normalize_proxy_mode_decision(
    mode: str | None,
    *,
    default: str = PROXY_MODE_TOKEN,
) -> ProxyModeDecision:
    """Normalize a user-provided proxy mode without side effects."""
    key = (mode or "").strip().lower()
    if not key:
        return ProxyModeDecision(raw=mode, key=key, normalized=default, used_default=True)

    normalized = MODE_ALIASES.get(key)
    if normalized is None:
        return ProxyModeDecision(
            raw=mode,
            key=key,
            normalized=default,
            used_default=True,
            unknown=True,
        )

    return ProxyModeDecision(
        raw=mode,
        key=key,
        normalized=normalized,
        alias_used=key != normalized,
    )


def normalize_proxy_mode_value(
    mode: str | None,
    *,
    default: str = PROXY_MODE_TOKEN,
) -> str:
    """Return only the canonical proxy mode value."""
    return normalize_proxy_mode_decision(mode, default=default).normalized

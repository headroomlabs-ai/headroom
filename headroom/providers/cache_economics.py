"""Provider-owned prefix-cache economics policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCacheEconomics:
    """Cost and cache behavior metadata for a provider."""

    provider: str
    read_multiplier: float
    write_multiplier: float
    label: str

    @property
    def read_discount(self) -> float:
        """Fraction of input price saved by a cache read."""
        return 1.0 - self.read_multiplier

    @property
    def write_penalty(self) -> float:
        """Fractional premium paid by a cache write."""
        return max(0.0, self.write_multiplier - 1.0)


_DEFAULT_PROVIDER = "anthropic"

_CACHE_ECONOMICS: dict[str, ProviderCacheEconomics] = {
    "anthropic": ProviderCacheEconomics(
        provider="anthropic",
        read_multiplier=0.1,
        write_multiplier=1.25,
        label="Explicit breakpoints, 5-min TTL",
    ),
    "openai": ProviderCacheEconomics(
        provider="openai",
        read_multiplier=0.5,
        write_multiplier=1.0,
        label="Automatic, no TTL control",
    ),
    "gemini": ProviderCacheEconomics(
        provider="gemini",
        read_multiplier=0.1,
        write_multiplier=1.0,
        label="Explicit cachedContent, configurable TTL",
    ),
    "bedrock": ProviderCacheEconomics(
        provider="bedrock",
        read_multiplier=0.1,
        write_multiplier=1.25,
        label="Same as Anthropic (Bedrock)",
    ),
}


def cache_economics_for_provider(provider: str) -> ProviderCacheEconomics:
    """Return provider cache economics, falling back to Anthropic policy."""
    return _CACHE_ECONOMICS.get(provider, _CACHE_ECONOMICS[_DEFAULT_PROVIDER])

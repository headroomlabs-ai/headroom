"""Headroom Cache Optimization Module.

This module provides a plugin-based architecture for cache optimization
across different LLM providers. Each provider has different caching
mechanisms and this module abstracts those differences.

Provider Caching Differences:
- Anthropic: Explicit cache_control blocks, 90% savings, 5-min TTL
- OpenAI: Automatic prefix caching, 50% savings, no user control
- Google: Separate CachedContent API, 75% savings + storage costs

Usage:
    from headroom.cache import CacheOptimizerRegistry, SemanticCacheLayer

    # Get provider-specific optimizer
    optimizer = CacheOptimizerRegistry.get("anthropic")
    result = optimizer.optimize(messages, context)

    # With semantic caching layer
    semantic = SemanticCacheLayer(optimizer, similarity_threshold=0.95)
    result = semantic.process(messages, context)

    # Register custom optimizer
    CacheOptimizerRegistry.register("my-provider", MyOptimizer)
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Expose concrete types to static analysis while keeping runtime imports lazy.
    from headroom.cache.anthropic import AnthropicCacheOptimizer  # noqa: F401
    from headroom.cache.base import (  # noqa: F401
        BaseCacheOptimizer,
        CacheBreakpoint,
        CacheConfig,
        CacheMetrics,
        CacheOptimizer,
        CacheResult,
        CacheStrategy,
        OptimizationContext,
    )
    from headroom.cache.compression_cache import CompressionCache  # noqa: F401
    from headroom.cache.dynamic_detector import (  # noqa: F401
        DetectorConfig,
        DynamicCategory,
        DynamicContentDetector,
        DynamicSpan,
        detect_dynamic_content,
    )
    from headroom.cache.google import GoogleCacheOptimizer  # noqa: F401
    from headroom.cache.openai import OpenAICacheOptimizer  # noqa: F401
    from headroom.cache.prefix_tracker import (  # noqa: F401
        FreezeStats,
        PrefixCacheTracker,
        PrefixFreezeConfig,
        SessionTrackerStore,
    )
    from headroom.cache.registry import CacheOptimizerRegistry  # noqa: F401
    from headroom.cache.semantic import SemanticCache, SemanticCacheLayer  # noqa: F401

__all__ = [
    # Base types
    "BaseCacheOptimizer",
    "CacheBreakpoint",
    "CacheConfig",
    "CacheMetrics",
    "CacheOptimizer",
    "CacheResult",
    "CacheStrategy",
    "OptimizationContext",
    # Dynamic content detection
    "DetectorConfig",
    "DynamicCategory",
    "DynamicContentDetector",
    "DynamicSpan",
    "detect_dynamic_content",
    # Registry
    "CacheOptimizerRegistry",
    # Provider implementations
    "AnthropicCacheOptimizer",
    "OpenAICacheOptimizer",
    "GoogleCacheOptimizer",
    # Semantic caching
    "SemanticCacheLayer",
    "SemanticCache",
    # Compression cache (token headroom mode)
    "CompressionCache",
    # Prefix cache tracking
    "PrefixCacheTracker",
    "PrefixFreezeConfig",
    "FreezeStats",
    "SessionTrackerStore",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Base types
    "BaseCacheOptimizer": ("headroom.cache.base", "BaseCacheOptimizer"),
    "CacheBreakpoint": ("headroom.cache.base", "CacheBreakpoint"),
    "CacheConfig": ("headroom.cache.base", "CacheConfig"),
    "CacheMetrics": ("headroom.cache.base", "CacheMetrics"),
    "CacheOptimizer": ("headroom.cache.base", "CacheOptimizer"),
    "CacheResult": ("headroom.cache.base", "CacheResult"),
    "CacheStrategy": ("headroom.cache.base", "CacheStrategy"),
    "OptimizationContext": ("headroom.cache.base", "OptimizationContext"),
    # Dynamic content detection
    "DetectorConfig": ("headroom.cache.dynamic_detector", "DetectorConfig"),
    "DynamicCategory": ("headroom.cache.dynamic_detector", "DynamicCategory"),
    "DynamicContentDetector": ("headroom.cache.dynamic_detector", "DynamicContentDetector"),
    "DynamicSpan": ("headroom.cache.dynamic_detector", "DynamicSpan"),
    "detect_dynamic_content": ("headroom.cache.dynamic_detector", "detect_dynamic_content"),
    # Registry
    "CacheOptimizerRegistry": ("headroom.cache.registry", "CacheOptimizerRegistry"),
    # Provider implementations
    "AnthropicCacheOptimizer": ("headroom.cache.anthropic", "AnthropicCacheOptimizer"),
    "OpenAICacheOptimizer": ("headroom.cache.openai", "OpenAICacheOptimizer"),
    "GoogleCacheOptimizer": ("headroom.cache.google", "GoogleCacheOptimizer"),
    # Semantic caching
    "SemanticCacheLayer": ("headroom.cache.semantic", "SemanticCacheLayer"),
    "SemanticCache": ("headroom.cache.semantic", "SemanticCache"),
    # Compression cache
    "CompressionCache": ("headroom.cache.compression_cache", "CompressionCache"),
    # Prefix cache tracking
    "PrefixCacheTracker": ("headroom.cache.prefix_tracker", "PrefixCacheTracker"),
    "PrefixFreezeConfig": ("headroom.cache.prefix_tracker", "PrefixFreezeConfig"),
    "FreezeStats": ("headroom.cache.prefix_tracker", "FreezeStats"),
    "SessionTrackerStore": ("headroom.cache.prefix_tracker", "SessionTrackerStore"),
}


def __getattr__(name: str) -> object:
    if name == "__path__":
        raise AttributeError(name)

    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

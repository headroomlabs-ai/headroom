"""
Cache Optimizer Registry.

Provides a plugin system for registering and retrieving cache optimizers.
This allows users to swap implementations and register custom optimizers.
"""

from __future__ import annotations

from .base import BaseCacheOptimizer, CacheConfig


class CacheOptimizerRegistry:
    """
    Registry for cache optimizer plugins.

    This registry allows:
    - Registration of custom optimizers
    - Retrieval by provider name
    - Tier-based selection (oss vs enterprise)

    Usage:
        # Get default optimizer for provider
        optimizer = CacheOptimizerRegistry.get("anthropic")

        # Get enterprise version if available
        optimizer = CacheOptimizerRegistry.get("anthropic", tier="enterprise")

        # Register custom optimizer
        CacheOptimizerRegistry.register("my-provider", MyOptimizer)
    """

    _optimizers: dict[str, type[BaseCacheOptimizer]] = {}
    _instances: dict[str, BaseCacheOptimizer] = {}

    @classmethod
    def register(
        cls,
        name: str,
        optimizer_class: type[BaseCacheOptimizer],
        *,
        override: bool = False,
    ) -> None:
        """
        Register a cache optimizer.

        Args:
            name: Name to register under (e.g., "anthropic", "anthropic-enterprise")
            optimizer_class: The optimizer class to register
            override: Whether to override existing registration

        Raises:
            ValueError: If name already registered and override=False
        """
        if name in cls._optimizers and not override:
            raise ValueError(
                f"Optimizer '{name}' already registered. Use override=True to replace."
            )
        cls._optimizers[name] = optimizer_class
        # Clear cached instance if exists
        cls._instances.pop(name, None)

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a cache optimizer.

        Args:
            name: Name to unregister
        """
        cls._optimizers.pop(name, None)
        cls._instances.pop(name, None)

    @classmethod
    def get(
        cls,
        provider: str,
        tier: str = "oss",
        config: CacheConfig | None = None,
        *,
        cached: bool = True,
    ) -> BaseCacheOptimizer:
        """
        Get a cache optimizer for a provider.

        Args:
            provider: Provider name (e.g., "anthropic", "openai", "google")
            tier: Tier to get ("oss" or "enterprise")
            config: Optional configuration
            cached: Whether to return cached instance

        Returns:
            Cache optimizer instance

        Raises:
            KeyError: If no optimizer registered for provider/tier
        """
        # Build the lookup key
        if tier != "oss":
            key = f"{provider}-{tier}"
            # Fall back to OSS if enterprise not available
            if key not in cls._optimizers:
                key = provider
        else:
            key = provider

        if key not in cls._optimizers:
            available = list(cls._optimizers.keys())
            raise KeyError(f"No optimizer registered for '{key}'. Available: {available}")

        # Return cached instance if requested
        cache_key = f"{key}:{id(config)}" if config else key
        if cached and cache_key in cls._instances:
            return cls._instances[cache_key]

        # Create new instance
        optimizer_class = cls._optimizers[key]
        instance = optimizer_class(config)

        if cached:
            cls._instances[cache_key] = instance

        return instance

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names (excluding tier suffixes)."""
        providers = set()
        for name in cls._optimizers:
            # Remove tier suffix if present
            base_name = name.split("-")[0]
            providers.add(base_name)
        return sorted(providers)

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered optimizer names."""
        return sorted(cls._optimizers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an optimizer is registered."""
        return name in cls._optimizers

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Mainly for testing."""
        cls._optimizers.clear()
        cls._instances.clear()

    @classmethod
    def reset_to_defaults(cls) -> None:
        """Reset to default registrations."""
        cls.clear()
        _register_defaults()


def _register_defaults() -> None:
    """Register default optimizers."""
    # Import here to avoid circular imports
    from .anthropic import AnthropicCacheOptimizer
    from .google import GoogleCacheOptimizer
    from .openai import OpenAICacheOptimizer

    CacheOptimizerRegistry.register("anthropic", AnthropicCacheOptimizer)
    CacheOptimizerRegistry.register("openai", OpenAICacheOptimizer)
    CacheOptimizerRegistry.register("google", GoogleCacheOptimizer)


# Auto-register defaults on module import
# Wrapped in try/except to allow partial imports during development
try:
    _register_defaults()
except ImportError:
    pass

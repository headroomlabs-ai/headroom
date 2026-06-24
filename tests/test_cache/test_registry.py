"""Tests for CacheOptimizerRegistry."""

import pytest

from headroom.cache import (
    AnthropicCacheOptimizer,
    CacheConfig,
    CacheOptimizerRegistry,
    GoogleCacheOptimizer,
    OpenAICacheOptimizer,
)
from headroom.cache.base import BaseCacheOptimizer, CacheResult, CacheStrategy


class MockOptimizer(BaseCacheOptimizer):
    """Mock optimizer for testing."""

    @property
    def name(self) -> str:
        return "mock-optimizer"

    @property
    def provider(self) -> str:
        return "mock"

    @property
    def strategy(self) -> CacheStrategy:
        return CacheStrategy.NONE

    def optimize(self, messages, context, config=None):
        return CacheResult(messages=messages)


class TestCacheOptimizerRegistry:
    """Test CacheOptimizerRegistry functionality."""

    def test_default_providers_registered(self):
        """Test that default providers are registered on import."""
        providers = CacheOptimizerRegistry.list_all()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "google" in providers

    def test_get_anthropic(self):
        """Test getting Anthropic optimizer."""
        optimizer = CacheOptimizerRegistry.get("anthropic")
        assert isinstance(optimizer, AnthropicCacheOptimizer)
        assert optimizer.provider == "anthropic"
        assert optimizer.strategy == CacheStrategy.EXPLICIT_BREAKPOINTS

    def test_get_openai(self):
        """Test getting OpenAI optimizer."""
        optimizer = CacheOptimizerRegistry.get("openai")
        assert isinstance(optimizer, OpenAICacheOptimizer)
        assert optimizer.provider == "openai"
        assert optimizer.strategy == CacheStrategy.PREFIX_STABILIZATION

    def test_get_google(self):
        """Test getting Google optimizer."""
        optimizer = CacheOptimizerRegistry.get("google")
        assert isinstance(optimizer, GoogleCacheOptimizer)
        assert optimizer.provider == "google"
        assert optimizer.strategy == CacheStrategy.CACHED_CONTENT

    def test_get_with_config(self):
        """Test getting optimizer with custom config."""
        config = CacheConfig(min_cacheable_tokens=2048)
        optimizer = CacheOptimizerRegistry.get("anthropic", config=config, cached=False)
        assert optimizer.config.min_cacheable_tokens >= 1024  # Anthropic enforces minimum

    def test_register_custom_optimizer(self):
        """Test registering a custom optimizer."""
        CacheOptimizerRegistry.register("mock", MockOptimizer)
        try:
            optimizer = CacheOptimizerRegistry.get("mock")
            assert isinstance(optimizer, MockOptimizer)
        finally:
            CacheOptimizerRegistry.unregister("mock")

    def test_register_duplicate_raises(self):
        """Test that registering duplicate without override raises."""
        CacheOptimizerRegistry.register("test-dup", MockOptimizer)
        try:
            with pytest.raises(ValueError):
                CacheOptimizerRegistry.register("test-dup", MockOptimizer)
        finally:
            CacheOptimizerRegistry.unregister("test-dup")

    def test_register_with_override(self):
        """Test registering with override."""
        CacheOptimizerRegistry.register("test-override", MockOptimizer)
        try:
            CacheOptimizerRegistry.register("test-override", MockOptimizer, override=True)
            optimizer = CacheOptimizerRegistry.get("test-override")
            assert isinstance(optimizer, MockOptimizer)
        finally:
            CacheOptimizerRegistry.unregister("test-override")

    def test_get_unknown_provider_raises(self):
        """Test getting unknown provider raises KeyError."""
        with pytest.raises(KeyError):
            CacheOptimizerRegistry.get("unknown-provider")

    def test_list_providers(self):
        """Test listing providers."""
        providers = CacheOptimizerRegistry.list_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "google" in providers

    def test_is_registered(self):
        """Test is_registered check."""
        assert CacheOptimizerRegistry.is_registered("anthropic")
        assert not CacheOptimizerRegistry.is_registered("nonexistent")

    def test_cached_instances(self):
        """Test that cached instances are reused."""
        opt1 = CacheOptimizerRegistry.get("anthropic", cached=True)
        opt2 = CacheOptimizerRegistry.get("anthropic", cached=True)
        assert opt1 is opt2

    def test_uncached_instances(self):
        """Test that uncached instances are not reused."""
        opt1 = CacheOptimizerRegistry.get("anthropic", cached=False)
        opt2 = CacheOptimizerRegistry.get("anthropic", cached=False)
        assert opt1 is not opt2

    def test_tier_based_selection(self):
        """Test tier-based optimizer selection."""
        # OSS tier should work
        oss_opt = CacheOptimizerRegistry.get("anthropic", tier="oss")
        assert oss_opt is not None

        # Enterprise tier falls back to OSS if not registered
        ent_opt = CacheOptimizerRegistry.get("anthropic", tier="enterprise")
        assert ent_opt is not None

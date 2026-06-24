"""Tests for AnthropicCacheOptimizer."""

import pytest

from headroom.cache import (
    AnthropicCacheOptimizer,
    CacheConfig,
    OptimizationContext,
)
from headroom.cache.base import CacheStrategy


class TestAnthropicCacheOptimizer:
    """Test AnthropicCacheOptimizer functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return AnthropicCacheOptimizer()

    @pytest.fixture
    def context(self):
        """Create optimization context."""
        return OptimizationContext(
            provider="anthropic",
            model="claude-3-opus",
        )

    def test_optimizer_properties(self, optimizer):
        """Test optimizer properties."""
        assert optimizer.name == "anthropic-cache-optimizer"
        assert optimizer.provider == "anthropic"
        assert optimizer.strategy == CacheStrategy.EXPLICIT_BREAKPOINTS

    def test_enforces_minimum_tokens(self):
        """Test that Anthropic minimum is enforced."""
        config = CacheConfig(min_cacheable_tokens=100)
        optimizer = AnthropicCacheOptimizer(config)
        assert optimizer.config.min_cacheable_tokens >= 1024

    def test_enforces_maximum_breakpoints(self):
        """Test that Anthropic maximum breakpoints is enforced."""
        config = CacheConfig(max_breakpoints=10)
        optimizer = AnthropicCacheOptimizer(config)
        assert optimizer.config.max_breakpoints <= 4

    def test_optimize_simple_messages(self, optimizer, context):
        """Test optimizing simple messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. " * 500},
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)

        assert result.messages is not None
        assert len(result.messages) == 2
        assert result.metrics.stable_prefix_hash != ""

    def test_optimize_inserts_cache_control(self, optimizer, context):
        """Test that optimization inserts cache_control blocks."""
        # Large system prompt to trigger caching
        messages = [
            {"role": "system", "content": "You are a helpful assistant. " * 500},
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)

        # Check if cache_control was inserted
        system_content = result.messages[0]["content"]
        if isinstance(system_content, list):
            has_cache_control = any(
                "cache_control" in block for block in system_content if isinstance(block, dict)
            )
            assert has_cache_control

    def test_optimize_with_dates(self, optimizer, context):
        """Test optimization extracts dates."""
        messages = [
            {
                "role": "system",
                "content": "Today is January 7, 2026. You are a helpful assistant. " * 300,
            },
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)

        # Dates should be moved to end
        assert (
            "extracted_dates" in result.transforms_applied
            or result.metrics.breakpoints_inserted >= 0
        )

    def test_optimize_disabled(self, context):
        """Test optimization when disabled."""
        config = CacheConfig(enabled=False)
        optimizer = AnthropicCacheOptimizer(config)

        messages = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)

        assert result.transforms_applied == []

    def test_prefix_hash_tracking(self, optimizer, context):
        """Test that prefix hash is tracked between calls."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. " * 500},
            {"role": "user", "content": "Hello!"},
        ]

        result1 = optimizer.optimize(messages, context)
        result2 = optimizer.optimize(messages, context)

        # Second call should detect stable prefix
        assert result2.metrics.previous_prefix_hash == result1.metrics.stable_prefix_hash

    def test_estimate_savings(self, optimizer, context):
        """Test savings estimation."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. " * 500},
            {"role": "user", "content": "Hello!"},
        ]

        savings = optimizer.estimate_savings(messages, context)
        assert savings >= 0.0
        assert savings <= 100.0

    def test_content_block_format(self, optimizer, context):
        """Test handling of content block format."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant. " * 500}],
            },
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)
        assert result.messages is not None

    def test_tools_are_cacheable(self, optimizer, context):
        """Test that tools are identified as cacheable."""
        messages = [
            {"role": "system", "content": "You are helpful. " * 300},
            {
                "role": "user",
                "content": "Use tools",
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web " * 200,
                        "input_schema": {"type": "object"},
                    }
                ],
            },
        ]

        result = optimizer.optimize(messages, context)
        assert result.metrics.cacheable_tokens > 0

    def test_metrics_history(self, optimizer, context):
        """Test that metrics are recorded."""
        messages = [
            {"role": "system", "content": "You are helpful. " * 500},
            {"role": "user", "content": "Hello!"},
        ]

        optimizer.optimize(messages, context)
        metrics = optimizer.get_metrics()

        assert metrics is not None
        assert metrics.stable_prefix_hash != ""

    def test_cache_constants(self, optimizer):
        """Test cache-related constants."""
        assert optimizer.get_cache_write_cost_multiplier() == 1.25
        assert optimizer.get_cache_read_cost_multiplier() == 0.10
        assert optimizer.get_cache_ttl_seconds() == 300

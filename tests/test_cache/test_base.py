"""Tests for cache base types and interfaces."""

from headroom.cache.base import (
    BreakpointLocation,
    CacheBreakpoint,
    CacheConfig,
    CacheMetrics,
    CacheResult,
    CacheStrategy,
    OptimizationContext,
)


class TestCacheStrategy:
    """Test CacheStrategy enum."""

    def test_strategies_exist(self):
        """Test all expected strategies exist."""
        assert CacheStrategy.PREFIX_STABILIZATION.value == "prefix_stabilization"
        assert CacheStrategy.EXPLICIT_BREAKPOINTS.value == "explicit_breakpoints"
        assert CacheStrategy.CACHED_CONTENT.value == "cached_content"
        assert CacheStrategy.NONE.value == "none"


class TestCacheConfig:
    """Test CacheConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.min_cacheable_tokens == 1024
        assert config.max_breakpoints == 4
        assert config.normalize_whitespace is True
        assert config.collapse_blank_lines is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = CacheConfig(
            enabled=False,
            min_cacheable_tokens=2048,
            max_breakpoints=2,
        )
        assert config.enabled is False
        assert config.min_cacheable_tokens == 2048
        assert config.max_breakpoints == 2

    def test_date_patterns(self):
        """Test date patterns are set."""
        config = CacheConfig()
        assert len(config.date_patterns) > 0
        assert any("Today" in p for p in config.date_patterns)


class TestCacheMetrics:
    """Test CacheMetrics dataclass."""

    def test_default_values(self):
        """Test default metrics values."""
        metrics = CacheMetrics()
        assert metrics.stable_prefix_tokens == 0
        assert metrics.breakpoints_inserted == 0
        assert metrics.estimated_cache_hit is False
        assert metrics.estimated_savings_percent == 0.0

    def test_custom_values(self):
        """Test custom metrics."""
        metrics = CacheMetrics(
            stable_prefix_tokens=5000,
            breakpoints_inserted=2,
            estimated_cache_hit=True,
            estimated_savings_percent=90.0,
        )
        assert metrics.stable_prefix_tokens == 5000
        assert metrics.breakpoints_inserted == 2
        assert metrics.estimated_cache_hit is True
        assert metrics.estimated_savings_percent == 90.0


class TestCacheBreakpoint:
    """Test CacheBreakpoint dataclass."""

    def test_breakpoint_creation(self):
        """Test creating a breakpoint."""
        bp = CacheBreakpoint(
            message_index=0,
            location=BreakpointLocation.AFTER_SYSTEM,
            tokens_at_breakpoint=2000,
            reason="System prompt is cacheable",
        )
        assert bp.message_index == 0
        assert bp.location == BreakpointLocation.AFTER_SYSTEM
        assert bp.tokens_at_breakpoint == 2000
        assert bp.content_index is None


class TestCacheResult:
    """Test CacheResult dataclass."""

    def test_result_creation(self):
        """Test creating a cache result."""
        messages = [{"role": "system", "content": "Hello"}]
        result = CacheResult(
            messages=messages,
            metrics=CacheMetrics(cacheable_tokens=1000),
            transforms_applied=["normalized_whitespace"],
        )
        assert result.messages == messages
        assert result.metrics.cacheable_tokens == 1000
        assert "normalized_whitespace" in result.transforms_applied

    def test_semantic_cache_hit(self):
        """Test semantic cache hit result."""
        result = CacheResult(
            messages=[],
            semantic_cache_hit=True,
            cached_response={"text": "cached response"},
        )
        assert result.semantic_cache_hit is True
        assert result.cached_response["text"] == "cached response"


class TestOptimizationContext:
    """Test OptimizationContext dataclass."""

    def test_context_creation(self):
        """Test creating optimization context."""
        context = OptimizationContext(
            provider="anthropic",
            model="claude-3-opus",
            request_id="req-123",
        )
        assert context.provider == "anthropic"
        assert context.model == "claude-3-opus"
        assert context.request_id == "req-123"

    def test_default_timestamp(self):
        """Test default timestamp is set."""
        context = OptimizationContext()
        assert context.timestamp is not None

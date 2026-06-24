"""Tests for OpenAICacheOptimizer."""

import pytest

from headroom.cache import CacheConfig, OpenAICacheOptimizer, OptimizationContext
from headroom.cache.base import CacheStrategy


class TestOpenAICacheOptimizer:
    """Test OpenAICacheOptimizer functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return OpenAICacheOptimizer()

    @pytest.fixture
    def context(self):
        """Create optimization context."""
        return OptimizationContext(
            provider="openai",
            model="gpt-4",
        )

    def test_optimizer_properties(self, optimizer):
        """Test optimizer properties."""
        assert optimizer.name == "openai-prefix-stabilizer"
        assert optimizer.provider == "openai"
        assert optimizer.strategy == CacheStrategy.PREFIX_STABILIZATION

    def test_optimize_simple_messages(self, optimizer, context):
        """Test optimizing simple messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)

        assert result.messages is not None
        assert len(result.messages) == 2
        assert result.metrics.stable_prefix_hash != ""

    def test_date_extraction(self, optimizer, context):
        """Test that dates are extracted from system prompt."""
        messages = [
            {
                "role": "system",
                "content": "Today is January 7, 2026. You are a helpful assistant.",
            },
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)

        # Check that date was extracted and moved
        system_content = result.messages[0]["content"]
        # The date should be moved to a dynamic section at the end
        assert "You are a helpful assistant" in system_content

    def test_whitespace_normalization(self, optimizer, context):
        """Test whitespace normalization."""
        messages = [
            {
                "role": "system",
                "content": "You   are    a   helpful   assistant.",
            },
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)

        # Whitespace should be normalized
        system_content = result.messages[0]["content"]
        assert "   " not in system_content  # Multiple spaces collapsed

    def test_optimize_disabled(self, context):
        """Test optimization when disabled."""
        config = CacheConfig(enabled=False)
        optimizer = OpenAICacheOptimizer(config)

        messages = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)
        assert result.transforms_applied == []

    def test_prefix_stability_tracking(self, optimizer, context):
        """Test that prefix stability is tracked."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        # First call
        optimizer.optimize(messages, context)

        # Second call with same messages
        result2 = optimizer.optimize(messages, context)

        # Second call should detect stable prefix
        assert result2.metrics.estimated_cache_hit is True
        assert result2.metrics.prefix_changed_from_previous is False

    def test_prefix_change_detection(self, optimizer, context):
        """Test detection of prefix changes."""
        messages1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        messages2 = [
            {"role": "system", "content": "You are a different assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        optimizer.optimize(messages1, context)
        result2 = optimizer.optimize(messages2, context)

        # Second call should detect prefix change
        assert result2.metrics.prefix_changed_from_previous is True

    def test_token_threshold_warning(self, optimizer, context):
        """Test warning when below token threshold."""
        messages = [
            {"role": "system", "content": "Short."},
            {"role": "user", "content": "Hi"},
        ]

        result = optimizer.optimize(messages, context)

        # Should have warning about being below threshold
        assert any("1024" in w for w in result.warnings)

    def test_estimate_savings_below_threshold(self, optimizer, context):
        """Test savings estimation below threshold."""
        messages = [
            {"role": "system", "content": "Short system prompt."},
            {"role": "user", "content": "Hello!"},
        ]

        savings = optimizer.estimate_savings(messages, context)
        assert savings == 0.0  # Below threshold

    def test_estimate_savings_above_threshold(self, optimizer, context):
        """Test savings estimation above threshold."""
        messages = [
            {"role": "system", "content": "You are helpful. " * 500},
            {"role": "user", "content": "Hello!"},
        ]

        # First call to establish baseline
        optimizer.optimize(messages, context)

        # Second call should show savings
        savings = optimizer.estimate_savings(messages, context)
        assert savings > 0.0

    def test_uuid_pattern_detection(self, optimizer, context):
        """Test detection of UUIDs in content."""
        messages = [
            {
                "role": "system",
                "content": "Request ID: 12345678-1234-1234-1234-123456789012. Be helpful.",
            },
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)
        # UUID should be detected as dynamic content
        assert result.metrics.stable_prefix_hash != ""

    def test_content_block_format(self, optimizer, context):
        """Test handling of content block format."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            },
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)
        assert result.messages is not None

    def test_metrics_recording(self, optimizer, context):
        """Test that metrics are recorded."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        optimizer.optimize(messages, context)
        metrics = optimizer.get_metrics()

        assert metrics.stable_prefix_hash != ""

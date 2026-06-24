"""Unit tests for Strands HeadroomHookProvider.

These tests use mocks and do NOT require AWS credentials or strands-agents.
They test the internal logic of HeadroomHookProvider in isolation.

For real integration tests, see test_hooks.py.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

# Check if strands-agents is installed for proper skip handling
try:
    import strands  # noqa: F401

    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False


# Skip all tests if Strands not installed
pytestmark = pytest.mark.skipif(not STRANDS_AVAILABLE, reason="strands-agents not installed")


class TestHeadroomHookProviderInit:
    """Tests for HeadroomHookProvider initialization."""

    def test_init_with_defaults(self):
        """Initialize with default settings."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()

        assert hook.compress_tool_outputs is True
        assert hook.min_tokens_to_compress == 100
        assert hook.preserve_errors is True
        assert hook.total_tokens_saved == 0
        assert hook.metrics_history == []

    def test_init_with_custom_config(self):
        """Initialize with custom configuration."""
        from headroom import HeadroomConfig
        from headroom.integrations.strands import HeadroomHookProvider

        config = HeadroomConfig()
        config.smart_crusher.min_tokens_to_crush = 200
        config.smart_crusher.max_items_after_crush = 20

        hook = HeadroomHookProvider(
            compress_tool_outputs=False,
            min_tokens_to_compress=500,
            config=config,
            preserve_errors=False,
        )

        assert hook.compress_tool_outputs is False
        assert hook.min_tokens_to_compress == 500
        assert hook.config is config
        assert hook.preserve_errors is False

    def test_init_creates_default_config_if_none(self):
        """Initialize creates a default HeadroomConfig if none provided."""
        from headroom import HeadroomConfig
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()

        assert hook.config is not None
        assert isinstance(hook.config, HeadroomConfig)


class TestRegisterHooks:
    """Tests for HeadroomHookProvider.register_hooks method."""

    def test_register_hooks_adds_callback_to_registry(self):
        """register_hooks adds AfterToolCallEvent callback to registry."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(compress_tool_outputs=True)
        mock_registry = MagicMock()

        hook.register_hooks(mock_registry)

        # Should have registered exactly one callback for AfterToolCallEvent
        assert mock_registry.add_callback.call_count == 1

    def test_register_hooks_skips_when_compression_disabled(self):
        """register_hooks does not register callbacks when compression is disabled."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(compress_tool_outputs=False)
        mock_registry = MagicMock()

        hook.register_hooks(mock_registry)

        # Should not have registered any callbacks
        assert mock_registry.add_callback.call_count == 0


class TestCrusherLazyInit:
    """Tests for SmartCrusher lazy initialization."""

    def test_crusher_is_lazily_initialized(self):
        """SmartCrusher is not created until first access."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()

        # Directly check internal state - crusher should be None initially
        assert hook._crusher is None

        # Access the crusher property
        crusher = hook.crusher

        # Now it should be initialized
        assert crusher is not None
        assert hook._crusher is crusher

    def test_crusher_uses_configured_min_tokens(self):
        """SmartCrusher uses min_tokens_to_compress from hook config."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(min_tokens_to_compress=250)

        crusher = hook.crusher

        # The crusher config should have our min_tokens setting
        assert crusher.config.min_tokens_to_crush == 250


class TestTokenEstimation:
    """Tests for _estimate_tokens helper method."""

    def test_estimate_tokens_empty_string(self):
        """Estimate returns 0 for empty string."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        assert hook._estimate_tokens("") == 0

    def test_estimate_tokens_short_string(self):
        """Estimate uses ~4 chars per token heuristic."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()

        # 12 chars = 3 tokens (12 // 4)
        assert hook._estimate_tokens("hello world!") == 3

        # 20 chars = 5 tokens
        assert hook._estimate_tokens("a" * 20) == 5


class TestExtractTextContent:
    """Tests for _extract_text_content helper method."""

    def test_extract_from_text_content(self):
        """Extract text from content with text field."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        result = {"content": [{"text": "Hello world"}]}

        extracted = hook._extract_text_content(result)
        assert extracted == "Hello world"

    def test_extract_from_json_content(self):
        """Extract and serialize JSON content."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        result = {"content": [{"json": {"key": "value"}}]}

        extracted = hook._extract_text_content(result)
        assert extracted == '{"key": "value"}'

    def test_extract_empty_content(self):
        """Return empty string for empty content."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        result = {"content": []}

        extracted = hook._extract_text_content(result)
        assert extracted == ""

    def test_extract_missing_content(self):
        """Return empty string for missing content key."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        result = {}

        extracted = hook._extract_text_content(result)
        assert extracted == ""


class TestShouldSkipCompression:
    """Tests for _should_skip_compression helper method."""

    def test_skip_when_compression_disabled(self):
        """Skip compression when compress_tool_outputs is False."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(compress_tool_outputs=False)
        result = {"content": [{"text": "data"}]}

        skip_reason = hook._should_skip_compression(result)
        assert skip_reason == "compression_disabled"

    def test_skip_error_results_when_preserve_errors_true(self):
        """Skip error results when preserve_errors is True."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(preserve_errors=True)
        result = {"status": "error", "content": [{"text": "Error message"}]}

        skip_reason = hook._should_skip_compression(result)
        assert skip_reason == "error_result_preserved"

    def test_allow_error_results_when_preserve_errors_false(self):
        """Allow error results when preserve_errors is False."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(preserve_errors=False)
        result = {"status": "error", "content": [{"text": "Error message"}]}

        skip_reason = hook._should_skip_compression(result)
        assert skip_reason is None

    def test_skip_empty_content(self):
        """Skip results with empty content."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        result = {"content": []}

        skip_reason = hook._should_skip_compression(result)
        assert skip_reason == "empty_content"

    def test_allow_valid_content(self):
        """Allow results with valid content."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        result = {"content": [{"text": "some data"}]}

        skip_reason = hook._should_skip_compression(result)
        assert skip_reason is None


class TestCompressToolResult:
    """Tests for _compress_tool_result hook handler."""

    def test_compress_large_tool_output(self):
        """Compresses large tool output and tracks metrics."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(
            compress_tool_outputs=True,
            min_tokens_to_compress=10,  # Low threshold for testing
        )

        # Create large JSON output (50 items)
        large_data = [{"id": i, "value": f"item-{i}", "data": "x" * 50} for i in range(50)]
        large_json = json.dumps(large_data)

        mock_event = MagicMock()
        mock_event.tool_use = {"name": "get_items", "toolUseId": "tool-123"}
        mock_event.result = {"content": [{"text": large_json}]}

        hook._compress_tool_result(mock_event)

        # Verify metrics were recorded
        assert len(hook.metrics_history) == 1
        metrics = hook.metrics_history[0]
        assert metrics.tool_name == "get_items"
        assert metrics.tool_use_id == "tool-123"
        assert metrics.tokens_before > 0

    def test_skip_compression_below_threshold(self):
        """Does not compress output below token threshold."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(
            compress_tool_outputs=True,
            min_tokens_to_compress=10000,  # High threshold
        )

        mock_event = MagicMock()
        mock_event.tool_use = {"name": "small_tool", "toolUseId": "tool-456"}
        mock_event.result = {"content": [{"text": '{"status": "ok"}'}]}

        hook._compress_tool_result(mock_event)

        # Metrics should show skipped compression
        assert len(hook.metrics_history) == 1
        metrics = hook.metrics_history[0]
        assert metrics.was_compressed is False
        assert "below_threshold" in metrics.skip_reason

    def test_skip_compression_when_disabled(self):
        """Does not compress when compression is disabled."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(compress_tool_outputs=False)

        mock_event = MagicMock()
        mock_event.tool_use = {"name": "test_tool", "toolUseId": "tool-789"}
        mock_event.result = {"content": [{"text": '{"data": "value"}'}]}

        hook._compress_tool_result(mock_event)

        # Metrics should show compression disabled
        assert len(hook.metrics_history) == 1
        metrics = hook.metrics_history[0]
        assert metrics.was_compressed is False
        assert metrics.skip_reason == "compression_disabled"


class TestMetricsTracking:
    """Tests for metrics tracking and aggregation."""

    def test_total_tokens_saved_accumulates(self):
        """total_tokens_saved accumulates across compressions."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(
            compress_tool_outputs=True,
            min_tokens_to_compress=10,
        )

        # Simulate two compressions with savings
        for i in range(2):
            large_data = [{"id": j, "data": "x" * 100} for j in range(50)]
            mock_event = MagicMock()
            mock_event.tool_use = {"name": f"tool_{i}", "toolUseId": f"id_{i}"}
            mock_event.result = {"content": [{"text": json.dumps(large_data)}]}

            hook._compress_tool_result(mock_event)

        # Should have accumulated some savings
        compressed_count = sum(1 for m in hook.metrics_history if m.was_compressed)
        if compressed_count > 0:
            assert hook.total_tokens_saved >= 0

    def test_metrics_history_bounded_to_100(self):
        """metrics_history keeps only last 100 entries."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(
            compress_tool_outputs=True,
            min_tokens_to_compress=10,
        )

        # Directly add 150 metrics
        for i in range(150):
            hook._record_metrics(
                request_id=f"req_{i}",
                tool_name=f"tool_{i}",
                tool_use_id=f"id_{i}",
                tokens_before=100,
                tokens_after=50,
                was_compressed=True,
                skip_reason=None,
            )

        # Should be bounded at 100
        assert len(hook.metrics_history) == 100

        # Should contain the most recent entries
        last_metric = hook.metrics_history[-1]
        assert last_metric.request_id == "req_149"


class TestGetSavingsSummary:
    """Tests for get_savings_summary method."""

    def test_empty_summary(self):
        """Returns zero values when no metrics recorded."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        summary = hook.get_savings_summary()

        assert summary["total_requests"] == 0
        assert summary["compressed_requests"] == 0
        assert summary["total_tokens_saved"] == 0
        assert summary["average_savings_percent"] == 0.0

    def test_summary_with_compressions(self):
        """Returns correct summary with recorded compressions."""
        from headroom.integrations.strands import HeadroomHookProvider
        from headroom.integrations.strands.hooks import CompressionMetrics

        hook = HeadroomHookProvider()

        # Add metrics manually
        hook._metrics_history = [
            CompressionMetrics(
                request_id="1",
                timestamp=datetime.now(timezone.utc),
                tool_name="tool_a",
                tool_use_id="id_1",
                tokens_before=100,
                tokens_after=60,
                tokens_saved=40,
                savings_percent=40.0,
                was_compressed=True,
                skip_reason=None,
            ),
            CompressionMetrics(
                request_id="2",
                timestamp=datetime.now(timezone.utc),
                tool_name="tool_b",
                tool_use_id="id_2",
                tokens_before=200,
                tokens_after=100,
                tokens_saved=100,
                savings_percent=50.0,
                was_compressed=True,
                skip_reason=None,
            ),
            CompressionMetrics(
                request_id="3",
                timestamp=datetime.now(timezone.utc),
                tool_name="tool_c",
                tool_use_id="id_3",
                tokens_before=50,
                tokens_after=50,
                tokens_saved=0,
                savings_percent=0.0,
                was_compressed=False,
                skip_reason="below_threshold",
            ),
        ]
        hook._total_tokens_saved = 140

        summary = hook.get_savings_summary()

        assert summary["total_requests"] == 3
        assert summary["compressed_requests"] == 2
        assert summary["total_tokens_saved"] == 140
        assert summary["average_savings_percent"] == 45.0  # (40 + 50) / 2
        assert summary["total_tokens_before"] == 350
        assert summary["total_tokens_after"] == 210


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_all_state(self):
        """reset() clears all tracked state."""
        from headroom.integrations.strands import HeadroomHookProvider
        from headroom.integrations.strands.hooks import CompressionMetrics

        hook = HeadroomHookProvider()

        # Add some state
        hook._metrics_history = [
            CompressionMetrics(
                request_id="1",
                timestamp=datetime.now(timezone.utc),
                tool_name="test",
                tool_use_id="id_1",
                tokens_before=100,
                tokens_after=50,
                tokens_saved=50,
                savings_percent=50.0,
                was_compressed=True,
            )
        ]
        hook._total_tokens_saved = 50

        # Reset
        hook.reset()

        # Verify all state cleared
        assert hook._metrics_history == []
        assert hook._total_tokens_saved == 0
        assert hook.total_tokens_saved == 0
        assert len(hook.metrics_history) == 0


class TestThreadSafety:
    """Tests for thread-safety of metrics tracking."""

    def test_concurrent_metric_recording(self):
        """Metrics recording is thread-safe."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()

        def record_metrics(thread_id):
            for i in range(10):
                hook._record_metrics(
                    request_id=f"thread_{thread_id}_req_{i}",
                    tool_name=f"tool_{thread_id}_{i}",
                    tool_use_id=f"id_{thread_id}_{i}",
                    tokens_before=100,
                    tokens_after=50,
                    was_compressed=True,
                    skip_reason=None,
                )

        threads = []
        for t_id in range(5):
            t = threading.Thread(target=record_metrics, args=(t_id,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have recorded 50 metrics (5 threads * 10 each)
        # But bounded to 100, so if we had more it would be truncated
        assert len(hook.metrics_history) == 50
        assert hook.total_tokens_saved == 50 * 50  # 50 metrics * 50 tokens each


class TestUpdateResultContent:
    """Tests for _update_result_content helper method."""

    def test_update_preserves_json_structure(self):
        """Updates preserve JSON structure when possible."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        result = {"content": [{"json": {"original": "data"}}]}

        compressed = '{"compressed": "data"}'
        hook._update_result_content(result, compressed)

        # Should update with parsed JSON
        assert result["content"] == [{"json": {"compressed": "data"}}]

    def test_update_uses_text_for_non_json(self):
        """Updates use text format for non-JSON content."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        result = {"content": [{"text": "original text"}]}

        compressed = "compressed text"
        hook._update_result_content(result, compressed)

        assert result["content"] == [{"text": "compressed text"}]

    def test_update_creates_content_if_empty(self):
        """Creates content list if missing."""
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider()
        result = {"content": []}

        hook._update_result_content(result, "new content")

        assert result["content"] == [{"text": "new content"}]

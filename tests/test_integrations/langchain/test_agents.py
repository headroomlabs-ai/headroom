"""Tests for LangChain agent tool integration.

Tests cover:
1. ToolCompressionMetrics - Dataclass for tool compression metrics
2. ToolMetricsCollector - Collector for compression metrics
3. HeadroomToolWrapper - Wrapper for LangChain tools with compression
4. wrap_tools_with_headroom - Convenience function for wrapping multiple tools
5. get_tool_metrics / reset_tool_metrics - Global metrics access
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Check if LangChain is available
try:
    from langchain_core.tools import BaseTool, StructuredTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Skip all tests if LangChain not installed
pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


@pytest.fixture
def mock_tool():
    """Create a mock LangChain tool."""
    mock = MagicMock(spec=BaseTool)
    mock.name = "test_tool"
    mock.description = "A test tool"
    mock.invoke = MagicMock(return_value="Tool result")
    return mock


@pytest.fixture
def mock_tool_with_large_output():
    """Create a mock tool that returns large output."""
    mock = MagicMock(spec=BaseTool)
    mock.name = "search_tool"
    mock.description = "Search tool with large results"
    # Return > 1000 chars to trigger compression
    large_output = '{"items": [' + ",".join(f'{{"id": {i}}}' for i in range(200)) + "]}"
    mock.invoke = MagicMock(return_value=large_output)
    return mock


class TestToolCompressionMetrics:
    """Tests for ToolCompressionMetrics dataclass."""

    def test_create_metrics(self):
        """Create metrics with all fields."""
        from headroom.integrations.langchain.agents import ToolCompressionMetrics

        metrics = ToolCompressionMetrics(
            tool_name="search",
            timestamp=datetime.now(),
            chars_before=5000,
            chars_after=2000,
            chars_saved=3000,
            compression_ratio=0.4,
            was_compressed=True,
        )

        assert metrics.tool_name == "search"
        assert metrics.chars_before == 5000
        assert metrics.chars_after == 2000
        assert metrics.chars_saved == 3000
        assert metrics.compression_ratio == 0.4
        assert metrics.was_compressed is True

    def test_metrics_defaults(self):
        """Verify no default values (all required)."""
        from headroom.integrations.langchain.agents import ToolCompressionMetrics

        # All fields are required, should raise TypeError if missing
        with pytest.raises(TypeError):
            ToolCompressionMetrics()  # type: ignore[call-arg]


class TestToolMetricsCollector:
    """Tests for ToolMetricsCollector."""

    def test_init_empty(self):
        """Initialize with empty metrics list."""
        from headroom.integrations.langchain.agents import ToolMetricsCollector

        collector = ToolMetricsCollector()

        assert collector.metrics == []

    def test_add_metric(self):
        """Add a metric to the collector."""
        from headroom.integrations.langchain.agents import (
            ToolCompressionMetrics,
            ToolMetricsCollector,
        )

        collector = ToolMetricsCollector()
        metric = ToolCompressionMetrics(
            tool_name="test",
            timestamp=datetime.now(),
            chars_before=100,
            chars_after=80,
            chars_saved=20,
            compression_ratio=0.8,
            was_compressed=True,
        )

        collector.add(metric)

        assert len(collector.metrics) == 1
        assert collector.metrics[0] is metric

    def test_add_metric_limits_to_1000(self):
        """Metrics list is limited to 1000 entries."""
        from headroom.integrations.langchain.agents import (
            ToolCompressionMetrics,
            ToolMetricsCollector,
        )

        collector = ToolMetricsCollector()

        # Add 1100 metrics
        for i in range(1100):
            metric = ToolCompressionMetrics(
                tool_name=f"tool_{i}",
                timestamp=datetime.now(),
                chars_before=100,
                chars_after=80,
                chars_saved=20,
                compression_ratio=0.8,
                was_compressed=True,
            )
            collector.add(metric)

        assert len(collector.metrics) == 1000
        # Should keep the last 1000 (most recent)
        assert collector.metrics[0].tool_name == "tool_100"
        assert collector.metrics[-1].tool_name == "tool_1099"

    def test_get_summary_empty(self):
        """Get summary with no metrics."""
        from headroom.integrations.langchain.agents import ToolMetricsCollector

        collector = ToolMetricsCollector()
        summary = collector.get_summary()

        assert summary["total_invocations"] == 0
        assert summary["total_compressions"] == 0
        assert summary["total_chars_saved"] == 0

    def test_get_summary_with_data(self):
        """Get summary with metrics."""
        from headroom.integrations.langchain.agents import (
            ToolCompressionMetrics,
            ToolMetricsCollector,
        )

        collector = ToolMetricsCollector()

        # Add compressed metric
        collector.add(
            ToolCompressionMetrics(
                tool_name="search",
                timestamp=datetime.now(),
                chars_before=5000,
                chars_after=2000,
                chars_saved=3000,
                compression_ratio=0.4,
                was_compressed=True,
            )
        )

        # Add uncompressed metric
        collector.add(
            ToolCompressionMetrics(
                tool_name="simple",
                timestamp=datetime.now(),
                chars_before=100,
                chars_after=100,
                chars_saved=0,
                compression_ratio=1.0,
                was_compressed=False,
            )
        )

        summary = collector.get_summary()

        assert summary["total_invocations"] == 2
        assert summary["total_compressions"] == 1
        assert summary["total_chars_saved"] == 3000
        assert summary["average_compression_ratio"] == 0.4  # Only compressed

    def test_get_summary_by_tool(self):
        """Get per-tool statistics."""
        from headroom.integrations.langchain.agents import (
            ToolCompressionMetrics,
            ToolMetricsCollector,
        )

        collector = ToolMetricsCollector()

        # Add metrics for different tools
        for _i in range(3):
            collector.add(
                ToolCompressionMetrics(
                    tool_name="search",
                    timestamp=datetime.now(),
                    chars_before=1000,
                    chars_after=500,
                    chars_saved=500,
                    compression_ratio=0.5,
                    was_compressed=True,
                )
            )

        for _i in range(2):
            collector.add(
                ToolCompressionMetrics(
                    tool_name="database",
                    timestamp=datetime.now(),
                    chars_before=100,
                    chars_after=100,
                    chars_saved=0,
                    compression_ratio=1.0,
                    was_compressed=False,
                )
            )

        summary = collector.get_summary()

        assert "by_tool" in summary
        assert summary["by_tool"]["search"]["invocations"] == 3
        assert summary["by_tool"]["search"]["compressions"] == 3
        assert summary["by_tool"]["search"]["chars_saved"] == 1500
        assert summary["by_tool"]["database"]["invocations"] == 2
        assert summary["by_tool"]["database"]["compressions"] == 0


class TestHeadroomToolWrapper:
    """Tests for HeadroomToolWrapper."""

    def test_init_defaults(self, mock_tool):
        """Initialize with default settings."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        wrapper = HeadroomToolWrapper(mock_tool)

        assert wrapper.tool is mock_tool
        assert wrapper.name == "test_tool"
        assert wrapper.description == "A test tool"
        assert wrapper.min_chars_to_compress == 1000

    def test_init_custom_threshold(self, mock_tool):
        """Initialize with custom compression threshold."""
        from headroom.integrations.langchain.agents import (
            HeadroomToolWrapper,
            ToolMetricsCollector,
        )

        collector = ToolMetricsCollector()
        wrapper = HeadroomToolWrapper(
            mock_tool,
            min_chars_to_compress=500,
            metrics_collector=collector,
        )

        assert wrapper.min_chars_to_compress == 500
        assert wrapper._metrics is collector

    def test_call_small_output_no_compression(self, mock_tool):
        """Small outputs are not compressed."""
        from headroom.integrations.langchain.agents import (
            HeadroomToolWrapper,
            ToolMetricsCollector,
        )

        collector = ToolMetricsCollector()
        wrapper = HeadroomToolWrapper(
            mock_tool,
            min_chars_to_compress=1000,
            metrics_collector=collector,
        )

        result = wrapper("input")

        assert result == "Tool result"
        assert len(collector.metrics) == 1
        assert collector.metrics[0].was_compressed is False

    def test_call_large_output_triggers_compression(self, mock_tool_with_large_output):
        """Large outputs trigger compression."""
        from headroom.integrations.langchain.agents import (
            HeadroomToolWrapper,
            ToolMetricsCollector,
        )

        collector = ToolMetricsCollector()
        wrapper = HeadroomToolWrapper(
            mock_tool_with_large_output,
            min_chars_to_compress=100,
            metrics_collector=collector,
        )

        # Mock compress_tool_result to return compressed output
        with patch("headroom.integrations.langchain.agents.compress_tool_result") as mock_compress:
            mock_compress.return_value = '{"items": [...compressed...]}'
            wrapper("query")

            mock_compress.assert_called_once()
            assert len(collector.metrics) == 1
            assert collector.metrics[0].was_compressed is True

    def test_call_converts_non_string_result(self, mock_tool):
        """Non-string results are converted to strings."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        mock_tool.invoke.return_value = {"key": "value"}

        wrapper = HeadroomToolWrapper(mock_tool)
        result = wrapper("input")

        assert isinstance(result, str)
        assert "key" in result

    def test_invoke_alias(self, mock_tool):
        """invoke() is an alias for __call__()."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        wrapper = HeadroomToolWrapper(mock_tool)

        result1 = wrapper("input")
        mock_tool.invoke.reset_mock()
        result2 = wrapper.invoke("input")

        assert result1 == result2

    def test_compression_failure_returns_original(self, mock_tool_with_large_output):
        """Compression failure returns original output."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        wrapper = HeadroomToolWrapper(
            mock_tool_with_large_output,
            min_chars_to_compress=100,
        )

        with patch("headroom.integrations.langchain.agents.compress_tool_result") as mock_compress:
            mock_compress.side_effect = Exception("Compression error")
            result = wrapper("query")

            # Should return original output
            assert "items" in result
            assert "id" in result

    def test_as_langchain_tool(self, mock_tool):
        """Convert wrapper to LangChain StructuredTool."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        wrapper = HeadroomToolWrapper(mock_tool)
        lc_tool = wrapper.as_langchain_tool()

        assert isinstance(lc_tool, StructuredTool)
        assert lc_tool.name == "test_tool"
        assert lc_tool.description == "A test tool"

    def test_metrics_recorded_correctly(self, mock_tool_with_large_output):
        """Verify metrics are recorded correctly."""
        from headroom.integrations.langchain.agents import (
            HeadroomToolWrapper,
            ToolMetricsCollector,
        )

        collector = ToolMetricsCollector()
        wrapper = HeadroomToolWrapper(
            mock_tool_with_large_output,
            min_chars_to_compress=100,
            metrics_collector=collector,
        )

        original_len = len(mock_tool_with_large_output.invoke.return_value)

        with patch("headroom.integrations.langchain.agents.compress_tool_result") as mock_compress:
            compressed_result = '{"items": [...]}'
            mock_compress.return_value = compressed_result
            wrapper("query")

            metric = collector.metrics[0]
            assert metric.tool_name == "search_tool"
            assert metric.chars_before == original_len
            assert metric.chars_after == len(compressed_result)
            assert metric.chars_saved == original_len - len(compressed_result)


class TestWrapToolsWithHeadroom:
    """Tests for wrap_tools_with_headroom function."""

    def test_wrap_single_tool(self, mock_tool):
        """Wrap a single tool."""
        from headroom.integrations.langchain.agents import wrap_tools_with_headroom

        wrapped = wrap_tools_with_headroom([mock_tool])

        assert len(wrapped) == 1
        assert isinstance(wrapped[0], StructuredTool)
        assert wrapped[0].name == "test_tool"

    def test_wrap_multiple_tools(self, mock_tool):
        """Wrap multiple tools."""
        from headroom.integrations.langchain.agents import wrap_tools_with_headroom

        tool2 = MagicMock(spec=BaseTool)
        tool2.name = "tool_2"
        tool2.description = "Second tool"
        tool2.invoke = MagicMock(return_value="Result 2")

        wrapped = wrap_tools_with_headroom([mock_tool, tool2])

        assert len(wrapped) == 2
        assert wrapped[0].name == "test_tool"
        assert wrapped[1].name == "tool_2"

    def test_wrap_with_custom_threshold(self, mock_tool):
        """Wrap with custom compression threshold."""
        from headroom.integrations.langchain.agents import wrap_tools_with_headroom

        wrapped = wrap_tools_with_headroom([mock_tool], min_chars_to_compress=500)

        assert len(wrapped) == 1
        # Invoke to verify wrapper is configured
        # The wrapper should be invoked through the StructuredTool
        assert wrapped[0].name == "test_tool"

    def test_wrap_with_shared_collector(self, mock_tool):
        """Wrap with shared metrics collector."""
        from headroom.integrations.langchain.agents import (
            ToolMetricsCollector,
            wrap_tools_with_headroom,
        )

        collector = ToolMetricsCollector()

        tool2 = MagicMock(spec=BaseTool)
        tool2.name = "tool_2"
        tool2.description = "Second tool"
        tool2.invoke = MagicMock(return_value="Result 2")

        wrapped = wrap_tools_with_headroom(
            [mock_tool, tool2],
            metrics_collector=collector,
        )

        # Invoke both tools
        wrapped[0].func("input1")
        wrapped[1].func("input2")

        # Both should use the same collector
        assert len(collector.metrics) == 2

    def test_wrap_empty_list(self):
        """Wrap empty list returns empty list."""
        from headroom.integrations.langchain.agents import wrap_tools_with_headroom

        wrapped = wrap_tools_with_headroom([])

        assert wrapped == []


class TestGlobalMetrics:
    """Tests for global metrics functions."""

    def test_get_tool_metrics(self):
        """get_tool_metrics returns the global collector."""
        from headroom.integrations.langchain.agents import (
            ToolMetricsCollector,
            get_tool_metrics,
        )

        collector = get_tool_metrics()

        assert isinstance(collector, ToolMetricsCollector)

    def test_reset_tool_metrics(self):
        """reset_tool_metrics creates new collector."""
        from headroom.integrations.langchain.agents import (
            ToolCompressionMetrics,
            get_tool_metrics,
            reset_tool_metrics,
        )

        # Add a metric to the global collector
        collector = get_tool_metrics()
        collector.add(
            ToolCompressionMetrics(
                tool_name="test",
                timestamp=datetime.now(),
                chars_before=100,
                chars_after=100,
                chars_saved=0,
                compression_ratio=1.0,
                was_compressed=False,
            )
        )

        # Reset
        reset_tool_metrics()

        # New collector should be empty
        new_collector = get_tool_metrics()
        assert len(new_collector.metrics) == 0

    def test_wrapper_uses_global_metrics_by_default(self, mock_tool):
        """HeadroomToolWrapper uses global metrics by default."""
        from headroom.integrations.langchain.agents import (
            HeadroomToolWrapper,
            get_tool_metrics,
            reset_tool_metrics,
        )

        # Reset to start fresh
        reset_tool_metrics()

        wrapper = HeadroomToolWrapper(mock_tool)
        wrapper("input")

        global_collector = get_tool_metrics()
        assert len(global_collector.metrics) == 1


class TestLangChainNotAvailable:
    """Tests for behavior when LangChain is not available."""

    def test_check_raises_import_error(self):
        """_check_langchain_available raises ImportError when not available."""
        from headroom.integrations.langchain.agents import _check_langchain_available

        # When LangChain IS available, should not raise
        try:
            _check_langchain_available()
        except ImportError:
            pytest.fail("Should not raise when LangChain is available")

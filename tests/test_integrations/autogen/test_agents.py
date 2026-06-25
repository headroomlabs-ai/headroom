"""Tests for AutoGen agent tool integration.

Tests cover:
1. ToolCompressionMetrics - Dataclass for tool compression metrics
2. ToolMetricsCollector - Collector for compression metrics
3. HeadroomToolWrapper - Wrapper for AutoGen FunctionTool with compression
4. wrap_tools_with_headroom - Convenience function for wrapping multiple tools
5. get_tool_metrics / reset_tool_metrics - Global metrics access
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import patch

import pytest

try:
    from autogen_core import CancellationToken
    from autogen_core.tools import FunctionTool

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

pytestmark = pytest.mark.skipif(not AUTOGEN_AVAILABLE, reason="AutoGen not installed")


def _make_large_output(n: int = 200) -> str:
    """Create a large JSON string to trigger compression."""
    return json.dumps({"items": [{"id": i, "data": "x" * 50} for i in range(n)]})


def _run_async(coro):
    """Helper to run async code in tests."""
    return asyncio.run(coro)


# Sample tool functions


def big_lookup(query: str) -> str:
    """Look up data and return a large result."""
    return _make_large_output()


def small_lookup(query: str) -> str:
    """Look up data and return a small result."""
    return "ok"


async def async_lookup(query: str) -> str:
    """Async tool that returns a large result."""
    return _make_large_output()


class TestToolCompressionMetrics:
    """Tests for ToolCompressionMetrics dataclass."""

    def test_create_metrics(self):
        from headroom.integrations.autogen.agents import ToolCompressionMetrics

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
        assert metrics.chars_saved == 3000
        assert metrics.was_compressed is True

    def test_metrics_all_fields_required(self):
        from headroom.integrations.autogen.agents import ToolCompressionMetrics

        with pytest.raises(TypeError):
            ToolCompressionMetrics()  # type: ignore[call-arg]


class TestToolMetricsCollector:
    """Tests for ToolMetricsCollector."""

    def test_empty_summary(self):
        from headroom.integrations.autogen.agents import ToolMetricsCollector

        collector = ToolMetricsCollector()
        summary = collector.get_summary()
        assert summary["total_invocations"] == 0
        assert summary["total_compressions"] == 0

    def test_add_and_summary(self):
        from headroom.integrations.autogen.agents import (
            ToolCompressionMetrics,
            ToolMetricsCollector,
        )

        collector = ToolMetricsCollector()
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

        summary = collector.get_summary()
        assert summary["total_invocations"] == 1
        assert summary["total_compressions"] == 1
        assert summary["total_chars_saved"] == 3000
        assert "search" in summary["by_tool"]


class TestGlobalMetrics:
    """Tests for global metrics functions."""

    def test_get_and_reset(self):
        from headroom.integrations.autogen.agents import get_tool_metrics, reset_tool_metrics

        metrics = get_tool_metrics()
        assert metrics is not None
        reset_tool_metrics()
        assert get_tool_metrics() is not metrics


class TestHeadroomToolWrapper:
    """Tests for HeadroomToolWrapper."""

    @patch("headroom.integrations.autogen.agents.compress_tool_result")
    def test_skips_short_output(self, mock_compress):
        from headroom.integrations.autogen.agents import HeadroomToolWrapper, ToolMetricsCollector

        tool = FunctionTool(small_lookup, description="Small", name="small_lookup")
        collector = ToolMetricsCollector()
        wrapper = HeadroomToolWrapper(tool, min_chars_to_compress=1000, metrics_collector=collector)

        result = _run_async(wrapper.wrapped_tool.run_json({"query": "test"}, CancellationToken()))
        assert str(result) == "ok"
        mock_compress.assert_not_called()
        assert collector.get_summary()["total_compressions"] == 0

    @patch("headroom.integrations.autogen.agents.compress_tool_result")
    def test_compresses_large_output(self, mock_compress):
        from headroom.integrations.autogen.agents import HeadroomToolWrapper, ToolMetricsCollector

        mock_compress.return_value = "compressed"
        tool = FunctionTool(big_lookup, description="Big", name="big_lookup")
        collector = ToolMetricsCollector()
        wrapper = HeadroomToolWrapper(tool, min_chars_to_compress=100, metrics_collector=collector)

        result = _run_async(wrapper.wrapped_tool.run_json({"query": "test"}, CancellationToken()))
        assert str(result) == "compressed"
        mock_compress.assert_called_once()
        assert collector.get_summary()["total_compressions"] == 1

    @patch(
        "headroom.integrations.autogen.agents.compress_tool_result",
        side_effect=RuntimeError("boom"),
    )
    def test_passes_through_on_error(self, mock_compress):
        from headroom.integrations.autogen.agents import HeadroomToolWrapper, ToolMetricsCollector

        large = _make_large_output()
        tool = FunctionTool(big_lookup, description="Big", name="big_lookup")
        collector = ToolMetricsCollector()
        wrapper = HeadroomToolWrapper(tool, min_chars_to_compress=100, metrics_collector=collector)

        result = _run_async(wrapper.wrapped_tool.run_json({"query": "test"}, CancellationToken()))
        assert str(result) == large
        assert collector.get_summary()["total_compressions"] == 0

    def test_preserves_tool_metadata(self):
        from headroom.integrations.autogen.agents import HeadroomToolWrapper

        tool = FunctionTool(big_lookup, description="Look up data", name="big_lookup")
        wrapper = HeadroomToolWrapper(tool)
        assert wrapper.name == "big_lookup"
        assert wrapper.description == "Look up data"
        assert wrapper.wrapped_tool.name == "big_lookup"

    @patch("headroom.integrations.autogen.agents.compress_tool_result")
    def test_wraps_async_tool(self, mock_compress):
        from headroom.integrations.autogen.agents import HeadroomToolWrapper, ToolMetricsCollector

        mock_compress.return_value = "compressed"
        tool = FunctionTool(async_lookup, description="Async", name="async_lookup")
        collector = ToolMetricsCollector()
        wrapper = HeadroomToolWrapper(tool, min_chars_to_compress=100, metrics_collector=collector)

        result = _run_async(wrapper.wrapped_tool.run_json({"query": "test"}, CancellationToken()))
        assert str(result) == "compressed"
        assert collector.get_summary()["total_compressions"] == 1


class TestWrapToolsWithHeadroom:
    """Tests for wrap_tools_with_headroom convenience function."""

    @patch("headroom.integrations.autogen.agents.compress_tool_result")
    def test_wraps_multiple_tools(self, mock_compress):
        from headroom.integrations.autogen.agents import wrap_tools_with_headroom

        tool1 = FunctionTool(big_lookup, description="Big", name="big_lookup")
        tool2 = FunctionTool(small_lookup, description="Small", name="small_lookup")

        wrapped = wrap_tools_with_headroom([tool1, tool2])
        assert len(wrapped) == 2
        assert wrapped[0].name == "big_lookup"
        assert wrapped[1].name == "small_lookup"

    @patch("headroom.integrations.autogen.agents.compress_tool_result")
    def test_shared_metrics(self, mock_compress):
        from headroom.integrations.autogen.agents import (
            ToolMetricsCollector,
            wrap_tools_with_headroom,
        )

        mock_compress.return_value = "compressed"
        tool1 = FunctionTool(big_lookup, description="Big", name="big_lookup")
        tool2 = FunctionTool(big_lookup, description="Big2", name="big_lookup_2")

        collector = ToolMetricsCollector()
        wrapped = wrap_tools_with_headroom(
            [tool1, tool2],
            min_chars_to_compress=100,
            metrics_collector=collector,
        )

        _run_async(wrapped[0].run_json({"query": "a"}, CancellationToken()))
        _run_async(wrapped[1].run_json({"query": "b"}, CancellationToken()))

        summary = collector.get_summary()
        assert summary["total_invocations"] == 2
        assert summary["total_compressions"] == 2

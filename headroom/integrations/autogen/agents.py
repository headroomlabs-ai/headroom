"""AutoGen agent tool integration with output compression.

This module provides HeadroomToolWrapper and wrap_tools_with_headroom
for wrapping AutoGen FunctionTool instances to automatically compress
their outputs and track per-tool compression metrics.

AutoGen (autogen-agentchat >=0.7) routes tool execution through a
Workbench abstraction. FunctionTool wraps a plain Python function;
the function's return value is stringified and becomes the
FunctionExecutionResult.content that enters model_context.

Interception strategy: wrap the callable inside FunctionTool so the
return value is compressed before AutoGen stringifies it. This is
the same pattern as the LangChain/CrewAI tool wrappers.

Note: AutoGen's ``tool_call_summary_formatter`` parameter on
AssistantAgent only controls the *final summary* emitted after the
tool loop, not what enters model_context. Wrapping the function
is the only clean, version-stable hook.

Example:
    from autogen_core.tools import FunctionTool
    from headroom.integrations.autogen import wrap_tools_with_headroom

    def search_database(query: str) -> str:
        \"\"\"Search the database.\"\"\"
        return json.dumps({"results": [...], "total": 1000})

    tool = FunctionTool(search_database, description="Search")
    wrapped = wrap_tools_with_headroom([tool])
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any

try:
    from autogen_core.tools import FunctionTool

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    FunctionTool = object  # type: ignore[misc,assignment]

# Re-exported: callers import ToolCompressionMetrics from this module.
from headroom.integrations._metrics import (  # noqa: F401
    ToolCompressionMetrics,
    ToolMetricsCollector,
    _record_tool_metrics,
)
from headroom.integrations.mcp import compress_tool_result

logger = logging.getLogger(__name__)


def _check_autogen_available() -> None:
    """Raise ImportError if AutoGen is not installed."""
    if not AUTOGEN_AVAILABLE:
        raise ImportError(
            "AutoGen is required for this integration. Install with: pip install autogen-agentchat"
        )


# Global metrics collector
_global_metrics = ToolMetricsCollector()


def get_tool_metrics() -> ToolMetricsCollector:
    """Get the global tool metrics collector.

    Returns:
        The global ToolMetricsCollector instance.
    """
    return _global_metrics


def reset_tool_metrics() -> None:
    """Reset global tool metrics."""
    global _global_metrics
    _global_metrics = ToolMetricsCollector()


def _compress_and_record(
    output: str,
    tool_name: str,
    min_chars: int,
    metrics: ToolMetricsCollector,
) -> str:
    """Compress output and record metrics.

    Args:
        output: Tool output string.
        tool_name: Name of the tool for logging.
        min_chars: Minimum chars to trigger compression.
        metrics: Collector for metrics.

    Returns:
        Compressed output, or original if below threshold or on error.
    """
    chars_before = len(output)

    if chars_before < min_chars:
        _record_metrics(metrics, tool_name, output, output, was_compressed=False)
        return output

    try:
        compressed = compress_tool_result(
            content=output,
            tool_name=tool_name,
        )
    except Exception as e:
        logger.debug("Tool compression failed for %s: %s", tool_name, e)
        _record_metrics(metrics, tool_name, output, output, was_compressed=False)
        return output

    _record_metrics(metrics, tool_name, output, compressed, was_compressed=True)
    return compressed


def _record_metrics(
    collector: ToolMetricsCollector,
    tool_name: str,
    original: str,
    compressed: str,
    was_compressed: bool,
) -> None:
    """Record compression metrics.

    Args:
        collector: The metrics collector.
        tool_name: Name of the tool.
        original: Original output.
        compressed: Compressed output.
        was_compressed: Whether compression was applied.
    """
    _record_tool_metrics(collector, tool_name, original, compressed, was_compressed, logger)


class HeadroomToolWrapper:
    """Wraps an AutoGen FunctionTool to compress its output.

    Creates a new FunctionTool whose internal function calls the original,
    stringifies the result, compresses it, and returns the compressed string.
    The original tool's name, description, and parameter schema are preserved.

    Example:
        from autogen_core.tools import FunctionTool
        from headroom.integrations.autogen import HeadroomToolWrapper

        def search(query: str) -> str:
            return json.dumps({"results": [...]})

        tool = FunctionTool(search, description="Search")
        wrapper = HeadroomToolWrapper(tool)
        wrapped_tool = wrapper.as_function_tool()

    Attributes:
        name: Tool name (from wrapped tool).
        description: Tool description (from wrapped tool).
        wrapped_tool: The new FunctionTool with compression.
    """

    def __init__(
        self,
        tool: FunctionTool,
        min_chars_to_compress: int = 1000,
        metrics_collector: ToolMetricsCollector | None = None,
    ) -> None:
        """Initialize HeadroomToolWrapper.

        Args:
            tool: The AutoGen FunctionTool to wrap.
            min_chars_to_compress: Minimum character count for output
                before compression is applied. Default 1000.
            metrics_collector: Collector for metrics. Uses global
                collector if not specified.
        """
        _check_autogen_available()

        self.name = tool.name
        self.description = tool.description
        self._min_chars = min_chars_to_compress
        self._metrics = metrics_collector or _global_metrics
        self.wrapped_tool = self._create_wrapped_tool(tool)

    def _create_wrapped_tool(self, tool: FunctionTool) -> FunctionTool:
        """Create a new FunctionTool with compression.

        Args:
            tool: The original FunctionTool.

        Returns:
            A new FunctionTool that compresses output.
        """
        original_func = tool._func
        tool_name = tool.name
        min_chars = self._min_chars
        metrics = self._metrics

        if asyncio.iscoroutinefunction(original_func):

            @functools.wraps(original_func)
            async def _compressed_func(*args: Any, **kwargs: Any) -> str:
                raw = await original_func(*args, **kwargs)
                return _compress_and_record(str(raw), tool_name, min_chars, metrics)
        else:

            @functools.wraps(original_func)
            def _compressed_func(*args: Any, **kwargs: Any) -> str:
                raw = original_func(*args, **kwargs)
                return _compress_and_record(str(raw), tool_name, min_chars, metrics)

        return FunctionTool(
            _compressed_func,
            description=tool.description,
            name=tool.name,
        )

    def as_function_tool(self) -> FunctionTool:
        """Return the wrapped FunctionTool.

        Returns:
            FunctionTool with compression applied.
        """
        return self.wrapped_tool


def wrap_tools_with_headroom(
    tools: list[FunctionTool],
    min_chars_to_compress: int = 1000,
    metrics_collector: ToolMetricsCollector | None = None,
) -> list[FunctionTool]:
    """Wrap multiple AutoGen FunctionTools with Headroom compression.

    Convenience function to wrap all tools in a list at once.
    Each wrapped tool preserves the original's name, description,
    and parameter schema.

    Args:
        tools: List of AutoGen FunctionTools to wrap.
        min_chars_to_compress: Minimum output size for compression.
        metrics_collector: Shared metrics collector for all tools.

    Returns:
        List of wrapped FunctionTools.

    Example:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_core.tools import FunctionTool
        from headroom.integrations.autogen import wrap_tools_with_headroom

        def search(query: str) -> str:
            return json.dumps(results)

        tool = FunctionTool(search, description="Search")
        wrapped = wrap_tools_with_headroom([tool])

        agent = AssistantAgent(
            name="researcher",
            model_client=model_client,
            tools=wrapped,
        )
    """
    _check_autogen_available()

    collector = metrics_collector or _global_metrics

    return [
        HeadroomToolWrapper(
            tool=t,
            min_chars_to_compress=min_chars_to_compress,
            metrics_collector=collector,
        ).as_function_tool()
        for t in tools
    ]

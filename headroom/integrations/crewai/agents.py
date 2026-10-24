"""CrewAI agent tool integration with output compression.

This module provides HeadroomToolWrapper and wrap_tools_with_headroom
for wrapping CrewAI tools to automatically compress their outputs
and track per-tool compression metrics.

Mirrors the LangChain agent integration pattern but targets CrewAI's
BaseTool interface (BaseTool.run -> _run -> result -> agent context).

Example:
    from crewai.tools.base_tool import tool
    from headroom.integrations.crewai import wrap_tools_with_headroom

    @tool
    def search_database(query: str) -> str:
        \"\"\"Search the database.\"\"\"
        return json.dumps({"results": [...], "total": 1000})

    wrapped = wrap_tools_with_headroom(
        [search_database],
        min_chars_to_compress=1000,
    )
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from crewai.tools.base_tool import BaseTool

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    BaseTool = object  # type: ignore[misc,assignment]

# Re-exported: callers import ToolCompressionMetrics from this module.
from headroom.integrations._metrics import (  # noqa: F401
    ToolCompressionMetrics,
    ToolMetricsCollector,
    _record_tool_metrics,
)
from headroom.integrations.mcp import compress_tool_result

logger = logging.getLogger(__name__)


def _check_crewai_available() -> None:
    """Raise ImportError if CrewAI is not installed."""
    if not CREWAI_AVAILABLE:
        raise ImportError(
            "CrewAI is required for this integration. Install with: pip install crewai"
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


class HeadroomToolWrapper(BaseTool):  # type: ignore[misc]
    """Wraps a CrewAI BaseTool to compress its output.

    Applies Headroom compression to tool outputs, particularly useful
    for tools that return large JSON arrays, search results, database
    query results, or verbose log output.

    The wrapper preserves the original tool's name, description, and
    argument schema so it can be used as a drop-in replacement.

    Example:
        from crewai.tools.base_tool import tool
        from headroom.integrations.crewai import HeadroomToolWrapper

        @tool
        def search(query: str) -> str:
            \"\"\"Search and return results.\"\"\"
            return json.dumps({"results": [...]})

        wrapped = HeadroomToolWrapper(search)
        result = wrapped.run(query="python tutorials")

    Attributes:
        name: Tool name (inherited from wrapped tool).
        description: Tool description (inherited from wrapped tool).
    """

    name: str = ""
    description: str = ""

    _inner: BaseTool
    _min_chars: int
    _metrics: ToolMetricsCollector

    def __init__(
        self,
        tool: BaseTool,
        min_chars_to_compress: int = 1000,
        metrics_collector: ToolMetricsCollector | None = None,
    ) -> None:
        """Initialize HeadroomToolWrapper.

        Args:
            tool: The CrewAI BaseTool to wrap.
            min_chars_to_compress: Minimum character count for output
                before compression is applied. Default 1000.
            metrics_collector: Collector for metrics. Uses global
                collector if not specified.
        """
        _check_crewai_available()

        original_description = tool.description

        super().__init__(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            result_schema=getattr(tool, "result_schema", None),
            cache_function=tool.cache_function,
            result_as_answer=tool.result_as_answer,
            max_usage_count=tool.max_usage_count,
        )
        # CrewAI's BaseTool rewrites description during construction
        # (appends schema text). Restore the original.
        self.description = original_description
        self._inner = tool
        self._min_chars = min_chars_to_compress
        self._metrics = metrics_collector or _global_metrics

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the wrapped tool and compress output.

        Args:
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            Compressed tool output as string.
        """
        raw = self._inner.run(*args, **kwargs)
        return self._compress_and_record(str(raw))

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the wrapped tool asynchronously and compress output.

        Args:
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            Compressed tool output as string.
        """
        raw = await self._inner.arun(*args, **kwargs)
        return self._compress_and_record(str(raw))

    def _compress_and_record(self, output: str) -> str:
        """Compress output and record metrics.

        Args:
            output: Tool output string.

        Returns:
            Compressed output, or original if below threshold or on error.
        """
        chars_before = len(output)

        if chars_before < self._min_chars:
            self._record_metrics(output, output, was_compressed=False)
            return output

        try:
            compressed = compress_tool_result(
                content=output,
                tool_name=self.name,
            )
        except Exception as e:
            logger.debug("Tool compression failed for %s: %s", self.name, e)
            self._record_metrics(output, output, was_compressed=False)
            return output

        self._record_metrics(output, compressed, was_compressed=True)
        return compressed

    def _record_metrics(self, original: str, compressed: str, was_compressed: bool) -> None:
        """Record compression metrics.

        Args:
            original: Original output.
            compressed: Compressed output.
            was_compressed: Whether compression was applied.
        """
        _record_tool_metrics(self._metrics, self.name, original, compressed, was_compressed, logger)


def wrap_tools_with_headroom(
    tools: list[BaseTool],
    min_chars_to_compress: int = 1000,
    metrics_collector: ToolMetricsCollector | None = None,
) -> list[BaseTool]:
    """Wrap multiple CrewAI tools with Headroom compression.

    Convenience function to wrap all tools in a list at once.
    Each wrapped tool preserves the original's name, description,
    and argument schema.

    Args:
        tools: List of CrewAI tools to wrap.
        min_chars_to_compress: Minimum output size for compression.
        metrics_collector: Shared metrics collector for all tools.

    Returns:
        List of wrapped tools.

    Example:
        from crewai import Agent
        from crewai.tools.base_tool import tool
        from headroom.integrations.crewai import wrap_tools_with_headroom

        @tool
        def search(query: str) -> str:
            \"\"\"Search the database.\"\"\"
            return json.dumps(results)

        wrapped = wrap_tools_with_headroom([search])
        agent = Agent(role="Researcher", tools=wrapped, ...)
    """
    _check_crewai_available()

    collector = metrics_collector or _global_metrics

    return [
        HeadroomToolWrapper(
            tool=t,
            min_chars_to_compress=min_chars_to_compress,
            metrics_collector=collector,
        )
        for t in tools
    ]

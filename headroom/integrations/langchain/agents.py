"""Agent tool integration for LangChain with output compression.

This module provides HeadroomToolWrapper and wrap_tools_with_headroom
for wrapping LangChain tools to automatically compress their outputs
and track per-tool compression metrics.

Example:
    from langchain_core.tools import tool
    from headroom.integrations import wrap_tools_with_headroom

    @tool
    def search(query: str) -> str:
        # docstring becomes the tool description
        return json.dumps(results)

    # Wrap with Headroom compression; the argument schema is preserved,
    # so the wrapped tool is a drop-in replacement.
    wrapped_tools = wrap_tools_with_headroom([search])

    # Use in any agent - outputs are compressed before re-entering context
    agent = create_agent(llm, wrapped_tools)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# LangChain imports - these are optional dependencies
try:
    from langchain_core.tools import BaseTool, StructuredTool, Tool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object  # type: ignore[misc,assignment]
    StructuredTool = object  # type: ignore[misc,assignment]
    Tool = object  # type: ignore[misc,assignment]

from headroom.integrations.mcp import compress_tool_result

logger = logging.getLogger(__name__)


def _check_langchain_available() -> None:
    """Raise ImportError if LangChain is not installed."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for this integration. "
            "Install with: pip install headroom[langchain] "
            "or: pip install langchain-core"
        )


@dataclass
class ToolCompressionMetrics:
    """Metrics from a single tool compression."""

    tool_name: str
    timestamp: datetime
    chars_before: int
    chars_after: int
    chars_saved: int
    compression_ratio: float
    was_compressed: bool


@dataclass
class ToolMetricsCollector:
    """Collects compression metrics across all tool invocations."""

    metrics: list[ToolCompressionMetrics] = field(default_factory=list)

    def add(self, metric: ToolCompressionMetrics) -> None:
        """Add a metric entry."""
        self.metrics.append(metric)
        # Keep only last 1000
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics:
            return {
                "total_invocations": 0,
                "total_compressions": 0,
                "total_chars_saved": 0,
            }

        compressed = [m for m in self.metrics if m.was_compressed]
        return {
            "total_invocations": len(self.metrics),
            "total_compressions": len(compressed),
            "total_chars_saved": sum(m.chars_saved for m in self.metrics),
            "average_compression_ratio": (
                sum(m.compression_ratio for m in compressed) / len(compressed) if compressed else 0
            ),
            "by_tool": self._get_by_tool_stats(),
        }

    def _get_by_tool_stats(self) -> dict[str, dict[str, Any]]:
        """Get per-tool statistics."""
        by_tool: dict[str, list[ToolCompressionMetrics]] = {}
        for m in self.metrics:
            if m.tool_name not in by_tool:
                by_tool[m.tool_name] = []
            by_tool[m.tool_name].append(m)

        result = {}
        for name, tool_metrics in by_tool.items():
            compressed = [m for m in tool_metrics if m.was_compressed]
            result[name] = {
                "invocations": len(tool_metrics),
                "compressions": len(compressed),
                "chars_saved": sum(m.chars_saved for m in tool_metrics),
            }
        return result


# Global metrics collector
_global_metrics = ToolMetricsCollector()


def get_tool_metrics() -> ToolMetricsCollector:
    """Get the global tool metrics collector."""
    return _global_metrics


def reset_tool_metrics() -> None:
    """Reset global tool metrics."""
    global _global_metrics
    _global_metrics = ToolMetricsCollector()


class HeadroomToolWrapper:
    """Wraps a LangChain tool to compress its output.

    Applies SmartCrusher compression to tool outputs, particularly
    useful for tools that return large JSON arrays (search results,
    database queries, etc.).

    Example:
        from langchain_core.tools import tool
        from headroom.integrations import HeadroomToolWrapper

        def search(query: str) -> str:
            # Returns large JSON with 1000 results
            return json.dumps({"results": [...1000 items...]})

        search_tool = Tool(name="search", func=search, description="Search")
        wrapped = HeadroomToolWrapper(search_tool)

        # Use wrapped tool - output automatically compressed
        result = wrapped("python tutorials")

    Attributes:
        tool: The wrapped LangChain tool
        min_chars_to_compress: Minimum output size to trigger compression
        metrics_collector: Collector for compression metrics
    """

    def __init__(
        self,
        tool: BaseTool,
        min_chars_to_compress: int = 1000,
        metrics_collector: ToolMetricsCollector | None = None,
    ):
        """Initialize HeadroomToolWrapper.

        Args:
            tool: The LangChain BaseTool to wrap.
            min_chars_to_compress: Minimum character count for output
                before compression is applied. Default 1000.
            metrics_collector: Collector for metrics. Uses global
                collector if not specified.
        """
        _check_langchain_available()

        self.tool = tool
        self.min_chars_to_compress = min_chars_to_compress
        self._metrics = metrics_collector or _global_metrics

        # Copy tool metadata
        self.name = tool.name
        self.description = tool.description

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Invoke the tool and compress output.

        Args:
            *args: Arguments to pass to the tool.
            **kwargs: Keyword arguments to pass to the tool.

        Returns:
            Compressed tool output as string.
        """
        # Invoke underlying tool
        result = self.tool.invoke(self._as_tool_input(*args, **kwargs))

        # Convert to string if needed
        if not isinstance(result, str):
            result = str(result)

        # Check if compression is needed
        if len(result) < self.min_chars_to_compress:
            self._record_metrics(result, result, was_compressed=False)
            return str(result)

        # Try to compress
        compressed = self._compress_output(result)
        self._record_metrics(result, compressed, was_compressed=True)

        return compressed

    @staticmethod
    def _as_tool_input(*args: Any, **kwargs: Any) -> Any:
        """Rebuild the single input argument ``BaseTool.invoke`` expects.

        ``StructuredTool`` calls its ``func`` with the parsed schema fields
        unpacked (``func(query="x")``), while ``BaseTool.invoke`` takes one
        positional ``input``. Passing the unpacked form straight through raises
        ``TypeError: BaseTool.invoke() missing 1 required positional argument``.
        """
        if args and kwargs:
            if len(args) == 1 and isinstance(args[0], dict):
                return {**args[0], **kwargs}
            raise TypeError(
                "Cannot map mixed positional and keyword arguments onto a tool "
                f"input: args={args!r}, kwargs={list(kwargs)!r}"
            )
        if kwargs:
            return kwargs
        if len(args) == 1:
            return args[0]
        if not args:
            return {}
        raise TypeError(f"Expected a single tool input, got {len(args)} positional arguments")

    def invoke(self, *args: Any, **kwargs: Any) -> str:
        """Invoke the tool (alias for __call__)."""
        return self(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> str:
        """Async counterpart of ``__call__``, used for ``ainvoke``."""
        result = await self.tool.ainvoke(self._as_tool_input(*args, **kwargs))

        if not isinstance(result, str):
            result = str(result)

        if len(result) < self.min_chars_to_compress:
            self._record_metrics(result, result, was_compressed=False)
            return str(result)

        compressed = self._compress_output(result)
        self._record_metrics(result, compressed, was_compressed=True)
        return compressed

    def _compress_output(self, output: str) -> str:
        """Apply compression to tool output.

        Args:
            output: Tool output string.

        Returns:
            Compressed output.
        """
        try:
            return compress_tool_result(
                content=output,
                tool_name=self.name,
            )
        except Exception as e:
            logger.debug(f"Tool compression failed: {e}")
            return output

    def _record_metrics(self, original: str, compressed: str, was_compressed: bool) -> None:
        """Record compression metrics.

        Args:
            original: Original output.
            compressed: Compressed output.
            was_compressed: Whether compression was applied.
        """
        chars_before = len(original)
        chars_after = len(compressed)
        chars_saved = chars_before - chars_after

        metric = ToolCompressionMetrics(
            tool_name=self.name,
            timestamp=datetime.now(timezone.utc),
            chars_before=chars_before,
            chars_after=chars_after,
            chars_saved=max(0, chars_saved),
            compression_ratio=chars_after / chars_before if chars_before > 0 else 1.0,
            was_compressed=was_compressed and chars_saved > 0,
        )

        self._metrics.add(metric)

        if was_compressed and chars_saved > 0:
            logger.info(
                f"HeadroomToolWrapper[{self.name}]: {chars_before} -> {chars_after} chars "
                f"({chars_saved} saved, {metric.compression_ratio:.1%} of original)"
            )

    @staticmethod
    def _usable_args_schema(tool: Any) -> Any:
        """Return the tool's args_schema only if LangChain will accept it.

        StructuredTool requires a pydantic model class or a JSON-schema dict.
        Tools built by hand (or test doubles) may carry something else, and
        passing that through turns a working tool into a construction error, so
        fall back to inferring the schema instead.
        """
        schema = getattr(tool, "args_schema", None)
        if schema is None:
            return None
        if isinstance(schema, dict):
            return schema
        try:
            from pydantic import BaseModel

            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema
        except Exception:  # pragma: no cover - pydantic always present with langchain
            pass
        logger.debug(
            "Tool %r has an args_schema LangChain cannot use (%s); inferring instead",
            getattr(tool, "name", tool),
            type(schema).__name__,
        )
        return None

    def as_langchain_tool(self) -> StructuredTool:
        """Convert wrapper back to a LangChain tool.

        The wrapped tool keeps the original tool's argument schema, so a model
        still sees the same typed parameters. Inferring the schema from
        ``__call__`` instead would advertise ``(*args, **kwargs)`` and leave the
        model with no parameters to fill in.

        Returns:
            StructuredTool that wraps this wrapper.
        """
        args_schema = self._usable_args_schema(self.tool)
        return StructuredTool.from_function(
            func=self.__call__,
            coroutine=self.acall,
            name=self.name,
            description=self.description,
            args_schema=args_schema,
            infer_schema=args_schema is None,
            return_direct=getattr(self.tool, "return_direct", False),
        )


def wrap_tools_with_headroom(
    tools: list[BaseTool],
    min_chars_to_compress: int = 1000,
    metrics_collector: ToolMetricsCollector | None = None,
) -> list[StructuredTool]:
    """Wrap multiple LangChain tools with Headroom compression.

    Convenience function to wrap all tools in a list at once.

    Args:
        tools: List of LangChain tools to wrap.
        min_chars_to_compress: Minimum output size for compression.
        metrics_collector: Shared metrics collector for all tools.

    Returns:
        List of wrapped tools as StructuredTools.

    Example:
        from langchain.agents import create_agent
        from headroom.integrations import wrap_tools_with_headroom

        tools = [search_tool, database_tool, api_tool]
        wrapped = wrap_tools_with_headroom(tools)

        # Use wrapped tools in agent
        agent = create_agent(llm, wrapped)
    """
    _check_langchain_available()

    collector = metrics_collector or _global_metrics

    wrapped = []
    for tool in tools:
        wrapper = HeadroomToolWrapper(
            tool=tool,
            min_chars_to_compress=min_chars_to_compress,
            metrics_collector=collector,
        )
        wrapped.append(wrapper.as_langchain_tool())

    return wrapped

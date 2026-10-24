"""Metric types shared by the framework integrations.

The LangChain, CrewAI and AutoGen tool wrappers record the same per-tool
compression metrics, and the LangChain, Agno and Strands model wrappers record
the same per-call optimization metrics. Each integration re-exports these names
from its own module, so ``headroom.integrations.<framework>`` keeps working.

Stdlib-only: importing this module must not pull in any agent framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class OptimizationMetrics:
    """Metrics from a single optimization pass."""

    request_id: str
    timestamp: datetime
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float
    transforms_applied: list[str]
    model: str


@dataclass
class ToolCompressionMetrics:
    """Metrics from a single tool compression.

    Attributes:
        tool_name: Name of the tool that was invoked.
        timestamp: When the compression occurred.
        chars_before: Character count of the original output.
        chars_after: Character count after compression.
        chars_saved: Characters removed by compression.
        compression_ratio: Ratio of compressed to original size.
        was_compressed: Whether compression was actually applied.
    """

    tool_name: str
    timestamp: datetime
    chars_before: int
    chars_after: int
    chars_saved: int
    compression_ratio: float
    was_compressed: bool


@dataclass
class ToolMetricsCollector:
    """Collects compression metrics across all tool invocations.

    Attributes:
        metrics: List of per-invocation metrics.
    """

    metrics: list[ToolCompressionMetrics] = field(default_factory=list)

    def add(self, metric: ToolCompressionMetrics) -> None:
        """Add a metric entry.

        Args:
            metric: The compression metrics to record.
        """
        self.metrics.append(metric)
        # Keep only last 1000
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dict with total_invocations, total_compressions,
            total_chars_saved, average_compression_ratio, and
            per-tool breakdown.
        """
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


def _record_tool_metrics(
    collector: ToolMetricsCollector,
    tool_name: str,
    original: str,
    compressed: str,
    was_compressed: bool,
    log: logging.Logger,
) -> None:
    """Record one tool invocation in ``collector`` and log real savings.

    Args:
        collector: The metrics collector.
        tool_name: Name of the tool.
        original: Original output.
        compressed: Compressed output.
        was_compressed: Whether compression was applied.
        log: The calling integration's logger, so the savings line keeps
            that integration's logger name.
    """
    chars_before = len(original)
    chars_after = len(compressed)
    chars_saved = chars_before - chars_after

    metric = ToolCompressionMetrics(
        tool_name=tool_name,
        timestamp=datetime.now(timezone.utc),
        chars_before=chars_before,
        chars_after=chars_after,
        chars_saved=max(0, chars_saved),
        compression_ratio=chars_after / chars_before if chars_before > 0 else 1.0,
        was_compressed=was_compressed and chars_saved > 0,
    )

    collector.add(metric)

    if was_compressed and chars_saved > 0:
        log.info(
            "HeadroomToolWrapper[%s]: %d -> %d chars (%d saved, %.1f%% of original)",
            tool_name,
            chars_before,
            chars_after,
            chars_saved,
            metric.compression_ratio * 100,
        )

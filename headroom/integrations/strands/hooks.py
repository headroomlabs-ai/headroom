"""Strands SDK hook provider for Headroom tool output compression.

This module provides HeadroomHookProvider, which implements Strands' HookProvider
interface to intercept tool outputs and compress them using Headroom's SmartCrusher.

Example:
    from strands import Agent
    from headroom.integrations.strands import HeadroomHookProvider

    # Create the hook provider
    hook_provider = HeadroomHookProvider(
        compress_tool_outputs=True,
        min_tokens_to_compress=100,
    )

    # Use with Strands agent
    agent = Agent(hooks=[hook_provider])
    response = agent("Search for documents about AI")

    # Check compression metrics
    print(f"Tokens saved: {hook_provider.total_tokens_saved}")
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

# Strands imports - these are optional dependencies
try:
    from strands.hooks import HookProvider, HookRegistry
    from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent
    from strands.types.tools import ToolResult

    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    # Type stubs for when strands is not installed
    HookProvider = object  # type: ignore[misc,assignment]
    HookRegistry = object  # type: ignore[misc,assignment]
    AfterToolCallEvent = object  # type: ignore[misc,assignment]
    BeforeToolCallEvent = object  # type: ignore[misc,assignment]
    ToolResult = dict  # type: ignore[misc,assignment]

from headroom import HeadroomConfig
from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

logger = logging.getLogger(__name__)


def _check_strands_available() -> None:
    """Raise ImportError if Strands is not installed."""
    if not STRANDS_AVAILABLE:
        raise ImportError(
            "Strands SDK is required for this integration. Install with: pip install strands-agents"
        )


def strands_available() -> bool:
    """Check if Strands SDK is installed.

    Returns:
        True if strands-agents package is available.
    """
    return STRANDS_AVAILABLE


@dataclass
class CompressionMetrics:
    """Metrics from a single tool output compression."""

    request_id: str
    timestamp: datetime
    tool_name: str
    tool_use_id: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float
    was_compressed: bool
    skip_reason: str | None = None


@dataclass
class HeadroomHookProvider(HookProvider):  # type: ignore[misc]
    """Strands HookProvider that compresses tool outputs using Headroom.

    This hook provider intercepts tool call results via AfterToolCallEvent
    and applies Headroom's SmartCrusher to compress large outputs, reducing
    token usage while preserving important information.

    The compression is intelligent and preserves:
    - Error items (containing error indicators)
    - Anomalous values (statistical outliers)
    - Items matching the user's query context
    - First/last items for context
    - Structural outliers (rare status values)

    Attributes:
        compress_tool_outputs: Whether to compress tool outputs.
        min_tokens_to_compress: Minimum token count before compression is applied.
        config: Headroom configuration.
        preserve_errors: If True, never compress results with error status.
        total_tokens_saved: Running total of tokens saved across all compressions.
        metrics_history: List of CompressionMetrics from recent compressions.

    Example:
        from strands import Agent
        from headroom.integrations.strands import HeadroomHookProvider

        hook = HeadroomHookProvider(min_tokens_to_compress=50)
        agent = Agent(hooks=[hook])

        # After running agent tasks...
        summary = hook.get_savings_summary()
        print(f"Total saved: {summary['total_tokens_saved']} tokens")
    """

    compress_tool_outputs: bool = True
    min_tokens_to_compress: int = 100
    config: HeadroomConfig | None = field(default=None)
    preserve_errors: bool = True

    # Internal state (not part of dataclass comparison)
    _crusher: SmartCrusher | None = field(default=None, repr=False, compare=False)
    _metrics_history: list[CompressionMetrics] = field(
        default_factory=list, repr=False, compare=False
    )
    _total_tokens_saved: int = field(default=0, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _initialized: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize the hook provider after dataclass construction."""
        _check_strands_available()

        if self.config is None:
            self.config = HeadroomConfig()

        self._initialized = True
        logger.debug(
            "HeadroomHookProvider initialized: compress=%s, min_tokens=%d, preserve_errors=%s",
            self.compress_tool_outputs,
            self.min_tokens_to_compress,
            self.preserve_errors,
        )

    @property
    def crusher(self) -> SmartCrusher:
        """Lazily initialize SmartCrusher (thread-safe).

        Returns:
            The SmartCrusher instance for compression.
        """
        if self._crusher is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._crusher is None:
                    # Use config from HeadroomConfig if available
                    if self.config and self.config.smart_crusher:
                        crusher_config = SmartCrusherConfig(
                            min_tokens_to_crush=self.min_tokens_to_compress,
                            max_items_after_crush=self.config.smart_crusher.max_items_after_crush,
                        )
                    else:
                        crusher_config = SmartCrusherConfig(
                            min_tokens_to_crush=self.min_tokens_to_compress
                        )
                    self._crusher = SmartCrusher(config=crusher_config)
                    logger.debug(
                        "SmartCrusher initialized with min_tokens=%d", self.min_tokens_to_compress
                    )
        return self._crusher

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all compressions.

        Returns:
            Cumulative token savings.
        """
        return self._total_tokens_saved

    @property
    def metrics_history(self) -> list[CompressionMetrics]:
        """History of compression metrics.

        Returns:
            Copy of the metrics history list.
        """
        return self._metrics_history.copy()

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks with the Strands HookRegistry.

        This method is called by Strands when the hook provider is added
        to an Agent. It registers the compression handler for AfterToolCallEvent.

        Args:
            registry: The Strands HookRegistry to register hooks with.
        """
        if not self.compress_tool_outputs:
            logger.debug("Tool output compression disabled, skipping hook registration")
            return

        # Register the after-tool-call hook for compression
        registry.add_callback(AfterToolCallEvent, self._compress_tool_result)
        logger.info(
            "HeadroomHookProvider registered: compressing tool outputs >= %d tokens",
            self.min_tokens_to_compress,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic of ~4 characters per token, which is
        reasonably accurate for English text and JSON content.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        # ~4 characters per token is a reasonable estimate
        return len(text) // 4

    def _extract_text_content(self, result: ToolResult) -> str:
        """Extract text content from a ToolResult.

        Handles both text and JSON content types in the result.

        Args:
            result: The ToolResult to extract content from.

        Returns:
            String representation of the content.
        """
        content = result.get("content", [])
        if not content:
            return ""

        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(str(item["text"]))
                elif "json" in item:
                    try:
                        text_parts.append(json.dumps(item["json"], indent=None))
                    except (TypeError, ValueError):
                        text_parts.append(str(item["json"]))
            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts)

    def _update_result_content(self, result: ToolResult, compressed_text: str) -> None:
        """Update the result content with compressed text.

        Modifies the result in place, preserving the original content structure
        (text vs json) where possible.

        Args:
            result: The ToolResult to update (modified in place).
            compressed_text: The compressed content to set.
        """
        content = result.get("content", [])

        if not content:
            # No existing content, create text content
            result["content"] = [{"text": compressed_text}]
            return

        # Try to preserve original structure
        first_item = content[0] if content else None

        if isinstance(first_item, dict):
            if "json" in first_item:
                # Try to parse compressed text back to JSON
                try:
                    parsed = json.loads(compressed_text)
                    result["content"] = [{"json": parsed}]
                except (json.JSONDecodeError, ValueError):
                    # Fall back to text if not valid JSON
                    result["content"] = [{"text": compressed_text}]
            else:
                # Text content
                result["content"] = [{"text": compressed_text}]
        else:
            # Unknown structure, use text
            result["content"] = [{"text": compressed_text}]

    def _compress_tool_result(self, event: AfterToolCallEvent) -> None:
        """Compress tool result content if it exceeds the token threshold.

        This is the main hook handler that intercepts AfterToolCallEvent
        and applies SmartCrusher compression to large tool outputs.

        Args:
            event: The AfterToolCallEvent containing the tool result.
                   The result field is writable and modified in place.
        """
        request_id = str(uuid4())
        result = event.result
        tool_name = event.tool_use.get("name", "unknown")
        tool_use_id = event.tool_use.get("toolUseId", "unknown")

        # Check if compression should be skipped
        skip_reason = self._should_skip_compression(result)
        if skip_reason:
            self._record_metrics(
                request_id=request_id,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                tokens_before=0,
                tokens_after=0,
                was_compressed=False,
                skip_reason=skip_reason,
            )
            logger.debug(
                "Skipping compression for tool %s (id=%s): %s",
                tool_name,
                tool_use_id,
                skip_reason,
            )
            return

        # Extract content and estimate tokens
        original_text = self._extract_text_content(result)
        tokens_before = self._estimate_tokens(original_text)

        # Check minimum token threshold
        if tokens_before < self.min_tokens_to_compress:
            self._record_metrics(
                request_id=request_id,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                was_compressed=False,
                skip_reason=f"below_threshold:{tokens_before}<{self.min_tokens_to_compress}",
            )
            logger.debug(
                "Tool %s output below threshold (%d < %d tokens), skipping compression",
                tool_name,
                tokens_before,
                self.min_tokens_to_compress,
            )
            return

        # Apply compression
        try:
            crush_result = self.crusher.crush(content=original_text, query="")
            compressed_text = crush_result.compressed
            was_modified = crush_result.was_modified
        except Exception as e:
            # Compression failed, keep original
            logger.warning(
                "Compression failed for tool %s (id=%s): %s. Keeping original.",
                tool_name,
                tool_use_id,
                str(e),
            )
            self._record_metrics(
                request_id=request_id,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                was_compressed=False,
                skip_reason=f"compression_error:{type(e).__name__}",
            )
            return

        tokens_after = self._estimate_tokens(compressed_text)

        # Only update if compression actually reduced tokens
        if was_modified and tokens_after < tokens_before:
            self._update_result_content(result, compressed_text)
            tokens_saved = tokens_before - tokens_after

            self._record_metrics(
                request_id=request_id,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                was_compressed=True,
                skip_reason=None,
            )

            logger.info(
                "Compressed tool %s output: %d -> %d tokens (%.1f%% saved)",
                tool_name,
                tokens_before,
                tokens_after,
                (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0,
            )
        else:
            # Compression didn't help
            self._record_metrics(
                request_id=request_id,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                was_compressed=False,
                skip_reason="no_reduction",
            )
            logger.debug(
                "Compression did not reduce tool %s output (%d tokens)",
                tool_name,
                tokens_before,
            )

    def _should_skip_compression(self, result: ToolResult) -> str | None:
        """Check if compression should be skipped for this result.

        Args:
            result: The tool result to check.

        Returns:
            Skip reason string if should skip, None if should compress.
        """
        # Skip if compression is disabled
        if not self.compress_tool_outputs:
            return "compression_disabled"

        # Skip error results if preserve_errors is True
        if self.preserve_errors and result.get("status") == "error":
            return "error_result_preserved"

        # Skip empty results
        content = result.get("content", [])
        if not content:
            return "empty_content"

        return None

    def _record_metrics(
        self,
        request_id: str,
        tool_name: str,
        tool_use_id: str,
        tokens_before: int,
        tokens_after: int,
        was_compressed: bool,
        skip_reason: str | None,
    ) -> None:
        """Record compression metrics (thread-safe).

        Args:
            request_id: Unique ID for this compression request.
            tool_name: Name of the tool that was called.
            tool_use_id: The toolUseId from the result.
            tokens_before: Token count before compression.
            tokens_after: Token count after compression.
            was_compressed: Whether compression was actually applied.
            skip_reason: Reason compression was skipped, if applicable.
        """
        tokens_saved = max(0, tokens_before - tokens_after)
        savings_percent = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0.0

        metrics = CompressionMetrics(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            savings_percent=savings_percent,
            was_compressed=was_compressed,
            skip_reason=skip_reason,
        )

        with self._lock:
            self._metrics_history.append(metrics)
            if was_compressed:
                self._total_tokens_saved += tokens_saved

            # Keep only last 100 metrics to bound memory
            if len(self._metrics_history) > 100:
                self._metrics_history = self._metrics_history[-100:]

    def get_savings_summary(self) -> dict[str, Any]:
        """Get summary of token savings across all compressions.

        Returns:
            Dictionary with compression statistics including:
            - total_requests: Number of tool outputs processed
            - compressed_requests: Number actually compressed
            - total_tokens_saved: Cumulative tokens saved
            - average_savings_percent: Mean compression ratio
            - total_tokens_before: Sum of all input tokens
            - total_tokens_after: Sum of all output tokens
        """
        if not self._metrics_history:
            return {
                "total_requests": 0,
                "compressed_requests": 0,
                "total_tokens_saved": 0,
                "average_savings_percent": 0.0,
                "total_tokens_before": 0,
                "total_tokens_after": 0,
            }

        compressed_metrics = [m for m in self._metrics_history if m.was_compressed]

        return {
            "total_requests": len(self._metrics_history),
            "compressed_requests": len(compressed_metrics),
            "total_tokens_saved": self._total_tokens_saved,
            "average_savings_percent": (
                sum(m.savings_percent for m in compressed_metrics) / len(compressed_metrics)
                if compressed_metrics
                else 0.0
            ),
            "total_tokens_before": sum(m.tokens_before for m in self._metrics_history),
            "total_tokens_after": sum(m.tokens_after for m in self._metrics_history),
        }

    def reset(self) -> None:
        """Reset all tracked metrics (thread-safe).

        Clears the metrics history and resets the total tokens saved counter.
        Useful for starting fresh measurements or between test runs.
        """
        with self._lock:
            self._metrics_history = []
            self._total_tokens_saved = 0
        logger.debug("HeadroomHookProvider metrics reset")

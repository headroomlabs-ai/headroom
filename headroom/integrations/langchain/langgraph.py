"""LangGraph integration for Headroom tool message compression.

This module provides a compress_tool_messages utility and a LangGraph-compatible
node factory for compressing ToolMessage content before it reaches the LLM,
solving context bloat from large tool outputs (JSON arrays, DB results, logs).

Addresses:
- LangGraph Issue #3717 (ToolMessage overflow)
- LangChain Issue #11405 (agent token limit)
- LangChain Issue #2140 (127K tokens from plugin)

Example:
    from langgraph.graph import StateGraph, MessagesState
    from headroom.integrations.langchain.langgraph import (
        compress_tool_messages,
        create_compress_tool_messages_node,
    )

    # Option 1: Use as a LangGraph node
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("compress", create_compress_tool_messages_node())
    graph.add_edge("tools", "compress")
    graph.add_edge("compress", "agent")

    # Option 2: Use as a standalone function
    compressed = compress_tool_messages(messages)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

# LangChain imports - optional dependencies
try:
    from langchain_core.messages import BaseMessage, ToolMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseMessage = object  # type: ignore[misc,assignment]
    ToolMessage = object  # type: ignore[misc,assignment]

from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

logger = logging.getLogger(__name__)


def _check_langchain_available() -> None:
    """Raise ImportError if LangChain is not installed."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for this integration. "
            "Install with: pip install headroom[langchain] "
            "or: pip install langchain-core"
        )


def _estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 characters per token heuristic."""
    if not text:
        return 0
    return len(text) // 4


@dataclass
class ToolMessageCompressionMetrics:
    """Metrics from compressing a single ToolMessage."""

    request_id: str
    timestamp: datetime
    tool_call_id: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float
    was_compressed: bool
    skip_reason: str | None = None


@dataclass
class CompressToolMessagesConfig:
    """Configuration for compress_tool_messages.

    Attributes:
        min_tokens_to_compress: Minimum estimated token count in a ToolMessage
            before compression is applied. Default 100.
        preserve_errors: If True, skip compression on ToolMessages whose content
            contains error indicators. Default True.
        error_indicators: Strings that indicate a ToolMessage contains an error.
    """

    min_tokens_to_compress: int = 100
    preserve_errors: bool = True
    error_indicators: tuple[str, ...] = ('"error"', '"ERROR"', "Error:", "Traceback")


@dataclass
class CompressToolMessagesResult:
    """Result from compress_tool_messages including metrics."""

    messages: list[Any]  # list[BaseMessage] but Any for when langchain not installed
    metrics: list[ToolMessageCompressionMetrics] = field(default_factory=list)

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all compressed messages."""
        return sum(m.tokens_saved for m in self.metrics if m.was_compressed)

    @property
    def messages_compressed(self) -> int:
        """Number of messages that were actually compressed."""
        return sum(1 for m in self.metrics if m.was_compressed)


class _CrusherSingleton:
    """Thread-safe lazy singleton for SmartCrusher."""

    def __init__(self, min_tokens: int) -> None:
        self._crusher: SmartCrusher | None = None
        self._min_tokens = min_tokens
        self._lock = threading.Lock()

    def get(self) -> SmartCrusher:
        if self._crusher is None:
            with self._lock:
                if self._crusher is None:
                    config = SmartCrusherConfig(
                        min_tokens_to_crush=self._min_tokens,
                    )
                    self._crusher = SmartCrusher(config=config)
        return self._crusher


# Module-level singleton, lazily initialized on first call
_crusher_singleton: _CrusherSingleton | None = None
_crusher_lock = threading.Lock()


def _get_crusher(min_tokens: int) -> SmartCrusher:
    """Get or create the module-level SmartCrusher singleton."""
    global _crusher_singleton
    if _crusher_singleton is None:
        with _crusher_lock:
            if _crusher_singleton is None:
                _crusher_singleton = _CrusherSingleton(min_tokens)
    return _crusher_singleton.get()


def _should_skip(
    content: str,
    config: CompressToolMessagesConfig,
) -> str | None:
    """Check if a ToolMessage should skip compression.

    Returns skip reason string, or None if it should be compressed.
    """
    if not content:
        return "empty_content"

    tokens = _estimate_tokens(content)
    if tokens < config.min_tokens_to_compress:
        return f"below_threshold:{tokens}<{config.min_tokens_to_compress}"

    if config.preserve_errors:
        for indicator in config.error_indicators:
            if indicator in content:
                return "error_content_preserved"

    return None


def compress_tool_messages(
    messages: list[BaseMessage],  # type: ignore[type-arg]
    *,
    min_tokens_to_compress: int = 100,
    preserve_errors: bool = True,
    config: CompressToolMessagesConfig | None = None,
) -> CompressToolMessagesResult:
    """Compress ToolMessage content in a list of LangChain messages.

    Iterates through messages, finds ToolMessages with large content,
    and compresses them using SmartCrusher. Non-tool messages are
    returned unchanged. tool_call_id is always preserved.

    Args:
        messages: List of LangChain BaseMessage objects.
        min_tokens_to_compress: Minimum estimated tokens to trigger compression.
        preserve_errors: If True, skip ToolMessages containing error indicators.
        config: Full configuration object (overrides other kwargs if provided).

    Returns:
        CompressToolMessagesResult with compressed messages and metrics.

    Example:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        messages = [
            HumanMessage(content="Get sales data"),
            AIMessage(content="", tool_calls=[{"id": "call_1", "name": "db", "args": {}}]),
            ToolMessage(content='[{"row": 1}, {"row": 2}, ...]', tool_call_id="call_1"),
        ]

        result = compress_tool_messages(messages)
        print(f"Saved {result.total_tokens_saved} tokens")
        compressed_messages = result.messages
    """
    _check_langchain_available()

    if config is None:
        config = CompressToolMessagesConfig(
            min_tokens_to_compress=min_tokens_to_compress,
            preserve_errors=preserve_errors,
        )

    crusher = _get_crusher(config.min_tokens_to_compress)
    result_messages: list[BaseMessage] = []
    metrics: list[ToolMessageCompressionMetrics] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            result_messages.append(msg)
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        request_id = str(uuid4())

        # Check if we should skip
        skip_reason = _should_skip(content, config)
        if skip_reason:
            result_messages.append(msg)
            tokens = _estimate_tokens(content)
            metrics.append(
                ToolMessageCompressionMetrics(
                    request_id=request_id,
                    timestamp=datetime.now(timezone.utc),
                    tool_call_id=getattr(msg, "tool_call_id", "unknown"),
                    tokens_before=tokens,
                    tokens_after=tokens,
                    tokens_saved=0,
                    savings_percent=0.0,
                    was_compressed=False,
                    skip_reason=skip_reason,
                )
            )
            logger.debug(
                "Skipping ToolMessage %s compression: %s",
                getattr(msg, "tool_call_id", "unknown"),
                skip_reason,
            )
            continue

        # Compress
        tokens_before = _estimate_tokens(content)
        try:
            crush_result = crusher.crush(content=content, query="")
            compressed_text = crush_result.compressed
            was_modified = crush_result.was_modified
        except Exception as e:
            logger.warning(
                "Compression failed for ToolMessage %s: %s. Keeping original.",
                getattr(msg, "tool_call_id", "unknown"),
                str(e),
            )
            result_messages.append(msg)
            metrics.append(
                ToolMessageCompressionMetrics(
                    request_id=request_id,
                    timestamp=datetime.now(timezone.utc),
                    tool_call_id=getattr(msg, "tool_call_id", "unknown"),
                    tokens_before=tokens_before,
                    tokens_after=tokens_before,
                    tokens_saved=0,
                    savings_percent=0.0,
                    was_compressed=False,
                    skip_reason=f"compression_error:{type(e).__name__}",
                )
            )
            continue

        tokens_after = _estimate_tokens(compressed_text)

        if was_modified and tokens_after < tokens_before:
            # Create new ToolMessage with compressed content, preserving tool_call_id
            compressed_msg = ToolMessage(
                content=compressed_text,
                tool_call_id=msg.tool_call_id,
            )
            result_messages.append(compressed_msg)
            tokens_saved = tokens_before - tokens_after

            metrics.append(
                ToolMessageCompressionMetrics(
                    request_id=request_id,
                    timestamp=datetime.now(timezone.utc),
                    tool_call_id=msg.tool_call_id,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    tokens_saved=tokens_saved,
                    savings_percent=(tokens_saved / tokens_before * 100)
                    if tokens_before > 0
                    else 0.0,
                    was_compressed=True,
                )
            )

            logger.info(
                "Compressed ToolMessage %s: %d -> %d tokens (%.1f%% saved)",
                msg.tool_call_id,
                tokens_before,
                tokens_after,
                (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0,
            )
        else:
            # Compression didn't help, keep original
            result_messages.append(msg)
            metrics.append(
                ToolMessageCompressionMetrics(
                    request_id=request_id,
                    timestamp=datetime.now(timezone.utc),
                    tool_call_id=msg.tool_call_id,
                    tokens_before=tokens_before,
                    tokens_after=tokens_before,
                    tokens_saved=0,
                    savings_percent=0.0,
                    was_compressed=False,
                    skip_reason="no_reduction",
                )
            )

    return CompressToolMessagesResult(messages=result_messages, metrics=metrics)


def create_compress_tool_messages_node(
    *,
    min_tokens_to_compress: int = 100,
    preserve_errors: bool = True,
    config: CompressToolMessagesConfig | None = None,
) -> Any:
    """Create a LangGraph node that compresses ToolMessages in graph state.

    Returns a function compatible with LangGraph's StateGraph that reads
    messages from state, compresses ToolMessages, and returns updated state.

    Args:
        min_tokens_to_compress: Minimum estimated tokens to trigger compression.
        preserve_errors: If True, skip ToolMessages containing error indicators.
        config: Full configuration object (overrides other kwargs if provided).

    Returns:
        A callable suitable for use as a LangGraph node.

    Example:
        from langgraph.graph import StateGraph, MessagesState

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_node("compress", create_compress_tool_messages_node(
            min_tokens_to_compress=200,
        ))

        # Wire: tools -> compress -> agent
        graph.add_edge("tools", "compress")
        graph.add_edge("compress", "agent")
    """
    _check_langchain_available()

    if config is None:
        config = CompressToolMessagesConfig(
            min_tokens_to_compress=min_tokens_to_compress,
            preserve_errors=preserve_errors,
        )

    def compress_node(state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph node that compresses ToolMessages in state.

        Args:
            state: LangGraph state dict containing a "messages" key.

        Returns:
            Updated state dict with compressed messages.
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        result = compress_tool_messages(messages, config=config)

        return {"messages": result.messages}

    return compress_node

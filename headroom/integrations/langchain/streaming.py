"""Streaming metrics tracking for LangChain.

This module provides StreamingMetricsTracker for tracking output tokens
during streaming responses from LangChain models.

Example:
    from langchain_openai import ChatOpenAI
    from headroom.integrations import HeadroomChatModel, StreamingMetricsTracker

    llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))
    tracker = StreamingMetricsTracker(model="gpt-4o")

    for chunk in llm.stream("Tell me a story"):
        tracker.add_chunk(chunk)
        print(chunk.content, end="", flush=True)

    print(f"\\nOutput tokens: {tracker.output_tokens}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# LangChain imports - these are optional dependencies
try:
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    AIMessageChunk = object  # type: ignore[misc,assignment]
    ChatGenerationChunk = object  # type: ignore[misc,assignment]

from headroom.providers import OpenAIProvider

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
class StreamingMetrics:
    """Metrics from a streaming response."""

    output_tokens: int
    chunk_count: int
    content_length: int
    start_time: datetime
    end_time: datetime | None
    duration_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_tokens": self.output_tokens,
            "chunk_count": self.chunk_count,
            "content_length": self.content_length,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
        }


class StreamingMetricsTracker:
    """Tracks output tokens and metrics during streaming.

    Accumulates content from streaming chunks and provides accurate
    token counting for the streamed output.

    Example:
        tracker = StreamingMetricsTracker(model="gpt-4o")

        async for chunk in llm.astream(messages):
            tracker.add_chunk(chunk)
            print(chunk.content, end="")

        print(f"\\nTokens: {tracker.output_tokens}")
        print(f"Duration: {tracker.duration_ms}ms")

    Attributes:
        model: Model name for token counting
        content: Accumulated content from all chunks
        output_tokens: Estimated token count for output
        chunk_count: Number of chunks received
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: Any = None,
    ):
        """Initialize StreamingMetricsTracker.

        Args:
            model: Model name for token counting. Default "gpt-4o".
            provider: Headroom provider for token counting. Uses
                OpenAIProvider if not specified.
        """
        _check_langchain_available()

        self._model = model
        self._provider = provider or OpenAIProvider()
        self._content = ""
        self._chunk_count = 0
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

    def add_chunk(self, chunk: Any) -> None:
        """Add a streaming chunk to the tracker.

        Extracts content from various chunk types:
        - AIMessageChunk
        - ChatGenerationChunk
        - dict with 'content' key
        - string

        Args:
            chunk: Streaming chunk from LangChain.
        """
        if self._start_time is None:
            self._start_time = datetime.now()

        self._chunk_count += 1

        # Extract content from various chunk types
        content = self._extract_content(chunk)
        if content:
            self._content += content

    def _extract_content(self, chunk: Any) -> str:
        """Extract string content from a chunk.

        Args:
            chunk: Streaming chunk of various types.

        Returns:
            Extracted content string.
        """
        # AIMessageChunk
        if hasattr(chunk, "content"):
            content = chunk.content
            if isinstance(content, str):
                return content
            return str(content) if content else ""

        # ChatGenerationChunk
        if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
            content = chunk.message.content
            if isinstance(content, str):
                return content
            return str(content) if content else ""

        # dict
        if isinstance(chunk, dict):
            return str(chunk.get("content", ""))

        # string
        if isinstance(chunk, str):
            return chunk

        return ""

    def finish(self) -> StreamingMetrics:
        """Mark streaming as complete and return final metrics.

        Returns:
            StreamingMetrics with final values.
        """
        self._end_time = datetime.now()

        duration_ms = None
        if self._start_time:
            duration_ms = (self._end_time - self._start_time).total_seconds() * 1000

        return StreamingMetrics(
            output_tokens=self.output_tokens,
            chunk_count=self._chunk_count,
            content_length=len(self._content),
            start_time=self._start_time or self._end_time,
            end_time=self._end_time,
            duration_ms=duration_ms,
        )

    @property
    def content(self) -> str:
        """Get accumulated content."""
        return self._content

    @property
    def output_tokens(self) -> int:
        """Get estimated output token count."""
        if not self._content:
            return 0
        token_counter = self._provider.get_token_counter(self._model)
        return token_counter.count_text(self._content)

    @property
    def chunk_count(self) -> int:
        """Get number of chunks received."""
        return self._chunk_count

    @property
    def duration_ms(self) -> float | None:
        """Get duration in milliseconds (after finish())."""
        if self._start_time is None or self._end_time is None:
            return None
        return (self._end_time - self._start_time).total_seconds() * 1000

    def reset(self) -> None:
        """Reset tracker for reuse."""
        self._content = ""
        self._chunk_count = 0
        self._start_time = None
        self._end_time = None


class StreamingMetricsCallback:
    """Context manager for tracking streaming metrics.

    Provides a clean interface for tracking a complete streaming
    response with automatic timing.

    Example:
        with StreamingMetricsCallback(model="gpt-4o") as tracker:
            for chunk in llm.stream(messages):
                tracker.add_chunk(chunk)
                print(chunk.content, end="")

        print(f"\\nMetrics: {tracker.metrics}")

    Attributes:
        tracker: The underlying StreamingMetricsTracker
        metrics: Final metrics after context exit
    """

    def __init__(self, model: str = "gpt-4o", provider: Any = None):
        """Initialize StreamingMetricsCallback.

        Args:
            model: Model name for token counting.
            provider: Headroom provider for token counting.
        """
        self._tracker = StreamingMetricsTracker(model=model, provider=provider)
        self._metrics: StreamingMetrics | None = None

    def __enter__(self) -> StreamingMetricsTracker:
        """Enter context, return tracker."""
        return self._tracker

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context, finalize metrics."""
        self._metrics = self._tracker.finish()

    @property
    def tracker(self) -> StreamingMetricsTracker:
        """Get the tracker."""
        return self._tracker

    @property
    def metrics(self) -> StreamingMetrics | None:
        """Get final metrics (after context exit)."""
        return self._metrics


def track_streaming_response(
    stream: Any,
    model: str = "gpt-4o",
    provider: Any = None,
) -> tuple[str, StreamingMetrics]:
    """Track a complete streaming response.

    Convenience function that consumes a stream and returns the
    accumulated content and metrics.

    Args:
        stream: Iterable of streaming chunks.
        model: Model name for token counting.
        provider: Headroom provider for token counting.

    Returns:
        Tuple of (accumulated_content, metrics).

    Example:
        content, metrics = track_streaming_response(
            llm.stream(messages),
            model="gpt-4o"
        )
        print(f"Content: {content}")
        print(f"Tokens: {metrics.output_tokens}")
    """
    tracker = StreamingMetricsTracker(model=model, provider=provider)

    for chunk in stream:
        tracker.add_chunk(chunk)

    metrics = tracker.finish()
    return tracker.content, metrics


async def track_async_streaming_response(
    stream: Any,
    model: str = "gpt-4o",
    provider: Any = None,
) -> tuple[str, StreamingMetrics]:
    """Track a complete async streaming response.

    Async version of track_streaming_response.

    Args:
        stream: Async iterable of streaming chunks.
        model: Model name for token counting.
        provider: Headroom provider for token counting.

    Returns:
        Tuple of (accumulated_content, metrics).

    Example:
        content, metrics = await track_async_streaming_response(
            llm.astream(messages),
            model="gpt-4o"
        )
    """
    tracker = StreamingMetricsTracker(model=model, provider=provider)

    async for chunk in stream:
        tracker.add_chunk(chunk)

    metrics = tracker.finish()
    return tracker.content, metrics

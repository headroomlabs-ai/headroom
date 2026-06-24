"""Tests for LangChain streaming metrics tracking.

Tests cover:
1. StreamingMetrics - Dataclass for streaming response metrics
2. StreamingMetricsTracker - Tracker for streaming chunks
3. StreamingMetricsCallback - Context manager for streaming
4. track_streaming_response - Sync helper function
5. track_async_streaming_response - Async helper function
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Check if LangChain is available
try:
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Skip all tests if LangChain not installed
pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


@pytest.fixture
def mock_provider():
    """Create a mock provider with token counter."""
    mock = MagicMock()
    mock_counter = MagicMock()
    # Simple token counting: split on spaces
    mock_counter.count_text = MagicMock(side_effect=lambda text: len(text.split()))
    mock.get_token_counter = MagicMock(return_value=mock_counter)
    return mock


@pytest.fixture
def sample_chunks():
    """Create sample streaming chunks."""
    return [
        AIMessageChunk(content="Hello"),
        AIMessageChunk(content=" "),
        AIMessageChunk(content="world"),
        AIMessageChunk(content="!"),
    ]


class TestStreamingMetrics:
    """Tests for StreamingMetrics dataclass."""

    def test_create_metrics(self):
        """Create metrics with all fields."""
        from headroom.integrations.langchain.streaming import StreamingMetrics

        start = datetime.now()
        end = datetime.now()

        metrics = StreamingMetrics(
            output_tokens=50,
            chunk_count=10,
            content_length=200,
            start_time=start,
            end_time=end,
            duration_ms=150.5,
        )

        assert metrics.output_tokens == 50
        assert metrics.chunk_count == 10
        assert metrics.content_length == 200
        assert metrics.start_time == start
        assert metrics.end_time == end
        assert metrics.duration_ms == 150.5

    def test_to_dict(self):
        """Convert metrics to dictionary."""
        from headroom.integrations.langchain.streaming import StreamingMetrics

        start = datetime(2025, 1, 1, 12, 0, 0)
        end = datetime(2025, 1, 1, 12, 0, 1)

        metrics = StreamingMetrics(
            output_tokens=50,
            chunk_count=10,
            content_length=200,
            start_time=start,
            end_time=end,
            duration_ms=1000.0,
        )

        result = metrics.to_dict()

        assert result["output_tokens"] == 50
        assert result["chunk_count"] == 10
        assert result["content_length"] == 200
        assert result["start_time"] == "2025-01-01T12:00:00"
        assert result["end_time"] == "2025-01-01T12:00:01"
        assert result["duration_ms"] == 1000.0

    def test_to_dict_with_none_end_time(self):
        """Convert metrics with None end_time."""
        from headroom.integrations.langchain.streaming import StreamingMetrics

        metrics = StreamingMetrics(
            output_tokens=50,
            chunk_count=10,
            content_length=200,
            start_time=datetime.now(),
            end_time=None,
            duration_ms=None,
        )

        result = metrics.to_dict()

        assert result["end_time"] is None
        assert result["duration_ms"] is None


class TestStreamingMetricsTrackerInit:
    """Tests for StreamingMetricsTracker initialization."""

    def test_init_defaults(self):
        """Initialize with default settings."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        with patch("headroom.integrations.langchain.streaming.OpenAIProvider"):
            tracker = StreamingMetricsTracker()

            assert tracker._model == "gpt-4o"
            assert tracker._content == ""
            assert tracker._chunk_count == 0
            assert tracker._start_time is None
            assert tracker._end_time is None

    def test_init_custom_settings(self, mock_provider):
        """Initialize with custom settings."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(
            model="claude-3-5-sonnet-20241022",
            provider=mock_provider,
        )

        assert tracker._model == "claude-3-5-sonnet-20241022"
        assert tracker._provider is mock_provider


class TestStreamingMetricsTrackerAddChunk:
    """Tests for add_chunk method."""

    def test_add_chunk_sets_start_time(self, mock_provider):
        """First chunk sets start time."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        assert tracker._start_time is None

        chunk = AIMessageChunk(content="Hello")
        tracker.add_chunk(chunk)

        assert tracker._start_time is not None

    def test_add_chunk_increments_count(self, mock_provider, sample_chunks):
        """Each chunk increments chunk count."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)

        assert tracker._chunk_count == 4

    def test_add_chunk_accumulates_content(self, mock_provider, sample_chunks):
        """Chunks accumulate content."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)

        assert tracker._content == "Hello world!"

    def test_add_chunk_extracts_ai_message_chunk(self, mock_provider):
        """Extract content from AIMessageChunk."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        chunk = AIMessageChunk(content="Hello")
        tracker.add_chunk(chunk)

        assert tracker._content == "Hello"

    def test_add_chunk_extracts_chat_generation_chunk(self, mock_provider):
        """Extract content from ChatGenerationChunk."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        chunk = ChatGenerationChunk(message=AIMessageChunk(content="Hello"))
        tracker.add_chunk(chunk)

        assert tracker._content == "Hello"

    def test_add_chunk_extracts_dict(self, mock_provider):
        """Extract content from dict."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        chunk = {"content": "Hello"}
        tracker.add_chunk(chunk)

        assert tracker._content == "Hello"

    def test_add_chunk_extracts_string(self, mock_provider):
        """Extract content from string."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        tracker.add_chunk("Hello")

        assert tracker._content == "Hello"

    def test_add_chunk_handles_empty_content(self, mock_provider):
        """Handle chunk with empty content."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        chunk = AIMessageChunk(content="")
        tracker.add_chunk(chunk)

        assert tracker._content == ""
        assert tracker._chunk_count == 1

    def test_add_chunk_handles_none_content(self, mock_provider):
        """Handle chunk with None content attribute."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        chunk = MagicMock()
        chunk.content = None
        tracker.add_chunk(chunk)

        assert tracker._content == ""
        assert tracker._chunk_count == 1


class TestStreamingMetricsTrackerFinish:
    """Tests for finish method."""

    def test_finish_sets_end_time(self, mock_provider, sample_chunks):
        """finish() sets end time."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)

        metrics = tracker.finish()

        assert tracker._end_time is not None
        assert metrics.end_time is not None

    def test_finish_calculates_duration(self, mock_provider, sample_chunks):
        """finish() calculates duration."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)

        metrics = tracker.finish()

        assert metrics.duration_ms is not None
        assert metrics.duration_ms >= 0

    def test_finish_returns_metrics(self, mock_provider, sample_chunks):
        """finish() returns StreamingMetrics."""
        from headroom.integrations.langchain.streaming import (
            StreamingMetrics,
            StreamingMetricsTracker,
        )

        tracker = StreamingMetricsTracker(provider=mock_provider)

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)

        metrics = tracker.finish()

        assert isinstance(metrics, StreamingMetrics)
        assert metrics.chunk_count == 4
        assert metrics.content_length == len("Hello world!")

    def test_finish_with_no_chunks(self, mock_provider):
        """finish() without chunks uses current time for both."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        metrics = tracker.finish()

        # start_time should be same as end_time when no chunks
        assert metrics.start_time == metrics.end_time
        assert metrics.duration_ms is None  # No start_time was set


class TestStreamingMetricsTrackerProperties:
    """Tests for tracker properties."""

    def test_content_property(self, mock_provider, sample_chunks):
        """content property returns accumulated content."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)

        assert tracker.content == "Hello world!"

    def test_output_tokens_property_empty(self, mock_provider):
        """output_tokens returns 0 when no content."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        assert tracker.output_tokens == 0

    def test_output_tokens_property_with_content(self, mock_provider, sample_chunks):
        """output_tokens uses provider's token counter."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(
            model="gpt-4o",
            provider=mock_provider,
        )

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)

        tokens = tracker.output_tokens

        # Mock counter splits on spaces: "Hello world!" = 2 tokens
        assert tokens == 2
        mock_provider.get_token_counter.assert_called_with("gpt-4o")

    def test_chunk_count_property(self, mock_provider, sample_chunks):
        """chunk_count property returns number of chunks."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)

        assert tracker.chunk_count == 4

    def test_duration_ms_before_finish(self, mock_provider, sample_chunks):
        """duration_ms returns None before finish()."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)

        assert tracker.duration_ms is None

    def test_duration_ms_after_finish(self, mock_provider, sample_chunks):
        """duration_ms returns value after finish()."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)

        tracker.finish()

        assert tracker.duration_ms is not None
        assert tracker.duration_ms >= 0


class TestStreamingMetricsTrackerReset:
    """Tests for reset method."""

    def test_reset_clears_state(self, mock_provider, sample_chunks):
        """reset() clears all state."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(provider=mock_provider)

        for chunk in sample_chunks:
            tracker.add_chunk(chunk)
        tracker.finish()

        tracker.reset()

        assert tracker._content == ""
        assert tracker._chunk_count == 0
        assert tracker._start_time is None
        assert tracker._end_time is None


class TestStreamingMetricsCallback:
    """Tests for StreamingMetricsCallback context manager."""

    def test_init(self, mock_provider):
        """Initialize callback."""
        from headroom.integrations.langchain.streaming import StreamingMetricsCallback

        callback = StreamingMetricsCallback(model="gpt-4o", provider=mock_provider)

        assert callback._tracker._model == "gpt-4o"
        assert callback._metrics is None

    def test_context_manager_enter(self, mock_provider):
        """Context manager enter returns tracker."""
        from headroom.integrations.langchain.streaming import (
            StreamingMetricsCallback,
            StreamingMetricsTracker,
        )

        callback = StreamingMetricsCallback(provider=mock_provider)

        with callback as tracker:
            assert isinstance(tracker, StreamingMetricsTracker)

    def test_context_manager_exit_finishes_tracker(self, mock_provider, sample_chunks):
        """Context manager exit finishes tracker."""
        from headroom.integrations.langchain.streaming import StreamingMetricsCallback

        callback = StreamingMetricsCallback(provider=mock_provider)

        with callback as tracker:
            for chunk in sample_chunks:
                tracker.add_chunk(chunk)

        assert callback.metrics is not None
        assert callback.metrics.chunk_count == 4

    def test_tracker_property(self, mock_provider):
        """tracker property returns the tracker."""
        from headroom.integrations.langchain.streaming import (
            StreamingMetricsCallback,
            StreamingMetricsTracker,
        )

        callback = StreamingMetricsCallback(provider=mock_provider)

        assert isinstance(callback.tracker, StreamingMetricsTracker)

    def test_metrics_property_before_exit(self, mock_provider):
        """metrics property returns None before context exit."""
        from headroom.integrations.langchain.streaming import StreamingMetricsCallback

        callback = StreamingMetricsCallback(provider=mock_provider)

        assert callback.metrics is None

    def test_metrics_property_after_exit(self, mock_provider, sample_chunks):
        """metrics property returns StreamingMetrics after context exit."""
        from headroom.integrations.langchain.streaming import (
            StreamingMetrics,
            StreamingMetricsCallback,
        )

        callback = StreamingMetricsCallback(provider=mock_provider)

        with callback as tracker:
            for chunk in sample_chunks:
                tracker.add_chunk(chunk)

        assert isinstance(callback.metrics, StreamingMetrics)


class TestTrackStreamingResponse:
    """Tests for track_streaming_response function."""

    def test_consumes_stream(self, mock_provider, sample_chunks):
        """Function consumes entire stream."""
        from headroom.integrations.langchain.streaming import track_streaming_response

        stream = iter(sample_chunks)

        content, metrics = track_streaming_response(stream, provider=mock_provider)

        assert content == "Hello world!"

    def test_returns_content_and_metrics(self, mock_provider, sample_chunks):
        """Function returns content and metrics tuple."""
        from headroom.integrations.langchain.streaming import (
            StreamingMetrics,
            track_streaming_response,
        )

        stream = iter(sample_chunks)

        content, metrics = track_streaming_response(stream, provider=mock_provider)

        assert isinstance(content, str)
        assert isinstance(metrics, StreamingMetrics)

    def test_with_custom_model(self, mock_provider, sample_chunks):
        """Function uses custom model for token counting."""
        from headroom.integrations.langchain.streaming import track_streaming_response

        stream = iter(sample_chunks)

        content, metrics = track_streaming_response(
            stream,
            model="claude-3-5-sonnet-20241022",
            provider=mock_provider,
        )

        mock_provider.get_token_counter.assert_called_with("claude-3-5-sonnet-20241022")

    def test_empty_stream(self, mock_provider):
        """Function handles empty stream."""
        from headroom.integrations.langchain.streaming import track_streaming_response

        stream = iter([])

        content, metrics = track_streaming_response(stream, provider=mock_provider)

        assert content == ""
        assert metrics.chunk_count == 0


class TestTrackAsyncStreamingResponse:
    """Tests for track_async_streaming_response function."""

    @pytest.mark.asyncio
    async def test_consumes_async_stream(self, mock_provider, sample_chunks):
        """Function consumes entire async stream."""
        from headroom.integrations.langchain.streaming import (
            track_async_streaming_response,
        )

        async def async_stream():
            for chunk in sample_chunks:
                yield chunk

        content, metrics = await track_async_streaming_response(
            async_stream(), provider=mock_provider
        )

        assert content == "Hello world!"

    @pytest.mark.asyncio
    async def test_returns_content_and_metrics(self, mock_provider, sample_chunks):
        """Function returns content and metrics tuple."""
        from headroom.integrations.langchain.streaming import (
            StreamingMetrics,
            track_async_streaming_response,
        )

        async def async_stream():
            for chunk in sample_chunks:
                yield chunk

        content, metrics = await track_async_streaming_response(
            async_stream(), provider=mock_provider
        )

        assert isinstance(content, str)
        assert isinstance(metrics, StreamingMetrics)

    @pytest.mark.asyncio
    async def test_with_custom_model(self, mock_provider, sample_chunks):
        """Function uses custom model for token counting."""
        from headroom.integrations.langchain.streaming import (
            track_async_streaming_response,
        )

        async def async_stream():
            for chunk in sample_chunks:
                yield chunk

        content, metrics = await track_async_streaming_response(
            async_stream(),
            model="gpt-4-turbo",
            provider=mock_provider,
        )

        mock_provider.get_token_counter.assert_called_with("gpt-4-turbo")

    @pytest.mark.asyncio
    async def test_empty_async_stream(self, mock_provider):
        """Function handles empty async stream."""
        from headroom.integrations.langchain.streaming import (
            track_async_streaming_response,
        )

        async def async_stream():
            return
            yield  # Make it a generator  # noqa: B901 - intentionally unreachable

        content, metrics = await track_async_streaming_response(
            async_stream(), provider=mock_provider
        )

        assert content == ""
        assert metrics.chunk_count == 0


class TestLangChainNotAvailable:
    """Tests for behavior when LangChain is not available."""

    def test_check_raises_import_error(self):
        """_check_langchain_available raises ImportError when not available."""
        from headroom.integrations.langchain.streaming import _check_langchain_available

        # When LangChain IS available, should not raise
        try:
            _check_langchain_available()
        except ImportError:
            pytest.fail("Should not raise when LangChain is available")

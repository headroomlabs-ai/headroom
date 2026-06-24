"""Tests for extended LangChain integration modules.

Tests cover:
1. langchain_providers - Provider auto-detection
2. langchain_memory - HeadroomChatMessageHistory
3. langchain_retriever - HeadroomDocumentCompressor
4. langchain_agents - HeadroomToolWrapper
5. langchain_langsmith - LangSmith integration
6. langchain_streaming - Streaming metrics
"""

import json
from unittest.mock import MagicMock

import pytest

# Check if LangChain is available
try:
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import StructuredTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Skip all tests if LangChain not installed
pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


class TestProviderDetection:
    """Tests for langchain_providers module."""

    def test_detect_openai_provider(self):
        """Detect OpenAI from ChatOpenAI class."""
        from headroom.integrations.langchain.providers import detect_provider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai.chat_models"

        provider = detect_provider(mock_model)
        assert provider == "openai"

    def test_detect_anthropic_provider(self):
        """Detect Anthropic from ChatAnthropic class."""
        from headroom.integrations.langchain.providers import detect_provider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatAnthropic"
        mock_model.__class__.__module__ = "langchain_anthropic.chat_models"

        provider = detect_provider(mock_model)
        assert provider == "anthropic"

    def test_detect_google_provider(self):
        """Detect Google from ChatGoogleGenerativeAI class."""
        from headroom.integrations.langchain.providers import detect_provider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_model.__class__.__module__ = "langchain_google_genai"

        provider = detect_provider(mock_model)
        assert provider == "google"

    def test_detect_fallback_to_openai(self):
        """Fall back to OpenAI for unknown models."""
        from headroom.integrations.langchain.providers import detect_provider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "CustomChatModel"
        mock_model.__class__.__module__ = "my_custom_module"

        provider = detect_provider(mock_model)
        assert provider == "openai"

    def test_detect_from_model_name_claude(self):
        """Detect Anthropic from model name containing 'claude'."""
        from headroom.integrations.langchain.providers import detect_provider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "CustomModel"
        mock_model.__class__.__module__ = "custom"
        mock_model.model_name = "claude-3-5-sonnet-20241022"

        provider = detect_provider(mock_model)
        assert provider == "anthropic"

    def test_get_headroom_provider_openai(self):
        """Get OpenAIProvider for OpenAI model."""
        from headroom.integrations.langchain.providers import get_headroom_provider
        from headroom.providers import OpenAIProvider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai"

        provider = get_headroom_provider(mock_model)
        assert isinstance(provider, OpenAIProvider)

    def test_get_headroom_provider_anthropic(self):
        """Get AnthropicProvider for Anthropic model."""
        from headroom.integrations.langchain.providers import get_headroom_provider
        from headroom.providers import AnthropicProvider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatAnthropic"
        mock_model.__class__.__module__ = "langchain_anthropic"

        provider = get_headroom_provider(mock_model)
        assert isinstance(provider, AnthropicProvider)

    def test_get_model_name_from_langchain(self):
        """Extract model name from LangChain model."""
        from headroom.integrations.langchain.providers import get_model_name_from_langchain

        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        name = get_model_name_from_langchain(mock_model)
        assert name == "gpt-4o"

    def test_get_model_name_fallback(self):
        """Fall back when model name not available."""
        from headroom.integrations.langchain.providers import get_model_name_from_langchain

        mock_model = MagicMock(spec=[])
        mock_model.__class__.__name__ = "ChatOpenAI"

        name = get_model_name_from_langchain(mock_model)
        assert name == "gpt-4o"  # Default for OpenAI


class TestHeadroomChatMessageHistory:
    """Tests for HeadroomChatMessageHistory memory wrapper."""

    def test_init(self):
        """Initialize with base history."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        mock_history = MagicMock()
        mock_history.messages = []

        wrapper = HeadroomChatMessageHistory(
            mock_history,
            compress_threshold_tokens=4000,
            keep_recent_turns=5,
        )

        assert wrapper._base is mock_history
        assert wrapper._threshold == 4000
        assert wrapper._keep_recent_turns == 5

    def test_messages_passthrough_under_threshold(self):
        """Messages pass through when under threshold."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        mock_history = MagicMock()
        mock_history.messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        wrapper = HeadroomChatMessageHistory(
            mock_history,
            compress_threshold_tokens=10000,  # High threshold
        )

        messages = wrapper.messages
        assert len(messages) == 2
        assert messages[0].content == "Hello"

    def test_add_message_delegates(self):
        """add_message delegates to base history."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        mock_history = MagicMock()
        mock_history.messages = []

        wrapper = HeadroomChatMessageHistory(mock_history)
        message = HumanMessage(content="Test")
        wrapper.add_message(message)

        mock_history.add_message.assert_called_once_with(message)

    def test_clear_delegates(self):
        """clear delegates to base history."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        mock_history = MagicMock()
        mock_history.messages = []

        wrapper = HeadroomChatMessageHistory(mock_history)
        wrapper.clear()

        mock_history.clear.assert_called_once()

    def test_get_compression_stats(self):
        """Get compression statistics."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        mock_history = MagicMock()
        mock_history.messages = []

        wrapper = HeadroomChatMessageHistory(mock_history)
        stats = wrapper.get_compression_stats()

        assert "compression_count" in stats
        assert "total_tokens_saved" in stats
        assert stats["compression_count"] == 0


class TestHeadroomDocumentCompressor:
    """Tests for HeadroomDocumentCompressor retriever integration."""

    def test_init(self):
        """Initialize with defaults."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        assert compressor.max_documents == 10
        assert compressor.min_relevance == 0.0
        assert compressor.prefer_diverse is False

    def test_init_custom(self):
        """Initialize with custom settings."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(
            max_documents=5,
            min_relevance=0.5,
            prefer_diverse=True,
        )

        assert compressor.max_documents == 5
        assert compressor.min_relevance == 0.5
        assert compressor.prefer_diverse is True

    def test_compress_passthrough_under_limit(self):
        """Pass through when under max_documents."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=10)

        docs = [
            Document(page_content="Python is a programming language."),
            Document(page_content="JavaScript runs in browsers."),
        ]

        result = compressor.compress_documents(docs, "What is Python?")

        assert len(result) == 2

    def test_compress_reduces_to_max(self):
        """Compress when over max_documents."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=2)

        docs = [
            Document(page_content="Python is a programming language."),
            Document(page_content="Java is also a language."),
            Document(page_content="Weather today is sunny."),
            Document(page_content="Cats are cute animals."),
        ]

        result = compressor.compress_documents(docs, "programming language")

        assert len(result) == 2

    def test_compress_prefers_relevant(self):
        """Keep most relevant documents."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=1)

        docs = [
            Document(page_content="Weather today is sunny."),
            Document(page_content="Python programming tutorial basics."),
            Document(page_content="Cats are cute animals."),
        ]

        result = compressor.compress_documents(docs, "Python tutorial")

        assert len(result) == 1
        assert "Python" in result[0].page_content

    def test_metrics_tracked(self):
        """Compression metrics are tracked."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=2)

        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
            Document(page_content="Doc 3"),
        ]

        compressor.compress_documents(docs, "query")

        metrics = compressor.last_metrics
        assert metrics is not None
        assert metrics.documents_before == 3
        assert metrics.documents_after == 2
        assert metrics.documents_removed == 1

    def test_get_compression_stats(self):
        """Get compression statistics."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=1)
        docs = [Document(page_content="A"), Document(page_content="B")]

        compressor.compress_documents(docs, "A")
        stats = compressor.get_compression_stats()

        assert "documents_before" in stats
        assert "documents_after" in stats
        assert "average_relevance" in stats


class TestHeadroomToolWrapper:
    """Tests for HeadroomToolWrapper agent integration."""

    def test_init(self):
        """Initialize wrapper."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"

        wrapper = HeadroomToolWrapper(mock_tool)

        assert wrapper.name == "test_tool"
        assert wrapper.description == "A test tool"

    def test_call_passthrough_small_output(self):
        """Small outputs pass through without compression."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        mock_tool = MagicMock()
        mock_tool.name = "test"
        mock_tool.description = "test"
        mock_tool.invoke.return_value = "small result"

        wrapper = HeadroomToolWrapper(mock_tool, min_chars_to_compress=1000)
        result = wrapper("query")

        assert result == "small result"

    def test_call_compresses_large_json(self):
        """Large JSON outputs get compressed."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "search"

        # Large JSON output
        large_output = json.dumps([{"id": i, "data": "x" * 100} for i in range(50)])
        mock_tool.invoke.return_value = large_output

        wrapper = HeadroomToolWrapper(mock_tool, min_chars_to_compress=100)
        result = wrapper("query")

        # Should be smaller after compression
        assert len(result) <= len(large_output)

    def test_as_langchain_tool(self):
        """Convert to LangChain tool."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        mock_tool = MagicMock()
        mock_tool.name = "test"
        mock_tool.description = "test tool"
        mock_tool.invoke.return_value = "result"

        wrapper = HeadroomToolWrapper(mock_tool)
        lc_tool = wrapper.as_langchain_tool()

        assert isinstance(lc_tool, StructuredTool)
        assert lc_tool.name == "test"

    def test_wrap_tools_with_headroom(self):
        """Wrap multiple tools at once."""
        from headroom.integrations.langchain.agents import wrap_tools_with_headroom

        tools = []
        for i in range(3):
            mock = MagicMock()
            mock.name = f"tool_{i}"
            mock.description = f"Tool {i}"
            mock.invoke.return_value = "result"
            tools.append(mock)

        wrapped = wrap_tools_with_headroom(tools)

        assert len(wrapped) == 3
        assert all(isinstance(t, StructuredTool) for t in wrapped)

    def test_metrics_collector(self):
        """Tool metrics are collected."""
        from headroom.integrations.langchain.agents import (
            HeadroomToolWrapper,
            ToolMetricsCollector,
        )

        collector = ToolMetricsCollector()

        mock_tool = MagicMock()
        mock_tool.name = "test"
        mock_tool.description = "test"
        mock_tool.invoke.return_value = "result"

        wrapper = HeadroomToolWrapper(mock_tool, metrics_collector=collector)
        wrapper("query")

        assert len(collector.metrics) == 1
        assert collector.metrics[0].tool_name == "test"


class TestHeadroomLangSmithCallbackHandler:
    """Tests for LangSmith integration."""

    def test_init(self):
        """Initialize handler."""
        from headroom.integrations.langchain.langsmith import (
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler(auto_update_runs=False)

        assert handler._auto_update is False
        assert handler._pending_metrics == {}

    def test_set_headroom_metrics(self):
        """Set metrics for a run."""
        from headroom.integrations.langchain.langsmith import (
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler(auto_update_runs=False)

        handler.set_headroom_metrics(
            run_id="test-run-123",
            tokens_before=1000,
            tokens_after=800,
            transforms_applied=["smart_crusher"],
        )

        assert "test-run-123" in handler._pending_metrics
        metrics = handler._pending_metrics["test-run-123"]
        assert metrics.tokens_before == 1000
        assert metrics.tokens_after == 800
        assert metrics.tokens_saved == 200
        assert metrics.savings_percent == 20.0

    def test_get_run_metrics(self):
        """Get metrics for a specific run."""
        from headroom.integrations.langchain.langsmith import (
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler(auto_update_runs=False)
        handler._run_metrics["run-1"] = {"headroom.tokens_saved": 100}

        metrics = handler.get_run_metrics("run-1")
        assert metrics["headroom.tokens_saved"] == 100

    def test_get_summary(self):
        """Get summary statistics."""
        from headroom.integrations.langchain.langsmith import (
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler(auto_update_runs=False)
        handler._run_metrics = {
            "run-1": {"headroom.tokens_saved": 100, "headroom.savings_percent": 20},
            "run-2": {"headroom.tokens_saved": 200, "headroom.savings_percent": 30},
        }

        summary = handler.get_summary()
        assert summary["total_runs"] == 2
        assert summary["total_tokens_saved"] == 300
        assert summary["average_savings_percent"] == 25.0

    def test_reset(self):
        """Reset clears all metrics."""
        from headroom.integrations.langchain.langsmith import (
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler(auto_update_runs=False)
        handler._run_metrics = {"run-1": {}}
        handler._pending_metrics = {"run-2": MagicMock()}

        handler.reset()

        assert handler._run_metrics == {}
        assert handler._pending_metrics == {}


class TestStreamingMetricsTracker:
    """Tests for streaming metrics tracking."""

    def test_init(self):
        """Initialize tracker."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(model="gpt-4o")

        assert tracker._model == "gpt-4o"
        assert tracker._content == ""
        assert tracker._chunk_count == 0

    def test_add_chunk_string(self):
        """Add string chunks."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker()
        tracker.add_chunk("Hello ")
        tracker.add_chunk("world!")

        assert tracker.content == "Hello world!"
        assert tracker.chunk_count == 2

    def test_add_chunk_with_content_attr(self):
        """Add chunks with content attribute."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker()

        chunk1 = MagicMock()
        chunk1.content = "Hello "
        chunk2 = MagicMock()
        chunk2.content = "world!"

        tracker.add_chunk(chunk1)
        tracker.add_chunk(chunk2)

        assert tracker.content == "Hello world!"

    def test_output_tokens(self):
        """Count output tokens."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker(model="gpt-4o")
        tracker.add_chunk("Hello world, this is a test message.")

        tokens = tracker.output_tokens
        assert tokens > 0

    def test_finish(self):
        """Finish tracking and get metrics."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker()
        tracker.add_chunk("Test content")
        metrics = tracker.finish()

        assert metrics.chunk_count == 1
        assert metrics.content_length == len("Test content")
        assert metrics.duration_ms is not None
        assert metrics.end_time is not None

    def test_reset(self):
        """Reset tracker for reuse."""
        from headroom.integrations.langchain.streaming import StreamingMetricsTracker

        tracker = StreamingMetricsTracker()
        tracker.add_chunk("Content")
        tracker.finish()

        tracker.reset()

        assert tracker.content == ""
        assert tracker.chunk_count == 0

    def test_streaming_metrics_callback(self):
        """Test context manager interface."""
        from headroom.integrations.langchain.streaming import StreamingMetricsCallback

        with StreamingMetricsCallback(model="gpt-4o") as tracker:
            tracker.add_chunk("Hello")
            tracker.add_chunk(" world")

        # After context exit, metrics should be available
        # (accessed via the callback object, not the tracker)

    def test_track_streaming_response(self):
        """Track a complete streaming response."""
        from headroom.integrations.langchain.streaming import track_streaming_response

        chunks = ["Hello ", "world", "!"]
        content, metrics = track_streaming_response(iter(chunks), model="gpt-4o")

        assert content == "Hello world!"
        assert metrics.chunk_count == 3


class TestAutoDetectProviderInChatModel:
    """Tests for auto_detect_provider in HeadroomChatModel."""

    def test_auto_detect_enabled_by_default(self):
        """auto_detect_provider is True by default."""
        from headroom.integrations import HeadroomChatModel

        mock_model = MagicMock()
        mock_model._llm_type = "test"
        mock_model._identifying_params = {}
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai"

        model = HeadroomChatModel(mock_model)
        assert model.auto_detect_provider is True

    def test_auto_detect_can_be_disabled(self):
        """auto_detect_provider can be set to False."""
        from headroom.integrations import HeadroomChatModel

        mock_model = MagicMock()
        mock_model._llm_type = "test"
        mock_model._identifying_params = {}

        model = HeadroomChatModel(mock_model, auto_detect_provider=False)
        assert model.auto_detect_provider is False

    def test_pipeline_uses_detected_provider(self):
        """Pipeline uses auto-detected provider."""
        from headroom.integrations import HeadroomChatModel
        from headroom.providers import AnthropicProvider

        mock_model = MagicMock()
        mock_model._llm_type = "test"
        mock_model._identifying_params = {}
        mock_model.__class__.__name__ = "ChatAnthropic"
        mock_model.__class__.__module__ = "langchain_anthropic"

        model = HeadroomChatModel(mock_model)
        _ = model.pipeline  # Force lazy init

        assert isinstance(model._provider, AnthropicProvider)

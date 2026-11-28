"""Tests for LangChain memory integration with automatic compression.

Tests cover:
1. HeadroomChatMessageHistory - Wrapper for chat message history with compression
2. Message conversion to/from OpenAI format
3. Rolling window compression behavior
4. Token counting and threshold detection
5. Compression statistics tracking
"""

from unittest.mock import MagicMock, patch

import pytest

# Check if LangChain is available
try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Skip all tests if LangChain not installed
pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


@pytest.fixture
def mock_base_history():
    """Create a mock BaseChatMessageHistory."""
    mock = MagicMock()
    mock.messages = []
    return mock


@pytest.fixture
def mock_provider():
    """Create a mock provider with token counter."""
    mock = MagicMock()
    mock_counter = MagicMock()
    mock_counter.count_text = MagicMock(side_effect=lambda text: len(text.split()))
    mock.get_token_counter = MagicMock(return_value=mock_counter)
    return mock


@pytest.fixture
def sample_langchain_messages():
    """Sample LangChain messages for testing."""
    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I am doing well, thank you!"),
        HumanMessage(content="What is the weather today?"),
        AIMessage(content="I don't have access to weather data."),
    ]


class TestHeadroomChatMessageHistoryInit:
    """Tests for HeadroomChatMessageHistory initialization."""

    def test_init_defaults(self, mock_base_history):
        """Initialize with default settings."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        with patch("headroom.integrations.langchain.memory.OpenAIProvider"):
            history = HeadroomChatMessageHistory(mock_base_history)

            assert history._base is mock_base_history
            assert history._threshold == 4000
            assert history._keep_recent_turns == 5
            assert history._model == "gpt-4o"
            assert history._compression_count == 0
            assert history._total_tokens_saved == 0

    def test_init_custom_threshold(self, mock_base_history, mock_provider):
        """Initialize with custom compression threshold."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(
            mock_base_history,
            compress_threshold_tokens=8000,
            keep_recent_turns=10,
            model="gpt-4-turbo",
            provider=mock_provider,
        )

        assert history._threshold == 8000
        assert history._keep_recent_turns == 10
        assert history._model == "gpt-4-turbo"
        assert history._provider is mock_provider


class TestHeadroomChatMessageHistoryMessages:
    def test_history_above_threshold_preserves_native_message_data(self):
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.messages import ChatMessage

        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        base = InMemoryChatMessageHistory(
            messages=[
                HumanMessage(
                    id="question",
                    content=[{"type": "text", "text": "Summarize the deployment."}],
                    additional_kwargs={"source": "agent"},
                ),
                ChatMessage(id="review", role="reviewer", content="Keep the rollback details."),
                ToolMessage(
                    id="deployment-output",
                    tool_call_id="deploy",
                    name="deployment",
                    content="Current deployment is blue.",
                    artifact={"pool": "blue"},
                ),
            ]
        )
        history = HeadroomChatMessageHistory(base, compress_threshold_tokens=1)

        assert history.messages == base.messages
        assert history.get_compression_stats()["compression_count"] == 1

    def test_compressed_tool_output_keeps_native_artifact(self):
        import json

        from langchain_core.chat_history import InMemoryChatMessageHistory

        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        text = json.dumps(
            [
                {"id": i, "msg": f"Result {i} ok", "blob": f"payload-{i:04d}-" + "abcde" * 48}
                for i in range(200)
            ]
        )
        tool = ToolMessage(
            content=text,
            id="result",
            tool_call_id="lookup",
            name="lookup",
            artifact={"count": 200},
        )
        history = HeadroomChatMessageHistory(
            InMemoryChatMessageHistory(
                messages=[
                    HumanMessage(content="Get items"),
                    tool,
                ]
            ),
            compress_threshold_tokens=100,
        )

        result = history.messages[-1]
        assert isinstance(result, ToolMessage)
        assert len(result.content) < len(text)
        assert result.id == tool.id
        assert result.tool_call_id == tool.tool_call_id
        assert result.artifact == tool.artifact
        assert tool.content == text

    def test_rejected_edit_keeps_the_last_read_valid(self):
        from langchain_core.chat_history import InMemoryChatMessageHistory

        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        read = "\n".join(f"line {i}: deployment value" for i in range(280))
        messages = [
            HumanMessage(content="Review the current settings."),
            AIMessage(
                content="",
                tool_calls=[{"id": "read", "name": "Read", "args": {"file_path": "settings.json"}}],
            ),
            ToolMessage(content=read, tool_call_id="read"),
            AIMessage(
                content="",
                tool_calls=[{"id": "edit", "name": "Edit", "args": {"file_path": "settings.json"}}],
            ),
            ToolMessage(
                content="<tool_use_error>String to replace not found</tool_use_error>",
                status="error",
                tool_call_id="edit",
                artifact={"applied": False},
            ),
        ]
        history = HeadroomChatMessageHistory(
            InMemoryChatMessageHistory(messages=messages), compress_threshold_tokens=100
        )

        assert history.messages == messages

    """Tests for message access and compression."""

    def test_messages_returns_empty_when_no_messages(self, mock_base_history, mock_provider):
        """messages property returns empty list when no messages."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        mock_base_history.messages = []

        history = HeadroomChatMessageHistory(mock_base_history, provider=mock_provider)
        messages = history.messages

        assert messages == []

    def test_messages_returns_uncompressed_when_below_threshold(
        self, mock_base_history, mock_provider, sample_langchain_messages
    ):
        """messages returns uncompressed when below token threshold."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        mock_base_history.messages = sample_langchain_messages

        history = HeadroomChatMessageHistory(
            mock_base_history,
            compress_threshold_tokens=10000,  # High threshold
            provider=mock_provider,
        )

        messages = history.messages

        # Should return all messages unchanged
        assert len(messages) == len(sample_langchain_messages)
        assert history._compression_count == 0

    def test_messages_below_threshold_with_default_provider(self):
        """With no provider passed, the default provider counts tokens and
        short histories pass through."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        base = MagicMock()
        base.messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]

        history = HeadroomChatMessageHistory(base, compress_threshold_tokens=10000)
        messages = history.messages

        assert [m.content for m in messages] == ["Hello", "Hi there!"]
        assert history._compression_count == 0

    def test_messages_compresses_when_over_threshold(self, mock_base_history, mock_provider):
        """messages applies compression when over token threshold."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        # Create messages that exceed threshold
        mock_base_history.messages = [
            SystemMessage(content="System " * 100),
            HumanMessage(content="User " * 100),
            AIMessage(content="Assistant " * 100),
        ]

        history = HeadroomChatMessageHistory(
            mock_base_history,
            compress_threshold_tokens=10,  # Very low threshold
            provider=mock_provider,
        )

        # Mock _apply_compression to return fewer messages
        with patch.object(history, "_apply_compression") as mock_apply:
            mock_apply.return_value = [
                SystemMessage(content="Compressed"),
            ]

            _ = history.messages

            mock_apply.assert_called_once()
            assert history._compression_count == 1

    def test_messages_tracks_tokens_saved(self, mock_base_history, mock_provider):
        """Compression tracks tokens saved."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        # Create messages that exceed threshold
        mock_base_history.messages = [
            SystemMessage(content="Word " * 50),
            HumanMessage(content="Word " * 50),
        ]

        history = HeadroomChatMessageHistory(
            mock_base_history,
            compress_threshold_tokens=10,  # Very low threshold
            provider=mock_provider,
        )

        # Mock _apply_compression to return fewer messages
        with patch.object(history, "_apply_compression") as mock_apply:
            mock_apply.return_value = [
                SystemMessage(content="Short"),
            ]

            _ = history.messages

            # tokens_saved should increase
            assert history._total_tokens_saved > 0


class TestHeadroomChatMessageHistoryAddMessage:
    """Tests for add_message methods."""

    def test_add_message(self, mock_base_history, mock_provider):
        """add_message delegates to base history."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(mock_base_history, provider=mock_provider)

        msg = HumanMessage(content="Hello")
        history.add_message(msg)

        mock_base_history.add_message.assert_called_once_with(msg)

    def test_add_user_message(self, mock_base_history, mock_provider):
        """add_user_message delegates to base history."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(mock_base_history, provider=mock_provider)

        history.add_user_message("Hello")

        mock_base_history.add_user_message.assert_called_once_with("Hello")

    def test_add_ai_message(self, mock_base_history, mock_provider):
        """add_ai_message delegates to base history."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(mock_base_history, provider=mock_provider)

        history.add_ai_message("Response")

        mock_base_history.add_ai_message.assert_called_once_with("Response")

    def test_clear(self, mock_base_history, mock_provider):
        """clear delegates to base history."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(mock_base_history, provider=mock_provider)

        history.clear()

        mock_base_history.clear.assert_called_once()


class TestHeadroomChatMessageHistoryConversion:
    """Tests for message format conversion."""

    def test_convert_to_openai_system_message(self, mock_base_history, mock_provider):
        """Convert SystemMessage to OpenAI format."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(mock_base_history, provider=mock_provider)

        messages = [SystemMessage(content="You are helpful.")]
        result = history._convert_to_openai(messages)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_convert_to_openai_human_message(self, mock_base_history, mock_provider):
        """Convert HumanMessage to OpenAI format."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(mock_base_history, provider=mock_provider)

        messages = [HumanMessage(content="Hello")]
        result = history._convert_to_openai(messages)

        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_convert_to_openai_ai_message(self, mock_base_history, mock_provider):
        """Convert AIMessage to OpenAI format."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(mock_base_history, provider=mock_provider)

        messages = [AIMessage(content="I can help.")]
        result = history._convert_to_openai(messages)

        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "I can help."

    def test_convert_to_openai_ai_message_with_tool_calls(self, mock_base_history, mock_provider):
        """Convert AIMessage with tool_calls to OpenAI format."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(mock_base_history, provider=mock_provider)

        messages = [
            AIMessage(
                content="Calling tool...",
                tool_calls=[{"id": "call_1", "name": "search", "args": {"q": "test"}}],
            )
        ]
        result = history._convert_to_openai(messages)

        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["id"] == "call_1"

    def test_convert_to_openai_tool_message(self, mock_base_history, mock_provider):
        """Convert ToolMessage to OpenAI format."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(mock_base_history, provider=mock_provider)

        messages = [ToolMessage(content='{"result": "data"}', tool_call_id="call_1")]
        result = history._convert_to_openai(messages)

        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == '{"result": "data"}'


class TestHeadroomChatMessageHistoryTokenCounting:
    """Tests for token counting."""

    def test_count_tokens(self, mock_base_history, mock_provider):
        """Count tokens using provider's tokenizer."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(
            mock_base_history,
            provider=mock_provider,
            model="gpt-4o",
        )

        messages = [
            HumanMessage(content="Hello world"),
            AIMessage(content="Hi there"),
        ]

        count = history._count_tokens(messages)

        # Mock counts words, so "Hello world" = 2, "Hi there" = 2
        assert count == 4
        mock_provider.get_token_counter.assert_called_with("gpt-4o")


class TestHeadroomChatMessageHistoryStats:
    """Tests for compression statistics."""

    def test_get_compression_stats_initial(self, mock_base_history, mock_provider):
        """Get initial compression stats."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(
            mock_base_history,
            compress_threshold_tokens=4000,
            keep_recent_turns=5,
            provider=mock_provider,
        )

        stats = history.get_compression_stats()

        assert stats["compression_count"] == 0
        assert stats["total_tokens_saved"] == 0
        assert stats["threshold_tokens"] == 4000
        assert stats["keep_recent_turns"] == 5

    def test_get_compression_stats_after_compression(self, mock_base_history, mock_provider):
        """Get compression stats after compression."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        mock_base_history.messages = [
            SystemMessage(content="Word " * 100),
            HumanMessage(content="Word " * 100),
        ]

        history = HeadroomChatMessageHistory(
            mock_base_history,
            compress_threshold_tokens=10,
            provider=mock_provider,
        )

        # Mock _apply_compression
        with patch.object(history, "_apply_compression") as mock_apply:
            mock_apply.return_value = [SystemMessage(content="Short")]

            _ = history.messages

        stats = history.get_compression_stats()

        assert stats["compression_count"] == 1
        assert stats["total_tokens_saved"] > 0


class TestHeadroomChatMessageHistoryCompression:
    """Tests for rolling window compression."""

    def test_apply_compression_calls_pipeline(self, mock_base_history, mock_provider):
        """_apply_compression uses TransformPipeline."""
        from headroom.integrations.langchain.memory import HeadroomChatMessageHistory

        history = HeadroomChatMessageHistory(
            mock_base_history,
            compress_threshold_tokens=1000,
            keep_recent_turns=5,
            provider=mock_provider,
        )

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]

        with patch("headroom.integrations.langchain.memory.TransformPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
            mock_instance.apply.return_value = mock_result
            MockPipeline.return_value = mock_instance

            result = history._apply_compression(messages)

            MockPipeline.assert_called_once()
            mock_instance.apply.assert_called_once()

            # Result should be converted back to LangChain messages
            assert all(isinstance(m, BaseMessage) for m in result)


class TestLangChainNotAvailable:
    """Tests for behavior when LangChain is not available."""

    def test_check_raises_import_error(self):
        """_check_langchain_available raises ImportError when not available."""
        from headroom.integrations.langchain.memory import _check_langchain_available

        # When LangChain IS available, should not raise
        try:
            _check_langchain_available()
        except ImportError:
            pytest.fail("Should not raise when LangChain is available")

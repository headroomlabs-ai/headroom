"""Tests for LangGraph tool message compression integration.

Tests cover:
1. compress_tool_messages - Compresses large ToolMessages in a message list
2. create_compress_tool_messages_node - LangGraph node factory
3. CompressToolMessagesConfig - Configuration options
4. CompressToolMessagesResult - Result with metrics
5. ToolMessageCompressionMetrics - Per-message metrics
"""

import json

import pytest

# Check if LangChain is available
try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Skip all tests if LangChain not installed
pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


def _make_large_tool_output(num_items: int = 200) -> str:
    """Generate a large JSON array string that will trigger compression."""
    items = [
        {"id": i, "name": f"item_{i}", "value": i * 1.5, "status": "ok"} for i in range(num_items)
    ]
    return json.dumps(items)


def _make_messages_with_tool_output(tool_content: str, tool_call_id: str = "call_1") -> list:
    """Create a typical message sequence with a tool call and result."""
    return [
        HumanMessage(content="Get the data"),
        AIMessage(content="", tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}]),
        ToolMessage(content=tool_content, tool_call_id=tool_call_id),
    ]


class TestCompressToolMessages:
    """Tests for the compress_tool_messages function."""

    def test_compresses_large_tool_message(self):
        """Large ToolMessage content should be compressed."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        large_output = _make_large_tool_output(200)
        messages = _make_messages_with_tool_output(large_output)

        result = compress_tool_messages(messages)

        # Should have same number of messages
        assert len(result.messages) == 3
        # ToolMessage should be smaller
        compressed_content = result.messages[2].content
        assert len(compressed_content) < len(large_output)

    def test_preserves_small_tool_messages(self):
        """Small ToolMessages should not be compressed."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        small_output = '{"result": "ok"}'
        messages = _make_messages_with_tool_output(small_output)

        result = compress_tool_messages(messages)

        # Content should be unchanged
        assert result.messages[2].content == small_output
        assert result.messages_compressed == 0

    def test_preserves_non_tool_messages(self):
        """HumanMessage and AIMessage should pass through unchanged."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        large_output = _make_large_tool_output(200)
        messages = _make_messages_with_tool_output(large_output)

        result = compress_tool_messages(messages)

        assert isinstance(result.messages[0], HumanMessage)
        assert result.messages[0].content == "Get the data"
        assert isinstance(result.messages[1], AIMessage)
        tool_call = result.messages[1].tool_calls[0]
        assert tool_call["id"] == "call_1"
        assert tool_call["name"] == "search"
        assert tool_call["args"] == {}

    def test_preserves_tool_call_id(self):
        """Compressed ToolMessages must keep their tool_call_id."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        large_output = _make_large_tool_output(200)
        messages = _make_messages_with_tool_output(large_output, tool_call_id="call_abc123")

        result = compress_tool_messages(messages)

        tool_msg = result.messages[2]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.tool_call_id == "call_abc123"

    def test_preserves_error_content_by_default(self):
        """ToolMessages with error indicators should be skipped by default."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        # Large content but contains error indicator
        error_output = json.dumps(
            {
                "error": "Database connection failed",
                "details": "x" * 2000,
            }
        )
        messages = _make_messages_with_tool_output(error_output)

        result = compress_tool_messages(messages)

        # Should be unchanged — error preserved
        assert result.messages[2].content == error_output
        assert result.metrics[0].skip_reason == "error_content_preserved"

    def test_compresses_error_content_when_disabled(self):
        """Error content should be compressed when preserve_errors=False."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        error_output = json.dumps(
            {
                "error": "fail",
                "data": [{"id": i} for i in range(200)],
            }
        )
        messages = _make_messages_with_tool_output(error_output)

        result = compress_tool_messages(messages, preserve_errors=False)

        # Should have attempted compression (no error_content_preserved skip)
        assert result.metrics[0].skip_reason != "error_content_preserved"

    def test_handles_empty_messages(self):
        """Empty message list should return empty result."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        result = compress_tool_messages([])

        assert result.messages == []
        assert result.metrics == []
        assert result.total_tokens_saved == 0

    def test_handles_no_tool_messages(self):
        """Message list with no ToolMessages should pass through."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        result = compress_tool_messages(messages)

        assert len(result.messages) == 2
        assert result.messages[0].content == "Hello"
        assert result.messages[1].content == "Hi there!"
        assert result.metrics == []

    def test_multiple_tool_messages(self):
        """Should compress multiple ToolMessages independently."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        large_output_1 = _make_large_tool_output(200)
        large_output_2 = _make_large_tool_output(150)

        messages = [
            HumanMessage(content="Get all data"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "search", "args": {}},
                    {"id": "call_2", "name": "database", "args": {}},
                ],
            ),
            ToolMessage(content=large_output_1, tool_call_id="call_1"),
            ToolMessage(content=large_output_2, tool_call_id="call_2"),
        ]

        result = compress_tool_messages(messages)

        assert len(result.messages) == 4
        # Both tool messages should have their correct tool_call_ids
        assert result.messages[2].tool_call_id == "call_1"
        assert result.messages[3].tool_call_id == "call_2"

    def test_min_tokens_to_compress_config(self):
        """Custom min_tokens_to_compress should be respected."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        # Content that's ~100 tokens (400 chars) — below a 200 token threshold
        medium_output = json.dumps({"data": "x" * 400})
        messages = _make_messages_with_tool_output(medium_output)

        result = compress_tool_messages(messages, min_tokens_to_compress=200)

        # Should be skipped due to being below threshold
        assert result.metrics[0].was_compressed is False
        assert "below_threshold" in (result.metrics[0].skip_reason or "")


class TestCompressToolMessagesResult:
    """Tests for CompressToolMessagesResult properties."""

    def test_total_tokens_saved(self):
        """total_tokens_saved should sum across compressed metrics."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        large_output = _make_large_tool_output(200)
        messages = _make_messages_with_tool_output(large_output)

        result = compress_tool_messages(messages)

        assert result.total_tokens_saved >= 0
        # If compression happened, tokens_saved should be positive
        if result.messages_compressed > 0:
            assert result.total_tokens_saved > 0

    def test_messages_compressed_count(self):
        """messages_compressed should count actually compressed messages."""
        from headroom.integrations.langchain.langgraph import compress_tool_messages

        messages = [
            HumanMessage(content="test"),
            ToolMessage(content='{"small": true}', tool_call_id="call_1"),
        ]

        result = compress_tool_messages(messages)

        assert result.messages_compressed == 0


class TestCompressToolMessagesConfig:
    """Tests for CompressToolMessagesConfig."""

    def test_config_object(self):
        """Config object should override kwargs."""
        from headroom.integrations.langchain.langgraph import (
            CompressToolMessagesConfig,
            compress_tool_messages,
        )

        config = CompressToolMessagesConfig(
            min_tokens_to_compress=500,
            preserve_errors=False,
        )

        medium_output = json.dumps({"data": "x" * 800})
        messages = _make_messages_with_tool_output(medium_output)

        result = compress_tool_messages(messages, config=config)

        # ~200 tokens, below the 500 threshold
        assert result.metrics[0].was_compressed is False

    def test_default_config(self):
        """Default config should have sensible defaults."""
        from headroom.integrations.langchain.langgraph import CompressToolMessagesConfig

        config = CompressToolMessagesConfig()
        assert config.min_tokens_to_compress == 100
        assert config.preserve_errors is True


class TestCreateCompressToolMessagesNode:
    """Tests for the LangGraph node factory."""

    def test_returns_callable(self):
        """Factory should return a callable node function."""
        from headroom.integrations.langchain.langgraph import create_compress_tool_messages_node

        node = create_compress_tool_messages_node()
        assert callable(node)

    def test_node_reads_messages_from_state(self):
        """Node should read messages from state dict and return updated state."""
        from headroom.integrations.langchain.langgraph import create_compress_tool_messages_node

        large_output = _make_large_tool_output(200)
        state = {
            "messages": _make_messages_with_tool_output(large_output),
        }

        node = create_compress_tool_messages_node()
        result_state = node(state)

        assert "messages" in result_state
        assert len(result_state["messages"]) == 3
        # ToolMessage should be compressed
        assert len(result_state["messages"][2].content) < len(large_output)

    def test_node_preserves_tool_call_id(self):
        """Node should preserve tool_call_id on compressed messages."""
        from headroom.integrations.langchain.langgraph import create_compress_tool_messages_node

        large_output = _make_large_tool_output(200)
        state = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="", tool_calls=[{"id": "call_xyz", "name": "db", "args": {}}]),
                ToolMessage(content=large_output, tool_call_id="call_xyz"),
            ],
        }

        node = create_compress_tool_messages_node()
        result_state = node(state)

        assert result_state["messages"][2].tool_call_id == "call_xyz"

    def test_node_handles_empty_state(self):
        """Node should handle empty messages gracefully."""
        from headroom.integrations.langchain.langgraph import create_compress_tool_messages_node

        node = create_compress_tool_messages_node()
        result_state = node({"messages": []})

        assert result_state == {"messages": []}

    def test_node_handles_missing_messages_key(self):
        """Node should handle state without messages key."""
        from headroom.integrations.langchain.langgraph import create_compress_tool_messages_node

        node = create_compress_tool_messages_node()
        result_state = node({})

        assert "messages" not in result_state or result_state.get("messages") == []

    def test_node_with_custom_config(self):
        """Node should respect custom configuration."""
        from headroom.integrations.langchain.langgraph import create_compress_tool_messages_node

        node = create_compress_tool_messages_node(min_tokens_to_compress=10000)

        large_output = _make_large_tool_output(200)
        state = {"messages": _make_messages_with_tool_output(large_output)}

        result_state = node(state)

        # With very high threshold, nothing should be compressed
        assert result_state["messages"][2].content == large_output

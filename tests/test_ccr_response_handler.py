"""Tests for CCR response handler.

These tests verify that:
1. CCR tool calls are correctly detected in responses
2. Retrieval execution works for both full and search modes
3. Continuation flow handles multiple rounds
4. Provider-specific formats are handled correctly
5. Streaming buffer detection works
"""

import json

import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.response_handler import (
    CCRResponseHandler,
    CCRToolCall,
    CCRToolResult,
    ResponseHandlerConfig,
    StreamingCCRBuffer,
)
from headroom.ccr.tool_injection import CCR_TOOL_NAME


class TestCCRToolCallDetection:
    """Test detection of CCR tool calls in responses."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_compression_store()
        yield
        reset_compression_store()

    def test_detect_anthropic_ccr_tool_call(self):
        """Detect CCR tool call in Anthropic format."""
        handler = CCRResponseHandler()

        response = {
            "content": [
                {"type": "text", "text": "Let me retrieve that data."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": "abc123"},
                },
            ]
        }

        assert handler.has_ccr_tool_calls(response, "anthropic")

    def test_detect_openai_ccr_tool_call(self):
        """Detect CCR tool call in OpenAI format."""
        handler = CCRResponseHandler()

        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me retrieve that data.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": CCR_TOOL_NAME,
                                    "arguments": '{"hash": "abc123"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        assert handler.has_ccr_tool_calls(response, "openai")

    def test_no_ccr_tool_call_anthropic(self):
        """No false positive when no CCR tool call present."""
        handler = CCRResponseHandler()

        response = {
            "content": [
                {"type": "text", "text": "Here is the data."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "some_other_tool",
                    "input": {"param": "value"},
                },
            ]
        }

        assert not handler.has_ccr_tool_calls(response, "anthropic")

    def test_no_ccr_tool_call_openai(self):
        """No false positive when no CCR tool call present in OpenAI format."""
        handler = CCRResponseHandler()

        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Here is the data.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "other_tool",
                                    "arguments": '{"param": "value"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        assert not handler.has_ccr_tool_calls(response, "openai")

    def test_text_only_response(self):
        """No false positive for text-only responses."""
        handler = CCRResponseHandler()

        response = {"content": [{"type": "text", "text": "Just plain text."}]}

        assert not handler.has_ccr_tool_calls(response, "anthropic")

    def test_empty_response(self):
        """Handle empty response gracefully."""
        handler = CCRResponseHandler()

        assert not handler.has_ccr_tool_calls({}, "anthropic")
        assert not handler.has_ccr_tool_calls({"content": []}, "anthropic")


class TestCCRToolCallParsing:
    """Test parsing of CCR tool calls."""

    def test_parse_anthropic_full_retrieval(self):
        """Parse full retrieval call from Anthropic format."""
        handler = CCRResponseHandler()

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": "abc123def456abc123def456"},
                }
            ]
        }

        ccr_calls, other_calls = handler._parse_ccr_tool_calls(response, "anthropic")

        assert len(ccr_calls) == 1
        assert ccr_calls[0].tool_call_id == "tool_123"
        assert ccr_calls[0].hash_key == "abc123def456abc123def456"
        assert not hasattr(ccr_calls[0], "query")
        assert len(other_calls) == 0

    def test_parse_anthropic_retrieval_ignores_query(self):
        """Retrieval parses the hash; any legacy ``query`` input is ignored."""
        handler = CCRResponseHandler()

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_456",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": "def456abc123def456abc123", "query": "authentication error"},
                }
            ]
        }

        ccr_calls, other_calls = handler._parse_ccr_tool_calls(response, "anthropic")

        assert len(ccr_calls) == 1
        assert ccr_calls[0].hash_key == "def456abc123def456abc123"
        assert not hasattr(ccr_calls[0], "query")

    def test_parse_mixed_tool_calls(self):
        """Parse response with both CCR and other tool calls."""
        handler = CCRResponseHandler()

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": "abc123def456abc123def456"},
                },
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "read_file",
                    "input": {"path": "/etc/config"},
                },
            ]
        }

        ccr_calls, other_calls = handler._parse_ccr_tool_calls(response, "anthropic")

        assert len(ccr_calls) == 1
        assert len(other_calls) == 1
        assert other_calls[0]["name"] == "read_file"


class TestCCRRetrievalExecution:
    """Test CCR retrieval execution."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_compression_store()
        yield
        reset_compression_store()

    def test_full_retrieval_success(self):
        """Successfully retrieve full content."""
        store = get_compression_store()
        original = json.dumps([{"id": i} for i in range(100)])
        compressed = json.dumps([{"id": i} for i in range(10)])

        hash_key = store.store(
            original=original,
            compressed=compressed,
            original_item_count=100,
            compressed_item_count=10,
        )

        handler = CCRResponseHandler()
        call = CCRToolCall(tool_call_id="test_id", hash_key=hash_key)

        result = handler._execute_retrieval(call)

        assert result.success
        assert result.items_retrieved == 100

        # Check content structure
        content = json.loads(result.content)
        assert content["hash"] == hash_key
        assert "original_content" in content

    def test_retrieval_returns_full_content_for_cached_hash(self):
        """Retrieval always returns the full original content (never empty)."""
        store = get_compression_store()
        items = [
            {"id": 1, "text": "Python programming language tutorial"},
            {"id": 2, "text": "JavaScript web development framework"},
            {"id": 3, "text": "Python data science machine learning"},
            {"id": 4, "text": "Ruby programming language basics"},
            {"id": 5, "text": "Python web framework django flask"},
        ]
        original = json.dumps(items)
        compressed = json.dumps(items[:1])

        hash_key = store.store(
            original=original,
            compressed=compressed,
            original_item_count=5,
            compressed_item_count=1,
        )

        handler = CCRResponseHandler()
        call = CCRToolCall(tool_call_id="test_id", hash_key=hash_key)

        result = handler._execute_retrieval(call)

        assert result.success
        assert result.items_retrieved == 5

        content = json.loads(result.content)
        assert content["hash"] == hash_key
        # Full content is always returned — the complete original round-trips.
        assert json.loads(content["original_content"]) == items

    def test_retrieval_nonexistent_hash(self):
        """Handle retrieval of nonexistent hash."""
        handler = CCRResponseHandler()
        call = CCRToolCall(tool_call_id="test_id", hash_key="nonexistent123")

        result = handler._execute_retrieval(call)

        assert not result.success
        assert result.items_retrieved == 0

        content = json.loads(result.content)
        assert "error" in content


class TestCCRToolResultMessage:
    """Test tool result message creation."""

    def test_anthropic_tool_result_format(self):
        """Create tool result message in Anthropic format."""
        handler = CCRResponseHandler()
        results = [
            CCRToolResult(
                tool_call_id="tool_123",
                content='{"data": "retrieved"}',
                success=True,
                items_retrieved=10,
            )
        ]

        message = handler._create_tool_result_message(results, "anthropic")

        assert message["role"] == "user"
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "tool_result"
        assert message["content"][0]["tool_use_id"] == "tool_123"

    def test_openai_tool_result_format(self):
        """Create tool result messages in OpenAI format."""
        handler = CCRResponseHandler()
        results = [
            CCRToolResult(
                tool_call_id="call_123",
                content='{"data": "retrieved"}',
                success=True,
            ),
            CCRToolResult(
                tool_call_id="call_456",
                content='{"data": "more data"}',
                success=True,
            ),
        ]

        message = handler._create_tool_result_message(results, "openai")

        assert "_openai_tool_results" in message
        assert len(message["_openai_tool_results"]) == 2
        assert message["_openai_tool_results"][0]["role"] == "tool"


class TestCCRResponseHandling:
    """Test the full response handling flow."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_compression_store()
        yield
        reset_compression_store()

    @pytest.mark.asyncio
    async def test_handle_response_no_ccr(self):
        """Handle response with no CCR calls (pass-through)."""
        handler = CCRResponseHandler()
        response = {"content": [{"type": "text", "text": "Just text."}]}

        async def mock_api_call(messages, tools):
            return {"content": [{"type": "text", "text": "Response"}]}

        result = await handler.handle_response(response, [], None, mock_api_call, "anthropic")

        # Should return original response unchanged
        assert result == response

    @pytest.mark.asyncio
    async def test_handle_response_with_ccr(self):
        """Handle response containing CCR tool call."""
        store = get_compression_store()
        original = json.dumps([{"id": i} for i in range(50)])
        hash_key = store.store(
            original=original,
            compressed="[]",
            original_item_count=50,
        )

        handler = CCRResponseHandler()

        # Initial response with CCR tool call
        initial_response = {
            "content": [
                {"type": "text", "text": "Let me get that data."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": hash_key},
                },
            ]
        }

        # Final response after tool result
        final_response = {"content": [{"type": "text", "text": "Here is all 50 items of data."}]}

        call_count = 0

        async def mock_api_call(messages, tools):
            nonlocal call_count
            call_count += 1
            return final_response

        result = await handler.handle_response(
            initial_response,
            [{"role": "user", "content": "Get me the data"}],
            None,
            mock_api_call,
            "anthropic",
        )

        # Should have made continuation call
        assert call_count == 1
        # Should return final response
        assert result == final_response

    @pytest.mark.asyncio
    async def test_handle_response_max_rounds(self):
        """Respects max retrieval rounds limit."""
        store = get_compression_store()
        hash_key = store.store(original="[1,2,3]", compressed="[]")

        config = ResponseHandlerConfig(max_retrieval_rounds=2)
        handler = CCRResponseHandler(config)

        # Response that always has CCR tool call (simulating infinite loop)
        ccr_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": hash_key},
                }
            ]
        }

        call_count = 0

        async def mock_api_call(messages, tools):
            nonlocal call_count
            call_count += 1
            return ccr_response

        await handler.handle_response(ccr_response, [], None, mock_api_call, "anthropic")

        # Should stop after max rounds
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_handle_response_disabled(self):
        """Disabled handler returns response unchanged."""
        config = ResponseHandlerConfig(enabled=False)
        handler = CCRResponseHandler(config)

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": "abc123"},
                }
            ]
        }

        async def mock_api_call(messages, tools):
            raise AssertionError("Should not be called")

        result = await handler.handle_response(response, [], None, mock_api_call, "anthropic")

        assert result == response

    @pytest.mark.asyncio
    async def test_handle_response_mixed_tools_skips_ccr(self):
        """When CCR and non-CCR tools are called together, skip CCR.

        Building a valid continuation is impossible without results for the
        non-CCR tools (Anthropic requires every tool_use to have a
        tool_result). Skipping CCR avoids a wasted 400 API call and returns
        the original response immediately so the client can resolve all
        tool calls itself.
        """
        store = get_compression_store()
        hash_key = store.store(original="[1,2,3]", compressed="[]")

        handler = CCRResponseHandler()

        mixed_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "ccr_call",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": hash_key},
                },
                {
                    "type": "tool_use",
                    "id": "user_call",
                    "name": "read_file",
                    "input": {"path": "/etc/config"},
                },
            ]
        }

        api_call_count = 0

        async def mock_api_call(messages, tools):
            nonlocal api_call_count
            api_call_count += 1
            return {"content": [{"type": "text", "text": "continuation"}]}

        result = await handler.handle_response(mixed_response, [], None, mock_api_call, "anthropic")

        # CCR skipped — no continuation call made (avoids the 400 API round-trip)
        assert api_call_count == 0, "should not attempt continuation with mixed tools"
        # Original response returned unchanged so client can handle all tool calls
        assert result is mixed_response


class TestCCRResponseHandlerStats:
    """Test handler statistics."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_compression_store()
        yield
        reset_compression_store()

    @pytest.mark.asyncio
    async def test_retrieval_count_tracking(self):
        """Track total retrieval count."""
        store = get_compression_store()
        hash_key = store.store(original="[1,2,3]", compressed="[]")

        handler = CCRResponseHandler()

        initial_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": CCR_TOOL_NAME,
                    "input": {"hash": hash_key},
                }
            ]
        }

        final_response = {"content": [{"type": "text", "text": "Done"}]}

        async def mock_api_call(messages, tools):
            return final_response

        await handler.handle_response(initial_response, [], None, mock_api_call, "anthropic")

        stats = handler.get_stats()
        assert stats["total_retrievals"] == 1


class TestStreamingCCRBuffer:
    """Test streaming buffer for CCR detection."""

    def test_buffer_accumulation(self):
        """Buffer accumulates chunks."""
        buffer = StreamingCCRBuffer()

        buffer.add_chunk(b"part1")
        buffer.add_chunk(b"part2")
        buffer.add_chunk(b"part3")

        assert buffer.get_accumulated() == b"part1part2part3"

    def test_detect_ccr_tool_in_stream(self):
        """Detect CCR tool call in streaming chunks."""
        buffer = StreamingCCRBuffer()

        # Simulate streaming response with tool_use
        chunk1 = b'{"type":"content_block_start","content_block":{"type":"tool_use"'
        chunk2 = f',"name":"{CCR_TOOL_NAME}"'.encode()

        detected = buffer.add_chunk(chunk1)
        assert not detected  # Not complete yet

        detected = buffer.add_chunk(chunk2)
        assert detected  # Now detected

        assert buffer.detected_ccr

    def test_no_false_positive_detection(self):
        """No false positive for non-CCR tool calls."""
        buffer = StreamingCCRBuffer()

        chunk = b'{"type":"content_block_start","content_block":{"type":"tool_use","name":"other_tool"}}'

        detected = buffer.add_chunk(chunk)
        assert not detected
        assert not buffer.detected_ccr

    def test_buffer_clear(self):
        """Buffer clears state correctly."""
        buffer = StreamingCCRBuffer()
        buffer.add_chunk(b"data")
        buffer.detected_ccr = True

        buffer.clear()

        assert buffer.get_accumulated() == b""
        assert not buffer.detected_ccr


class TestResponseHandlerConfig:
    """Test response handler configuration."""

    def test_default_config(self):
        """Default config values."""
        config = ResponseHandlerConfig()

        assert config.enabled is True
        assert config.max_retrieval_rounds == 3
        assert config.strip_ccr_from_response is True
        assert config.continuation_timeout_ms == 120000

    def test_custom_config(self):
        """Custom config values."""
        config = ResponseHandlerConfig(
            enabled=False,
            max_retrieval_rounds=5,
        )

        assert config.enabled is False
        assert config.max_retrieval_rounds == 5


class TestCCRToolCallDataClass:
    """Test CCRToolCall dataclass."""

    def test_full_retrieval_call(self):
        """Create full retrieval call."""
        call = CCRToolCall(
            tool_call_id="test_123",
            hash_key="abc123",
        )

        assert call.tool_call_id == "test_123"
        assert call.hash_key == "abc123"
        assert not hasattr(call, "query")


class TestCCRToolResultDataClass:
    """Test CCRToolResult dataclass."""

    def test_successful_result(self):
        """Create successful result."""
        result = CCRToolResult(
            tool_call_id="test_123",
            content='{"data": "content"}',
            success=True,
            items_retrieved=50,
        )

        assert result.success
        assert result.items_retrieved == 50
        assert not hasattr(result, "was_search")

    def test_failed_result(self):
        """Create failed result."""
        result = CCRToolResult(
            tool_call_id="test_789",
            content='{"error": "not found"}',
            success=False,
        )

        assert not result.success
        assert result.items_retrieved == 0


class TestExtractAssistantMessage:
    """Test extraction of assistant messages from responses."""

    def test_extract_anthropic_message(self):
        """Extract assistant message from Anthropic response."""
        handler = CCRResponseHandler()

        response = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "id": "123", "name": "test", "input": {}},
            ]
        }

        message = handler._extract_assistant_message(response, "anthropic")

        assert message["role"] == "assistant"
        assert message["content"] == response["content"]

    def test_extract_openai_message(self):
        """Extract assistant message from OpenAI response."""
        handler = CCRResponseHandler()

        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "tool_calls": [{"id": "123"}],
                    }
                }
            ]
        }

        message = handler._extract_assistant_message(response, "openai")

        assert message["role"] == "assistant"
        assert message["content"] == "Hello"
        assert message["tool_calls"] == [{"id": "123"}]


class TestExtractAssistantMessageEdgeCases:
    """Regression: `_extract_assistant_message` must not crash on an empty or
    malformed OpenAI `choices` array (OpenAI-compatible gateways can send
    `choices: []` or `[null]` on content-filtered / usage-only responses)."""

    def test_openai_empty_choices_does_not_crash(self):
        handler = CCRResponseHandler()
        msg = handler._extract_assistant_message({"choices": []}, "openai")
        assert msg == {"role": "assistant", "content": None, "tool_calls": None}

    def test_openai_null_first_choice_does_not_crash(self):
        handler = CCRResponseHandler()
        msg = handler._extract_assistant_message({"choices": [None]}, "openai")
        assert msg == {"role": "assistant", "content": None, "tool_calls": None}

    def test_openai_absent_choices_does_not_crash(self):
        handler = CCRResponseHandler()
        msg = handler._extract_assistant_message({}, "openai")
        assert msg == {"role": "assistant", "content": None, "tool_calls": None}

    def test_openai_normal_choice_still_extracts(self):
        handler = CCRResponseHandler()
        resp = {"choices": [{"message": {"content": "hi", "tool_calls": [{"id": "1"}]}}]}
        msg = handler._extract_assistant_message(resp, "openai")
        assert msg == {"role": "assistant", "content": "hi", "tool_calls": [{"id": "1"}]}

"""Tests for CCR batch result processor.

These tests verify that:
1. BatchResultProcessor class initialization works correctly
2. Result parsing for Anthropic, OpenAI, and Google batch formats
3. CCR tool call detection in batch results
4. Continuation call handling works for all providers
5. Error cases and edge cases are handled gracefully
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from headroom.ccr.batch_processor import (
    BatchResultProcessor,
    BatchResultProcessorConfig,
    ProcessedBatchResult,
    process_batch_results,
)
from headroom.ccr.batch_store import (
    BatchContext,
    BatchContextStore,
    BatchRequestContext,
    reset_batch_context_store,
)
from headroom.ccr.tool_injection import CCR_TOOL_NAME


class TestBatchResultProcessorConfig:
    """Test BatchResultProcessorConfig dataclass."""

    def test_default_config(self):
        """Default config values."""
        config = BatchResultProcessorConfig()

        assert config.enabled is True
        assert config.continuation_timeout == 120
        assert config.max_continuation_rounds == 3

    def test_custom_config(self):
        """Custom config values."""
        config = BatchResultProcessorConfig(
            enabled=False,
            continuation_timeout=60,
            max_continuation_rounds=5,
        )

        assert config.enabled is False
        assert config.continuation_timeout == 60
        assert config.max_continuation_rounds == 5


class TestProcessedBatchResult:
    """Test ProcessedBatchResult dataclass."""

    def test_default_values(self):
        """Default values for ProcessedBatchResult."""
        result = ProcessedBatchResult(
            custom_id="req_123",
            result={"content": "test"},
        )

        assert result.custom_id == "req_123"
        assert result.result == {"content": "test"}
        assert result.was_processed is False
        assert result.continuation_rounds == 0
        assert result.error is None

    def test_processed_result(self):
        """ProcessedBatchResult with CCR processing."""
        result = ProcessedBatchResult(
            custom_id="req_456",
            result={"content": "processed"},
            was_processed=True,
            continuation_rounds=2,
        )

        assert result.was_processed is True
        assert result.continuation_rounds == 2

    def test_error_result(self):
        """ProcessedBatchResult with error."""
        result = ProcessedBatchResult(
            custom_id="req_789",
            result={"content": "partial"},
            error="Retrieval failed",
        )

        assert result.error == "Retrieval failed"


class TestBatchResultProcessorInit:
    """Test BatchResultProcessor initialization."""

    def test_default_initialization(self):
        """Initialize with default config."""
        http_client = MagicMock(spec=httpx.AsyncClient)
        processor = BatchResultProcessor(http_client)

        assert processor.http_client == http_client
        assert processor.config.enabled is True
        assert processor.config.continuation_timeout == 120
        assert processor.ccr_handler is not None

    def test_custom_config_initialization(self):
        """Initialize with custom config."""
        http_client = MagicMock(spec=httpx.AsyncClient)
        config = BatchResultProcessorConfig(
            enabled=False,
            continuation_timeout=60,
        )
        processor = BatchResultProcessor(http_client, config)

        assert processor.config.enabled is False
        assert processor.config.continuation_timeout == 60

    def test_api_urls_set(self):
        """API URLs are set for all providers."""
        http_client = MagicMock(spec=httpx.AsyncClient)
        processor = BatchResultProcessor(http_client)

        assert "anthropic" in processor.api_urls
        assert "openai" in processor.api_urls
        assert "google" in processor.api_urls
        assert processor.api_urls["anthropic"] == "https://api.anthropic.com"
        assert processor.api_urls["openai"] == "https://api.openai.com"
        assert processor.api_urls["google"] == "https://generativelanguage.googleapis.com"


class TestCustomIdExtraction:
    """Test _get_custom_id method for different providers."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        http_client = MagicMock(spec=httpx.AsyncClient)
        return BatchResultProcessor(http_client)

    def test_anthropic_custom_id(self, processor):
        """Extract custom_id from Anthropic batch result."""
        result = {"custom_id": "anthropic_req_123", "result": {"message": {}}}
        custom_id = processor._get_custom_id(result, "anthropic")
        assert custom_id == "anthropic_req_123"

    def test_openai_custom_id(self, processor):
        """Extract custom_id from OpenAI batch result."""
        result = {"custom_id": "openai_req_456", "response": {"body": {}}}
        custom_id = processor._get_custom_id(result, "openai")
        assert custom_id == "openai_req_456"

    def test_google_custom_id(self, processor):
        """Extract custom_id from Google batch result (metadata.key)."""
        result = {"metadata": {"key": "google_req_789"}, "response": {}}
        custom_id = processor._get_custom_id(result, "google")
        assert custom_id == "google_req_789"

    def test_google_missing_metadata(self, processor):
        """Handle missing metadata in Google result."""
        result = {"response": {}}
        custom_id = processor._get_custom_id(result, "google")
        assert custom_id == ""

    def test_unknown_provider_fallback(self, processor):
        """Fallback extraction for unknown provider."""
        result = {"custom_id": "unknown_req", "id": "backup_id"}
        custom_id = processor._get_custom_id(result, "unknown")
        assert custom_id == "unknown_req"

    def test_unknown_provider_uses_id_fallback(self, processor):
        """Unknown provider falls back to 'id' field."""
        result = {"id": "id_field_value"}
        custom_id = processor._get_custom_id(result, "unknown")
        assert custom_id == "id_field_value"


class TestResponseExtraction:
    """Test _extract_response method for different providers."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        http_client = MagicMock(spec=httpx.AsyncClient)
        return BatchResultProcessor(http_client)

    def test_anthropic_response_extraction(self, processor):
        """Extract response from Anthropic batch result."""
        result = {
            "custom_id": "req_1",
            "result": {
                "type": "message",
                "message": {
                    "content": [{"type": "text", "text": "Hello"}],
                    "stop_reason": "end_turn",
                },
            },
        }
        response = processor._extract_response(result, "anthropic")
        assert response is not None
        assert response["content"] == [{"type": "text", "text": "Hello"}]

    def test_openai_response_extraction(self, processor):
        """Extract response from OpenAI batch result."""
        result = {
            "custom_id": "req_2",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": "Hello"}}],
                },
            },
        }
        response = processor._extract_response(result, "openai")
        assert response is not None
        assert response["choices"][0]["message"]["content"] == "Hello"

    def test_google_response_extraction(self, processor):
        """Extract response from Google batch result."""
        result = {
            "metadata": {"key": "req_3"},
            "response": {
                "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
            },
        }
        response = processor._extract_response(result, "google")
        assert response is not None
        assert response["candidates"][0]["content"]["parts"][0]["text"] == "Hello"

    def test_anthropic_missing_result(self, processor):
        """Handle missing result in Anthropic format."""
        result = {"custom_id": "req_4"}
        response = processor._extract_response(result, "anthropic")
        assert response is None

    def test_openai_missing_body(self, processor):
        """Handle missing body in OpenAI format."""
        result = {"custom_id": "req_5", "response": {"status_code": 500}}
        response = processor._extract_response(result, "openai")
        assert response is None

    def test_invalid_response_type(self, processor):
        """Handle non-dict response."""
        result = {"custom_id": "req_6", "result": {"message": "not_a_dict"}}
        response = processor._extract_response(result, "anthropic")
        assert response is None


class TestCCRToolCallDetectionInBatch:
    """Test CCR tool call detection within batch results."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        http_client = MagicMock(spec=httpx.AsyncClient)
        return BatchResultProcessor(http_client)

    def test_detect_anthropic_ccr_in_batch(self, processor):
        """Detect CCR tool call in Anthropic batch result."""
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
        assert processor.ccr_handler.has_ccr_tool_calls(response, "anthropic")

    def test_detect_openai_ccr_in_batch(self, processor):
        """Detect CCR tool call in OpenAI batch result."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Retrieving data...",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": CCR_TOOL_NAME,
                                    "arguments": '{"hash": "def456"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        assert processor.ccr_handler.has_ccr_tool_calls(response, "openai")

    def test_detect_google_ccr_in_batch(self, processor):
        """Detect CCR tool call in Google batch result."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Retrieving data..."},
                            {
                                "functionCall": {
                                    "name": CCR_TOOL_NAME,
                                    "args": {"hash": "ghi789"},
                                }
                            },
                        ]
                    }
                }
            ]
        }
        assert processor.ccr_handler.has_ccr_tool_calls(response, "google")

    def test_no_ccr_in_text_only_response(self, processor):
        """No false positive for text-only response."""
        response = {"content": [{"type": "text", "text": "Just a text response."}]}
        assert not processor.ccr_handler.has_ccr_tool_calls(response, "anthropic")

    def test_no_ccr_for_other_tools(self, processor):
        """No false positive for other tool calls."""
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_xyz",
                    "name": "read_file",
                    "input": {"path": "/etc/config"},
                }
            ]
        }
        assert not processor.ccr_handler.has_ccr_tool_calls(response, "anthropic")


class TestResultUpdate:
    """Test _update_result method for different providers."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        http_client = MagicMock(spec=httpx.AsyncClient)
        return BatchResultProcessor(http_client)

    def test_update_anthropic_result(self, processor):
        """Update Anthropic batch result with final response."""
        original = {
            "custom_id": "req_1",
            "result": {
                "type": "tool_use",
                "message": {"content": [{"type": "tool_use", "name": CCR_TOOL_NAME}]},
            },
        }
        final_response = {
            "content": [{"type": "text", "text": "Final answer"}],
            "stop_reason": "end_turn",
        }

        updated = processor._update_result(original, final_response, "anthropic")

        assert updated["result"]["message"] == final_response
        assert updated["result"]["type"] == "succeeded"
        assert updated["custom_id"] == "req_1"

    def test_update_openai_result(self, processor):
        """Update OpenAI batch result with final response."""
        original = {
            "custom_id": "req_2",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {"message": {"tool_calls": [{"function": {"name": CCR_TOOL_NAME}}]}}
                    ]
                },
            },
        }
        final_response = {"choices": [{"message": {"content": "Final answer"}}]}

        updated = processor._update_result(original, final_response, "openai")

        assert updated["response"]["body"] == final_response
        assert updated["custom_id"] == "req_2"

    def test_update_google_result(self, processor):
        """Update Google batch result with final response."""
        original = {
            "metadata": {"key": "req_3"},
            "response": {
                "candidates": [{"content": {"parts": [{"functionCall": {"name": CCR_TOOL_NAME}}]}}]
            },
        }
        final_response = {"candidates": [{"content": {"parts": [{"text": "Final answer"}]}}]}

        updated = processor._update_result(original, final_response, "google")

        assert updated["response"] == final_response

    def test_update_creates_missing_containers(self, processor):
        """Update creates missing result/response containers."""
        original_anthropic = {"custom_id": "req_1"}
        original_openai = {"custom_id": "req_2"}
        final = {"content": "test"}

        updated_anthropic = processor._update_result(original_anthropic, final, "anthropic")
        updated_openai = processor._update_result(original_openai, final, "openai")

        assert "result" in updated_anthropic
        assert "response" in updated_openai


class TestMessagesToGoogleContents:
    """Test _messages_to_google_contents conversion."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        http_client = MagicMock(spec=httpx.AsyncClient)
        return BatchResultProcessor(http_client)

    def test_convert_simple_text_message(self, processor):
        """Convert simple text messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        contents = processor._messages_to_google_contents(messages)

        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "Hello"}]
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "Hi there"}]

    def test_skip_system_messages(self, processor):
        """System messages are skipped (handled separately in Google)."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        contents = processor._messages_to_google_contents(messages)

        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_convert_tool_result_content(self, processor):
        """Convert structured content with tool results."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool_123", "content": "Result data"}
                ],
            }
        ]

        contents = processor._messages_to_google_contents(messages)

        assert len(contents) == 1
        assert contents[0]["parts"][0]["functionResponse"]["response"]["content"] == "Result data"

    def test_convert_tool_use_content(self, processor):
        """Convert content with tool_use blocks."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "read_file", "input": {"path": "/test"}}],
            }
        ]

        contents = processor._messages_to_google_contents(messages)

        assert len(contents) == 1
        assert contents[0]["role"] == "model"
        assert contents[0]["parts"][0]["functionCall"]["name"] == "read_file"

    def test_preserve_google_format_messages(self, processor):
        """Messages already in Google format are preserved."""
        messages = [{"role": "model", "parts": [{"text": "Already Google format"}]}]

        contents = processor._messages_to_google_contents(messages)

        assert len(contents) == 1
        assert contents[0]["parts"] == [{"text": "Already Google format"}]


class TestProcessResults:
    """Test process_results method."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_batch_context_store()
        yield
        reset_batch_context_store()

    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        http_client = AsyncMock(spec=httpx.AsyncClient)
        return BatchResultProcessor(http_client)

    @pytest.mark.asyncio
    async def test_disabled_processor_passthrough(self, processor):
        """Disabled processor passes through results unchanged."""
        processor.config.enabled = False

        results = [{"custom_id": "req_1", "result": {"message": {"content": "test"}}}]

        processed = await processor.process_results("batch_123", results, "anthropic")

        assert len(processed) == 1
        assert processed[0].custom_id == "req_1"
        assert processed[0].was_processed is False

    @pytest.mark.asyncio
    async def test_missing_batch_context_passthrough(self, processor):
        """Missing batch context passes through results unchanged."""
        results = [{"custom_id": "req_1", "result": {"message": {"content": "test"}}}]

        processed = await processor.process_results("nonexistent_batch", results, "anthropic")

        assert len(processed) == 1
        assert processed[0].was_processed is False

    @pytest.mark.asyncio
    async def test_no_ccr_tool_calls_passthrough(self):
        """Results without CCR tool calls pass through."""
        http_client = AsyncMock(spec=httpx.AsyncClient)
        processor = BatchResultProcessor(http_client)

        # Set up batch context
        store = BatchContextStore()
        context = BatchContext(batch_id="batch_123", provider="anthropic")
        context.add_request(
            BatchRequestContext(
                custom_id="req_1",
                messages=[{"role": "user", "content": "Hi"}],
                model="claude-3-opus",
            )
        )
        await store.store(context)

        with patch(
            "headroom.ccr.batch_processor.get_batch_context_store",
            return_value=store,
        ):
            results = [
                {
                    "custom_id": "req_1",
                    "result": {
                        "message": {
                            "content": [{"type": "text", "text": "Hello!"}],
                            "stop_reason": "end_turn",
                        }
                    },
                }
            ]

            processed = await processor.process_results("batch_123", results, "anthropic")

            assert len(processed) == 1
            assert processed[0].was_processed is False

    @pytest.mark.asyncio
    async def test_missing_request_context_passthrough(self):
        """Missing request context for custom_id passes through."""
        http_client = AsyncMock(spec=httpx.AsyncClient)
        processor = BatchResultProcessor(http_client)

        # Set up batch context without matching request
        store = BatchContextStore()
        context = BatchContext(batch_id="batch_123", provider="anthropic")
        # Don't add any requests
        await store.store(context)

        with patch(
            "headroom.ccr.batch_processor.get_batch_context_store",
            return_value=store,
        ):
            results = [
                {
                    "custom_id": "unknown_req",
                    "result": {"message": {"content": [{"type": "text", "text": "Test"}]}},
                }
            ]

            processed = await processor.process_results("batch_123", results, "anthropic")

            assert len(processed) == 1
            assert processed[0].custom_id == "unknown_req"
            assert processed[0].was_processed is False


class TestContinuationCalls:
    """Test continuation API calls for different providers."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_batch_context_store()
        yield
        reset_batch_context_store()

    @pytest.mark.asyncio
    async def test_anthropic_continuation_call(self):
        """Test Anthropic continuation call format."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"type": "text", "text": "Final answer"}]}
        mock_response.raise_for_status = MagicMock()

        http_client = AsyncMock(spec=httpx.AsyncClient)
        http_client.post.return_value = mock_response

        processor = BatchResultProcessor(http_client)

        request_context = BatchRequestContext(
            custom_id="req_1",
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-opus-20240229",
            extras={"max_tokens": 1000},
        )
        batch_context = BatchContext(
            batch_id="batch_123",
            provider="anthropic",
            api_key="test_api_key",
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Tool result here"},
        ]

        await processor._anthropic_continuation(messages, None, request_context, batch_context)

        # Verify API call
        http_client.post.assert_called_once()
        call_args = http_client.post.call_args

        assert "api.anthropic.com" in call_args.args[0]
        assert call_args.kwargs["headers"]["x-api-key"] == "test_api_key"
        assert call_args.kwargs["json"]["model"] == "claude-3-opus-20240229"

    @pytest.mark.asyncio
    async def test_openai_continuation_call(self):
        """Test OpenAI continuation call format."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Final answer"}}]}
        mock_response.raise_for_status = MagicMock()

        http_client = AsyncMock(spec=httpx.AsyncClient)
        http_client.post.return_value = mock_response

        processor = BatchResultProcessor(http_client)

        request_context = BatchRequestContext(
            custom_id="req_1",
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
        )
        batch_context = BatchContext(
            batch_id="batch_123",
            provider="openai",
            api_key="sk-test123",
        )

        await processor._openai_continuation(
            [{"role": "user", "content": "Hello"}],
            None,
            request_context,
            batch_context,
        )

        # Verify API call
        http_client.post.assert_called_once()
        call_args = http_client.post.call_args

        assert "api.openai.com" in call_args.args[0]
        assert "Bearer sk-test123" in call_args.kwargs["headers"]["Authorization"]
        assert call_args.kwargs["json"]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_google_continuation_call(self):
        """Test Google continuation call format."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Final answer"}]}}]
        }
        mock_response.raise_for_status = MagicMock()

        http_client = AsyncMock(spec=httpx.AsyncClient)
        http_client.post.return_value = mock_response

        processor = BatchResultProcessor(http_client)

        request_context = BatchRequestContext(
            custom_id="req_1",
            messages=[{"role": "user", "content": "Hello"}],
            model="gemini-pro",
            system_instruction="Be helpful.",
        )
        batch_context = BatchContext(
            batch_id="batch_123",
            provider="google",
            api_key="google_api_key",
        )

        await processor._google_continuation(
            [{"role": "user", "content": "Hello"}],
            [{"name": "test_tool", "parameters": {}}],
            request_context,
            batch_context,
        )

        # Verify API call
        http_client.post.assert_called_once()
        call_args = http_client.post.call_args

        assert "generativelanguage.googleapis.com" in call_args.args[0]
        assert "gemini-pro" in call_args.args[0]
        assert "key=google_api_key" in call_args.args[0]
        assert "contents" in call_args.kwargs["json"]
        assert "systemInstruction" in call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_continuation_with_tools(self):
        """Test continuation includes tools when present."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"type": "text", "text": "Done"}]}
        mock_response.raise_for_status = MagicMock()

        http_client = AsyncMock(spec=httpx.AsyncClient)
        http_client.post.return_value = mock_response

        processor = BatchResultProcessor(http_client)

        request_context = BatchRequestContext(
            custom_id="req_1",
            messages=[],
            model="claude-3-opus",
            extras={"max_tokens": 1000},
        )
        batch_context = BatchContext(
            batch_id="batch_123",
            provider="anthropic",
            api_key="test_key",
        )

        tools = [
            {"name": "read_file", "input_schema": {"type": "object"}},
            {"name": CCR_TOOL_NAME, "input_schema": {"type": "object"}},
        ]

        await processor._anthropic_continuation(
            [{"role": "user", "content": "test"}],
            tools,
            request_context,
            batch_context,
        )

        call_args = http_client.post.call_args
        assert "tools" in call_args.kwargs["json"]
        assert len(call_args.kwargs["json"]["tools"]) == 2
        sent_names = [tool.get("name") for tool in call_args.kwargs["json"]["tools"]]
        assert sent_names == sorted(sent_names)

    @pytest.mark.asyncio
    async def test_make_continuation_call_unknown_provider(self):
        """Test continuation call raises for unknown provider."""
        http_client = AsyncMock(spec=httpx.AsyncClient)
        processor = BatchResultProcessor(http_client)

        request_context = BatchRequestContext(
            custom_id="req_1",
            messages=[],
            model="unknown-model",
        )
        batch_context = BatchContext(
            batch_id="batch_123",
            provider="unknown",
        )

        with pytest.raises(ValueError, match="Unknown provider"):
            await processor._make_continuation_call(
                [],
                None,
                request_context,
                batch_context,
                "unknown",
            )


class TestProcessSingleResult:
    """Test _process_single_result method."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_batch_context_store()
        yield
        reset_batch_context_store()

    @pytest.mark.asyncio
    async def test_process_single_result_with_ccr(self):
        """Process a single result containing CCR tool call."""
        # Mock the CCR handler to simulate CCR processing
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Final processed answer"}]
        }
        mock_response.raise_for_status = MagicMock()

        http_client = AsyncMock(spec=httpx.AsyncClient)
        http_client.post.return_value = mock_response

        processor = BatchResultProcessor(http_client)

        # Mock the CCR handler's handle_response
        processor.ccr_handler.handle_response = AsyncMock(
            return_value={"content": [{"type": "text", "text": "Final processed answer"}]}
        )

        original_result = {
            "custom_id": "req_1",
            "result": {
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_123",
                            "name": CCR_TOOL_NAME,
                            "input": {"hash": "abc123"},
                        }
                    ]
                }
            },
        }
        response = original_result["result"]["message"]

        request_context = BatchRequestContext(
            custom_id="req_1",
            messages=[{"role": "user", "content": "Get data"}],
            tools=[{"name": CCR_TOOL_NAME}],
            model="claude-3-opus",
        )
        batch_context = BatchContext(
            batch_id="batch_123",
            provider="anthropic",
            api_key="test_key",
        )

        processed = await processor._process_single_result(
            original_result,
            response,
            request_context,
            batch_context,
            "anthropic",
        )

        assert processed.custom_id == "req_1"
        assert processed.was_processed is True


class TestConvenienceFunction:
    """Test process_batch_results convenience function."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_batch_context_store()
        yield
        reset_batch_context_store()

    @pytest.mark.asyncio
    async def test_process_batch_results_function(self):
        """Test the convenience function."""
        http_client = AsyncMock(spec=httpx.AsyncClient)

        results = [
            {
                "custom_id": "req_1",
                "result": {"message": {"content": [{"type": "text", "text": "Hello"}]}},
            }
        ]

        processed = await process_batch_results(
            "batch_123",
            results,
            "anthropic",
            http_client,
        )

        assert len(processed) == 1
        assert processed[0].custom_id == "req_1"


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_batch_context_store()
        yield
        reset_batch_context_store()

    @pytest.mark.asyncio
    async def test_continuation_api_error(self):
        """Handle API errors during continuation."""
        http_client = AsyncMock(spec=httpx.AsyncClient)
        http_client.post.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        processor = BatchResultProcessor(http_client)

        request_context = BatchRequestContext(
            custom_id="req_1",
            messages=[],
            model="claude-3-opus",
            extras={"max_tokens": 1000},
        )
        batch_context = BatchContext(
            batch_id="batch_123",
            provider="anthropic",
            api_key="test_key",
        )

        with pytest.raises(httpx.HTTPStatusError):
            await processor._anthropic_continuation(
                [{"role": "user", "content": "test"}],
                None,
                request_context,
                batch_context,
            )

    @pytest.mark.asyncio
    async def test_processing_error_captured(self):
        """Processing errors are captured in result."""
        http_client = AsyncMock(spec=httpx.AsyncClient)
        processor = BatchResultProcessor(http_client)

        # Set up batch context
        store = BatchContextStore()
        context = BatchContext(batch_id="batch_123", provider="anthropic")
        context.add_request(
            BatchRequestContext(
                custom_id="req_1",
                messages=[{"role": "user", "content": "Hi"}],
                model="claude-3-opus",
            )
        )
        await store.store(context)

        # Mock the handler to raise an error
        processor.ccr_handler.has_ccr_tool_calls = MagicMock(return_value=True)
        processor._process_single_result = AsyncMock(side_effect=Exception("Processing failed"))

        with patch(
            "headroom.ccr.batch_processor.get_batch_context_store",
            return_value=store,
        ):
            results = [
                {
                    "custom_id": "req_1",
                    "result": {
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "tool_1",
                                    "name": CCR_TOOL_NAME,
                                    "input": {"hash": "abc"},
                                }
                            ]
                        }
                    },
                }
            ]

            processed = await processor.process_results("batch_123", results, "anthropic")

            assert len(processed) == 1
            assert processed[0].error == "Processing failed"
            assert processed[0].was_processed is False


class TestMultipleResultsProcessing:
    """Test processing multiple results in a batch."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_batch_context_store()
        yield
        reset_batch_context_store()

    @pytest.mark.asyncio
    async def test_mixed_results_processing(self):
        """Process batch with mix of CCR and non-CCR results."""
        http_client = AsyncMock(spec=httpx.AsyncClient)
        processor = BatchResultProcessor(http_client)

        # Set up batch context with multiple requests
        store = BatchContextStore()
        context = BatchContext(batch_id="batch_123", provider="anthropic")
        context.add_request(
            BatchRequestContext(
                custom_id="req_1",
                messages=[{"role": "user", "content": "Request 1"}],
                model="claude-3-opus",
            )
        )
        context.add_request(
            BatchRequestContext(
                custom_id="req_2",
                messages=[{"role": "user", "content": "Request 2"}],
                model="claude-3-opus",
            )
        )
        context.add_request(
            BatchRequestContext(
                custom_id="req_3",
                messages=[{"role": "user", "content": "Request 3"}],
                model="claude-3-opus",
            )
        )
        await store.store(context)

        with patch(
            "headroom.ccr.batch_processor.get_batch_context_store",
            return_value=store,
        ):
            results = [
                # Non-CCR result
                {
                    "custom_id": "req_1",
                    "result": {
                        "message": {"content": [{"type": "text", "text": "Simple response"}]}
                    },
                },
                # Non-CCR result
                {
                    "custom_id": "req_2",
                    "result": {
                        "message": {"content": [{"type": "text", "text": "Another response"}]}
                    },
                },
                # Non-CCR result (other tool)
                {
                    "custom_id": "req_3",
                    "result": {
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "tool_1",
                                    "name": "read_file",
                                    "input": {"path": "/test"},
                                }
                            ]
                        }
                    },
                },
            ]

            processed = await processor.process_results("batch_123", results, "anthropic")

            assert len(processed) == 3
            # All should not be processed (no CCR tools)
            assert all(not r.was_processed for r in processed)
            assert processed[0].custom_id == "req_1"
            assert processed[1].custom_id == "req_2"
            assert processed[2].custom_id == "req_3"


class TestProviderSpecificFormats:
    """Test provider-specific batch result formats."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_batch_context_store()
        yield
        reset_batch_context_store()

    @pytest.mark.asyncio
    async def test_anthropic_batch_format(self):
        """Test full Anthropic batch result format handling."""
        http_client = AsyncMock(spec=httpx.AsyncClient)
        processor = BatchResultProcessor(http_client)

        # Typical Anthropic batch result format
        results = [
            {
                "custom_id": "my-request-1",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Response text"}],
                        "model": "claude-3-opus-20240229",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 20},
                    },
                },
            }
        ]

        processed = await processor.process_results("batch_1", results, "anthropic")

        assert processed[0].custom_id == "my-request-1"
        assert processed[0].result["result"]["message"]["content"][0]["text"] == "Response text"

    @pytest.mark.asyncio
    async def test_openai_batch_format(self):
        """Test full OpenAI batch result format handling."""
        http_client = AsyncMock(spec=httpx.AsyncClient)
        processor = BatchResultProcessor(http_client)

        # Typical OpenAI batch result format
        results = [
            {
                "id": "batch_req_123",
                "custom_id": "request-1",
                "response": {
                    "status_code": 200,
                    "request_id": "req_abc",
                    "body": {
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "created": 1234567890,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "OpenAI response",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    },
                },
                "error": None,
            }
        ]

        processed = await processor.process_results("batch_1", results, "openai")

        assert processed[0].custom_id == "request-1"

    @pytest.mark.asyncio
    async def test_google_batch_format(self):
        """Test full Google batch result format handling."""
        http_client = AsyncMock(spec=httpx.AsyncClient)
        processor = BatchResultProcessor(http_client)

        # Typical Google batch result format
        results = [
            {
                "metadata": {
                    "key": "request-abc",
                },
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Google response text"}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                            "safetyRatings": [],
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 20,
                    },
                },
            }
        ]

        processed = await processor.process_results("batch_1", results, "google")

        assert processed[0].custom_id == "request-abc"

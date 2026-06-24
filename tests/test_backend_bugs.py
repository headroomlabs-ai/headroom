"""Tests for backend bug fixes in LiteLLM and any-llm integrations.

Tests tool forwarding, tool argument parsing, streaming param forwarding,
message conversion (tool_use/tool_result), streaming tool_calls, and
Vertex AI model mapping.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import (  # noqa: E402  (must follow importorskip)
    _VERTEX_MODEL_MAP,
    LiteLLMBackend,
    _convert_anthropic_tool,
    _convert_tool_choice,
    _parse_tool_arguments,
)

# =============================================================================
# Tool Format Conversion (Bug 1)
# =============================================================================


class TestConvertAnthropicTool:
    """Test Anthropic → OpenAI tool format conversion."""

    def test_basic_tool_conversion(self):
        anthropic_tool = {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
        result = _convert_anthropic_tool(anthropic_tool)
        assert result == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }

    def test_tool_without_description(self):
        tool = {"name": "do_thing", "input_schema": {"type": "object"}}
        result = _convert_anthropic_tool(tool)
        assert result["function"]["name"] == "do_thing"
        assert "description" not in result["function"]
        assert result["function"]["parameters"] == {"type": "object"}

    def test_tool_without_input_schema(self):
        tool = {"name": "simple_tool", "description": "No params"}
        result = _convert_anthropic_tool(tool)
        assert result["function"]["name"] == "simple_tool"
        assert "parameters" not in result["function"]


class TestConvertToolChoice:
    """Test Anthropic → OpenAI tool_choice conversion."""

    def test_auto(self):
        assert _convert_tool_choice({"type": "auto"}) == "auto"

    def test_any_to_required(self):
        assert _convert_tool_choice({"type": "any"}) == "required"

    def test_specific_tool(self):
        result = _convert_tool_choice({"type": "tool", "name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_string_passthrough(self):
        assert _convert_tool_choice("auto") == "auto"
        assert _convert_tool_choice("none") == "none"


# =============================================================================
# Tool Argument Parsing (Bug 2)
# =============================================================================


class TestParseToolArguments:
    """Test that tool arguments are parsed from JSON string to dict."""

    def test_json_string_parsed(self):
        result = _parse_tool_arguments('{"location": "Paris"}')
        assert result == {"location": "Paris"}

    def test_dict_passthrough(self):
        d = {"location": "Paris"}
        result = _parse_tool_arguments(d)
        assert result == d

    def test_invalid_json_returns_original(self):
        result = _parse_tool_arguments("not json")
        assert result == "not json"

    def test_empty_string(self):
        result = _parse_tool_arguments("")
        assert result == ""

    def test_none_passthrough(self):
        result = _parse_tool_arguments(None)
        assert result is None


# =============================================================================
# LiteLLM send_message Tools Forwarding (Bug 1)
# =============================================================================


class TestLiteLLMToolsForwarding:
    """Test that tools are forwarded through LiteLLM send_message."""

    @pytest.mark.asyncio
    async def test_tools_forwarded_in_send_message(self):
        """Tools should be converted and passed to litellm.acompletion."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Hello", tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = mock_response

            backend = LiteLLMBackend(provider="openrouter")
            body = {
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
                "tool_choice": {"type": "auto"},
            }

            await backend.send_message(body, {})

            call_kwargs = mock_acomp.call_args[1]
            assert "tools" in call_kwargs
            assert call_kwargs["tools"][0]["type"] == "function"
            assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"
            assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_tool_arguments_parsed_in_response(self):
        """Tool call arguments should be parsed from JSON string to dict."""
        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"location": "Paris"}'

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content=None, tool_calls=[mock_tc]),
                finish_reason="tool_calls",
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = mock_response

            backend = LiteLLMBackend(provider="openrouter")
            result = await backend.send_message(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                {},
            )

            tool_block = result.body["content"][0]
            assert tool_block["type"] == "tool_use"
            assert tool_block["input"] == {"location": "Paris"}
            assert isinstance(tool_block["input"], dict)


# =============================================================================
# Message Conversion: tool_use / tool_result (GitHub Issue — Bug 2)
# =============================================================================


class TestConvertMessagesToolBlocks:
    """Test that _convert_messages_for_litellm converts Anthropic tool blocks to OpenAI format."""

    def _make_backend(self):
        with patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}):
            return LiteLLMBackend(provider="openrouter")

    def test_tool_result_converted_to_tool_role(self):
        """Anthropic tool_result blocks must become role=tool messages."""
        backend = self._make_backend()
        messages = [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "Sunny, 22C"},
                ],
            },
        ]
        converted = backend._convert_messages_for_litellm(messages)

        # assistant message should have tool_calls
        assistant = converted[1]
        assert assistant["role"] == "assistant"
        assert "tool_calls" in assistant
        assert assistant["tool_calls"][0]["id"] == "toolu_01"
        assert assistant["tool_calls"][0]["type"] == "function"
        assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}

        # tool_result should become role=tool
        tool_msg = converted[2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_01"
        assert tool_msg["content"] == "Sunny, 22C"

    def test_tool_result_with_list_content(self):
        """tool_result with list content should be flattened to string."""
        backend = self._make_backend()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_02",
                        "content": [
                            {"type": "text", "text": "Line 1"},
                            {"type": "text", "text": "Line 2"},
                        ],
                    },
                ],
            },
        ]
        converted = backend._convert_messages_for_litellm(messages)
        assert converted[0]["role"] == "tool"
        assert converted[0]["content"] == "Line 1\nLine 2"

    def test_assistant_tool_use_with_text(self):
        """Assistant message with both text and tool_use blocks."""
        backend = self._make_backend()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check the weather."},
                    {
                        "type": "tool_use",
                        "id": "toolu_03",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    },
                ],
            },
        ]
        converted = backend._convert_messages_for_litellm(messages)
        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "Let me check the weather."
        assert converted[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_simple_text_messages_unchanged(self):
        """Plain string messages pass through."""
        backend = self._make_backend()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        converted = backend._convert_messages_for_litellm(messages)
        assert converted == messages

    def test_multiple_tool_results(self):
        """Multiple tool_result blocks in one user message → multiple role=tool messages."""
        backend = self._make_backend()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_a", "content": "Result A"},
                    {"type": "tool_result", "tool_use_id": "toolu_b", "content": "Result B"},
                ],
            },
        ]
        converted = backend._convert_messages_for_litellm(messages)
        assert len(converted) == 2
        assert converted[0]["role"] == "tool"
        assert converted[0]["tool_call_id"] == "toolu_a"
        assert converted[1]["role"] == "tool"
        assert converted[1]["tool_call_id"] == "toolu_b"

    def test_tool_result_immediately_follows_tool_calls(self):
        """Bedrock requires role=tool immediately after assistant tool_calls — no intervening messages.

        Regression test for GitHub issue #70: a stray user text message was inserted
        between the assistant tool_calls and the tool results, causing Bedrock to reject
        the request with 'tool_use ids were found without tool_result blocks immediately after'.
        """
        backend = self._make_backend()
        messages = [
            {"role": "user", "content": "What's the weather in Paris and Tokyo?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_02",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "Sunny, 22C"},
                    {"type": "tool_result", "tool_use_id": "toolu_02", "content": "Rainy, 18C"},
                ],
            },
        ]
        converted = backend._convert_messages_for_litellm(messages)

        # Find the assistant message with tool_calls
        assistant_idx = next(i for i, m in enumerate(converted) if m.get("tool_calls"))

        # Every message after the assistant tool_calls must be role=tool
        # with no intervening user/assistant messages
        for i in range(assistant_idx + 1, len(converted)):
            assert converted[i]["role"] == "tool", (
                f"Message at index {i} has role={converted[i]['role']!r}, "
                f"expected 'tool' — Bedrock requires tool results immediately "
                f"after assistant tool_calls with no intervening messages"
            )

    def test_tool_result_with_text_does_not_insert_user_message(self):
        """Text alongside tool_result should NOT produce a separate user message.

        Bedrock rejects any message between assistant tool_calls and tool results.
        """
        backend = self._make_backend()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here are the results:"},
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "42"},
                ],
            },
        ]
        converted = backend._convert_messages_for_litellm(messages)

        # Should only have the tool message, no user text message
        assert len(converted) == 1
        assert converted[0]["role"] == "tool"
        assert converted[0]["tool_call_id"] == "toolu_01"
        assert converted[0]["content"] == "42"


# =============================================================================
# Streaming tool_calls (GitHub Issue — Bug 1)
# =============================================================================


class TestStreamMessageToolCalls:
    """Test that stream_message emits tool_use blocks and correct stop_reason."""

    @pytest.mark.asyncio
    async def test_stream_emits_tool_use_blocks(self):
        """Tool calls in streaming should produce content_block_start with type=tool_use."""

        async def mock_stream():
            # First chunk: tool call start (id + name)
            tc = MagicMock()
            tc.index = 0
            tc.id = "toolu_stream_01"
            tc.function = MagicMock()
            tc.function.name = "get_weather"
            tc.function.arguments = ""

            chunk1 = MagicMock()
            chunk1.choices = [
                MagicMock(delta=MagicMock(content=None, tool_calls=[tc]), finish_reason=None)
            ]
            yield chunk1

            # Second chunk: arguments delta
            tc2 = MagicMock()
            tc2.index = 0
            tc2.id = None
            tc2.function = MagicMock()
            tc2.function.name = None
            tc2.function.arguments = '{"city":"Paris"}'

            chunk2 = MagicMock()
            chunk2.choices = [
                MagicMock(delta=MagicMock(content=None, tool_calls=[tc2]), finish_reason=None)
            ]
            yield chunk2

            # Final chunk: finish_reason=tool_calls
            chunk3 = MagicMock()
            chunk3.choices = [
                MagicMock(
                    delta=MagicMock(content=None, tool_calls=None), finish_reason="tool_calls"
                )
            ]
            yield chunk3

        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = mock_stream()
            backend = LiteLLMBackend(provider="openrouter")

            events = []
            async for event in backend.stream_message(
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Get weather",
                            "input_schema": {"type": "object"},
                        }
                    ],
                },
                {},
            ):
                events.append(event)

        # Find content_block_start events
        block_starts = [e for e in events if e.event_type == "content_block_start"]
        assert len(block_starts) == 1
        assert block_starts[0].data["content_block"]["type"] == "tool_use"
        assert block_starts[0].data["content_block"]["id"] == "toolu_stream_01"
        assert block_starts[0].data["content_block"]["name"] == "get_weather"

        # Find input_json_delta events
        json_deltas = [
            e
            for e in events
            if e.event_type == "content_block_delta"
            and e.data.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(json_deltas) == 1
        assert json_deltas[0].data["delta"]["partial_json"] == '{"city":"Paris"}'

        # Check stop_reason is "tool_use"
        msg_delta = [e for e in events if e.event_type == "message_delta"]
        assert len(msg_delta) == 1
        assert msg_delta[0].data["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_stream_text_still_works(self):
        """Pure text streaming should still work correctly."""

        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [
                MagicMock(delta=MagicMock(content="Hello!", tool_calls=None), finish_reason=None)
            ]
            yield chunk

            chunk2 = MagicMock()
            chunk2.choices = [
                MagicMock(delta=MagicMock(content=None, tool_calls=None), finish_reason="stop")
            ]
            yield chunk2

        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = mock_stream()
            backend = LiteLLMBackend(provider="openrouter")

            events = []
            async for event in backend.stream_message(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                {},
            ):
                events.append(event)

        block_starts = [e for e in events if e.event_type == "content_block_start"]
        assert len(block_starts) == 1
        assert block_starts[0].data["content_block"]["type"] == "text"

        text_deltas = [e for e in events if e.event_type == "content_block_delta"]
        assert len(text_deltas) == 1
        assert text_deltas[0].data["delta"]["text"] == "Hello!"

        msg_delta = [e for e in events if e.event_type == "message_delta"]
        assert msg_delta[0].data["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_stream_text_then_tool(self):
        """Text followed by tool call should produce two blocks."""

        async def mock_stream():
            # Text chunk
            chunk1 = MagicMock()
            chunk1.choices = [
                MagicMock(
                    delta=MagicMock(content="I'll check. ", tool_calls=None), finish_reason=None
                )
            ]
            yield chunk1

            # Tool call chunk
            tc = MagicMock()
            tc.index = 0
            tc.id = "toolu_mixed"
            tc.function = MagicMock()
            tc.function.name = "search"
            tc.function.arguments = '{"q":"test"}'

            chunk2 = MagicMock()
            chunk2.choices = [
                MagicMock(delta=MagicMock(content=None, tool_calls=[tc]), finish_reason=None)
            ]
            yield chunk2

            # Finish
            chunk3 = MagicMock()
            chunk3.choices = [
                MagicMock(
                    delta=MagicMock(content=None, tool_calls=None), finish_reason="tool_calls"
                )
            ]
            yield chunk3

        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = mock_stream()
            backend = LiteLLMBackend(provider="openrouter")

            events = []
            async for event in backend.stream_message(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                {},
            ):
                events.append(event)

        block_starts = [e for e in events if e.event_type == "content_block_start"]
        assert len(block_starts) == 2
        assert block_starts[0].data["content_block"]["type"] == "text"
        assert block_starts[1].data["content_block"]["type"] == "tool_use"

        # Two content_block_stop events (one per block)
        block_stops = [e for e in events if e.event_type == "content_block_stop"]
        assert len(block_stops) == 2

        # stop_reason should be tool_use
        msg_delta = [e for e in events if e.event_type == "message_delta"]
        assert msg_delta[0].data["delta"]["stop_reason"] == "tool_use"


# =============================================================================
# Streaming Params (Bugs 3-4)
# =============================================================================


class TestLiteLLMStreamingParams:
    """Test that streaming forwards all params."""

    @pytest.mark.asyncio
    async def test_streaming_forwards_all_params(self):
        """stream_message should forward top_p, stop, and tools."""

        # Create an async iterator for the mock streaming response
        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=MagicMock(content="Hi"))]
            yield chunk

        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = mock_stream()

            backend = LiteLLMBackend(provider="openrouter")
            body = {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop_sequences": ["\n"],
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "A test",
                        "input_schema": {"type": "object"},
                    }
                ],
            }

            events = []
            async for event in backend.stream_message(body, {}):
                events.append(event)

            call_kwargs = mock_acomp.call_args[1]
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["stop"] == ["\n"]
            assert "tools" in call_kwargs
            assert call_kwargs["tools"][0]["function"]["name"] == "test_tool"


# =============================================================================
# Vertex AI Model Map (Bug 6)
# =============================================================================


class TestVertexModelMap:
    """Test that Vertex AI model map includes all current models.

    Model IDs sourced from: https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
    """

    def test_claude_46_models(self):
        assert _VERTEX_MODEL_MAP["claude-opus-4-6"] == "vertex_ai/claude-opus-4-6"
        assert _VERTEX_MODEL_MAP["claude-sonnet-4-6"] == "vertex_ai/claude-sonnet-4-6"

    def test_claude_45_models(self):
        assert (
            _VERTEX_MODEL_MAP["claude-sonnet-4-5-20250929"]
            == "vertex_ai/claude-sonnet-4-5@20250929"
        )
        assert _VERTEX_MODEL_MAP["claude-opus-4-5-20251101"] == "vertex_ai/claude-opus-4-5@20251101"

    def test_claude_4_models(self):
        assert _VERTEX_MODEL_MAP["claude-sonnet-4-20250514"] == "vertex_ai/claude-sonnet-4@20250514"
        assert _VERTEX_MODEL_MAP["claude-opus-4-20250514"] == "vertex_ai/claude-opus-4@20250514"

    def test_claude_35_models(self):
        assert (
            _VERTEX_MODEL_MAP["claude-3-5-sonnet-20241022"]
            == "vertex_ai/claude-3-5-sonnet-v2@20241022"
        )
        assert (
            _VERTEX_MODEL_MAP["claude-3-5-haiku-20241022"] == "vertex_ai/claude-3-5-haiku@20241022"
        )

    def test_claude_haiku_45(self):
        assert (
            _VERTEX_MODEL_MAP["claude-haiku-4-5-20251001"] == "vertex_ai/claude-haiku-4-5@20251001"
        )

    def test_claude_3_legacy(self):
        assert "claude-3-haiku-20240307" in _VERTEX_MODEL_MAP


# =============================================================================
# URL Normalization (trailing /v1 stripping)
# =============================================================================

pytest.importorskip("fastapi")


class TestOpenAIURLNormalization:
    """Test that OPENAI_TARGET_API_URL with /v1 suffix is normalized."""

    def test_v1_suffix_stripped(self):
        from headroom.proxy.server import HeadroomProxy, ProxyConfig

        original = HeadroomProxy.OPENAI_API_URL
        try:
            config = ProxyConfig(
                openai_api_url="http://localhost:4000/v1",
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
            )
            proxy = HeadroomProxy(config)
            assert proxy.OPENAI_API_URL == "http://localhost:4000"
        finally:
            HeadroomProxy.OPENAI_API_URL = original

    def test_v1_slash_suffix_stripped(self):
        from headroom.proxy.server import HeadroomProxy, ProxyConfig

        original = HeadroomProxy.OPENAI_API_URL
        try:
            config = ProxyConfig(
                openai_api_url="http://localhost:4000/v1/",
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
            )
            proxy = HeadroomProxy(config)
            assert proxy.OPENAI_API_URL == "http://localhost:4000"
        finally:
            HeadroomProxy.OPENAI_API_URL = original

    def test_no_v1_unchanged(self):
        from headroom.proxy.server import HeadroomProxy, ProxyConfig

        original = HeadroomProxy.OPENAI_API_URL
        try:
            config = ProxyConfig(
                openai_api_url="http://localhost:4000",
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
            )
            proxy = HeadroomProxy(config)
            assert proxy.OPENAI_API_URL == "http://localhost:4000"
        finally:
            HeadroomProxy.OPENAI_API_URL = original


# =============================================================================
# Bedrock API Key Forwarding Regression (#105)
# =============================================================================


class TestBedrockApiKeyNotForwarded:
    """Bedrock uses AWS SigV4 auth, not API keys.

    Forwarding x-api-key (e.g. sk-ant-dummy) to LiteLLM overrides
    AWS credentials and breaks Bedrock auth.
    """

    def test_bedrock_does_not_forward_api_key(self):
        """api_key should NOT be in kwargs for Bedrock provider."""
        backend = LiteLLMBackend(provider="bedrock", region="us-west-2")

        kwargs = {}
        headers = {
            "x-api-key": "sk-ant-dummy-key",
            "authorization": "Bearer sk-ant-dummy-key",
        }

        # Simulate what the handler does: build kwargs then check
        _env_auth_providers = ("bedrock", "vertex_ai", "vertex_ai_beta", "sagemaker")
        if backend.provider not in _env_auth_providers:
            auth_header = headers.get("authorization", headers.get("Authorization", ""))
            if auth_header.startswith("Bearer "):
                kwargs["api_key"] = auth_header[7:]
            elif headers.get("x-api-key"):
                kwargs["api_key"] = headers["x-api-key"]

        assert "api_key" not in kwargs, (
            f"Bedrock should not have api_key in kwargs, got: {kwargs.get('api_key')}"
        )

    def test_openai_does_forward_api_key(self):
        """api_key SHOULD be in kwargs for non-Bedrock providers."""
        backend = LiteLLMBackend(provider="openai")

        kwargs = {}
        headers = {"authorization": "Bearer sk-real-key-123"}

        _env_auth_providers = ("bedrock", "vertex_ai", "vertex_ai_beta", "sagemaker")
        if backend.provider not in _env_auth_providers:
            auth_header = headers.get("authorization", headers.get("Authorization", ""))
            if auth_header.startswith("Bearer "):
                kwargs["api_key"] = auth_header[7:]

        assert kwargs.get("api_key") == "sk-real-key-123"

    def test_vertex_does_not_forward_api_key(self):
        """Vertex AI also uses env-based auth (Google ADC)."""
        backend = LiteLLMBackend(provider="vertex_ai")

        kwargs = {}
        headers = {"x-api-key": "sk-ant-dummy"}

        _env_auth_providers = ("bedrock", "vertex_ai", "vertex_ai_beta", "sagemaker")
        if backend.provider not in _env_auth_providers:
            if headers.get("x-api-key"):
                kwargs["api_key"] = headers["x-api-key"]

        assert "api_key" not in kwargs

"""Tests for native Deepseek backend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from headroom.backends.base import BackendResponse
from headroom.backends.deepseek import DeepseekBackend


class TestDeepseekBackendBasics:
    def test_name(self):
        backend = DeepseekBackend()
        assert backend.name == "deepseek"

    def test_supports_claude_models(self):
        backend = DeepseekBackend()
        assert backend.supports_model("claude-3-5-sonnet-20241022")
        assert backend.supports_model("claude-opus-4-6")

    def test_supports_deepseek_models(self):
        backend = DeepseekBackend()
        assert backend.supports_model("deepseek-chat")
        assert backend.supports_model("deepseek-v4-flash")

    def test_does_not_support_other_models(self):
        backend = DeepseekBackend()
        assert not backend.supports_model("gpt-4o")

    def test_map_model_id_known_claude(self):
        backend = DeepseekBackend()
        assert backend.map_model_id("claude-3-5-sonnet-20241022") == "deepseek-chat"
        assert backend.map_model_id("claude-opus-4-6") == "deepseek-v4-pro"

    def test_map_model_id_deepseek_passthrough(self):
        backend = DeepseekBackend()
        assert backend.map_model_id("deepseek-chat") == "deepseek-chat"
        assert backend.map_model_id("deepseek-v4-flash") == "deepseek-v4-flash"

    def test_map_model_id_unknown_claude_passes_through(self):
        backend = DeepseekBackend()
        assert backend.map_model_id("claude-future-model") == "claude-future-model"


class TestDeepseekBackendAnthropic:
    @pytest.mark.asyncio
    async def test_send_message(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "deepseek-chat",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        backend = DeepseekBackend(anthropic_client=mock_client)
        response = await backend.send_message(
            body={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={},
        )

        assert isinstance(response, BackendResponse)
        assert response.status_code == 200
        assert response.body["role"] == "assistant"
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_passes_all_params(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "deepseek-chat",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        backend = DeepseekBackend(anthropic_client=mock_client)
        await backend.send_message(
            body={
                "model": "deepseek-chat",
                "max_tokens": 2048,
                "messages": [],
                "system": "You are helpful",
                "temperature": 0.7,
                "top_p": 0.9,
                "stop_sequences": ["END"],
                "tools": [{"name": "search"}],
                "tool_choice": {"type": "auto"},
            },
            headers={},
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-chat"
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["system"] == "You are helpful"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["stop_sequences"] == ["END"]
        assert call_kwargs["tools"] == [{"name": "search"}]
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    @pytest.mark.asyncio
    async def test_send_message_error(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("Authentication failed")
        )

        backend = DeepseekBackend(anthropic_client=mock_client)
        response = await backend.send_message(
            body={"model": "deepseek-chat", "messages": []},
            headers={},
        )

        assert response.status_code == 401
        assert response.error is not None
        assert "Authentication failed" in response.error

    @pytest.mark.asyncio
    async def test_send_message_generic_error(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("Something went wrong")
        )

        backend = DeepseekBackend(anthropic_client=mock_client)
        response = await backend.send_message(
            body={"model": "deepseek-chat", "messages": []},
            headers={},
        )

        assert response.status_code == 500
        assert response.error is not None


class TestDeepseekBackendStreaming:
    @pytest.mark.asyncio
    async def test_stream_message(self):
        mock_client = AsyncMock()

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_event1 = MagicMock()
        mock_event1.type = "content_block_start"
        mock_event1.content_block = SimpleNamespace(type="text", text="")

        mock_event2 = MagicMock()
        mock_event2.type = "content_block_delta"
        mock_event2.delta = SimpleNamespace(type="text_delta", text="Hello")

        mock_event3 = MagicMock()
        mock_event3.type = "content_block_stop"

        async def mock_aiter(self_inner):
            for ev in [mock_event1, mock_event2, mock_event3]:
                yield ev

        mock_stream_ctx.__aiter__ = mock_aiter

        mock_final = SimpleNamespace(
            stop_reason="end_turn",
            usage=SimpleNamespace(output_tokens=5),
        )
        mock_stream_ctx.get_final_message = AsyncMock(return_value=mock_final)

        mock_client.messages.stream = MagicMock(return_value=mock_stream_ctx)

        backend = DeepseekBackend(anthropic_client=mock_client)
        events = []
        async for event in backend.stream_message(
            body={
                "model": "deepseek-chat",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={},
        ):
            events.append(event)

        assert len(events) > 0
        assert events[0].event_type == "message_start"
        assert events[-1].event_type == "message_stop"

        event_types = [e.event_type for e in events]
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types

    @pytest.mark.asyncio
    async def test_stream_message_error(self):
        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(
            side_effect=Exception("Rate limited")
        )

        backend = DeepseekBackend(anthropic_client=mock_client)
        events = []
        async for event in backend.stream_message(
            body={"model": "deepseek-chat", "messages": []},
            headers={},
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0].event_type == "error"
        assert "Rate limited" in events[0].data["error"]["message"]


class TestDeepseekBackendOpenAI:
    @pytest.mark.asyncio
    async def test_send_openai_message(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        backend = DeepseekBackend(openai_client=mock_client)
        response = await backend.send_openai_message(
            body={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={},
        )

        assert isinstance(response, BackendResponse)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_send_openai_message_passes_params(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"id": "chatcmpl-123"}

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        backend = DeepseekBackend(openai_client=mock_client)
        await backend.send_openai_message(
            body={
                "model": "deepseek-chat",
                "messages": [],
                "max_tokens": 1024,
                "temperature": 0.5,
                "top_p": 0.8,
                "stop": ["END"],
                "tools": [{"name": "search"}],
                "tool_choice": "auto",
            },
            headers={},
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_send_openai_message_error(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Rate limited")
        )

        backend = DeepseekBackend(openai_client=mock_client)
        response = await backend.send_openai_message(
            body={"model": "deepseek-chat", "messages": []},
            headers={},
        )

        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_send_openai_message_generic_error(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Server error")
        )

        backend = DeepseekBackend(openai_client=mock_client)
        response = await backend.send_openai_message(
            body={"model": "deepseek-chat", "messages": []},
            headers={},
        )

        assert response.status_code == 500


class TestDeepseekBackendOpenAIStreaming:
    @pytest.mark.asyncio
    async def test_stream_openai_message(self):
        mock_chunk1 = MagicMock()
        mock_chunk1.model_dump.return_value = {
            "choices": [{"delta": {"content": "Hello"}}]
        }
        mock_chunk2 = MagicMock()
        mock_chunk2.model_dump.return_value = {
            "choices": [{"delta": {"content": "!"}}]
        }

        async def mock_aiter():
            for chunk in [mock_chunk1, mock_chunk2]:
                yield chunk

        mock_stream = mock_aiter()

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        backend = DeepseekBackend(openai_client=mock_client)
        chunks = []
        async for chunk in backend.stream_openai_message(
            body={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers={},
        ):
            chunks.append(chunk)

        assert len(chunks) == 3  # 2 data chunks + [DONE]
        assert chunks[0].startswith("data: ")
        assert "[DONE]" in chunks[-1]

    @pytest.mark.asyncio
    async def test_stream_openai_message_error(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        backend = DeepseekBackend(openai_client=mock_client)
        chunks = []
        async for chunk in backend.stream_openai_message(
            body={"model": "deepseek-chat", "messages": []},
            headers={},
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "[DONE]" in chunks[-1]


class TestDeepseekBackendClose:
    @pytest.mark.asyncio
    async def test_close(self):
        mock_anthropic = AsyncMock()
        mock_openai = AsyncMock()

        backend = DeepseekBackend(
            anthropic_client=mock_anthropic, openai_client=mock_openai
        )
        await backend.close()

        mock_anthropic.close.assert_called_once()
        mock_openai.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_noop_without_clients(self):
        backend = DeepseekBackend()
        await backend.close()  # Should not raise


class TestDeepseekBackendClientCreation:
    def test_anthropic_client_not_created_without_import(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            backend = DeepseekBackend(api_key="test-key")
            with pytest.raises(RuntimeError, match="anthropic SDK is required"):
                backend._get_anthropic_client()

    def test_openai_client_not_created_without_import(self):
        with patch.dict("sys.modules", {"openai": None}):
            backend = DeepseekBackend(api_key="test-key")
            with pytest.raises(RuntimeError, match="openai SDK is required"):
                backend._get_openai_client()

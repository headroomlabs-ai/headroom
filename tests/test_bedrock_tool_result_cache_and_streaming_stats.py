"""Two Bedrock Converse gaps left after #1390 (system-prompt/text-block cache_control):

1. ``_convert_messages_for_litellm`` preserved ``cache_control`` on text blocks
   and the system prompt, but not on ``tool_result`` blocks. Claude Code's
   moving cache breakpoint lands on the tail ``tool_result`` far more often
   than on the system prompt or a plain text block, so this is the case that
   matters most in agent loops.

2. ``LiteLLMBackend.stream_message`` never requested
   ``stream_options.include_usage``, so LiteLLM/Bedrock never returned a usage
   chunk over SSE — cache_read/cache_write always reported 0 downstream even
   when the Bedrock prompt cache was genuinely engaged. The ``message_start``
   emitted before streaming begins is necessarily sent before any usage is
   known (hardcoded ``input_tokens: 0``, no cache fields); once the real
   values are captured from the trailing usage chunk, the terminal
   ``message_delta.usage`` carries them so the public Anthropic event order
   remains valid.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402


def _backend(provider: str = "bedrock") -> LiteLLMBackend:
    with patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}):
        return LiteLLMBackend(provider=provider, region="us-east-1")


class TestToolResultCacheControlPreserved:
    """tool_result blocks must carry cache_control onto the emitted tool message."""

    def test_tool_result_with_cache_control_preserved(self):
        backend = _backend()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": "big tool output",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ]
        converted = backend._convert_messages_for_litellm(messages)
        assert len(converted) == 1
        assert converted[0]["role"] == "tool"
        assert converted[0]["cache_control"] == {"type": "ephemeral"}

    def test_tool_result_without_cache_control_unaffected(self):
        backend = _backend()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_02", "content": "no marker"},
                ],
            },
        ]
        converted = backend._convert_messages_for_litellm(messages)
        assert "cache_control" not in converted[0]

    def test_multiple_tool_results_only_marked_one_preserved(self):
        backend = _backend()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_a", "content": "Result A"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_b",
                        "content": "Result B",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ]
        converted = backend._convert_messages_for_litellm(messages)
        assert len(converted) == 2
        assert "cache_control" not in converted[0]
        assert converted[1]["cache_control"] == {"type": "ephemeral"}

    def test_tool_result_cache_control_not_forwarded_for_non_bedrock(self):
        """The conversion is provider-agnostic; other providers get the same
        block shape, and LiteLLM/that provider's own transformation is
        responsible for ignoring cache_control it doesn't understand."""
        backend = _backend(provider="openrouter")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_03",
                        "content": "output",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ]
        converted = backend._convert_messages_for_litellm(messages)
        assert converted[0]["cache_control"] == {"type": "ephemeral"}


class TestStreamingCacheStatsCompletion:
    """stream_message must request usage and re-surface real cache stats."""

    def _mock_stream_with_final_usage(
        self, cache_read: int = 0, cache_write: int = 0, prompt_tokens: int = 0
    ):
        async def mock_stream():
            chunk1 = MagicMock()
            chunk1.usage = None
            chunk1.choices = [
                MagicMock(delta=MagicMock(content="hi", tool_calls=None), finish_reason=None)
            ]
            yield chunk1

            # Final chunk: content-less, usage-carrying (mirrors
            # stream_options.include_usage's trailing chunk shape).
            chunk2 = MagicMock()
            chunk2.choices = [
                MagicMock(delta=MagicMock(content=None, tool_calls=None), finish_reason="stop")
            ]
            chunk2.usage = MagicMock(
                prompt_tokens=prompt_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_write,
            )
            yield chunk2

        return mock_stream

    @pytest.mark.asyncio
    async def test_stream_options_include_usage_requested(self):
        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = self._mock_stream_with_final_usage()()
            backend = LiteLLMBackend(provider="bedrock", region="us-east-1")

            events = [
                e
                async for e in backend.stream_message(
                    {"model": "test", "messages": [{"role": "user", "content": "hi"}]}, {}
                )
            ]
            assert events

            call_kwargs = mock_acomp.call_args[1]
            assert call_kwargs["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_terminal_message_delta_carries_real_cache_stats(self):
        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = self._mock_stream_with_final_usage(
                cache_read=1200, cache_write=300, prompt_tokens=1500
            )()
            backend = LiteLLMBackend(provider="bedrock", region="us-east-1")

            events = [
                e
                async for e in backend.stream_message(
                    {"model": "test", "messages": [{"role": "user", "content": "hi"}]}, {}
                )
            ]

            message_starts = [e for e in events if e.event_type == "message_start"]
            assert len(message_starts) == 1, "stream should keep a single initial message_start"

            first_usage = message_starts[0].data["message"]["usage"]
            assert first_usage["input_tokens"] == 0
            assert "cache_read_input_tokens" not in first_usage

            message_deltas = [e for e in events if e.event_type == "message_delta"]
            assert len(message_deltas) == 1
            final_usage = message_deltas[0].data["usage"]
            assert final_usage["input_tokens"] == 1500
            assert final_usage["cache_read_input_tokens"] == 1200
            assert final_usage["cache_creation_input_tokens"] == 300

    @pytest.mark.asyncio
    async def test_no_extra_message_start_when_usage_never_reported(self):
        """Non-Bedrock LiteLLM providers (or any response that never carries a
        usage chunk) must not get a spurious trailing message_start."""

        async def mock_stream():
            chunk = MagicMock()
            chunk.usage = None
            chunk.choices = [
                MagicMock(delta=MagicMock(content="hi", tool_calls=None), finish_reason="stop")
            ]
            yield chunk

        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = mock_stream()
            backend = LiteLLMBackend(provider="openrouter")

            events = [
                e
                async for e in backend.stream_message(
                    {"model": "test", "messages": [{"role": "user", "content": "hi"}]}, {}
                )
            ]

            message_starts = [e for e in events if e.event_type == "message_start"]
            assert len(message_starts) == 1

    @pytest.mark.asyncio
    async def test_terminal_message_delta_omits_zero_cache_fields(self):
        """If only input_tokens came back (no caching engaged), the trailing
        message_delta should include input_tokens but no empty cache_* keys."""
        with (
            patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp,
            patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}),
        ):
            mock_acomp.return_value = self._mock_stream_with_final_usage(
                cache_read=0, cache_write=0, prompt_tokens=42
            )()
            backend = LiteLLMBackend(provider="bedrock", region="us-east-1")

            events = [
                e
                async for e in backend.stream_message(
                    {"model": "test", "messages": [{"role": "user", "content": "hi"}]}, {}
                )
            ]

            message_starts = [e for e in events if e.event_type == "message_start"]
            assert len(message_starts) == 1
            message_deltas = [e for e in events if e.event_type == "message_delta"]
            usage = message_deltas[0].data["usage"]
            assert usage["input_tokens"] == 42
            assert "cache_read_input_tokens" not in usage
            assert "cache_creation_input_tokens" not in usage


class TestSystemFieldCacheControl:
    """The top-level Anthropic `system` field must not be flattened to a
    plain string when it carries per-block cache_control, or Bedrock prompt
    caching of the system prefix silently breaks (see module docstring,
    #1390's uncovered case)."""

    def test_list_system_with_cache_control_preserved(self):
        backend = _backend()
        system = [
            {"type": "text", "text": "You are a helpful assistant."},
            {"type": "text", "text": "Long static prefix.", "cache_control": {"type": "ephemeral"}},
        ]
        msg = backend._system_field_to_message(system)
        assert msg["role"] == "system"
        assert isinstance(msg["content"], list)
        assert msg["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_string_system_unaffected(self):
        backend = _backend()
        msg = backend._system_field_to_message("You are a helpful assistant.")
        assert msg == {"role": "system", "content": "You are a helpful assistant."}
        assert isinstance(msg["content"], str)

    def test_list_system_without_cache_control_has_no_cache_control_keys(self):
        backend = _backend()
        system = [
            {"type": "text", "text": "First block."},
            {"type": "text", "text": "Second block."},
        ]
        msg = backend._system_field_to_message(system)
        assert isinstance(msg["content"], list)
        assert all("cache_control" not in block for block in msg["content"])

    def test_bedrock_converse_transform_emits_cachepoint_for_list_with_cache_control(self):
        from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

        backend = _backend()
        system = [
            {"type": "text", "text": "You are a helpful assistant."},
            {"type": "text", "text": "Long static prefix.", "cache_control": {"type": "ephemeral"}},
        ]
        system_msg = backend._system_field_to_message(system)
        messages = [system_msg, {"role": "user", "content": "hi"}]

        _, system_blocks = AmazonConverseConfig()._transform_system_message(
            messages, model="global.anthropic.claude-sonnet-5"
        )
        assert any("cachePoint" in block for block in system_blocks)

    def test_bedrock_converse_transform_omits_cachepoint_without_cache_control(self):
        from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

        backend = _backend()
        system = [
            {"type": "text", "text": "First block."},
            {"type": "text", "text": "Second block."},
        ]
        system_msg = backend._system_field_to_message(system)
        messages = [system_msg, {"role": "user", "content": "hi"}]

        _, system_blocks = AmazonConverseConfig()._transform_system_message(
            messages, model="global.anthropic.claude-sonnet-5"
        )
        assert not any("cachePoint" in block for block in system_blocks)

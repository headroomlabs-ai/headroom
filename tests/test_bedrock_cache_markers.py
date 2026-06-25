"""Regression tests for Bedrock cache_control → cachePoint marker preservation.

Issue #1345: --mode cache with --backend bedrock froze prefix turns but never
injected cachePoint markers because _convert_messages_for_litellm stripped
cache_control from text blocks before LiteLLM could translate them.
"""

from __future__ import annotations

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402


def _backend() -> LiteLLMBackend:
    return LiteLLMBackend(provider="bedrock", region="us-east-1")


class TestConvertMessagesPreservesCacheControl:
    def test_text_block_without_cache_control_joins_to_string(self) -> None:
        """Baseline: text blocks without cache_control stay as plain strings."""
        backend = _backend()
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = backend._convert_messages_for_litellm(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_text_block_with_cache_control_kept_as_list(self) -> None:
        """cache_control on text block must survive conversion as a content block list."""
        backend = _backend()
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "summarise this",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
        result = backend._convert_messages_for_litellm(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list), "cache_control blocks must stay as list"
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert content[0]["text"] == "summarise this"

    def test_multiple_text_blocks_one_with_cache_control(self) -> None:
        """When any text block has cache_control, all text blocks are kept as list."""
        backend = _backend()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first part"},
                    {
                        "type": "text",
                        "text": "second part",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ]
        result = backend._convert_messages_for_litellm(msgs)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        # cache_control preserved on the block that had it
        assert "cache_control" in content[1]
        assert "cache_control" not in content[0]

    def test_cache_control_not_on_text_block_still_plain_string(self) -> None:
        """String content is unchanged."""
        backend = _backend()
        msgs = [{"role": "user", "content": "plain string"}]
        result = backend._convert_messages_for_litellm(msgs)
        assert result == [{"role": "user", "content": "plain string"}]

    def test_tool_result_blocks_unaffected(self) -> None:
        """tool_result conversion path is unchanged by this fix."""
        backend = _backend()
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "output",
                    }
                ],
            }
        ]
        result = backend._convert_messages_for_litellm(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"


class TestSystemPromptPreservesCacheControl:
    """send_message and stream_message must pass system cache_control to LiteLLM."""

    def test_system_list_without_cache_control_joins_to_string(self) -> None:
        """Baseline: list system without cache_control joined to plain string."""
        import asyncio
        from unittest.mock import patch

        backend = _backend()
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "system": [{"type": "text", "text": "be helpful"}],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }

        captured: dict = {}

        async def fake_acompletion(**kwargs):  # noqa: ANN003
            captured.update(kwargs)
            raise RuntimeError("stop")

        with patch("headroom.backends.litellm.acompletion", fake_acompletion):
            try:
                asyncio.run(backend.send_message(body, {}))
            except RuntimeError:
                pass

        sys_msg = next(m for m in captured["messages"] if m["role"] == "system")
        assert isinstance(sys_msg["content"], str)
        assert sys_msg["content"] == "be helpful"

    def test_system_list_without_dict_entries_keeps_legacy_stringification(self) -> None:
        """Non-dict system list entries were stringified before the cache_control fix."""
        backend = _backend()
        assert backend._convert_system_for_litellm(["alpha", {"type": "text", "text": "beta"}]) == (
            "alpha beta"
        )

    def test_system_list_with_cache_control_kept_as_list(self) -> None:
        """System blocks with cache_control must reach LiteLLM as content block list."""
        import asyncio
        from unittest.mock import patch

        backend = _backend()
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "system": [
                {
                    "type": "text",
                    "text": "you are a helpful assistant",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }

        captured: dict = {}

        async def fake_acompletion(**kwargs):  # noqa: ANN003
            captured.update(kwargs)
            raise RuntimeError("stop")

        with patch("headroom.backends.litellm.acompletion", fake_acompletion):
            try:
                asyncio.run(backend.send_message(body, {}))
            except RuntimeError:
                pass

        sys_msg = next(m for m in captured["messages"] if m["role"] == "system")
        assert isinstance(sys_msg["content"], list), "cache_control system must stay as list"
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert sys_msg["content"][0]["text"] == "you are a helpful assistant"

    def test_system_list_with_cache_control_preserves_non_dict_entries_as_text_blocks(self) -> None:
        """Mixed system lists keep non-dict entries when cache_control requires block mode."""
        backend = _backend()
        result = backend._convert_system_for_litellm(
            [
                "legacy preface",
                {
                    "type": "text",
                    "text": "cache this",
                    "cache_control": {"type": "ephemeral"},
                },
            ]
        )
        assert result == [
            {"type": "text", "text": "legacy preface"},
            {
                "type": "text",
                "text": "cache this",
                "cache_control": {"type": "ephemeral"},
            },
        ]

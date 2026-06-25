"""Regression tests for #1077: SmartCrusher must not re-compress headroom_retrieve
tool results.

When the proxy's CCR path returns content via headroom_retrieve, the client sends
it back as a tool_result.  Without the fix, SmartCrusher.apply() would compress
that content again → new <<ccr:hash>> marker → infinite retrieval loop.
"""

from __future__ import annotations

import json

import pytest

from headroom import OpenAIProvider, Tokenizer
from headroom.ccr.tool_injection import CCR_TOOL_NAME
from headroom.config import SmartCrusherConfig


def _build_extension() -> None:
    try:
        from headroom._core import SmartCrusher  # noqa: F401
    except ImportError:
        pytest.skip(
            "headroom._core not built — run `bash scripts/build_rust_extension.sh`",
            allow_module_level=True,
        )


_build_extension()

_provider = OpenAIProvider()


def _get_tokenizer(model: str = "gpt-4o") -> Tokenizer:
    return Tokenizer(_provider.get_token_counter(model), model)


from headroom.transforms.smart_crusher import SmartCrusher  # noqa: E402


def _make_crusher(min_tokens: int = 0) -> SmartCrusher:
    return SmartCrusher(config=SmartCrusherConfig(min_tokens_to_crush=min_tokens))


def _big_content() -> str:
    """Return a JSON array large enough to trigger compression."""
    return json.dumps([{"id": i, "value": "x" * 20} for i in range(60)])


class TestHeadroomRetrieveExemptionOpenAI:
    """OpenAI-style role=tool messages from headroom_retrieve must not be crushed."""

    def test_retrieve_result_not_compressed(self):
        """headroom_retrieve tool result is skipped even when content is large."""
        content = _big_content()
        messages = [
            # The assistant turn that called headroom_retrieve
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "function": {"name": CCR_TOOL_NAME, "arguments": '{"hash":"abc"}'},
                    }
                ],
            },
            # The tool result — this must NOT be re-compressed
            {"role": "tool", "tool_call_id": "call_abc", "content": content},
        ]
        crusher = _make_crusher(min_tokens=0)
        tokenizer = _get_tokenizer()
        result = crusher.apply(messages, tokenizer)

        # Content must be byte-for-byte unchanged
        tool_msg = result.messages[1]
        assert tool_msg["content"] == content, (
            "headroom_retrieve tool result was re-compressed (infinite loop bug #1077)"
        )
        # No CCR transforms should have fired
        assert not any("smart_crush" in t for t in result.transforms_applied)

    def test_non_retrieve_tool_still_compressed(self):
        """Normal tool results are still compressed — exemption is narrow."""
        content = _big_content()
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "function": {"name": "Bash", "arguments": '{"cmd":"ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_xyz", "content": content},
        ]
        crusher = _make_crusher(min_tokens=0)
        tokenizer = _get_tokenizer()
        result = crusher.apply(messages, tokenizer)

        tool_msg = result.messages[1]
        # Content should have been modified (compressed)
        assert tool_msg["content"] != content or result.tokens_after <= result.tokens_before


class TestHeadroomRetrieveExemptionAnthropic:
    """Anthropic-style tool_result content blocks from headroom_retrieve must not be crushed."""

    def test_retrieve_result_not_compressed(self):
        """Anthropic tool_result block for headroom_retrieve is skipped."""
        content = _big_content()
        messages = [
            # assistant turn with tool_use block calling headroom_retrieve
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_ccr_1",
                        "name": CCR_TOOL_NAME,
                        "input": {"hash": "abc123def456"},
                    }
                ],
            },
            # user turn with tool_result — must NOT be re-compressed
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_ccr_1",
                        "content": content,
                    }
                ],
            },
        ]
        crusher = _make_crusher(min_tokens=0)
        tokenizer = _get_tokenizer()
        result = crusher.apply(messages, tokenizer)

        tool_result_block = result.messages[1]["content"][0]
        assert tool_result_block["content"] == content, (
            "headroom_retrieve Anthropic tool_result was re-compressed (#1077)"
        )
        assert not any("smart_crush" in t for t in result.transforms_applied)

    def test_non_retrieve_anthropic_tool_still_compressed(self):
        """Normal Anthropic tool_result blocks are still compressed."""
        content = _big_content()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_bash_1",
                        "name": "Bash",
                        "input": {"cmd": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_bash_1",
                        "content": content,
                    }
                ],
            },
        ]
        crusher = _make_crusher(min_tokens=0)
        tokenizer = _get_tokenizer()
        result = crusher.apply(messages, tokenizer)

        # Either the content changed (compressed) or tokens went down
        tool_result_block = result.messages[1]["content"][0]
        assert (
            tool_result_block["content"] != content or result.tokens_after <= result.tokens_before
        )

    def test_mixed_retrieve_and_normal_only_normal_compressed(self):
        """With two tool_results in one user turn, only the non-CCR one is compressed."""
        content = _big_content()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_ccr_2",
                        "name": CCR_TOOL_NAME,
                        "input": {"hash": "abc123def456"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_bash_2",
                        "name": "Bash",
                        "input": {"cmd": "ls"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_ccr_2",
                        "content": content,  # must NOT be compressed
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_bash_2",
                        "content": content,  # MAY be compressed
                    },
                ],
            },
        ]
        crusher = _make_crusher(min_tokens=0)
        tokenizer = _get_tokenizer()
        result = crusher.apply(messages, tokenizer)

        blocks = result.messages[1]["content"]
        ccr_block = blocks[0]
        bash_block = blocks[1]

        # headroom_retrieve result must be untouched
        assert ccr_block["content"] == content, (
            "headroom_retrieve result was compressed — infinite loop bug #1077"
        )
        # The bash result should differ (or at least the crush count > 0)
        assert bash_block["content"] != content or result.tokens_after < result.tokens_before

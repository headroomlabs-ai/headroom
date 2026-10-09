"""Regression tests for qualified CCR retrieval tool names."""

from __future__ import annotations

import json

import pytest

from headroom import OpenAIProvider, Tokenizer
from headroom.ccr.tool_injection import CCR_TOOL_NAME
from headroom.config import SmartCrusherConfig

try:
    from headroom._core import SmartCrusher as _RustSmartCrusher  # noqa: F401
except ImportError:
    pytest.skip("headroom._core not built", allow_module_level=True)

from headroom.transforms.smart_crusher import SmartCrusher


def _big_content() -> str:
    return json.dumps([{"id": i, "value": "x" * 20} for i in range(60)])


def _apply_for_tool(tool_name: str):
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": _big_content()},
    ]
    tokenizer = Tokenizer(OpenAIProvider().get_token_counter("gpt-4o"), "gpt-4o")
    result = SmartCrusher(config=SmartCrusherConfig(min_tokens_to_crush=0)).apply(
        messages, tokenizer
    )
    return messages[1]["content"], result


@pytest.mark.parametrize(
    "tool_name",
    ["mcp__Headroom__headroom_retrieve", "mcp_Headroom_headroom_retrieve"],
)
def test_qualified_ccr_retrieval_result_is_preserved(tool_name: str) -> None:
    original, result = _apply_for_tool(tool_name)

    assert result.messages[1]["content"] == original
    assert not any("smart_crush" in transform for transform in result.transforms_applied)


def test_near_match_ccr_tool_name_still_compresses() -> None:
    original, result = _apply_for_tool("mcp__Headroom__headroom_retrieve_extra")

    assert result.messages[1]["content"] != original or result.tokens_after < result.tokens_before


def test_bare_ccr_tool_name_remains_preserved() -> None:
    original, result = _apply_for_tool(CCR_TOOL_NAME)

    assert result.messages[1]["content"] == original


def _apply_anthropic_for_tool(tool_name: str):
    """Anthropic block shape: tool_use in the assistant turn, tool_result in the user turn."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": tool_name, "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": _big_content()},
            ],
        },
    ]
    tokenizer = Tokenizer(OpenAIProvider().get_token_counter("gpt-4o"), "gpt-4o")
    result = SmartCrusher(config=SmartCrusherConfig(min_tokens_to_crush=0)).apply(
        messages, tokenizer
    )
    return messages[1]["content"][0]["content"], result


@pytest.mark.parametrize(
    "tool_name",
    [
        "mcp__Headroom__headroom_retrieve",
        "mcp_Headroom_headroom_retrieve",
        CCR_TOOL_NAME,
    ],
)
def test_qualified_ccr_tool_result_block_is_preserved(tool_name: str) -> None:
    original, result = _apply_anthropic_for_tool(tool_name)

    assert result.messages[1]["content"][0]["content"] == original
    assert not any("smart" in transform for transform in result.transforms_applied)


@pytest.mark.parametrize(
    "tool_name",
    ["mcp__Headroom__headroom_retrieve", "mcp_Headroom_headroom_retrieve", CCR_TOOL_NAME],
)
def test_mcp_compressor_preserves_qualified_ccr_output(tool_name: str) -> None:
    """`HeadroomMCPCompressor.compress` is the production entry point issue #2656 names.

    It drives `SmartCrusher.apply` with a `role=tool` message, so the guard has to
    hold through that wrapper and not only on a directly built message list.
    """
    from headroom.integrations.mcp.server import HeadroomMCPCompressor

    content = json.dumps({"results": [{"id": i, "value": "x" * 40} for i in range(80)]})
    result = HeadroomMCPCompressor().compress(content, tool_name=tool_name)

    assert result.compressed_content == content


def test_mcp_compressor_still_compresses_a_near_match_name() -> None:
    from headroom.integrations.mcp.server import HeadroomMCPCompressor

    content = json.dumps({"results": [{"id": i, "value": "x" * 40} for i in range(80)]})
    result = HeadroomMCPCompressor().compress(
        content, tool_name="mcp__Headroom__headroom_retrieve_extra"
    )

    assert result.compressed_content != content


def test_near_match_ccr_tool_result_block_still_compresses() -> None:
    original, result = _apply_anthropic_for_tool("mcp__Headroom__headroom_retrieve_extra")

    assert (
        result.messages[1]["content"][0]["content"] != original
        or result.tokens_after < result.tokens_before
    )

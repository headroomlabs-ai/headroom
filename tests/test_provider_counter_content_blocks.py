"""Provider token counters must price every content block, not just ``text``.

Each counter in ``headroom/providers/`` had grown its own shortened content-block
walker handling only the shapes its provider was expected to send. Everything else
fell through and contributed nothing. Measured on one 6,800-char block
(``count_messages`` of a single-block message, so 7-8 is message overhead alone):

    block type          OpenAI ctr   Anthropic ctr
    text (control)            3409            3748
    tool_result                  8            3748
    thinking                     8               7
    document                     8               7
    mcp_tool_result              8               7
    output_text                  8               7
    refusal                      8               7

Note each counter zeroed blocks from its OWN provider — ``output_text`` and
``refusal`` are OpenAI Responses shapes, ``thinking`` and ``document`` are
Anthropic's. And these counters are what the LIVE proxy pipelines use
(``proxy/server.py`` builds them with ``AnthropicProvider`` / ``OpenAIProvider``),
so this was the main request path, not an edge case.

The counters now delegate to ``count_content_blocks``, which reuses
``BaseTokenizer._count_content_parts``. That matters beyond coverage: a naive
``count_text(str(block))`` catch-all would serialize a base64 image and price it
as text — a 1MB screenshot reads as ~330K phantom tokens. The shared walker gives
media a pixel/byte-based estimate instead.
"""

from __future__ import annotations

import pytest

from headroom.providers.anthropic import AnthropicProvider
from headroom.providers.openai import OpenAIProvider
from headroom.tokenizers.base import count_content_blocks

_BIG = "x " * 3400  # ~6,800 chars


def _counters():
    return {
        "openai": OpenAIProvider().get_token_counter("gpt-4o"),
        "anthropic": AnthropicProvider(warn=False).get_token_counter("claude-sonnet-4-6"),
    }


@pytest.mark.parametrize(
    "block",
    [
        {"type": "tool_result", "tool_use_id": "t", "content": _BIG},
        {"type": "tool_use", "id": "t", "name": "grep", "input": {"pattern": _BIG}},
        {"type": "thinking", "thinking": _BIG},
        {"type": "document", "source": {"data": _BIG}},
        {"type": "mcp_tool_result", "content": _BIG},
        {"type": "output_text", "text": _BIG},
        {"type": "refusal", "refusal": _BIG},
        {"type": "search_result", "content": _BIG},
    ],
    ids=lambda b: str(b.get("type")),
)
def test_no_provider_counter_prices_a_large_block_at_zero(block: dict) -> None:
    """Every one of these returned 7-8 tokens — message overhead only."""
    message = {"role": "user", "content": [block]}
    for name, counter in _counters().items():
        got = counter.count_messages([message])
        assert got > 1_000, f"{name} priced a ~6,800-char {block['type']} block at {got}"


def test_base64_media_is_not_priced_as_text() -> None:
    """The reason a str(block) catch-all would have been the wrong fix."""
    image = {"type": "image", "source": {"type": "base64", "data": "A" * 200_000}}
    message = {"role": "user", "content": [image]}

    for name, counter in _counters().items():
        got = counter.count_messages([message])
        # 200KB of base64 as text would be ~50,000 tokens; the pixel estimate is 1600.
        assert got < 5_000, f"{name} priced a 200KB base64 image as text: {got}"
        assert got > 1_000, f"{name} ignored a declared image entirely: {got}"


def test_plain_text_blocks_are_unchanged() -> None:
    """The control: the shape both counters already handled must not move."""
    message = {"role": "user", "content": [{"type": "text", "text": _BIG}]}
    for name, counter in _counters().items():
        text_form = {"role": "user", "content": _BIG}
        block_form = counter.count_messages([message])
        # A text block and the equivalent string should agree closely.
        assert abs(block_form - counter.count_messages([text_form])) <= 5, name


def test_shared_walker_ignores_non_block_parts() -> None:
    """A bare int is not a block and must contribute nothing."""
    assert count_content_blocks([123], len) == 0
    assert count_content_blocks([], len) == 0


def test_shared_walker_counts_nested_tool_result_blocks() -> None:
    """A tool that returns blocks nests them; they must be walked, not serialized."""
    nested = {
        "type": "tool_result",
        "tool_use_id": "t",
        "content": [{"type": "text", "text": _BIG}],
    }
    flat = {"type": "tool_result", "tool_use_id": "t", "content": _BIG}
    assert abs(count_content_blocks([nested], len) - count_content_blocks([flat], len)) < 100

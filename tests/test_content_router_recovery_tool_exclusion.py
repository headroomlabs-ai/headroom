"""Regression tests: other tools' recovery output passes through byte-for-byte (#4010).

``caveman_retrieve`` returns the stored original of something Caveman compressed
upstream of Headroom. Compressing that result again hands the model a lossy copy
of the "complete original" plus a fresh retrieve marker, so recovery never
finishes in one call.

The payload is indented JSON on purpose: a name that is only in
``DEFAULT_EXCLUDE_TOOLS`` still gets the excluded-tool lossless fold, which
minifies it. Only ``DEFAULT_VERBATIM_EXCLUDE_TOOLS`` keeps it byte-exact.
"""

from __future__ import annotations

import json

import pytest

from headroom.config import DEFAULT_VERBATIM_EXCLUDE_TOOLS, is_tool_excluded
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


def _get_tokenizer():
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    return Tokenizer(token_counter, "gpt-4o")


def _big_indented_json() -> str:
    return json.dumps(
        [{"id": i, "value": "x" * 20, "active": i % 2 == 0} for i in range(60)], indent=2
    )


@pytest.mark.parametrize("tool_name", ["caveman_retrieve", "mcp__caveman__caveman_retrieve"])
def test_anthropic_recovery_result_is_verbatim(tool_name: str) -> None:
    content = _big_indented_json()
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_rec_1",
                    "name": tool_name,
                    "input": {"handle": "ccr://abc123"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_rec_1", "content": content}],
        },
    ]

    result = ContentRouter(ContentRouterConfig()).apply(messages, _get_tokenizer())

    out = result.messages[1]["content"][0]["content"]
    assert out == content
    assert "hash=" not in out and "<<ccr:" not in out


def test_openai_recovery_result_is_verbatim() -> None:
    content = _big_indented_json()
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_rec_1",
                    "type": "function",
                    "function": {
                        "name": "caveman_retrieve",
                        "arguments": '{"handle":"ccr://abc123"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_rec_1", "content": content},
    ]

    result = ContentRouter(ContentRouterConfig()).apply(messages, _get_tokenizer())

    assert result.messages[1]["content"] == content


def test_recovery_tool_in_verbatim_set() -> None:
    # The Responses API guard and cross-turn dedup key off this set directly.
    assert is_tool_excluded("caveman_retrieve", DEFAULT_VERBATIM_EXCLUDE_TOOLS)
    assert is_tool_excluded("mcp__caveman__caveman_retrieve", DEFAULT_VERBATIM_EXCLUDE_TOOLS)

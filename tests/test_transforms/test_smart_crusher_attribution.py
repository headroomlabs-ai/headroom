"""Tool-name attribution on the ``smart_crush`` transform tag.

When `SmartCrusher` crushes tool outputs it enriches the
``smart_crush:<count>`` tag with the names of the tools whose output was
crushed: ``smart_crush:<count>:<name1,name2,...>``. Names are resolved
from the assistant's ``tool_calls`` (OpenAI) / ``tool_use`` blocks
(Anthropic). When no name resolves, the tag falls back to the legacy
count-only shape.
"""

from __future__ import annotations

import json

import pytest

from headroom import OpenAIProvider, Tokenizer


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


def get_tokenizer(model: str = "gpt-4o") -> Tokenizer:
    token_counter = _provider.get_token_counter(model)
    return Tokenizer(token_counter, model)


def _make_crusher():
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    return SmartCrusher(SmartCrusherConfig(min_tokens_to_crush=10))


# Uniform / tabular payloads the Rust crusher reliably compacts.
_LARGE_A = {"items": [{"id": i, "v": "x" * 10} for i in range(40)]}
_LARGE_B = {"rows": list(range(200))}


class TestSmartCrushAttribution:
    def test_transform_tag_includes_tool_names_openai(self):
        """Tag shape is ``smart_crush:<count>:<name1,name2>`` for OpenAI format."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "Grep", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(_LARGE_A)},
            {"role": "tool", "tool_call_id": "c2", "content": json.dumps(_LARGE_B)},
        ]

        result = _make_crusher().apply(messages, get_tokenizer())

        tags = [t for t in result.transforms_applied if t.startswith("smart_crush:")]
        assert len(tags) == 1
        parts = tags[0].split(":", 2)
        assert parts[0] == "smart_crush"
        assert parts[1] == "2"
        # Order follows first-crushed-first.
        assert parts[2] == "Bash,Grep"

    def test_transform_tag_includes_tool_names_anthropic(self):
        """Anthropic tool_use blocks feed the tool-name index."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "u1", "name": "Read", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "u1", "content": json.dumps(_LARGE_A)},
                ],
            },
        ]

        result = _make_crusher().apply(messages, get_tokenizer())

        assert "smart_crush:1:Read" in result.transforms_applied

    def test_transform_tag_dedupes_repeated_tool(self):
        """Same tool crushed twice shows once in the tag."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(_LARGE_A)},
            {"role": "tool", "tool_call_id": "c2", "content": json.dumps(_LARGE_A)},
        ]

        result = _make_crusher().apply(messages, get_tokenizer())

        assert "smart_crush:2:Bash" in result.transforms_applied

    def test_tool_name_index_skips_entries_missing_id_or_name(self):
        """tool_calls / tool_use blocks missing id or name are skipped, other
        blocks (text, etc.) are skipped, and the tag still reflects the entries
        that DO have both."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "NamelessRead"},  # no id → skipped
                    {"type": "tool_use", "id": "u0"},  # no name → skipped
                    {"type": "text", "text": "thinking..."},  # not tool_use → skipped
                    {"type": "tool_use", "id": "u1", "name": "Grep", "input": {}},  # good
                ],
                "tool_calls": [
                    {"id": "", "function": {"name": "Empty"}},  # no id → skipped
                    {"id": "c1", "function": {"name": ""}},  # no name → skipped
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "u1", "content": json.dumps(_LARGE_A)},
                ],
            },
        ]

        result = _make_crusher().apply(messages, get_tokenizer())

        assert "smart_crush:1:Grep" in result.transforms_applied

    def test_transform_tag_falls_back_when_no_names(self):
        """Crushed tool with no resolvable name keeps legacy ``smart_crush:<n>`` shape."""
        # No assistant message → no name index entries.
        messages = [
            {"role": "tool", "tool_call_id": "orphan", "content": json.dumps(_LARGE_A)},
        ]

        result = _make_crusher().apply(messages, get_tokenizer())

        assert "smart_crush:1" in result.transforms_applied

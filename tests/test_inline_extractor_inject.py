"""Tests for inline_extractor.inject_memory_instruction system-prompt handling.

The system prompt's ``content`` can be a plain string or a list of content-part
dicts (OpenAI allows a list; Anthropic system prompts are commonly a list of
``{"type": "text", ...}`` blocks). The instruction must be appended without
crashing on the list form.
"""

from __future__ import annotations

from headroom.memory.inline_extractor import (
    MEMORY_INSTRUCTION,
    MEMORY_INSTRUCTION_SHORT,
    inject_memory_instruction,
)


def test_str_system_content_is_concatenated() -> None:
    out = inject_memory_instruction(
        [{"role": "system", "content": "You are X."}, {"role": "user", "content": "hi"}],
        short=True,
    )
    assert out[0]["content"] == "You are X." + MEMORY_INSTRUCTION_SHORT


def test_list_system_content_appends_a_text_part() -> None:
    """Regression: a list-shaped system content used to raise
    ``TypeError: can only concatenate list (not "str") to list``."""
    out = inject_memory_instruction(
        [
            {"role": "system", "content": [{"type": "text", "text": "You are X."}]},
            {"role": "user", "content": "hi"},
        ],
        short=True,
    )
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "You are X."}
    assert content[-1] == {"type": "text", "text": MEMORY_INSTRUCTION_SHORT}
    # The original list is not mutated in place.
    assert len(content) == 2


def test_list_system_content_long_instruction() -> None:
    out = inject_memory_instruction(
        [{"role": "system", "content": [{"type": "text", "text": "sys"}]}],
        short=False,
    )
    assert out[0]["content"][-1] == {"type": "text", "text": MEMORY_INSTRUCTION}


def test_missing_system_prepends_default() -> None:
    out = inject_memory_instruction([{"role": "user", "content": "hi"}], short=True)
    assert out[0]["role"] == "system"
    assert MEMORY_INSTRUCTION_SHORT in out[0]["content"]


def test_original_messages_not_mutated() -> None:
    original = [{"role": "system", "content": [{"type": "text", "text": "sys"}]}]
    inject_memory_instruction(original, short=True)
    assert original[0]["content"] == [{"type": "text", "text": "sys"}]

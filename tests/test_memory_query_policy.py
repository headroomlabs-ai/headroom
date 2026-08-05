"""Tests for pure memory query construction policy."""

from __future__ import annotations

from headroom.proxy.memory_query_policy import (
    extract_memory_query_sources,
    render_embedding_input,
)


def test_render_embedding_input_orders_sources_for_embedding() -> None:
    rendered = render_embedding_input(
        user_text="latest user",
        recent_tool_outputs=("tool output",),
        recent_assistant_turns=("assistant context",),
    )

    assert rendered.index("assistant context") < rendered.index("tool output")
    assert rendered.index("tool output") < rendered.index("latest user")


def test_extract_sources_uses_latest_user_and_recent_context_in_order() -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
        {"role": "tool", "content": "t2"},
        {"role": "user", "content": "second"},
    ]

    user_text, tool_outputs, assistant_turns = extract_memory_query_sources(
        messages,
        lookback_assistant=2,
        lookback_tools=2,
    )

    assert user_text == "second"
    assert tool_outputs == ("t1", "t2")
    assert assistant_turns == ("a1", "a2")


def test_extract_sources_handles_anthropic_tool_result_without_user_text() -> None:
    messages = [
        {"role": "user", "content": "real user"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "content": [{"type": "text", "text": "nested"}]}],
        },
    ]

    user_text, tool_outputs, assistant_turns = extract_memory_query_sources(messages)

    assert user_text == "real user"
    assert tool_outputs == ("nested",)
    assert assistant_turns == ()


def test_extract_sources_captures_anthropic_user_text_blocks() -> None:
    """Anthropic user turns carry the prompt as text blocks (the standard Claude
    Code shape). The user's question must be captured — not dropped — so memory
    retrieval keys on it."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "help me refactor auth"}]},
    ]

    user_text, _tool_outputs, _assistant_turns = extract_memory_query_sources(messages)

    assert user_text == "help me refactor auth"


def test_extract_sources_captures_user_text_alongside_tool_result() -> None:
    """A user turn mixing a tool_result and a text block yields both: the text as
    the user query and the tool output as context."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": "exit 0"},
                {"type": "text", "text": "did the tests pass?"},
            ],
        },
    ]

    user_text, tool_outputs, _assistant_turns = extract_memory_query_sources(messages)

    assert user_text == "did the tests pass?"
    assert tool_outputs == ("exit 0",)

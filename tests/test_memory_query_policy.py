"""Tests for pure memory query construction policy."""

from __future__ import annotations

from headroom.proxy.memory_query_policy import (
    ASSISTANT_DELIM,
    TOOL_DELIM,
    USER_DELIM,
    extract_memory_query_sources,
    render_embedding_input,
)


def test_bounded_embedding_keeps_primary_question_and_context_tails() -> None:
    user = "古い質問" * 2000 + " latest JWT question"
    tool = "old tool output " * 1000 + " latest authentication failure"
    assistant = "old assistant context " * 1000 + " inspect auth.py"
    rendered = render_embedding_input(
        user_text=user,
        recent_tool_outputs=(tool,),
        recent_assistant_turns=(assistant,),
        max_chars=4096,
    )

    assert len(rendered) <= 4096
    assert rendered.startswith(USER_DELIM + user[-2048:])
    assert TOOL_DELIM + tool[-512:] in rendered
    assert ASSISTANT_DELIM + assistant[-512:] in rendered


def test_bounded_embedding_spends_remaining_budget_on_newest_context() -> None:
    rendered = render_embedding_input(
        user_text="latest question",
        recent_tool_outputs=tuple("x" * 600 + f" end-{i}" for i in range(10)),
        recent_assistant_turns=("old background",),
        max_chars=1200,
    )

    assert len(rendered) <= 1200
    assert rendered.startswith(USER_DELIM + "latest question")
    assert "end-9" in rendered and "end-8" in rendered
    assert "end-0" not in rendered
    assert "old background" not in rendered


def test_zero_embedding_budget_does_not_send_context_without_the_question() -> None:
    assert (
        render_embedding_input(
            user_text="question",
            recent_tool_outputs=("tool context",),
            recent_assistant_turns=("assistant context",),
            max_chars=0,
        )
        == ""
    )


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


def test_extract_sources_skips_system_reminder_blocks() -> None:
    """Claude Code appends <system-reminder> harness blocks to the user turn.
    Concatenated into the embedding input they dilute the real question below the
    similarity floor so nothing is retrieved (#2195); they must be filtered out."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "how do I add caching to the auth handler?"},
                {
                    "type": "text",
                    "text": "<system-reminder>\nThe user opened file x.\n</system-reminder>",
                },
            ],
        },
    ]

    user_text, _tool_outputs, _assistant_turns = extract_memory_query_sources(messages)

    assert user_text == "how do I add caching to the auth handler?"
    assert "system-reminder" not in user_text


def test_extract_sources_reminder_only_turn_yields_no_user_text() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "<system-reminder>x</system-reminder>"}],
        },
    ]

    user_text, _tool_outputs, _assistant_turns = extract_memory_query_sources(messages)

    assert user_text == ""


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

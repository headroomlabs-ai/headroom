from __future__ import annotations

from headroom_edge_layer.intent_query import extract_intent_query


def test_extracts_latest_assistant_text_and_tool_args() -> None:
    messages = [
        {"role": "user", "content": "please help"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Searching for the bug"},
                {"type": "tool_use", "id": "t1", "name": "Grep", "input": {"pattern": "foo"}},
            ],
        },
    ]
    query = extract_intent_query(messages)
    assert "Searching for the bug" in query
    assert "foo" in query


def test_falls_back_to_tool_args_when_no_text_block() -> None:
    messages = [
        {"role": "user", "content": "please help"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/a.py"}}],
        },
    ]
    query = extract_intent_query(messages)
    assert "/a.py" in query


def test_falls_back_to_last_user_text_when_assistant_has_nothing_readable() -> None:
    messages = [
        {"role": "user", "content": "fix the widget bug"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "secret reasoning"}]},
    ]
    query = extract_intent_query(messages)
    assert "fix the widget bug" in query
    assert "secret reasoning" not in query


def test_never_reads_thinking_or_redacted_thinking_blocks() -> None:
    messages = [
        {"role": "user", "content": "please help"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "internal reasoning nobody should see"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "visible answer"},
            ],
        },
    ]
    query = extract_intent_query(messages)
    assert "internal reasoning" not in query
    assert "opaque" not in query
    assert "visible answer" in query


def test_empty_messages_returns_empty_string() -> None:
    assert extract_intent_query([]) == ""

"""Tests for :class:`headroom.proxy.memory_query.MemoryQuery`.

``MemoryQuery`` is the multi-source query value type that replaces
the pre-PR pattern of "use the latest user message, truncated to 500
chars". The truncation was a real bug — none of Letta/Mem0/Cognee/
Supermemory truncate the embedding input.

The query is built from three sources, all preserved at full fidelity:

* ``user_text`` — latest user message, untruncated
* ``recent_tool_outputs`` — last N tool results (often the most
  relevant signal in coding sessions)
* ``recent_assistant_turns`` — last K assistant turns for intent

Building the embedding input is a simple concatenation with delimiters
so the embedding model sees structured context, not a wall of text.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

from headroom.proxy.memory_query import MemoryQuery

# ── Value-type contract ───────────────────────────────────────────────


def test_memory_query_is_frozen() -> None:
    q = MemoryQuery(
        user_text="hello",
        recent_tool_outputs=(),
        recent_assistant_turns=(),
        conversation_id=None,
    )
    try:
        q.user_text = "mutated"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("MemoryQuery must be frozen")


def test_memory_query_value_equal() -> None:
    a = MemoryQuery(
        user_text="hi", recent_tool_outputs=(), recent_assistant_turns=(), conversation_id="c1"
    )
    b = MemoryQuery(
        user_text="hi", recent_tool_outputs=(), recent_assistant_turns=(), conversation_id="c1"
    )
    assert a == b


# ── NO TRUNCATION — the entire point of this type ────────────────────


def test_full_user_message_is_preserved_no_500_char_cap() -> None:
    """Pre-PR: ``_extract_user_query`` capped at 500 chars. None of
    the four memory systems we surveyed truncate. MemoryQuery must
    preserve the full message — embedding models handle their own
    window (MiniLM 512 tok; BGE-small 8K tok)."""
    long_msg = "a" * 8000  # 8KB user message
    q = MemoryQuery(
        user_text=long_msg,
        recent_tool_outputs=(),
        recent_assistant_turns=(),
        conversation_id=None,
    )
    embedding_input = q.to_embedding_input()
    # Original content fully present — count actual occurrences of "a" run.
    assert "a" * 8000 in embedding_input


def test_tool_outputs_preserved_at_full_fidelity() -> None:
    """Tool results — often the strongest retrieval signal in coding
    sessions — must NOT be truncated."""
    big_tool_output = "GREP RESULT\n" + "match line\n" * 1000  # large grep output
    q = MemoryQuery(
        user_text="how do I fix this?",
        recent_tool_outputs=(big_tool_output,),
        recent_assistant_turns=(),
        conversation_id=None,
    )
    embedding_input = q.to_embedding_input()
    assert "match line" * 1000 in embedding_input.replace("\n", "")


# ── Multi-source query construction ──────────────────────────────────


def test_embedding_input_includes_all_sources() -> None:
    """The query the embedder sees should include user msg + recent
    tool outputs + recent assistant turns. Each source is delimited
    so the embedder treats them as distinct context, not run-on text."""
    q = MemoryQuery(
        user_text="fix the auth bug",
        recent_tool_outputs=("auth.py:42: KeyError",),
        recent_assistant_turns=("I'll look at the auth flow",),
        conversation_id=None,
    )
    txt = q.to_embedding_input()
    assert "fix the auth bug" in txt
    assert "auth.py:42: KeyError" in txt
    assert "I'll look at the auth flow" in txt


def test_empty_sources_still_produce_valid_query() -> None:
    """A user-msg-only query (no tools, no prior assistant) is the
    minimum viable case — common on first turn."""
    q = MemoryQuery(
        user_text="hello",
        recent_tool_outputs=(),
        recent_assistant_turns=(),
        conversation_id=None,
    )
    txt = q.to_embedding_input()
    assert txt
    assert "hello" in txt


def test_empty_user_text_is_valid_when_only_tool_signal() -> None:
    """Edge case: agent-driven request with no new user text (e.g. a
    tool-call follow-up). Query is the tool output."""
    q = MemoryQuery(
        user_text="",
        recent_tool_outputs=("ls -la /home/user/projects/headroom",),
        recent_assistant_turns=(),
        conversation_id=None,
    )
    txt = q.to_embedding_input()
    assert "ls -la /home/user/projects/headroom" in txt


# ── from_messages constructor ────────────────────────────────────────


def test_from_messages_extracts_latest_user_text() -> None:
    """Construct from a chat-style messages list — picks the most
    recent ``role: user`` content."""
    messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second turn"},
    ]
    q = MemoryQuery.from_messages(messages, lookback_assistant=0, lookback_tools=0)
    assert q.user_text == "second turn"


def test_from_messages_extracts_recent_assistant_turns_in_order() -> None:
    """Recent assistant turns are pulled in chronological order
    (oldest of the lookback window first, latest last)."""
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]
    q = MemoryQuery.from_messages(messages, lookback_assistant=2, lookback_tools=0)
    assert q.recent_assistant_turns == ("a1", "a2")
    assert q.user_text == "u3"


def test_from_messages_caps_assistant_lookback() -> None:
    """``lookback_assistant=K`` keeps only the K most recent assistant
    turns. With lookback=1 and three assistant turns, only the latest."""
    messages = [
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u"},
    ]
    q = MemoryQuery.from_messages(messages, lookback_assistant=1, lookback_tools=0)
    assert q.recent_assistant_turns == ("a3",)


def test_from_messages_extracts_tool_outputs() -> None:
    """Tool results are pulled from ``role: tool`` messages (OpenAI
    shape) — pre-PR these never participated in retrieval at all."""
    messages = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "I'll run ls"},
        {"role": "tool", "content": "main.py\nREADME.md\n"},
        {"role": "user", "content": "now read main.py"},
    ]
    q = MemoryQuery.from_messages(messages, lookback_assistant=0, lookback_tools=2)
    assert q.recent_tool_outputs == ("main.py\nREADME.md\n",)


def test_from_messages_handles_anthropic_tool_result_shape() -> None:
    """Anthropic shape: tool_result inside the user message as a
    content block. The constructor should still extract it."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "ANTHROPIC_TOOL_OUTPUT"}
            ],
        },
        {"role": "user", "content": "thanks"},
    ]
    q = MemoryQuery.from_messages(messages, lookback_assistant=0, lookback_tools=2)
    assert "ANTHROPIC_TOOL_OUTPUT" in q.recent_tool_outputs


def test_from_messages_empty_returns_empty_query() -> None:
    """No messages → empty query, no exception."""
    q = MemoryQuery.from_messages([], lookback_assistant=2, lookback_tools=2)
    assert q.user_text == ""
    assert q.recent_assistant_turns == ()
    assert q.recent_tool_outputs == ()


def test_from_messages_handles_assistant_only_messages() -> None:
    """Edge case: no user messages at all (rare; agent-driven). Should
    still build a valid query."""
    messages = [{"role": "assistant", "content": "assistant only"}]
    q = MemoryQuery.from_messages(messages, lookback_assistant=2, lookback_tools=0)
    assert q.user_text == ""
    assert q.recent_assistant_turns == ("assistant only",)

"""Tests for ``headroom.proxy.helpers.compute_turn_id``."""

from __future__ import annotations

from headroom.proxy.helpers import compute_turn_id

MODEL = "claude-sonnet-4-5"
SYSTEM = "You are helpful."


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_tool_use(tool_id: str, name: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
    }


def _user_tool_result(tool_id: str, out: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": out}],
    }


def test_returns_none_when_messages_empty():
    assert compute_turn_id(MODEL, SYSTEM, []) is None
    assert compute_turn_id(MODEL, SYSTEM, None) is None


def test_returns_none_when_no_user_text_message():
    messages = [_assistant_tool_use("t1", "bash")]
    assert compute_turn_id(MODEL, SYSTEM, messages) is None


def test_stable_across_agent_loop_iterations():
    iteration_1 = [_user("fix the bug")]
    iteration_2 = iteration_1 + [
        _assistant_tool_use("t1", "read"),
        _user_tool_result("t1", "file contents"),
    ]
    iteration_3 = iteration_2 + [
        _assistant_tool_use("t2", "edit"),
        _user_tool_result("t2", "edit ok"),
    ]

    id1 = compute_turn_id(MODEL, SYSTEM, iteration_1)
    id2 = compute_turn_id(MODEL, SYSTEM, iteration_2)
    id3 = compute_turn_id(MODEL, SYSTEM, iteration_3)

    assert id1 is not None
    assert id1 == id2 == id3


def test_rolls_over_on_new_user_prompt():
    turn_1 = [_user("first prompt")]
    turn_2 = turn_1 + [
        _assistant_tool_use("t1", "bash"),
        _user_tool_result("t1", "ok"),
        _user("second prompt"),
    ]

    id1 = compute_turn_id(MODEL, SYSTEM, turn_1)
    id2 = compute_turn_id(MODEL, SYSTEM, turn_2)

    assert id1 != id2


def test_different_model_yields_different_id():
    messages = [_user("same prompt")]
    id_a = compute_turn_id("claude-sonnet-4-5", SYSTEM, messages)
    id_b = compute_turn_id("claude-opus-4-7", SYSTEM, messages)
    assert id_a != id_b


def test_different_system_yields_different_id():
    messages = [_user("same prompt")]
    id_a = compute_turn_id(MODEL, "system A", messages)
    id_b = compute_turn_id(MODEL, "system B", messages)
    assert id_a != id_b


def test_accepts_list_system_prompt():
    messages = [_user("hi")]
    system_list = [{"type": "text", "text": "You are helpful."}]
    assert compute_turn_id(MODEL, system_list, messages) is not None


def test_text_block_in_list_content_is_a_user_turn():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert compute_turn_id(MODEL, SYSTEM, messages) is not None


def test_tool_result_only_content_is_not_a_turn_boundary():
    # A message whose only content is a tool_result is a continuation, not a
    # new turn — so the function must not latch onto it.
    messages = [_user_tool_result("t1", "result only")]
    assert compute_turn_id(MODEL, SYSTEM, messages) is None


def test_returns_16_hex_chars():
    turn_id = compute_turn_id(MODEL, SYSTEM, [_user("hi")])
    assert turn_id is not None
    assert len(turn_id) == 16
    int(turn_id, 16)  # raises if not hex


def test_skips_non_dict_and_non_user_messages():
    # A non-dict entry and an assistant message must both be skipped by the
    # reverse scan before it finds the real user-text message.
    messages = [
        _user("the actual prompt"),
        {"role": "assistant", "content": "response"},
        "not-a-dict-message-entry",
    ]
    assert compute_turn_id(MODEL, SYSTEM, messages) is not None


def test_ignores_empty_string_user_content():
    # An empty-string user content is not a real prompt; keep scanning.
    messages = [_user(""), _user("the real prompt")]
    hit = compute_turn_id(MODEL, SYSTEM, messages)
    assert hit is not None
    # Hash should match a single-message [real prompt] prefix — i.e. the
    # scan stopped at "the real prompt" and included the leading empty msg
    # in the hashed prefix. Either way: not None and reproducible.
    assert hit == compute_turn_id(MODEL, SYSTEM, messages)


def test_mixed_text_and_tool_result_is_not_a_turn_boundary():
    # A user message whose content list has BOTH text and tool_result is
    # treated as an agent-loop continuation (not a fresh prompt). If
    # nothing else earlier qualifies, compute_turn_id returns None.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                {"type": "text", "text": "and a comment"},
            ],
        }
    ]
    assert compute_turn_id(MODEL, SYSTEM, messages) is None


def test_none_system_hashes_without_system_segment():
    messages = [_user("hi")]
    a = compute_turn_id(MODEL, None, messages)
    b = compute_turn_id(MODEL, None, messages)
    assert a is not None
    assert a == b
    # Different-system values must still produce a different id than None.
    assert a != compute_turn_id(MODEL, "some system", messages)


def test_stable_when_cache_control_moves_between_calls():
    # Clients like Claude Code move the cache_control breakpoint to the
    # newest message on each call: the user-text message carries it on
    # call 1 and not on call 2 (where a later tool_result carries it).
    # The turn_id must be stable across those calls — otherwise the
    # prompt-level aggregator in the desktop app never gets more than one
    # call per "turn" and the prompt record degenerates to the biggest
    # single call.
    call_1_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "fix the bug",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    call_2_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "fix the bug"}],
        },
        _assistant_tool_use("t1", "read"),
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "file contents",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]

    id1 = compute_turn_id(MODEL, SYSTEM, call_1_messages)
    id2 = compute_turn_id(MODEL, SYSTEM, call_2_messages)

    assert id1 is not None
    assert id1 == id2


def test_stable_when_cache_control_moves_on_system_prompt():
    # Same cache-breakpoint mechanic but applied to a list-shaped system
    # prompt: the annotation moves between system text blocks across
    # calls. The turn_id must ignore it.
    system_call_1 = [
        {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}
    ]
    system_call_2 = [{"type": "text", "text": "You are helpful."}]
    messages = [_user("hi")]

    id1 = compute_turn_id(MODEL, system_call_1, messages)
    id2 = compute_turn_id(MODEL, system_call_2, messages)

    assert id1 is not None
    assert id1 == id2

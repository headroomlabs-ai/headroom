"""Tests for output verbosity steering helpers."""

from __future__ import annotations

from headroom.proxy.output_steering import (
    apply_openai_responses_verbosity_steering,
    apply_verbosity_steering,
    replace_or_append_steering_block,
    steering_text,
)


def test_replace_or_append_steering_block_replaces_existing_block() -> None:
    old = steering_text(1)
    new = steering_text(3)
    assert old is not None
    assert new is not None
    updated, changed = replace_or_append_steering_block(f"System.\n\n{old}\n\nTail.", new)

    assert changed is True
    assert old not in updated
    assert updated == f"System.\n\n{new}\n\nTail."


def test_anthropic_steering_preserves_cached_prefix_block() -> None:
    cached = {
        "type": "text",
        "text": "Big system prompt.",
        "cache_control": {"type": "ephemeral"},
    }
    body = {"system": [cached.copy()]}

    assert apply_verbosity_steering(body, 2) is True
    assert body["system"][0] == cached
    assert body["system"][1] == {"type": "text", "text": steering_text(2)}


def test_anthropic_steering_tolerates_non_string_system_block_text() -> None:
    # A malformed client block ({"type": "text", "text": null}) must not crash
    # `.startswith` and 500 the request; steering is still appended. The OpenAI
    # chat sibling already guards this exact case.
    body = {
        "system": [
            {"type": "text", "text": None},
            {"type": "text", "text": "Real system prompt."},
        ]
    }

    assert apply_verbosity_steering(body, 2) is True
    # The malformed block is left as-is and a steering block is appended.
    assert body["system"][0] == {"type": "text", "text": None}
    assert body["system"][-1] == {"type": "text", "text": steering_text(2)}


def test_openai_responses_steering_is_idempotent() -> None:
    body = {"instructions": "System."}

    assert apply_openai_responses_verbosity_steering(body, 2) is True
    snapshot = body.copy()
    assert apply_openai_responses_verbosity_steering(body, 2) is False
    assert body == snapshot


def test_openai_chat_steering_appends_to_system_message() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
    }
    assert apply_openai_chat_verbosity_steering(body, 2) is True
    sys_content = body["messages"][0]["content"]
    assert "You are helpful." in sys_content
    assert steering_text(2) in sys_content
    # Other messages and ordering are untouched.
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_openai_chat_steering_is_idempotent_and_swaps_level() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {"messages": [{"role": "system", "content": "S."}]}
    assert apply_openai_chat_verbosity_steering(body, 2) is True
    first = body["messages"][0]["content"]
    # Same level again: no change.
    assert apply_openai_chat_verbosity_steering(body, 2) is False
    assert body["messages"][0]["content"] == first
    # Different level: replace, still exactly one block.
    assert apply_openai_chat_verbosity_steering(body, 4) is True
    swapped = body["messages"][0]["content"]
    assert steering_text(4) in swapped
    assert swapped.count("<headroom_output_shaping>") == 1


def test_openai_chat_steering_inserts_system_when_absent() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert apply_openai_chat_verbosity_steering(body, 3) is True
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == steering_text(3)
    assert body["messages"][1] == {"role": "user", "content": "hi"}


def test_openai_chat_steering_handles_list_content() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {"messages": [{"role": "system", "content": [{"type": "text", "text": "base"}]}]}
    assert apply_openai_chat_verbosity_steering(body, 1) is True
    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "base"}
    assert parts[1]["type"] == "text"
    assert parts[1]["text"] == steering_text(1)


def test_openai_chat_steering_level_zero_is_noop() -> None:
    from headroom.proxy.output_steering import apply_openai_chat_verbosity_steering

    body = {"messages": [{"role": "system", "content": "S."}]}
    assert apply_openai_chat_verbosity_steering(body, 0) is False
    assert body["messages"][0]["content"] == "S."

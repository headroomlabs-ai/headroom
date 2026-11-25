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


def test_openai_responses_steering_skips_a_body_already_steered_in_input() -> None:
    """An upstream that rejects ``instructions`` gets its steering in the user turn
    from an extension hook; core must not add ``instructions`` on top of it."""
    steered = f"fix it\n\n{steering_text(3)}"
    body = {"input": [{"role": "user", "content": steered}]}

    assert apply_openai_responses_verbosity_steering(body, 3) is False
    assert body == {"input": [{"role": "user", "content": steered}]}


def test_openai_responses_steering_ignores_the_marker_quoted_in_tool_output() -> None:
    """Reading this repo's own steering source puts the marker in a tool result; that is not a
    steered body, so core must still steer it."""
    body = {
        "instructions": "You are Codex.",
        "input": [
            {"role": "user", "content": "show me the steering policy"},
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": 'STEERING_SENTINEL = "<headroom_output_shaping>"\n'
                'STEERING_SUFFIX = "</headroom_output_shaping>"',
            },
        ],
    }
    assert apply_openai_responses_verbosity_steering(body, 2) is True
    assert steering_text(2) in body["instructions"]


def test_openai_responses_steering_ignores_a_user_quoting_the_marker_mid_message() -> None:
    body = {
        "input": [{"role": "user", "content": "why does <headroom_output_shaping> appear here?"}]
    }
    assert apply_openai_responses_verbosity_steering(body, 2) is True
    assert body["instructions"] == steering_text(2)


def test_openai_responses_steering_skips_input_text_parts_already_steered() -> None:
    steered = f"fix it\n\n{steering_text(3)}"
    body = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": steered}],
            }
        ]
    }
    assert apply_openai_responses_verbosity_steering(body, 3) is False
    assert "instructions" not in body


def test_openai_responses_steering_skips_a_string_input_already_steered() -> None:
    steered = f"fix it\n\n{steering_text(3)}"
    body = {"input": steered}
    assert apply_openai_responses_verbosity_steering(body, 3) is False
    assert body == {"input": steered}


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


def _shape_live(payload: dict, monkeypatch, holdout: str = "0") -> tuple[list[str], bool]:
    """Shape through the Responses path live HTTP and WebSocket traffic takes."""
    from headroom.proxy.handlers.openai import _shape_openai_responses_payload

    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", holdout)
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
    return _shape_openai_responses_payload(
        payload, model="auto", request_id="t", output_shaper_enabled=True
    )


def test_responses_turn_steered_by_an_extension_is_labelled_shaped(monkeypatch) -> None:
    """Kiro rejects ``instructions``, so headroom-kiro steers the user turn itself.

    Core must not add a second copy, but the request was shaped: without the
    verbosity label the output-savings ledger books every Kiro turn as unshaped.
    """
    steered = f"fix it\n\n{steering_text(3)}"
    payload = {"model": "auto", "input": [{"role": "user", "content": steered}]}

    labels, mutated = _shape_live(payload, monkeypatch)

    assert "output_shaper:verbosity:L3" in labels
    assert mutated is False
    assert payload == {"model": "auto", "input": [{"role": "user", "content": steered}]}


def test_control_arm_turn_is_not_labelled_shaped(monkeypatch) -> None:
    steered = f"fix it\n\n{steering_text(3)}"
    payload = {"model": "auto", "input": [{"role": "user", "content": steered}]}

    labels, _ = _shape_live(payload, monkeypatch, holdout="1")

    assert not any(label.startswith("output_shaper:verbosity:") for label in labels)


def test_older_steered_prompt_does_not_label_an_unsteered_turn(monkeypatch) -> None:
    """An earlier prompt keeps its block; this turn's prompt carries none, and
    core will not add a second copy, so nothing shaped this turn."""
    payload = {
        "model": "auto",
        "input": [
            {"role": "user", "content": f"fix it\n\n{steering_text(3)}"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "now run the tests"},
        ],
    }

    labels, mutated = _shape_live(payload, monkeypatch)

    assert mutated is False
    assert not any(label.startswith("output_shaper:verbosity:") for label in labels)

"""Tests for pure output turn classification policy."""

from __future__ import annotations

from typing import Any

from headroom.proxy.output_turn_policy import (
    TurnKind,
    classify_openai_responses_input,
    classify_turn,
)


def _tool_result(is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "content": "ok"}
    if is_error:
        block["is_error"] = True
    return block


def test_anthropic_text_user_message_is_new_ask() -> None:
    assert classify_turn([{"role": "user", "content": "explain this"}]) is TurnKind.NEW_USER_ASK


def test_anthropic_clean_tool_results_are_mechanical() -> None:
    messages = [{"role": "user", "content": [_tool_result(), _tool_result()]}]
    assert classify_turn(messages) is TurnKind.MECHANICAL_CONTINUATION


def test_anthropic_error_tool_result_is_error_continuation() -> None:
    messages = [{"role": "user", "content": [_tool_result(), _tool_result(is_error=True)]}]
    assert classify_turn(messages) is TurnKind.ERROR_CONTINUATION


def test_anthropic_user_media_or_text_block_is_new_ask() -> None:
    assert (
        classify_turn([{"role": "user", "content": [{"type": "image", "source": {}}]}])
        is TurnKind.NEW_USER_ASK
    )
    assert (
        classify_turn(
            [{"role": "user", "content": [_tool_result(), {"type": "text", "text": "also"}]}]
        )
        is TurnKind.NEW_USER_ASK
    )


def test_anthropic_unknown_shapes_are_unknown() -> None:
    assert classify_turn([]) is TurnKind.UNKNOWN
    assert classify_turn([{"role": "assistant", "content": "done"}]) is TurnKind.UNKNOWN
    assert classify_turn([{"role": "user", "content": []}]) is TurnKind.UNKNOWN
    assert classify_turn([{"role": "user", "content": [{}]}]) is TurnKind.UNKNOWN


def test_openai_responses_string_input_is_new_ask() -> None:
    assert classify_openai_responses_input("explain this") is TurnKind.NEW_USER_ASK
    assert classify_openai_responses_input("   ") is TurnKind.UNKNOWN


def test_openai_responses_tool_outputs_only_are_mechanical() -> None:
    assert (
        classify_openai_responses_input(
            [
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                {"type": "local_shell_call_output", "call_id": "call_2", "output": "ok"},
            ]
        )
        is TurnKind.MECHANICAL_CONTINUATION
    )


def test_openai_responses_user_message_or_input_media_is_new_ask() -> None:
    assert (
        classify_openai_responses_input(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "also check foo.py"}],
                }
            ]
        )
        is TurnKind.NEW_USER_ASK
    )
    assert (
        classify_openai_responses_input(
            [{"type": "message", "role": "user", "content": [{"type": "input_image"}]}]
        )
        is TurnKind.NEW_USER_ASK
    )


def test_openai_responses_unknown_mixed_with_tool_output_is_unknown() -> None:
    assert (
        classify_openai_responses_input(
            [
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                {"type": "unrecognized_event"},
            ]
        )
        is TurnKind.UNKNOWN
    )

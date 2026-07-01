"""Tests for headroom.proxy.output_shaper.

Covers turn classification (structural only), cache-safe verbosity steering,
effort routing on mechanical continuations, and the env-driven gate.
"""

from __future__ import annotations

import copy
from typing import Any

from headroom.proxy.output_shaper import (
    LEGACY_THINKING_FLOOR,
    OutputShaperSettings,
    TurnKind,
    apply_openai_responses_verbosity_steering,
    apply_verbosity_steering,
    classify_openai_responses_input,
    classify_turn,
    route_effort,
    route_openai_reasoning_effort,
    route_openai_text_verbosity,
    shape_openai_responses_request,
    shape_request,
    steering_text,
)

ENABLED = OutputShaperSettings(enabled=True)


def _tool_result(is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": "toolu_01",
        "content": "ok",
    }
    if is_error:
        block["is_error"] = True
    return block


def _mechanical_messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "fix the bug in foo.py"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading the file."},
                {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {}},
            ],
        },
        {"role": "user", "content": [_tool_result()]},
    ]


# ---------------------------------------------------------------------------
# classify_turn
# ---------------------------------------------------------------------------


class TestClassifyTurn:
    def test_string_user_message_is_new_ask(self):
        assert classify_turn([{"role": "user", "content": "explain this"}]) == TurnKind.NEW_USER_ASK

    def test_clean_tool_result_is_mechanical(self):
        assert classify_turn(_mechanical_messages()) == TurnKind.MECHANICAL_CONTINUATION

    def test_multiple_clean_tool_results_are_mechanical(self):
        msgs = _mechanical_messages()
        msgs[-1]["content"].append(_tool_result())
        assert classify_turn(msgs) == TurnKind.MECHANICAL_CONTINUATION

    def test_error_tool_result_is_error_continuation(self):
        msgs = _mechanical_messages()
        msgs[-1]["content"] = [_tool_result(), _tool_result(is_error=True)]
        assert classify_turn(msgs) == TurnKind.ERROR_CONTINUATION

    def test_text_block_alongside_tool_result_is_new_ask(self):
        msgs = _mechanical_messages()
        msgs[-1]["content"].append({"type": "text", "text": "also check bar.py"})
        assert classify_turn(msgs) == TurnKind.NEW_USER_ASK

    def test_image_block_is_new_ask(self):
        msgs = [{"role": "user", "content": [{"type": "image", "source": {}}]}]
        assert classify_turn(msgs) == TurnKind.NEW_USER_ASK

    def test_assistant_last_is_unknown(self):
        msgs = [{"role": "assistant", "content": "hello"}]
        assert classify_turn(msgs) == TurnKind.UNKNOWN

    def test_empty_messages_is_unknown(self):
        assert classify_turn([]) == TurnKind.UNKNOWN

    def test_empty_content_list_is_unknown(self):
        assert classify_turn([{"role": "user", "content": []}]) == TurnKind.UNKNOWN

    def test_whitespace_string_content_is_unknown(self):
        assert classify_turn([{"role": "user", "content": "  "}]) == TurnKind.UNKNOWN


# ---------------------------------------------------------------------------
# apply_verbosity_steering
# ---------------------------------------------------------------------------


class TestVerbositySteering:
    def test_level_zero_is_noop(self):
        body = {"system": "You are helpful."}
        assert apply_verbosity_steering(body, 0) is False
        assert body["system"] == "You are helpful."

    def test_string_system_converted_to_blocks_with_original_bytes_first(self):
        body = {"system": "You are helpful."}
        assert apply_verbosity_steering(body, 2) is True
        assert body["system"][0] == {"type": "text", "text": "You are helpful."}
        assert body["system"][1]["text"] == steering_text(2)

    def test_missing_system_creates_steering_only_block(self):
        body: dict[str, Any] = {}
        assert apply_verbosity_steering(body, 2) is True
        assert body["system"] == [{"type": "text", "text": steering_text(2)}]

    def test_block_system_appends_after_cache_control(self):
        cached = {
            "type": "text",
            "text": "Big system prompt.",
            "cache_control": {"type": "ephemeral"},
        }
        body = {"system": [copy.deepcopy(cached)]}
        assert apply_verbosity_steering(body, 2) is True
        # The cached block is byte-identical and still first — prefix intact.
        assert body["system"][0] == cached
        assert body["system"][1] == {"type": "text", "text": steering_text(2)}
        # Our block carries no cache_control (breakpoints are a scarce resource).
        assert "cache_control" not in body["system"][1]

    def test_idempotent_at_same_level(self):
        body = {"system": [{"type": "text", "text": "Sys."}]}
        assert apply_verbosity_steering(body, 2) is True
        snapshot = copy.deepcopy(body)
        assert apply_verbosity_steering(body, 2) is False
        assert body == snapshot

    def test_level_change_replaces_block_in_place(self):
        body = {"system": [{"type": "text", "text": "Sys."}]}
        apply_verbosity_steering(body, 2)
        assert apply_verbosity_steering(body, 4) is True
        steering_blocks = [
            b for b in body["system"] if b["text"].startswith("<headroom_output_shaping>")
        ]
        assert len(steering_blocks) == 1
        assert steering_blocks[0]["text"] == steering_text(4)

    def test_steering_text_is_deterministic(self):
        for level in (1, 2, 3, 4):
            assert steering_text(level) == steering_text(level)


# ---------------------------------------------------------------------------
# route_effort
# ---------------------------------------------------------------------------


class TestRouteEffort:
    def test_lowers_explicit_effort_on_mechanical_turn(self):
        body = {"output_config": {"effort": "xhigh"}}
        labels = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert body["output_config"]["effort"] == "low"
        assert labels == ["output_shaper:effort:xhigh->low"]

    def test_never_injects_effort_when_absent(self):
        body: dict[str, Any] = {"messages": []}
        labels = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert "output_config" not in body
        assert labels == []

    def test_effort_untouched_on_new_ask(self):
        body = {"output_config": {"effort": "xhigh"}}
        assert route_effort(body, TurnKind.NEW_USER_ASK, ENABLED) == []
        assert body["output_config"]["effort"] == "xhigh"

    def test_effort_untouched_on_error_continuation(self):
        body = {"output_config": {"effort": "xhigh"}}
        assert route_effort(body, TurnKind.ERROR_CONTINUATION, ENABLED) == []
        assert body["output_config"]["effort"] == "xhigh"

    def test_effort_already_at_target_untouched(self):
        body = {"output_config": {"effort": "low"}}
        assert route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED) == []

    def test_unknown_effort_value_untouched(self):
        body = {"output_config": {"effort": "turbo"}}
        assert route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED) == []
        assert body["output_config"]["effort"] == "turbo"

    def test_configurable_mechanical_effort(self):
        settings = OutputShaperSettings(enabled=True, mechanical_effort="medium")
        body = {"output_config": {"effort": "xhigh"}}
        route_effort(body, TurnKind.MECHANICAL_CONTINUATION, settings)
        assert body["output_config"]["effort"] == "medium"

    def test_legacy_thinking_budget_clamped(self):
        body = {"thinking": {"type": "enabled", "budget_tokens": 32000}}
        labels = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert body["thinking"]["budget_tokens"] == LEGACY_THINKING_FLOOR
        assert body["thinking"]["type"] == "enabled"  # never toggled
        assert labels == [f"output_shaper:thinking_budget:32000->{LEGACY_THINKING_FLOOR}"]

    def test_legacy_budget_at_floor_untouched(self):
        body = {"thinking": {"type": "enabled", "budget_tokens": LEGACY_THINKING_FLOOR}}
        assert route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED) == []

    def test_adaptive_thinking_untouched(self):
        body = {"thinking": {"type": "adaptive"}}
        assert route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED) == []
        assert body["thinking"] == {"type": "adaptive"}


# ---------------------------------------------------------------------------
# shape_request (end to end)
# ---------------------------------------------------------------------------


class TestShapeRequest:
    def test_disabled_is_noop(self):
        body = {
            "system": "Sys.",
            "messages": _mechanical_messages(),
            "output_config": {"effort": "xhigh"},
        }
        snapshot = copy.deepcopy(body)
        result = shape_request(body, OutputShaperSettings(enabled=False))
        assert result.changed is False
        assert body == snapshot

    def test_enabled_applies_steering_and_effort_routing(self):
        body = {
            "system": "Sys.",
            "messages": _mechanical_messages(),
            "output_config": {"effort": "xhigh"},
            "thinking": {"type": "adaptive"},
        }
        result = shape_request(body, ENABLED)
        assert result.changed is True
        assert result.labels == [
            "output_shaper:verbosity:L2",
            "output_shaper:effort:xhigh->low",
        ]
        assert body["output_config"]["effort"] == "low"
        assert body["system"][1]["text"] == steering_text(2)

    def test_new_ask_gets_steering_but_keeps_effort(self):
        body = {
            "system": "Sys.",
            "messages": [{"role": "user", "content": "design a cache layer"}],
            "output_config": {"effort": "xhigh"},
        }
        result = shape_request(body, ENABLED)
        assert result.labels == ["output_shaper:verbosity:L2"]
        assert body["output_config"]["effort"] == "xhigh"

    def test_second_pass_is_stable(self):
        body = {"system": "Sys.", "messages": _mechanical_messages()}
        shape_request(body, ENABLED)
        snapshot = copy.deepcopy(body)
        result = shape_request(body, ENABLED)
        assert result.changed is False
        assert body == snapshot

    def test_from_env_defaults_off(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_OUTPUT_SHAPER", raising=False)
        assert OutputShaperSettings.from_env().enabled is False

    def test_from_env_enabled_with_overrides(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
        monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
        monkeypatch.setenv("HEADROOM_MECHANICAL_EFFORT", "medium")
        settings = OutputShaperSettings.from_env()
        assert settings.enabled is True
        assert settings.verbosity_level == 3
        assert settings.mechanical_effort == "medium"

    def test_from_env_clamps_bad_values(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "true")
        monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "99")
        monkeypatch.setenv("HEADROOM_MECHANICAL_EFFORT", "bogus")
        settings = OutputShaperSettings.from_env()
        assert settings.verbosity_level == 4
        assert settings.mechanical_effort == "low"


class TestOpenAIResponsesClassify:
    def test_string_input_is_new_ask(self):
        assert classify_openai_responses_input("explain this") == TurnKind.NEW_USER_ASK

    def test_function_call_output_only_is_mechanical(self):
        input_data = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            }
        ]
        assert (
            classify_openai_responses_input(input_data)
            == TurnKind.MECHANICAL_CONTINUATION
        )

    def test_mixed_user_message_and_tool_output_is_new_ask(self):
        input_data = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "also check foo.py"}],
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            },
        ]
        assert classify_openai_responses_input(input_data) == TurnKind.NEW_USER_ASK


class TestOpenAIResponsesSteering:
    def test_instructions_steering_is_idempotent_and_replaced(self):
        body = {"instructions": f"System.\n\n{steering_text(1)}"}

        assert apply_openai_responses_verbosity_steering(body, 2) is True
        assert body["instructions"].count("<headroom_output_shaping>") == 1
        assert steering_text(1) not in body["instructions"]
        assert steering_text(2) in body["instructions"]

        snapshot = copy.deepcopy(body)
        assert apply_openai_responses_verbosity_steering(body, 2) is False
        assert body == snapshot


class TestOpenAIResponsesReasoning:
    def test_reasoning_effort_lowers_only_for_mechanical_continuations(self):
        body = {"reasoning": {"effort": "xhigh"}}
        labels = route_openai_reasoning_effort(
            body,
            TurnKind.MECHANICAL_CONTINUATION,
            ENABLED,
        )
        assert labels == ["output_shaper:reasoning_effort:xhigh->low"]
        assert body["reasoning"]["effort"] == "low"

        new_ask = {"reasoning": {"effort": "xhigh"}}
        assert route_openai_reasoning_effort(new_ask, TurnKind.NEW_USER_ASK, ENABLED) == []
        assert new_ask["reasoning"]["effort"] == "xhigh"

    def test_reasoning_effort_is_not_injected_when_absent(self):
        body: dict[str, Any] = {}
        labels = route_openai_reasoning_effort(
            body,
            TurnKind.MECHANICAL_CONTINUATION,
            ENABLED,
        )
        assert labels == []
        assert "reasoning" not in body


class TestOpenAIResponsesTextVerbosity:
    def test_text_verbosity_set_for_gpt5_family(self):
        body = {"model": "gpt-5.1"}
        labels = route_openai_text_verbosity(body)
        assert labels == ["output_shaper:text_verbosity:unset->low"]
        assert body["text"] == {"verbosity": "low"}

    def test_text_verbosity_not_injected_for_non_gpt5(self):
        body = {"model": "gpt-4o"}
        assert route_openai_text_verbosity(body) == []
        assert "text" not in body

    def test_existing_text_verbosity_is_lowered_for_any_model(self):
        body = {"model": "gpt-4o", "text": {"verbosity": "medium"}}
        labels = route_openai_text_verbosity(body)
        assert labels == ["output_shaper:text_verbosity:medium->low"]
        assert body["text"]["verbosity"] == "low"

    def test_shape_openai_responses_combines_steering_native_knobs(self):
        body = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                }
            ],
            "instructions": "System.",
            "reasoning": {"effort": "xhigh"},
            "text": {"verbosity": "medium"},
        }
        result = shape_openai_responses_request(body, ENABLED)

        assert result.changed is True
        assert result.labels == [
            "output_shaper:verbosity:L2",
            "output_shaper:reasoning_effort:xhigh->low",
            "output_shaper:text_verbosity:medium->low",
        ]
        assert steering_text(2) in body["instructions"]
        assert body["reasoning"]["effort"] == "low"
        assert body["text"]["verbosity"] == "low"

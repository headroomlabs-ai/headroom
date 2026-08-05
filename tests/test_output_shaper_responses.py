"""Tests for the OpenAI Responses side of headroom.proxy.output_shaper.

Covers turn classification on the Responses ``input`` item list (structural
only, incl. the JSON-field error sniff on tool outputs), cache-safe verbosity
steering on the ``instructions`` tail, ``reasoning.effort`` routing on
mechanical continuations, the conversation-stable holdout key, and the
handler-level shaping helper used by the /v1/responses HTTP + WS paths.
"""

from __future__ import annotations

import json
from typing import Any

from headroom.proxy.handlers.openai import _shape_openai_responses_payload
from headroom.proxy.output_savings import conversation_key_from_responses_body
from headroom.proxy.output_shaper import (
    OutputShaperSettings,
    TurnKind,
    apply_responses_verbosity_steering,
    classify_responses_turn,
    route_responses_effort,
    shape_responses_request,
    steering_text,
)

ENABLED = OutputShaperSettings(enabled=True)


def _fn_output(output: Any = "ok", item_type: str = "function_call_output") -> dict[str, Any]:
    return {"type": item_type, "call_id": "call_01", "output": output}


def _user_message(text: str = "fix the bug") -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def _mechanical_input() -> list[dict[str, Any]]:
    return [
        _user_message(),
        {"type": "function_call", "call_id": "call_01", "name": "read", "arguments": "{}"},
        _fn_output(),
    ]


# ---------------------------------------------------------------------------
# classify_responses_turn
# ---------------------------------------------------------------------------


class TestClassifyResponsesTurn:
    def test_string_input_is_new_ask(self):
        assert classify_responses_turn("explain this") == TurnKind.NEW_USER_ASK

    def test_blank_string_input_is_unknown(self):
        assert classify_responses_turn("   ") == TurnKind.UNKNOWN

    def test_empty_or_non_list_is_unknown(self):
        assert classify_responses_turn([]) == TurnKind.UNKNOWN
        assert classify_responses_turn(None) == TurnKind.UNKNOWN
        assert classify_responses_turn({"role": "user"}) == TurnKind.UNKNOWN

    def test_trailing_function_call_output_is_mechanical(self):
        assert classify_responses_turn(_mechanical_input()) == TurnKind.MECHANICAL_CONTINUATION

    def test_multiple_trailing_tool_outputs_are_mechanical(self):
        items = _mechanical_input() + [_fn_output(), _fn_output()]
        assert classify_responses_turn(items) == TurnKind.MECHANICAL_CONTINUATION

    def test_custom_tool_call_output_is_mechanical(self):
        items = _mechanical_input()[:-1] + [_fn_output(item_type="custom_tool_call_output")]
        assert classify_responses_turn(items) == TurnKind.MECHANICAL_CONTINUATION

    def test_apply_patch_call_output_is_mechanical(self):
        items = _mechanical_input()[:-1] + [_fn_output(item_type="apply_patch_call_output")]
        assert classify_responses_turn(items) == TurnKind.MECHANICAL_CONTINUATION

    def test_trailing_user_message_is_new_ask(self):
        items = _mechanical_input() + [_user_message("also check bar.py")]
        assert classify_responses_turn(items) == TurnKind.NEW_USER_ASK

    def test_role_only_user_item_is_new_ask(self):
        # Codex sends bare {"role": "user", "content": [...]} items without type.
        items = [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
        assert classify_responses_turn(items) == TurnKind.NEW_USER_ASK

    def test_trailing_assistant_message_is_unknown(self):
        items = [_user_message(), {"type": "message", "role": "assistant", "content": []}]
        assert classify_responses_turn(items) == TurnKind.UNKNOWN

    def test_non_dict_item_is_unknown(self):
        assert classify_responses_turn([_user_message(), "garbage"]) == TurnKind.UNKNOWN

    def test_nonzero_exit_code_is_error(self):
        items = _mechanical_input()[:-1] + [
            _fn_output(json.dumps({"output": "boom", "metadata": {"exit_code": 1}}))
        ]
        assert classify_responses_turn(items) == TurnKind.ERROR_CONTINUATION

    def test_zero_exit_code_is_mechanical(self):
        items = _mechanical_input()[:-1] + [
            _fn_output(json.dumps({"output": "fine", "metadata": {"exit_code": 0}}))
        ]
        assert classify_responses_turn(items) == TurnKind.MECHANICAL_CONTINUATION

    def test_success_false_is_error(self):
        items = _mechanical_input()[:-1] + [_fn_output(json.dumps({"success": False}))]
        assert classify_responses_turn(items) == TurnKind.ERROR_CONTINUATION

    def test_error_field_is_error(self):
        items = _mechanical_input()[:-1] + [_fn_output({"error": "ENOENT"})]
        assert classify_responses_turn(items) == TurnKind.ERROR_CONTINUATION

    def test_any_error_in_trailing_run_wins(self):
        items = _mechanical_input() + [_fn_output(json.dumps({"exit_code": 2}))]
        assert classify_responses_turn(items) == TurnKind.ERROR_CONTINUATION

    def test_prose_output_mentioning_error_is_mechanical(self):
        # Structural only: prose content is never inspected.
        items = _mechanical_input()[:-1] + [_fn_output("error: this is just prose")]
        assert classify_responses_turn(items) == TurnKind.MECHANICAL_CONTINUATION


# ---------------------------------------------------------------------------
# apply_responses_verbosity_steering
# ---------------------------------------------------------------------------


class TestResponsesVerbositySteering:
    def test_appends_to_instructions_tail(self):
        body = {"instructions": "You are Codex."}
        assert apply_responses_verbosity_steering(body, 2) is True
        assert body["instructions"].startswith("You are Codex.")
        assert body["instructions"].endswith(steering_text(2))

    def test_missing_instructions_becomes_steering(self):
        body: dict[str, Any] = {}
        assert apply_responses_verbosity_steering(body, 2) is True
        assert body["instructions"] == steering_text(2)

    def test_level_zero_is_noop(self):
        body = {"instructions": "You are Codex."}
        assert apply_responses_verbosity_steering(body, 0) is False
        assert body["instructions"] == "You are Codex."

    def test_idempotent_at_same_level(self):
        body = {"instructions": "You are Codex."}
        apply_responses_verbosity_steering(body, 2)
        snapshot = body["instructions"]
        assert apply_responses_verbosity_steering(body, 2) is False
        assert body["instructions"] == snapshot

    def test_level_change_replaces_block_in_place(self):
        body = {"instructions": "You are Codex."}
        apply_responses_verbosity_steering(body, 2)
        assert apply_responses_verbosity_steering(body, 3) is True
        assert body["instructions"].count("<headroom_output_shaping>") == 1
        assert steering_text(3) in body["instructions"]
        assert body["instructions"].startswith("You are Codex.")

    def test_non_string_instructions_untouched(self):
        body = {"instructions": ["not", "a", "string"]}
        assert apply_responses_verbosity_steering(body, 2) is False
        assert body["instructions"] == ["not", "a", "string"]

    def test_byte_stable_across_turns(self):
        # The same level produces identical instructions bytes on every turn
        # of a conversation — the prefix-cache contract.
        a = {"instructions": "You are Codex."}
        b = {"instructions": "You are Codex."}
        apply_responses_verbosity_steering(a, 2)
        apply_responses_verbosity_steering(b, 2)
        assert a["instructions"] == b["instructions"]


# ---------------------------------------------------------------------------
# route_responses_effort
# ---------------------------------------------------------------------------


class TestRouteResponsesEffort:
    def test_lowers_high_to_low_on_mechanical(self):
        body = {"reasoning": {"effort": "high", "summary": "auto"}}
        labels = route_responses_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert body["reasoning"]["effort"] == "low"
        assert labels == ["output_shaper:effort:high->low"]
        assert body["reasoning"]["summary"] == "auto"  # only effort is touched

    def test_never_raises_effort(self):
        body = {"reasoning": {"effort": "minimal"}}
        labels = route_responses_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert body["reasoning"]["effort"] == "minimal"
        assert labels == []

    def test_never_injects_reasoning(self):
        body: dict[str, Any] = {"model": "gpt-5.5"}
        labels = route_responses_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert "reasoning" not in body
        assert labels == []

    def test_new_ask_keeps_full_effort(self):
        body = {"reasoning": {"effort": "high"}}
        assert route_responses_effort(body, TurnKind.NEW_USER_ASK, ENABLED) == []
        assert body["reasoning"]["effort"] == "high"

    def test_error_continuation_keeps_full_effort(self):
        body = {"reasoning": {"effort": "high"}}
        assert route_responses_effort(body, TurnKind.ERROR_CONTINUATION, ENABLED) == []
        assert body["reasoning"]["effort"] == "high"

    def test_unknown_effort_value_untouched(self):
        body = {"reasoning": {"effort": "turbo"}}
        assert route_responses_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED) == []
        assert body["reasoning"]["effort"] == "turbo"

    def test_respects_mechanical_effort_setting(self):
        settings = OutputShaperSettings(enabled=True, mechanical_effort="medium")
        body = {"reasoning": {"effort": "high"}}
        labels = route_responses_effort(body, TurnKind.MECHANICAL_CONTINUATION, settings)
        assert body["reasoning"]["effort"] == "medium"
        assert labels == ["output_shaper:effort:high->medium"]


# ---------------------------------------------------------------------------
# shape_responses_request
# ---------------------------------------------------------------------------


def _mechanical_body() -> dict[str, Any]:
    return {
        "model": "gpt-5.5",
        "instructions": "You are Codex.",
        "input": _mechanical_input(),
        "reasoning": {"effort": "high"},
        "tools": [{"type": "function", "name": "read"}],
    }


class TestShapeResponsesRequest:
    def test_disabled_is_noop(self):
        body = _mechanical_body()
        result = shape_responses_request(body, OutputShaperSettings(enabled=False))
        assert result.changed is False
        assert body == _mechanical_body()

    def test_enabled_applies_both_levers(self):
        body = _mechanical_body()
        result = shape_responses_request(body, OutputShaperSettings(enabled=True))
        assert result.changed is True
        assert "output_shaper:verbosity:L2" in (result.labels or [])
        assert "output_shaper:effort:high->low" in (result.labels or [])
        assert body["reasoning"]["effort"] == "low"
        assert body["instructions"].startswith("You are Codex.")

    def test_level_override_wins(self):
        body = _mechanical_body()
        result = shape_responses_request(body, OutputShaperSettings(enabled=True), level_override=3)
        assert "output_shaper:verbosity:L3" in (result.labels or [])

    def test_new_ask_only_steers(self):
        body = _mechanical_body()
        body["input"] = [_user_message("new question")]
        result = shape_responses_request(body, OutputShaperSettings(enabled=True))
        assert result.changed is True
        assert body["reasoning"]["effort"] == "high"
        assert result.labels == ["output_shaper:verbosity:L2"]


# ---------------------------------------------------------------------------
# conversation_key_from_responses_body
# ---------------------------------------------------------------------------


class TestResponsesConversationKey:
    def test_stable_as_conversation_grows(self):
        turn1 = {"model": "gpt-5.5", "input": [_user_message("task A")]}
        turn2 = {"model": "gpt-5.5", "input": [_user_message("task A"), _fn_output()]}
        assert conversation_key_from_responses_body(turn1) == conversation_key_from_responses_body(
            turn2
        )

    def test_differs_by_first_user_text(self):
        a = {"model": "gpt-5.5", "input": [_user_message("task A")]}
        b = {"model": "gpt-5.5", "input": [_user_message("task B")]}
        assert conversation_key_from_responses_body(a) != conversation_key_from_responses_body(b)

    def test_string_input_supported(self):
        a = {"model": "gpt-5.5", "input": "task A"}
        b = {"model": "gpt-5.5", "input": "task B"}
        assert conversation_key_from_responses_body(a) != conversation_key_from_responses_body(b)


# ---------------------------------------------------------------------------
# _shape_openai_responses_payload (handler-level helper)
# ---------------------------------------------------------------------------


class TestShapeHandlerHelper:
    def test_disabled_returns_nothing(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_OUTPUT_SHAPER", raising=False)
        labels, mutated = _shape_openai_responses_payload(
            _mechanical_body(), model="gpt-5.5", request_id="t1"
        )
        assert labels == [] and mutated is False

    def test_treatment_shapes_and_labels(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
        monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "2")
        monkeypatch.delenv("HEADROOM_OUTPUT_HOLDOUT", raising=False)
        body = _mechanical_body()
        labels, mutated = _shape_openai_responses_payload(body, model="gpt-5.5", request_id="t2")
        assert mutated is True
        assert any(label.startswith("output_shaper:stratum:") for label in labels)
        assert "output_shaper:effort:high->low" in labels
        assert body["reasoning"]["effort"] == "low"

    def test_full_holdout_labels_without_mutation(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
        monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "1.0")
        body = _mechanical_body()
        snapshot = json.dumps(body, sort_keys=True)
        labels, mutated = _shape_openai_responses_payload(body, model="gpt-5.5", request_id="t3")
        assert mutated is False
        assert json.dumps(body, sort_keys=True) == snapshot  # control arm untouched
        assert len(labels) == 1
        assert labels[0].startswith("output_shaper:control:")


class TestHandlerPathControlLabels:
    """Regression: control-arm labels must reach the request outcome even
    when the forwarded payload bytes are unchanged (review feedback on the
    ``if _modified:`` gating in the /v1/responses HTTP handler)."""

    def test_http_handler_records_control_label_without_mutation(self, monkeypatch):
        import anyio

        from tests.test_openai_codex_routing import (
            _build_request,
            _DummyOpenAIHandler,
            _DummyTokenizer,
        )

        monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
        monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "1.0")
        monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

        body = _mechanical_body()
        snapshot = json.dumps(body, sort_keys=True)

        outcomes: list[Any] = []

        class _Handler(_DummyOpenAIHandler):
            async def _record_request_outcome(self, outcome) -> None:
                outcomes.append(outcome)

        handler = _Handler()
        handler.config.optimize = True

        request = _build_request(body, {"Authorization": "Bearer sk-test"})
        response = anyio.run(handler.handle_openai_responses, request)

        assert response.status_code == 200
        assert handler.captured_request is not None
        forwarded_body = handler.captured_request[3]
        assert json.dumps(forwarded_body, sort_keys=True) == snapshot

        assert outcomes, "handler must record a request outcome"
        transforms = list(outcomes[0].transforms_applied)
        assert any(t.startswith("output_shaper:control:") for t in transforms)
        assert not any(t.startswith("output_shaper:verbosity:") for t in transforms)

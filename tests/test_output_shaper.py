"""Tests for headroom.proxy.output_shaper.

Covers turn classification (structural only), cache-safe verbosity steering,
effort routing on mechanical continuations, and the env-driven gate.
"""

from __future__ import annotations

import copy
from typing import Any

from headroom.proxy.output_shaper import (
    OutputShaperSettings,
    TurnKind,
    apply_openai_responses_verbosity_steering,
    apply_verbosity_steering,
    classify_openai_responses_input,
    classify_turn,
    shape_openai_chat_request,
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

    def test_steering_is_the_only_lever(self):
        """Steering applies; request params are left exactly as the client sent
        them. Effort routing was removed after measurement: on mechanical turns
        it saved ~$0.0007 while a switch cost ~$0.011 in cache re-writes, and
        the model's own API now rejects the legacy thinking form outright."""
        body = {
            "system": "Sys.",
            "messages": _mechanical_messages(),
            "output_config": {"effort": "xhigh"},
            "thinking": {"type": "adaptive"},
        }
        result = shape_request(body, ENABLED)
        assert result.changed is True
        assert result.labels == ["output_shaper:verbosity:L3"]
        assert body["output_config"]["effort"] == "xhigh", "must not touch effort"
        assert body["thinking"] == {"type": "adaptive"}, "must not touch thinking"
        assert body["system"][1]["text"] == steering_text(3)

    def test_new_ask_gets_steering_but_keeps_effort(self):
        body = {
            "system": "Sys.",
            "messages": [{"role": "user", "content": "design a cache layer"}],
            "output_config": {"effort": "xhigh"},
        }
        result = shape_request(body, ENABLED)
        assert result.labels == ["output_shaper:verbosity:L3"]
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
        settings = OutputShaperSettings.from_env()
        assert settings.enabled is True
        assert settings.verbosity_level == 3

    def test_from_env_clamps_bad_values(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "true")
        monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "99")
        settings = OutputShaperSettings.from_env()
        assert settings.verbosity_level == 4


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
        assert classify_openai_responses_input(input_data) == TurnKind.MECHANICAL_CONTINUATION

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


class TestShapeOpenAIChatRequest:
    def test_disabled_is_noop(self):
        body = {"messages": [{"role": "system", "content": "Sys."}]}
        snapshot = copy.deepcopy(body)
        result = shape_openai_chat_request(body, OutputShaperSettings(enabled=False))
        assert result.changed is False
        assert body == snapshot

    def test_enabled_applies_verbosity_steering(self):
        body = {
            "messages": [
                {"role": "system", "content": "Sys."},
                {"role": "user", "content": "hi"},
            ]
        }
        result = shape_openai_chat_request(body, ENABLED)
        assert result.changed is True
        assert result.labels == ["output_shaper:verbosity:L3"]
        assert steering_text(3) in body["messages"][0]["content"]
        # User turn is untouched.
        assert body["messages"][1] == {"role": "user", "content": "hi"}

    def test_level_override_supersedes_settings(self):
        body = {"messages": [{"role": "system", "content": "Sys."}]}
        result = shape_openai_chat_request(body, ENABLED, level_override=4)
        assert result.labels == ["output_shaper:verbosity:L4"]
        assert steering_text(4) in body["messages"][0]["content"]

    def test_second_pass_is_stable(self):
        body = {"messages": [{"role": "system", "content": "Sys."}]}
        shape_openai_chat_request(body, ENABLED)
        snapshot = copy.deepcopy(body)
        second = shape_openai_chat_request(body, ENABLED)
        assert second.changed is False
        assert body == snapshot


class TestShaperEnabledFor:
    """The gate. Steering is opt-in: it appends a block to the system prompt,
    and at L3 that is a visible behaviour change, so a user who did not ask
    for it must not get it."""

    @staticmethod
    def _config(*, optimize: bool, env: dict[str, str] | None = None):
        from types import SimpleNamespace

        from headroom.rollout import resolve_rollout

        return SimpleNamespace(optimize=optimize, rollout=resolve_rollout(env or {}))

    def test_off_without_an_explicit_opt_in(self):
        from headroom.proxy.output_shaper import shaper_enabled_for

        assert shaper_enabled_for(self._config(optimize=True)) is False

    def test_on_when_explicitly_enabled(self):
        from headroom.proxy.output_shaper import shaper_enabled_for

        cfg = self._config(optimize=True, env={"HEADROOM_OUTPUT_SHAPER": "1"})
        assert shaper_enabled_for(cfg) is True

    def test_explicit_request_shapes_even_with_optimize_off(self):
        """Shaping without input compression is a supported combination."""
        from headroom.proxy.output_shaper import shaper_enabled_for

        cfg = self._config(optimize=False, env={"HEADROOM_OUTPUT_SHAPER": "1"})
        assert shaper_enabled_for(cfg) is True

    def test_kill_switch_wins(self):
        from headroom.proxy.output_shaper import shaper_enabled_for

        for env in (
            {"HEADROOM_OUTPUT_SHAPER": "0"},
            {"HEADROOM_DISABLE_FEATURES": "proxy_output_shaper"},
        ):
            assert shaper_enabled_for(self._config(optimize=True, env=env)) is False

    def test_default_on_would_still_respect_optimize_off(self, monkeypatch):
        """A guard for a default that does not exist yet.

        The feature is opt-in, so `reason is DEFAULT` with `enabled` true
        cannot currently occur. The branch is kept because turning the default
        back on would otherwise silently reintroduce the byte-faithful
        forwarding bug: an operator running `optimize=False` would start
        getting a steering block appended, and on a body with no `system`
        field, one created.
        """
        from headroom.proxy.output_shaper import shaper_enabled_for
        from headroom.rollout import FeatureDecisionReason

        class _Decision:
            enabled = True
            reason = FeatureDecisionReason.DEFAULT

        class _Rollout:
            def decision(self, name):
                return _Decision()

        from types import SimpleNamespace

        assert shaper_enabled_for(SimpleNamespace(optimize=False, rollout=_Rollout())) is False
        assert shaper_enabled_for(SimpleNamespace(optimize=True, rollout=_Rollout())) is True

    def test_no_rollout_snapshot_falls_back_to_the_env_var(self):
        """SDK/test callers build a config without a snapshot; returning None
        preserves OutputShaperSettings.from_env's own resolution."""
        from types import SimpleNamespace

        from headroom.proxy.output_shaper import shaper_enabled_for

        assert shaper_enabled_for(SimpleNamespace(optimize=True, rollout=None)) is None
        assert shaper_enabled_for(None) is None


class TestCacheModeSuppressesSteeringOnly:
    """``mode="cache"`` freezes prior turns for prefix-cache stability.

    Steering is the one lever that writes into that key: it appends to the
    system-prompt tail, and on a body carrying no ``system`` field it creates
    one, displacing ``messages[0]``. Effort routing and the thinking budget
    ride request parameters outside the key, so they must keep working — the
    point is a targeted suppression, not switching the feature off.
    """

    def test_steering_allowed_for_reads_the_mode(self):
        from types import SimpleNamespace

        from headroom.proxy.output_shaper import steering_allowed_for

        assert steering_allowed_for(SimpleNamespace(mode="token")) is True
        assert steering_allowed_for(SimpleNamespace(mode="cache")) is False
        assert steering_allowed_for(None) is True, "absent config must not disable levers"

    def test_cache_mode_resolves_level_zero(self):
        from headroom.proxy.output_shaper import OutputShaperSettings, resolve_verbosity_level

        settings = OutputShaperSettings(enabled=True, verbosity_level=3, steering_enabled=False)
        assert resolve_verbosity_level(settings) == (0, "cache_mode")

    def test_cache_mode_outranks_the_manual_level_override(self, monkeypatch):
        """An env-set level must not reintroduce the prefix mutation."""
        from headroom.proxy import runtime_env
        from headroom.proxy.output_shaper import OutputShaperSettings, resolve_verbosity_level

        monkeypatch.setattr(runtime_env, "getenv", lambda k, d="": "4" if "VERBOSITY" in k else d)
        settings = OutputShaperSettings(enabled=True, verbosity_level=4, steering_enabled=False)
        assert resolve_verbosity_level(settings)[0] == 0

    def test_effort_routing_survives_cache_mode(self):
        """The savings that do not touch the cache key must still apply."""
        from headroom.proxy.output_shaper import OutputShaperSettings, shape_request

        body = {
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
            ],
        }
        settings = OutputShaperSettings(enabled=True, verbosity_level=2, steering_enabled=False)
        result = shape_request(body, settings, level_override=0)
        assert "system" not in body, "cache mode must not create a system block"
        assert result is not None

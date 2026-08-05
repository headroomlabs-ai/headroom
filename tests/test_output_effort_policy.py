"""Tests for pure output effort policy decisions."""

from __future__ import annotations

from headroom.proxy.output_effort_policy import (
    LEGACY_THINKING_FLOOR,
    can_create_openai_text_verbosity,
    clamp_legacy_thinking_budget,
    lower_effort_value,
    lower_text_verbosity_value,
)


def test_lower_effort_value_lowers_known_higher_effort_to_target() -> None:
    assert lower_effort_value("xhigh", "low") == "low"
    assert lower_effort_value("max", "medium") == "medium"


def test_lower_effort_value_keeps_lower_equal_unknown_or_non_string_values() -> None:
    assert lower_effort_value("low", "medium") is None
    assert lower_effort_value("medium", "medium") is None
    assert lower_effort_value("turbo", "low") is None
    assert lower_effort_value("high", "turbo") is None
    assert lower_effort_value(None, "low") is None


def test_clamp_legacy_thinking_budget_only_clamps_enabled_over_floor() -> None:
    assert (
        clamp_legacy_thinking_budget(
            thinking_type="enabled",
            budget_tokens=32_000,
        )
        == LEGACY_THINKING_FLOOR
    )
    assert (
        clamp_legacy_thinking_budget(
            thinking_type="enabled",
            budget_tokens=LEGACY_THINKING_FLOOR,
        )
        is None
    )
    assert clamp_legacy_thinking_budget(thinking_type="adaptive", budget_tokens=32_000) is None
    assert clamp_legacy_thinking_budget(thinking_type="enabled", budget_tokens="32000") is None


def test_can_create_openai_text_verbosity_only_for_gpt5_family() -> None:
    assert can_create_openai_text_verbosity("gpt-5")
    assert can_create_openai_text_verbosity("GPT-5.1")
    assert not can_create_openai_text_verbosity("gpt-4o")
    assert not can_create_openai_text_verbosity(None)


def test_lower_text_verbosity_value_lowers_existing_verbose_values() -> None:
    assert lower_text_verbosity_value("medium") == "low"
    assert lower_text_verbosity_value("high") == "low"
    assert lower_text_verbosity_value("low") is None
    assert lower_text_verbosity_value("chatty") is None
    assert lower_text_verbosity_value(None) is None

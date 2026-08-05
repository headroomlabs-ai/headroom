"""Tests for pure memory-injection decision policy helpers."""

from __future__ import annotations

from headroom.proxy.memory_decision_policy import (
    apply_memory_skip_reason,
    decide_memory_injection,
)


def test_decide_memory_injection_allows_happy_path() -> None:
    decision = decide_memory_injection(
        headers={},
        memory_handler_present=True,
        memory_user_id_present=True,
        mode_name="auto_tail",
    )

    assert decision.inject is True
    assert decision.skip_reason is None


def test_decide_memory_injection_uses_canonical_precedence() -> None:
    decision = decide_memory_injection(
        headers={"x-headroom-bypass": "true"},
        memory_handler_present=False,
        memory_user_id_present=False,
        mode_name="disabled",
    )

    assert decision.inject is False
    assert decision.skip_reason == "bypass_header"
    assert decision.bypass_header_set is True
    assert decision.memory_handler_present is False
    assert decision.memory_user_id_present is False


def test_decide_memory_injection_reports_mode_reasons() -> None:
    disabled = decide_memory_injection(
        headers={},
        memory_handler_present=True,
        memory_user_id_present=True,
        mode_name="disabled",
    )
    tool = decide_memory_injection(
        headers={},
        memory_handler_present=True,
        memory_user_id_present=True,
        mode_name="tool",
    )

    assert disabled.skip_reason == "mode_disabled"
    assert tool.skip_reason == "mode_tool"


def test_apply_memory_skip_reason_stamps_only_when_skipping() -> None:
    tags: dict[str, str] = {}
    apply_memory_skip_reason(tags, None)
    assert tags == {}

    apply_memory_skip_reason(tags, "no_user_id")
    assert tags == {"memory_skip_reason": "no_user_id"}

"""Tests for the install-result display helper."""

from __future__ import annotations

from headroom.mcp_registry.base import RegisterResult, RegisterStatus
from headroom.mcp_registry.display import (
    any_succeeded,
    format_result,
    format_results,
)

# ----------------------------------------------------------------------
# format_result
# ----------------------------------------------------------------------


def test_registered_includes_restart_hint() -> None:
    line = format_result(
        "claude",
        RegisterResult(RegisterStatus.REGISTERED, "via CLI"),
        restart_hint="restart Claude Code if it was running",
    )
    assert line is not None
    assert "claude" in line
    assert "registered" in line
    assert "restart Claude Code" in line


def test_already_silent_when_not_verbose() -> None:
    line = format_result(
        "claude",
        RegisterResult(RegisterStatus.ALREADY, "matches"),
        verbose=False,
    )
    assert line is None


def test_already_emits_when_verbose() -> None:
    line = format_result(
        "claude",
        RegisterResult(RegisterStatus.ALREADY, "matches"),
        verbose=True,
    )
    assert line is not None
    assert "already registered" in line


def test_not_detected_emits_skipped() -> None:
    line = format_result(
        "cursor",
        RegisterResult(RegisterStatus.NOT_DETECTED, "Cursor not found"),
    )
    assert line is not None
    assert "cursor" in line
    assert "not detected" in line
    assert "skipped" in line


def test_mismatch_emits_overwrite_hint() -> None:
    line = format_result(
        "claude",
        RegisterResult(RegisterStatus.MISMATCH, "env differs"),
        overwrite_hint="headroom mcp install --force",
    )
    assert line is not None
    assert "differs" in line
    assert "headroom mcp install --force" in line


def test_mismatch_omits_hint_when_empty() -> None:
    line = format_result(
        "claude",
        RegisterResult(RegisterStatus.MISMATCH, "env differs"),
        overwrite_hint="",
    )
    assert line is not None
    assert "To update" not in line


def test_no_sdk_points_at_pip_extras() -> None:
    line = format_result(
        "claude",
        RegisterResult(RegisterStatus.NO_SDK, "missing"),
    )
    assert line is not None
    assert "MCP SDK" in line
    assert "headroom-ai[mcp]" in line


def test_failed_includes_detail() -> None:
    line = format_result(
        "claude",
        RegisterResult(RegisterStatus.FAILED, "connection refused"),
    )
    assert line is not None
    assert "failed" in line
    assert "connection refused" in line


def test_label_overrides_agent_name() -> None:
    line = format_result(
        "claude",
        RegisterResult(RegisterStatus.REGISTERED, "ok"),
        label="MCP retrieve tool",
    )
    assert line is not None
    assert "MCP retrieve tool" in line
    assert "claude" not in line


# ----------------------------------------------------------------------
# format_results
# ----------------------------------------------------------------------


def test_format_results_filters_silent_lines() -> None:
    results = {
        "claude": RegisterResult(RegisterStatus.REGISTERED, "ok"),
        "cursor": RegisterResult(RegisterStatus.ALREADY, "matches"),
    }
    lines = format_results(results, verbose=False)
    # ALREADY suppressed when verbose=False, so only one line.
    assert len(lines) == 1
    assert "claude" in lines[0]


def test_format_results_label_for_remaps_agent_name() -> None:
    results = {
        "claude": RegisterResult(RegisterStatus.REGISTERED, "ok"),
    }
    labels = {"claude": "Claude Code"}
    lines = format_results(results, label_for=labels.get)
    assert len(lines) == 1
    assert "Claude Code" in lines[0]


def test_format_results_preserves_iteration_order() -> None:
    results = {
        "a": RegisterResult(RegisterStatus.REGISTERED, "ok"),
        "b": RegisterResult(RegisterStatus.NOT_DETECTED, "missing"),
        "c": RegisterResult(RegisterStatus.MISMATCH, "env differs"),
    }
    lines = format_results(results, verbose=True)
    assert len(lines) == 3
    # Same order as input dict iteration.
    assert lines[0].strip().startswith("a:")
    assert lines[1].strip().startswith("b:")
    assert lines[2].strip().startswith("c:")


# ----------------------------------------------------------------------
# any_succeeded
# ----------------------------------------------------------------------


def test_any_succeeded_true_when_one_registered() -> None:
    results = {
        "a": RegisterResult(RegisterStatus.REGISTERED, "ok"),
        "b": RegisterResult(RegisterStatus.NOT_DETECTED, "missing"),
    }
    assert any_succeeded(results) is True


def test_any_succeeded_true_when_already_registered() -> None:
    results = {"a": RegisterResult(RegisterStatus.ALREADY, "matches")}
    assert any_succeeded(results) is True


def test_any_succeeded_false_when_all_failed_or_skipped() -> None:
    results = {
        "a": RegisterResult(RegisterStatus.NOT_DETECTED, "missing"),
        "b": RegisterResult(RegisterStatus.FAILED, "boom"),
        "c": RegisterResult(RegisterStatus.MISMATCH, "differs"),
    }
    assert any_succeeded(results) is False


def test_any_succeeded_empty_dict_is_false() -> None:
    assert any_succeeded({}) is False

"""Issue #746: keep Claude Code's on-demand tool loading active through the proxy.

Covers the two halves of the fix:

* ``headroom wrap claude`` injects ``ENABLE_TOOL_SEARCH`` into the launched
  Claude Code environment (with correct precedence / validation), and
* the proxy detects a Claude Code request that is *not* deferring tools and
  emits a single actionable hint for users who run ``claude`` manually.
"""

from __future__ import annotations

import pytest

from headroom.cli.wrap import (
    _TOOL_SEARCH_DEFAULT,
    _TOOL_SEARCH_ENV,
    _configure_tool_search_env,
    _normalize_tool_search_mode,
)
from headroom.proxy.helpers import (
    claude_code_tool_search_inactive,
    format_tool_search_disabled_hint,
    reset_tool_search_hint_state,
    take_tool_search_hint_slot,
    tool_search_hint_pending,
)

# ---------------------------------------------------------------------------
# wrap: ENABLE_TOOL_SEARCH value normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", "true"),
        ("TRUE", "true"),
        (" on ", "on"),
        ("1", "1"),
        ("false", "false"),
        ("off", "off"),
        ("auto", "auto"),
        ("auto:0", "auto:0"),
        ("auto:50", "auto:50"),
        ("auto:100", "auto:100"),
    ],
)
def test_normalize_tool_search_mode_accepts_valid(value: str, expected: str) -> None:
    assert _normalize_tool_search_mode(value) == expected


@pytest.mark.parametrize("value", ["yep", "auto:", "auto:101", "auto:-1", "auto:abc", ""])
def test_normalize_tool_search_mode_rejects_invalid(value: str) -> None:
    import click

    with pytest.raises(click.ClickException):
        _normalize_tool_search_mode(value)


# ---------------------------------------------------------------------------
# wrap: ENABLE_TOOL_SEARCH injection precedence
# ---------------------------------------------------------------------------


def test_configure_injects_default_when_unset() -> None:
    env: dict[str, str] = {}
    result = _configure_tool_search_env(env, None)
    assert result == _TOOL_SEARCH_DEFAULT
    assert env[_TOOL_SEARCH_ENV] == _TOOL_SEARCH_DEFAULT


def test_configure_respects_existing_env_value() -> None:
    env = {_TOOL_SEARCH_ENV: "auto:30"}
    result = _configure_tool_search_env(env, None)
    # None signals "left the user's value untouched".
    assert result is None
    assert env[_TOOL_SEARCH_ENV] == "auto:30"


def test_configure_flag_overrides_existing_env_value() -> None:
    env = {_TOOL_SEARCH_ENV: "false"}
    result = _configure_tool_search_env(env, "auto")
    assert result == "auto"
    assert env[_TOOL_SEARCH_ENV] == "auto"


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_configure_overrides_blank_env_value(blank: str) -> None:
    # Claude Code treats an empty ENABLE_TOOL_SEARCH as unset, so a blank value
    # must be replaced with the default rather than forwarded as a no-op.
    env = {_TOOL_SEARCH_ENV: blank}
    result = _configure_tool_search_env(env, None)
    assert result == _TOOL_SEARCH_DEFAULT
    assert env[_TOOL_SEARCH_ENV] == _TOOL_SEARCH_DEFAULT


def test_configure_flag_validated() -> None:
    import click

    with pytest.raises(click.ClickException):
        _configure_tool_search_env({}, "nonsense")


# ---------------------------------------------------------------------------
# proxy: detect a Claude Code request that is not deferring tools
# ---------------------------------------------------------------------------

_TOOLS = [
    {"name": "Read", "description": "read a file", "input_schema": {"type": "object"}},
    {"name": "Bash", "description": "run a command", "input_schema": {"type": "object"}},
]


def test_inactive_true_for_eager_claude_code() -> None:
    assert claude_code_tool_search_inactive(client="claude-code", tools=_TOOLS, anthropic_beta=None)


def test_inactive_false_when_tool_search_tool_present() -> None:
    tools = [*_TOOLS, {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}]
    assert not claude_code_tool_search_inactive(
        client="claude-code", tools=tools, anthropic_beta=None
    )


def test_inactive_false_when_beta_header_present() -> None:
    assert not claude_code_tool_search_inactive(
        client="claude-code",
        tools=_TOOLS,
        anthropic_beta="context-1m-2025-08-07,advanced-tool-use-2025-11-20",
    )


def test_inactive_false_for_other_clients() -> None:
    assert not claude_code_tool_search_inactive(client="codex", tools=_TOOLS, anthropic_beta=None)
    assert not claude_code_tool_search_inactive(client=None, tools=_TOOLS, anthropic_beta=None)


def test_inactive_false_when_no_tools() -> None:
    assert not claude_code_tool_search_inactive(client="claude-code", tools=[], anthropic_beta=None)
    assert not claude_code_tool_search_inactive(
        client="claude-code", tools=None, anthropic_beta=None
    )


# ---------------------------------------------------------------------------
# proxy: hint content + one-time guard
# ---------------------------------------------------------------------------


def test_hint_message_is_actionable() -> None:
    msg = format_tool_search_disabled_hint(_TOOLS)
    assert "ENABLE_TOOL_SEARCH=true" in msg
    assert "746" in msg
    assert str(len(_TOOLS)) in msg


def test_hint_slot_fires_once() -> None:
    reset_tool_search_hint_state()
    try:
        assert tool_search_hint_pending() is True
        assert take_tool_search_hint_slot() is True
        # Once consumed, the cheap gate flips so the hot path stops scanning.
        assert tool_search_hint_pending() is False
        assert take_tool_search_hint_slot() is False
        assert take_tool_search_hint_slot() is False
    finally:
        reset_tool_search_hint_state()

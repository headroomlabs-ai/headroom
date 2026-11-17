"""Focused tests for Claude wrap tool-search defaults."""

from __future__ import annotations

import pytest

from headroom.cli.wrap import (
    _TOOL_SEARCH_DEFAULT,
    _TOOL_SEARCH_ENV,
    _configure_tool_search_env,
)


def test_foundry_without_override_disables_tool_search() -> None:
    env = {"CLAUDE_CODE_USE_FOUNDRY": "1"}

    result = _configure_tool_search_env(env, None)

    assert result == "false"
    assert env[_TOOL_SEARCH_ENV] == "false"


@pytest.mark.parametrize("flag_value", ["auto", "false"])
def test_explicit_flag_wins_in_foundry(flag_value: str) -> None:
    env = {"CLAUDE_CODE_USE_FOUNDRY": "1"}

    result = _configure_tool_search_env(env, flag_value)

    assert result == flag_value
    assert env[_TOOL_SEARCH_ENV] == flag_value


def test_existing_environment_value_wins_in_foundry() -> None:
    env = {"CLAUDE_CODE_USE_FOUNDRY": "1", _TOOL_SEARCH_ENV: "auto:30"}

    result = _configure_tool_search_env(env, None)

    assert result is None
    assert env[_TOOL_SEARCH_ENV] == "auto:30"


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_environment_uses_mode_default(blank: str) -> None:
    foundry_env = {"CLAUDE_CODE_USE_FOUNDRY": "1", _TOOL_SEARCH_ENV: blank}
    generic_env = {_TOOL_SEARCH_ENV: blank}

    assert _configure_tool_search_env(foundry_env, None) == "false"
    assert foundry_env[_TOOL_SEARCH_ENV] == "false"
    assert _configure_tool_search_env(generic_env, None) == _TOOL_SEARCH_DEFAULT
    assert generic_env[_TOOL_SEARCH_ENV] == _TOOL_SEARCH_DEFAULT


def test_non_foundry_without_override_keeps_generic_default() -> None:
    env: dict[str, str] = {}

    result = _configure_tool_search_env(env, None)

    assert result == _TOOL_SEARCH_DEFAULT
    assert env[_TOOL_SEARCH_ENV] == _TOOL_SEARCH_DEFAULT

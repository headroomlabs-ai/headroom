"""Issue #4951: routed `wrap claude` sessions start without Claude Code's Code Mode.

Claude Code does not offer Code Mode (experimental) through a gateway or proxy,
so a session routed through Headroom starts without it. The check is in the
client; Headroom's fix is a one-line notice for users who turned Code Mode on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.providers.claude.runtime import (
    CODE_MODE_ENV,
    CODE_MODE_MIN_VERSION,
    code_mode_gate_short_message,
    code_mode_requested,
)
from tests.test_cli.test_wrap_claude_vertex_proxy_env import _invoke_wrap_claude

_ON = {CODE_MODE_ENV: "1"}
_OFF = {CODE_MODE_ENV: "0"}


def test_off_by_default() -> None:
    assert not code_mode_requested(shell={}, user_settings={}, managed_settings={})


def test_shell_or_user_settings_turn_it_on() -> None:
    assert code_mode_requested(shell=_ON, user_settings={}, managed_settings={})
    assert code_mode_requested(shell={}, user_settings=_ON, managed_settings={})


def test_managed_settings_can_block_but_not_enable() -> None:
    assert not code_mode_requested(shell={}, user_settings={}, managed_settings=_ON)
    assert not code_mode_requested(shell=_ON, user_settings=_ON, managed_settings=_OFF)


def test_zero_in_shell_or_user_settings_wins() -> None:
    assert not code_mode_requested(shell=_OFF, user_settings=_ON, managed_settings={})
    assert not code_mode_requested(shell=_ON, user_settings=_OFF, managed_settings={})


def test_message_names_the_proxy_gate_and_the_way_out() -> None:
    msg = code_mode_gate_short_message(version=(2, 1, 332))
    assert "Code Mode (experimental) is off in this session" in msg
    assert "gateway or proxy" in msg
    assert "without wrap" in msg
    old = code_mode_gate_short_message(version=(2, 1, 300))
    assert ".".join(map(str, CODE_MODE_MIN_VERSION)) in old


def _isolate_claude_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, user_env: dict[str, str]
) -> None:
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(json.dumps({"env": user_env}), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(wrap_mod, "_claude_managed_settings_path", lambda: tmp_path / "none.json")


def test_wrap_prints_notice_when_user_settings_turn_code_mode_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_claude_settings(monkeypatch, tmp_path, _ON)
    _captured, output = _invoke_wrap_claude(CliRunner(), monkeypatch, env={})
    # The shared harness pins Claude Code 2.1.196, below Code Mode's minimum.
    assert "Code Mode (experimental) needs Claude Code 2.1.323+" in output
    assert "gateway or proxy" in output


def test_wrap_stays_silent_when_code_mode_not_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_claude_settings(monkeypatch, tmp_path, {})
    monkeypatch.delenv(CODE_MODE_ENV, raising=False)
    _captured, output = _invoke_wrap_claude(CliRunner(), monkeypatch, env={})
    assert "Code Mode" not in output


def test_wrap_stays_silent_when_shell_turns_it_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_claude_settings(monkeypatch, tmp_path, _ON)
    _captured, output = _invoke_wrap_claude(CliRunner(), monkeypatch, env=_OFF)
    assert "Code Mode" not in output

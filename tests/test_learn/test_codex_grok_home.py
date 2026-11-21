"""Regression tests: `headroom learn` honors CODEX_HOME and GROK_HOME.

Codex and Grok relocate their home when these variables are set (the rest of
Headroom already honors them). The learn plugins hardcoded ``~/.codex`` and
``~/.grok``, so they scanned a stale or missing directory. Twin of the
CLAUDE_CONFIG_DIR fix (issue #1630).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.learn.plugins.codex import CodexPlugin
from headroom.learn.plugins.grok import GrokPlugin


def _make_codex_session(home: Path) -> None:
    sessions = home / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "rollout.jsonl").write_text("{}\n", encoding="utf-8")


def _make_grok_session(home: Path) -> None:
    session = home / "sessions" / "%2Ftmp%2Fproj" / "s1"
    session.mkdir(parents=True)
    (session / "updates.jsonl").write_text("{}\n", encoding="utf-8")


def test_codex_honors_codex_home(tmp_path, monkeypatch) -> None:
    custom = tmp_path / "custom-codex"
    _make_codex_session(custom)
    (custom / "AGENTS.md").write_text("x", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(custom))
    plugin = CodexPlugin()
    assert plugin.codex_dir == custom
    assert plugin.sessions_dir == custom / "sessions"
    assert plugin.detect()
    projects = plugin.discover_projects()
    assert [p.context_file for p in projects] == [custom / "AGENTS.md"]


def test_codex_unset_falls_back_to_home(monkeypatch) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert CodexPlugin().codex_dir == Path.home() / ".codex"


@pytest.mark.parametrize("blank", ["", "   "])
def test_codex_blank_falls_back_to_home(monkeypatch, blank) -> None:
    monkeypatch.setenv("CODEX_HOME", blank)
    assert CodexPlugin().codex_dir == Path.home() / ".codex"


def test_codex_explicit_dir_wins_over_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "env"))
    assert CodexPlugin(codex_dir=tmp_path / "explicit").codex_dir == tmp_path / "explicit"


def test_grok_honors_grok_home(tmp_path, monkeypatch) -> None:
    custom = tmp_path / "custom-grok"
    _make_grok_session(custom)
    monkeypatch.setenv("GROK_HOME", str(custom))
    plugin = GrokPlugin()
    assert plugin.grok_dir == custom
    assert plugin.sessions_dir == custom / "sessions"
    assert plugin.detect()
    assert len(plugin.discover_projects()) == 1


def test_grok_unset_falls_back_to_home(monkeypatch) -> None:
    monkeypatch.delenv("GROK_HOME", raising=False)
    assert GrokPlugin().grok_dir == Path.home() / ".grok"


@pytest.mark.parametrize("blank", ["", "   "])
def test_grok_blank_falls_back_to_home(monkeypatch, blank) -> None:
    monkeypatch.setenv("GROK_HOME", blank)
    assert GrokPlugin().grok_dir == Path.home() / ".grok"


def test_grok_explicit_dir_wins_over_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "env"))
    assert GrokPlugin(grok_dir=tmp_path / "explicit").grok_dir == tmp_path / "explicit"

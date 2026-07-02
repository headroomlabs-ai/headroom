"""Regression tests: `headroom learn` honors CLAUDE_CONFIG_DIR (issue #1630).

Claude Code relocates its config (conversation logs under ``projects/`` and the
global ``CLAUDE.md``) when ``CLAUDE_CONFIG_DIR`` is set. The learn scanner and
memory writer previously hardcoded ``~/.claude``, so they scanned/wrote the
wrong directory and detected no projects for such users.
"""

from __future__ import annotations

from pathlib import Path

from headroom.learn._shared import claude_config_dir
from headroom.learn.models import ProjectInfo
from headroom.learn.plugins.claude import ClaudeCodePlugin
from headroom.learn.writer import ClaudeCodeWriter


def test_config_dir_honors_env(tmp_path, monkeypatch) -> None:
    custom = tmp_path / "custom-claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
    assert claude_config_dir() == custom


def test_config_dir_defaults_to_home(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert claude_config_dir() == Path.home() / ".claude"


def test_plugin_scans_config_dir_override(tmp_path, monkeypatch) -> None:
    custom = tmp_path / "custom-claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
    plugin = ClaudeCodePlugin()
    assert plugin.claude_dir == custom
    assert plugin.projects_dir == custom / "projects"


def test_plugin_explicit_dir_wins_over_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "env"))
    explicit = tmp_path / "explicit"
    plugin = ClaudeCodePlugin(claude_dir=explicit)
    assert plugin.claude_dir == explicit


def test_writer_home_memory_follows_config_dir(tmp_path, monkeypatch) -> None:
    custom = tmp_path / "custom-claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
    writer = ClaudeCodeWriter()
    project = ProjectInfo(
        name="home",
        project_path=Path.home(),
        data_path=custom / "projects",
    )
    assert writer._resolve_context_path(project) == custom / "CLAUDE.md"

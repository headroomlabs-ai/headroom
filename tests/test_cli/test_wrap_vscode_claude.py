"""CLI coverage for Claude Code inside VS Code."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from headroom.cli.main import main


def test_wrap_vscode_claude_configures_actual_port(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    captured = {}

    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        kwargs["print_setup_lines"](9999)

    with patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(main, ["wrap", "vscode-claude", "--settings-file", str(path)])

    assert result.exit_code == 0, result.output
    env = json.loads(path.read_text(encoding="utf-8"))["env"]
    assert env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:9999/p/")
    assert env["ENABLE_TOOL_SEARCH"] == "true"
    assert "Reload VS Code" in result.output
    assert captured["agent_type"] == "claude"


def test_wrap_vscode_claude_no_configure_prints_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"

    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        kwargs["print_setup_lines"](8787)

    with patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(
            main,
            ["wrap", "vscode-claude", "--no-configure", "--settings-file", str(path)],
        )

    assert result.exit_code == 0, result.output
    assert not path.exists()
    assert "ANTHROPIC_BASE_URL" in result.output
    assert "ENABLE_TOOL_SEARCH" in result.output


def test_unwrap_vscode_claude_restores_previous_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = {"env": {"KEEP": "1"}, "permissions": {"allow": ["Read"]}}
    path.write_text(json.dumps(original), encoding="utf-8")

    from headroom.providers.claude.vscode import configure_vscode_claude_settings

    configure_vscode_claude_settings(path, "http://127.0.0.1:8787/p/demo")
    result = CliRunner().invoke(main, ["unwrap", "vscode-claude", "--settings-file", str(path)])

    assert result.exit_code == 0, result.output
    assert json.loads(path.read_text(encoding="utf-8")) == original

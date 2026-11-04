"""CLI coverage for Claude Code inside VS Code."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
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
    assert env["ENABLE_TOOL_SEARCH"] == "false"
    assert "Reload VS Code" in result.output
    assert captured["agent_type"] == "claude"


def test_wrap_vscode_claude_help_exposes_1m() -> None:
    result = CliRunner().invoke(main, ["wrap", "vscode-claude", "--help"])

    assert result.exit_code == 0, result.output
    assert "--1m" in result.output


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


def test_wrap_vscode_claude_1m_persists_model(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"model": "claude-sonnet-5", "custom": {"keep": True}}), encoding="utf-8"
    )

    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        kwargs["print_setup_lines"](8787)

    with patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(
            main,
            ["wrap", "vscode-claude", "--1m", "--settings-file", str(path)],
        )

    assert result.exit_code == 0, result.output
    configured = json.loads(path.read_text(encoding="utf-8"))
    assert configured["model"] == "claude-sonnet-5[1m]"
    assert configured["custom"] == {"keep": True}
    assert "1M context model persisted: claude-sonnet-5[1m]" in result.output


def test_wrap_vscode_claude_default_leaves_model_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"model": "claude-sonnet-5", "custom": {"keep": True}}), encoding="utf-8"
    )

    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        kwargs["print_setup_lines"](8787)

    with patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(
            main,
            ["wrap", "vscode-claude", "--settings-file", str(path)],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "claude-sonnet-5"
    assert "1M context model persisted:" not in result.output


def test_wrap_vscode_claude_no_configure_1m_is_write_free(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = '{"model":"claude-opus-5","custom":{"keep":true}}'
    path.write_text(original, encoding="utf-8")
    state_path = tmp_path / ".headroom-vscode-claude.json"
    state_path.write_text('{"sentinel":true}', encoding="utf-8")

    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        kwargs["print_setup_lines"](8787)

    with patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(
            main,
            ["wrap", "vscode-claude", "--no-configure", "--1m", "--settings-file", str(path)],
        )

    assert result.exit_code == 0, result.output
    assert '"model": "claude-opus-5[1m]"' in result.output
    assert (
        '  "ENABLE_TOOL_SEARCH": "true"\n'
        f"  Add this top-level setting to {path}:\n"
        '  "model": "claude-opus-5[1m]"'
    ) in result.output
    assert path.read_text(encoding="utf-8") == original
    assert state_path.read_text(encoding="utf-8") == '{"sentinel":true}'


@pytest.mark.parametrize("contents", ["{broken", '{"model":{"name":"opus"}}'])
def test_wrap_vscode_claude_no_configure_1m_falls_back_for_invalid_settings(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(contents, encoding="utf-8")
    state_path = tmp_path / ".headroom-vscode-claude.json"
    state_path.write_text('{"sentinel":true}', encoding="utf-8")

    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        kwargs["print_setup_lines"](8787)

    with patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(
            main,
            ["wrap", "vscode-claude", "--no-configure", "--1m", "--settings-file", str(path)],
        )

    assert result.exit_code == 0, result.output
    assert (
        '  "ENABLE_TOOL_SEARCH": "true"\n'
        f"  Add this top-level setting to {path}:\n"
        '  "model": "claude-opus-5[1m]"'
    ) in result.output
    assert '  "ENABLE_TOOL_SEARCH": "true",\n' not in result.output
    assert path.read_text(encoding="utf-8") == contents
    assert state_path.read_text(encoding="utf-8") == '{"sentinel":true}'


def test_unwrap_vscode_claude_restores_previous_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = {"env": {"KEEP": "1"}, "permissions": {"allow": ["Read"]}}
    path.write_text(json.dumps(original), encoding="utf-8")

    from headroom.providers.claude.vscode import configure_vscode_claude_settings

    configure_vscode_claude_settings(path, "http://127.0.0.1:8787/p/demo")
    result = CliRunner().invoke(main, ["unwrap", "vscode-claude", "--settings-file", str(path)])

    assert result.exit_code == 0, result.output
    assert json.loads(path.read_text(encoding="utf-8")) == original

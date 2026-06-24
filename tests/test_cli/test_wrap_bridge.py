"""Tests for Docker-bridge wrap preparation flows."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.cli.wrap import _setup_lean_ctx_agent


@pytest.fixture(autouse=True)
def _default_context_tool(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    monkeypatch.delenv("LEAN_CTX_AGENT", raising=False)
    monkeypatch.delenv("LEAN_CTX_DATA_DIR", raising=False)


def _set_test_home(monkeypatch, tmp_path: Path) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)


def test_wrap_claude_prepare_only_skips_host_binary_lookup() -> None:
    runner = CliRunner()

    with patch("headroom.cli.wrap._prepare_wrap_rtk") as prepare_rtk:
        with patch("headroom.cli.wrap.shutil.which") as which_mock:
            result = runner.invoke(main, ["wrap", "claude", "--prepare-only"])

    assert result.exit_code == 0, result.output
    prepare_rtk.assert_called_once()
    which_mock.assert_not_called()


def test_wrap_claude_prepare_only_uses_lean_ctx_when_configured(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")

    with patch("headroom.cli.wrap._prepare_wrap_rtk") as prepare_rtk:
        with patch(
            "headroom.cli.wrap._setup_lean_ctx_agent",
            return_value=Path("lean-ctx"),
        ) as setup:
            result = runner.invoke(main, ["wrap", "claude", "--prepare-only"])

    assert result.exit_code == 0, result.output
    prepare_rtk.assert_not_called()
    setup.assert_called_once_with("claude", verbose=False)


def test_setup_lean_ctx_agent_runs_outside_project_root(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".git").mkdir()
    lean_ctx = tmp_path / "lean-ctx"
    lean_ctx.write_text("#!/bin/sh\n", encoding="utf-8")
    calls: list[dict] = []

    def fake_run(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.chdir(project_root)
    monkeypatch.setattr("headroom.lean_ctx.get_lean_ctx_path", lambda: lean_ctx)
    monkeypatch.setattr("headroom.cli.wrap.subprocess.run", fake_run)

    assert _setup_lean_ctx_agent("codex") == lean_ctx

    assert calls
    cwd = Path(calls[0]["kwargs"]["cwd"])
    assert cwd != project_root
    assert project_root not in cwd.parents


def test_wrap_codex_prepare_only_updates_config(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    config_file = tmp_path / ".codex" / "config.toml"
    assert config_file.exists()
    assert 'model_provider = "headroom"' in config_file.read_text()
    assert 'base_url = "http://127.0.0.1:8787/v1"' in config_file.read_text()


def test_wrap_codex_prepare_only_uses_lean_ctx_when_configured(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with patch("headroom.cli.wrap._ensure_rtk_binary") as ensure_rtk:
            with patch(
                "headroom.cli.wrap._setup_lean_ctx_agent",
                return_value=Path("lean-ctx"),
            ) as setup:
                result = runner.invoke(
                    main,
                    ["wrap", "codex", "--prepare-only", "--no-mcp", "--no-serena"],
                )

        assert result.exit_code == 0, result.output
        ensure_rtk.assert_not_called()
        setup.assert_called_once_with("codex", verbose=False)
        assert not Path("AGENTS.md").exists()


def test_wrap_codex_prepare_only_accepts_no_context_tool_alias(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with patch("headroom.cli.wrap._ensure_rtk_binary") as ensure_rtk:
            with patch("headroom.cli.wrap._setup_lean_ctx_agent") as setup:
                result = runner.invoke(
                    main,
                    [
                        "wrap",
                        "codex",
                        "--prepare-only",
                        "--no-context-tool",
                        "--no-mcp",
                        "--no-serena",
                    ],
                )

        assert result.exit_code == 0, result.output
        ensure_rtk.assert_not_called()
        setup.assert_not_called()


def test_wrap_aider_prepare_only_injects_conventions(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=Path("rtk")):
            result = runner.invoke(main, ["wrap", "aider", "--prepare-only"])

        assert result.exit_code == 0, result.output
        conventions = Path("CONVENTIONS.md")
        assert conventions.exists()
        assert "headroom:rtk-instructions" in conventions.read_text()


def test_wrap_cursor_prepare_only_injects_cursorrules(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=Path("rtk")):
            result = runner.invoke(main, ["wrap", "cursor", "--prepare-only"])

        assert result.exit_code == 0, result.output
        cursorrules = Path(".cursorrules")
        assert cursorrules.exists()
        assert "headroom:rtk-instructions" in cursorrules.read_text()


def test_wrap_cursor_prepare_only_uses_lean_ctx_when_configured(
    monkeypatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with patch("headroom.cli.wrap._ensure_rtk_binary") as ensure_rtk:
            with patch(
                "headroom.cli.wrap._setup_lean_ctx_agent",
                return_value=Path("lean-ctx"),
            ) as setup:
                result = runner.invoke(main, ["wrap", "cursor", "--prepare-only"])

        assert result.exit_code == 0, result.output
        ensure_rtk.assert_not_called()
        setup.assert_called_once_with("cursor", verbose=False)
        assert not Path(".cursorrules").exists()


def test_wrap_openclaw_prepare_only_emits_config_without_python_default() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "wrap",
            "openclaw",
            "--prepare-only",
            "--gateway-provider-id",
            "codex",
            "--gateway-provider-id",
            "anthropic",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["enabled"] is True
    assert payload["config"]["proxyPort"] == 8787
    assert payload["config"]["gatewayProviderIds"] == ["codex", "anthropic"]
    assert "pythonPath" not in payload["config"]


def test_unwrap_openclaw_prepare_only_preserves_unmanaged_config() -> None:
    runner = CliRunner()
    existing_entry = json.dumps(
        {
            "enabled": True,
            "config": {
                "pythonPath": "C:\\Python312\\python.exe",
                "proxyPort": 8787,
                "customFlag": True,
            },
        }
    )

    result = runner.invoke(
        main,
        [
            "unwrap",
            "openclaw",
            "--prepare-only",
            "--existing-entry-json",
            existing_entry,
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"enabled": False, "config": {"customFlag": True}}

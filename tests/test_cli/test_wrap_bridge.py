"""Tests for Docker-bridge wrap preparation flows."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture(autouse=True)
def _no_retired_context_tool_env(monkeypatch) -> None:
    """Keep a developer's exported HEADROOM_CONTEXT_TOOL from failing every test.

    The var is now rejected outright, so leaving it set in the ambient
    environment would abort each wrap invocation below.
    """
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)


def _set_test_home(monkeypatch, tmp_path: Path) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)


def test_wrap_claude_prepare_only_skips_host_binary_lookup() -> None:
    runner = CliRunner()

    with patch("headroom.cli.wrap.shutil.which") as which_mock:
        result = runner.invoke(main, ["wrap", "claude", "--prepare-only"])

    assert result.exit_code == 0, result.output
    which_mock.assert_not_called()


def test_wrap_codex_prepare_only_updates_config(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    config_file = tmp_path / ".codex" / "config.toml"
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert 'model_provider = "headroom"' in content
    assert 'base_url = "http://127.0.0.1:8787/v1"' in content


def test_wrap_grok_build_uses_actual_proxy_port(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    def fake_watcher(**kwargs) -> None:
        kwargs["print_setup_lines"](9999)

    monkeypatch.setattr("headroom.cli.wrap._run_proxy_only_watcher", fake_watcher)

    result = runner.invoke(main, ["wrap", "grok-build", "--port", "8787"])

    assert result.exit_code == 0, result.output
    config_file = tmp_path / ".grok" / "config.toml"
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert 'base_url = "http://127.0.0.1:9999/' in content
    assert "http://127.0.0.1:8787/" not in content
    assert "http://127.0.0.1:9999/" in result.output
    assert "http://127.0.0.1:8787/" not in result.output


def test_wrap_rejects_retired_context_tool_flag(monkeypatch, tmp_path: Path) -> None:
    """A surviving --context-tool must fail loudly, not be silently ignored.

    rtk / lean-ctx are gone, but the flag lives on in shell profiles, scripts and
    CI jobs. Accepting it as a no-op would look like Headroom had quietly stopped
    filtering; the user needs to be told the feature was removed.
    """
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        result = runner.invoke(
            main,
            ["wrap", "codex", "--prepare-only", "--no-context-tool", "--no-mcp", "--no-serena"],
        )

    assert result.exit_code != 0
    assert "have been removed from Headroom" in result.output


def test_wrap_rejects_retired_context_tool_env(monkeypatch, tmp_path: Path) -> None:
    """An exported HEADROOM_CONTEXT_TOOL fails too, with the same message.

    The env var is the form most likely to be left behind in a shell rc, where
    it would otherwise never surface.
    """
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--no-mcp", "--no-serena"])

    assert result.exit_code != 0
    assert "have been removed from Headroom" in result.output


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

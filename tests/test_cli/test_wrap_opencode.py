"""Tests for headroom wrap opencode and headroom unwrap opencode CLI commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def opencode_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up isolated OpenCode environment for testing."""
    config_dir = tmp_path / ".config" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "opencode.json"
    config_file.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "model": "opencode/mimo-v2.5-pro",
        "provider": {
            "mimo": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "MiMo",
                "options": {
                    "baseURL": "https://custom.api.com/v1",
                    "apiKey": "sk-test"
                }
            }
        }
    }))
    auth_dir = tmp_path / ".local" / "share" / "opencode"
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth_file = auth_dir / "auth.json"
    auth_file.write_text(json.dumps({
        "opencode": {"type": "api", "key": "sk-test-zen"},
        "opencode-go": {"type": "api", "key": "sk-test-go"},
        "deepseek": {"type": "api", "key": "sk-test-ds"}
    }))
    monkeypatch.setattr(
        "headroom.providers.opencode.runtime._AUTH_PATH",
        auth_file,
    )
    monkeypatch.setattr(
        "headroom.providers.opencode.runtime._USER_CONFIG_CANDIDATES",
        (config_file,),
    )
    monkeypatch.setattr(
        "headroom.providers.opencode.config._opencode_config_path",
        lambda: config_file,
    )
    monkeypatch.setattr(
        "headroom.install.paths.opencode_config_path",
        lambda: config_file,
    )
    return config_file, auth_file


# ============================================================================
# wrap opencode --help
# ============================================================================


class TestWrapOpencodeHelp:
    def test_help_shows_routing_mode(self, runner: CliRunner):
        result = runner.invoke(main, ["wrap", "opencode", "--help"])
        assert result.exit_code == 0
        assert "routing-mode" in result.output
        assert "multi" in result.output
        assert "single" in result.output

    def test_help_shows_all_flags(self, runner: CliRunner):
        result = runner.invoke(main, ["wrap", "opencode", "--help"])
        assert result.exit_code == 0
        assert "--no-rtk" in result.output or "--no-context-tool" in result.output
        assert "--no-mcp" in result.output
        assert "--no-serena" in result.output
        assert "--memory" in result.output
        assert "--learn" in result.output


# ============================================================================
# wrap opencode --prepare-only
# ============================================================================


class TestWrapOpencodePrepareOnly:
    def test_prepare_only_exits_cleanly(
        self, runner: CliRunner, opencode_env: tuple
    ):
        config_file, auth_file = opencode_env
        with patch(
            "headroom.cli.wrap._setup_context_tool_for_agent",
            return_value=None,
        ), patch(
            "headroom.cli.wrap._ensure_rtk_binary",
            return_value=None,
        ), patch(
            "headroom.cli.wrap._inject_rtk_instructions",
            return_value=False,
        ):
            result = runner.invoke(
                main,
                ["wrap", "opencode", "--prepare-only", "--no-mcp", "--no-serena"],
            )
            assert result.exit_code == 0

    def test_prepare_only_snapshots_config(
        self, runner: CliRunner, opencode_env: tuple
    ):
        config_file, auth_file = opencode_env
        backup_file = config_file.with_suffix(".json.headroom-backup")
        with patch(
            "headroom.cli.wrap._setup_context_tool_for_agent",
            return_value=None,
        ), patch(
            "headroom.cli.wrap._ensure_rtk_binary",
            return_value=None,
        ), patch(
            "headroom.cli.wrap._inject_rtk_instructions",
            return_value=False,
        ):
            result = runner.invoke(
                main,
                ["wrap", "opencode", "--prepare-only", "--no-mcp", "--no-serena"],
            )
            assert backup_file.exists()


# ============================================================================
# unwrap opencode --help
# ============================================================================


class TestUnwrapOpencodeHelp:
    def test_help_shows_options(self, runner: CliRunner):
        result = runner.invoke(main, ["unwrap", "opencode", "--help"])
        assert result.exit_code == 0
        assert "--no-stop-proxy" in result.output
        assert "--port" in result.output


# ============================================================================
# unwrap opencode
# ============================================================================


class TestUnwrapOpencode:
    def test_unwrap_noop_when_no_config(self, runner: CliRunner, tmp_path: Path):
        monkeypatch = pytest.MonkeyPatch()
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "opencode.json"
        monkeypatch.setattr(
            "headroom.providers.opencode.config._opencode_config_path",
            lambda: config_file,
        )
        monkeypatch.setattr(
            "headroom.cli.wrap.opencode_config_paths",
            lambda: (config_file, config_file.with_suffix(".json.headroom-backup")),
        )
        with patch(
            "headroom.cli.wrap.find_opencode_ports",
            return_value=[],
        ), patch(
            "headroom.mcp_registry.OpencodeRegistrar.detect",
            return_value=False,
        ):
            result = runner.invoke(main, ["unwrap", "opencode", "--no-stop-proxy"])
            assert result.exit_code == 0
            assert "Nothing to undo" in result.output
        monkeypatch.undo()

    def test_unwrap_restores_from_backup(self, runner: CliRunner, opencode_env: tuple):
        config_file, auth_file = opencode_env
        backup_file = config_file.with_suffix(".json.headroom-backup")
        original_content = '{"model": "original"}'
        backup_file.write_text(original_content)

        with patch(
            "headroom.cli.wrap.find_opencode_ports",
            return_value=[],
        ), patch(
            "headroom.mcp_registry.OpencodeRegistrar.detect",
            return_value=False,
        ):
            result = runner.invoke(main, ["unwrap", "opencode", "--no-stop-proxy"])
            assert result.exit_code == 0
            assert "Restored" in result.output
            assert config_file.read_text() == original_content
            assert not backup_file.exists()


# ============================================================================
# build_launch_env integration
# ============================================================================


class TestBuildLaunchEnvIntegration:
    def test_env_contains_opencode_config_content(
        self, opencode_env: tuple, monkeypatch: pytest.MonkeyPatch
    ):
        from headroom.providers.opencode import build_launch_env

        env, display = build_launch_env(8787)
        assert "OPENCODE_CONFIG_CONTENT" in env
        content = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert "provider" in content

    def test_env_preserves_existing_openai_base_url(
        self, opencode_env: tuple
    ):
        from headroom.providers.opencode import build_launch_env

        environ = {"OPENAI_BASE_URL": "https://custom.openai.com/v1"}
        env, display = build_launch_env(8787, environ=environ)
        assert env["OPENAI_BASE_URL"] == "https://custom.openai.com/v1"

    def test_env_preserves_existing_anthropic_base_url(
        self, opencode_env: tuple
    ):
        from headroom.providers.opencode import build_launch_env

        environ = {"ANTHROPIC_BASE_URL": "https://custom.anthropic.com/v1"}
        env, display = build_launch_env(8787, environ=environ)
        assert env["ANTHROPIC_BASE_URL"] == "https://custom.anthropic.com/v1"

    def test_single_routing_mode_header_format(self, opencode_env: tuple):
        from headroom.providers.opencode import build_launch_env

        env, display = build_launch_env(8787, routing_mode="single")
        content = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        for name, entry in content["provider"].items():
            if name != "openai":
                assert "headers" in entry["options"]
                assert "x-headroom-base-url" in entry["options"]["headers"]

    def test_multi_routing_mode_no_headers(self, opencode_env: tuple):
        from headroom.providers.opencode import build_launch_env

        env, display = build_launch_env(8787, routing_mode="multi")
        content = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        for name, entry in content["provider"].items():
            assert "headers" not in entry["options"]

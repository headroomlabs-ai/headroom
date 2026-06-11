"""Tests for the OpenCode provider integration.

Covers:
- Runtime environment construction (build_launch_env)
- Provider-scope config patch/revert (apply_provider_scope, revert_provider_scope)
- MCP registrar (register/unregister, malformed config handling)
- wrap --prepare-only CLI path
- Learn plugin with a small SQLite fixture
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from headroom.mcp_registry.base import RegisterStatus, ServerSpec
from headroom.mcp_registry.opencode import MalformedConfigError, OpenCodeRegistrar, _read_json
from headroom.providers.opencode.runtime import build_launch_env

# ======================================================================
# build_launch_env
# ======================================================================


class TestBuildLaunchEnv:
    def test_sets_anthropic_and_openai_urls(self) -> None:
        env, display = build_launch_env(8787, {})
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
        assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"

    def test_does_not_set_github_copilot_host(self) -> None:
        env, display = build_launch_env(8787, {})
        assert "GITHUB_COPILOT_HOST" not in env

    def test_preserves_existing_env(self) -> None:
        env, _ = build_launch_env(9000, {"MY_VAR": "keep"})
        assert env["MY_VAR"] == "keep"
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"

    def test_display_lines_match_env(self) -> None:
        env, display = build_launch_env(8787, {})
        assert f"ANTHROPIC_BASE_URL={env['ANTHROPIC_BASE_URL']}" in display
        assert f"OPENAI_BASE_URL={env['OPENAI_BASE_URL']}" in display

    def test_custom_port(self) -> None:
        env, _ = build_launch_env(1234, {})
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1234"
        assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:1234/v1"


# ======================================================================
# apply_provider_scope / revert_provider_scope
# ======================================================================


class TestProviderScope:
    def _manifest(self, port: int = 8787, scope: str = "provider"):  # noqa: ANN202
        from headroom.install.models import DeploymentManifest

        return DeploymentManifest(
            profile="default",
            preset="default",
            runtime_kind="local",
            supervisor_kind="none",
            scope=scope,
            provider_mode="wrap",
            targets=["opencode"],
            port=port,
            host="127.0.0.1",
            backend="github-copilot",
        )

    def test_apply_creates_config_with_base_urls(self, tmp_path: Path) -> None:
        from headroom.providers.opencode.install import apply_provider_scope

        config_path = tmp_path / "opencode.json"
        with patch(
            "headroom.providers.opencode.install.opencode_config_path", return_value=config_path
        ):
            result = apply_provider_scope(self._manifest())

        assert result is not None
        payload = json.loads(config_path.read_text())
        assert (
            payload["provider"]["github-copilot"]["options"]["baseURL"] == "http://127.0.0.1:8787"
        )
        assert payload["provider"]["anthropic"]["options"]["baseURL"] == "http://127.0.0.1:8787"
        assert payload["provider"]["openai"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"

    def test_apply_preserves_existing_keys(self, tmp_path: Path) -> None:
        from headroom.providers.opencode.install import apply_provider_scope

        config_path = tmp_path / "opencode.json"
        config_path.write_text(json.dumps({"mcp": {"headroom": {}}, "theme": "dark"}))

        with patch(
            "headroom.providers.opencode.install.opencode_config_path", return_value=config_path
        ):
            apply_provider_scope(self._manifest())

        payload = json.loads(config_path.read_text())
        assert payload["mcp"] == {"headroom": {}}
        assert payload["theme"] == "dark"

    def test_revert_restores_previous_urls(self, tmp_path: Path) -> None:
        from headroom.providers.opencode.install import apply_provider_scope, revert_provider_scope

        config_path = tmp_path / "opencode.json"
        # Pre-existing config with a custom base URL for anthropic.
        config_path.write_text(
            json.dumps({"provider": {"anthropic": {"options": {"baseURL": "https://custom.api"}}}})
        )

        with patch(
            "headroom.providers.opencode.install.opencode_config_path", return_value=config_path
        ):
            manifest = self._manifest()
            mutation = apply_provider_scope(manifest)
            assert mutation is not None

            # Now revert.
            revert_provider_scope(mutation, manifest)

        payload = json.loads(config_path.read_text())
        # anthropic should be restored to its previous value.
        assert payload["provider"]["anthropic"]["options"]["baseURL"] == "https://custom.api"
        # github-copilot had no previous value, so baseURL should be removed.
        assert "baseURL" not in payload["provider"].get("github-copilot", {}).get("options", {})

    def test_apply_returns_none_for_wrong_scope(self) -> None:
        from headroom.providers.opencode.install import apply_provider_scope

        assert apply_provider_scope(self._manifest(scope="global")) is None


# ======================================================================
# MCP Registrar
# ======================================================================


def _spec(env: dict[str, str] | None = None) -> ServerSpec:
    return ServerSpec(
        name="headroom",
        command="headroom",
        args=("mcp", "serve"),
        env=env or {},
    )


class TestOpenCodeRegistrar:
    def test_detect_true_when_config_exists(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        config.write_text("{}")
        reg = OpenCodeRegistrar(config_path=config)
        assert reg.detect() is True

    def test_detect_false_when_no_config_no_binary(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        with patch("shutil.which", return_value=None):
            reg = OpenCodeRegistrar(config_path=config)
            assert reg.detect() is False

    def test_register_creates_mcp_entry(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        reg = OpenCodeRegistrar(config_path=config)
        result = reg.register_server(_spec())
        assert result.status == RegisterStatus.REGISTERED
        payload = json.loads(config.read_text())
        assert payload["mcp"]["headroom"]["command"] == ["headroom", "mcp", "serve"]
        assert payload["mcp"]["headroom"]["enabled"] is True

    def test_register_already_when_identical(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        reg = OpenCodeRegistrar(config_path=config)
        reg.register_server(_spec())
        result = reg.register_server(_spec())
        assert result.status == RegisterStatus.ALREADY

    def test_register_mismatch_when_different(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        reg = OpenCodeRegistrar(config_path=config)
        reg.register_server(_spec(env={"FOO": "bar"}))
        result = reg.register_server(_spec())
        assert result.status == RegisterStatus.MISMATCH

    def test_register_force_overwrites(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        reg = OpenCodeRegistrar(config_path=config)
        reg.register_server(_spec(env={"FOO": "bar"}))
        result = reg.register_server(_spec(), force=True)
        assert result.status == RegisterStatus.REGISTERED
        payload = json.loads(config.read_text())
        assert "environment" not in payload["mcp"]["headroom"]

    def test_unregister_removes_entry(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        reg = OpenCodeRegistrar(config_path=config)
        reg.register_server(_spec())
        assert reg.unregister_server("headroom") is True
        payload = json.loads(config.read_text())
        assert "headroom" not in payload.get("mcp", {})

    def test_unregister_returns_false_when_missing(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        config.write_text("{}")
        reg = OpenCodeRegistrar(config_path=config)
        assert reg.unregister_server("headroom") is False

    def test_register_fails_on_malformed_config(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        config.write_text("not valid json {{{{")
        reg = OpenCodeRegistrar(config_path=config)
        result = reg.register_server(_spec())
        assert result.status == RegisterStatus.FAILED
        assert "malformed" in (result.detail or "").lower()

    def test_get_server_returns_none_on_malformed_config(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        config.write_text("[1, 2, 3]")
        reg = OpenCodeRegistrar(config_path=config)
        assert reg.get_server("headroom") is None

    def test_unregister_returns_false_on_malformed_config(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        config.write_text("broken json")
        reg = OpenCodeRegistrar(config_path=config)
        assert reg.unregister_server("headroom") is False

    def test_get_server_round_trip(self, tmp_path: Path) -> None:
        config = tmp_path / "opencode.json"
        spec = _spec(env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"})
        reg = OpenCodeRegistrar(config_path=config)
        reg.register_server(spec)
        got = reg.get_server("headroom")
        assert got is not None
        assert got.command == "headroom"
        assert got.args == ("mcp", "serve")
        assert got.env == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"}


# ======================================================================
# _read_json raises on malformed config
# ======================================================================


class TestReadJson:
    def test_returns_empty_dict_for_missing_file(self, tmp_path: Path) -> None:
        assert _read_json(tmp_path / "nope.json") == {}

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all")
        with pytest.raises(MalformedConfigError) as exc_info:
            _read_json(bad)
        assert "Cannot parse" in str(exc_info.value)

    def test_raises_when_top_level_is_not_dict(self, tmp_path: Path) -> None:
        arr = tmp_path / "array.json"
        arr.write_text("[1, 2, 3]")
        with pytest.raises(MalformedConfigError):
            _read_json(arr)

    def test_returns_dict_for_valid_json(self, tmp_path: Path) -> None:
        good = tmp_path / "good.json"
        good.write_text('{"mcp": {}}')
        assert _read_json(good) == {"mcp": {}}


# ======================================================================
# Learn plugin
# ======================================================================


class TestOpenCodeLearnPlugin:
    def _create_fixture_db(self, db_path: Path) -> None:
        """Create a minimal OpenCode SQLite DB with one project and session."""
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE project (
                id TEXT PRIMARY KEY,
                name TEXT,
                worktree TEXT
            );
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                time_created INTEGER
            );
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT,
                data TEXT,
                time_created INTEGER
            );

            INSERT INTO project VALUES ('proj1', 'test-project', '/tmp/test-project');
            INSERT INTO session VALUES ('sess1', 'proj1', 1700000000000);
            INSERT INTO message VALUES ('msg1', 'sess1', 'assistant');
            INSERT INTO part VALUES ('part1', 'msg1', '{"type":"tool","tool":"bash","callID":"call1","state":{"status":"error","input":{"command":"cat missing.txt"},"output":"cat: missing.txt: No such file or directory"}}', 1700000001000);
            INSERT INTO part VALUES ('part2', 'msg1', '{"type":"tool","tool":"bash","callID":"call2","state":{"status":"completed","input":{"command":"echo hi"},"output":"hi"}}', 1700000002000);
            """
        )
        conn.close()

    def test_detect_true_when_db_exists(self, tmp_path: Path) -> None:
        from headroom.learn.plugins.opencode import OpenCodePlugin

        db = tmp_path / "opencode.db"
        self._create_fixture_db(db)
        plugin = OpenCodePlugin(db_path=db)
        assert plugin.detect() is True

    def test_detect_false_when_db_missing(self, tmp_path: Path) -> None:
        from headroom.learn.plugins.opencode import OpenCodePlugin

        plugin = OpenCodePlugin(db_path=tmp_path / "nope.db")
        assert plugin.detect() is False

    def test_discover_projects(self, tmp_path: Path) -> None:
        from headroom.learn.plugins.opencode import OpenCodePlugin

        db = tmp_path / "opencode.db"
        self._create_fixture_db(db)
        plugin = OpenCodePlugin(db_path=db)
        projects = plugin.discover_projects()
        assert len(projects) == 1
        assert projects[0].name == "test-project"
        assert projects[0].project_path == Path("/tmp/test-project")

    def test_scan_project_finds_error_tool_calls(self, tmp_path: Path) -> None:
        from headroom.learn.plugins.opencode import OpenCodePlugin

        db = tmp_path / "opencode.db"
        self._create_fixture_db(db)
        plugin = OpenCodePlugin(db_path=db)
        projects = plugin.discover_projects()
        sessions = plugin.scan_project(projects[0])
        assert len(sessions) == 1
        # Should have 2 tool calls, one error and one success.
        assert len(sessions[0].tool_calls) == 2
        assert sessions[0].tool_calls[0].is_error is True
        assert sessions[0].tool_calls[0].name == "bash"
        assert sessions[0].tool_calls[1].is_error is False

    def test_plugin_name(self, tmp_path: Path) -> None:
        from headroom.learn.plugins.opencode import OpenCodePlugin

        plugin = OpenCodePlugin(db_path=tmp_path / "nope.db")
        assert plugin.name == "opencode"
        assert plugin.display_name == "OpenCode"


# ======================================================================
# wrap opencode --prepare-only
# ======================================================================


class TestWrapPrepareOnly:
    def test_prepare_only_exits_without_launching(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from headroom.cli.wrap import wrap

        runner = CliRunner()
        with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
            result = runner.invoke(
                wrap,
                ["opencode", "--prepare-only", "--no-proxy"],
                catch_exceptions=False,
            )
        # --prepare-only should return early without errors.
        assert result.exit_code == 0

    def test_prepare_only_does_not_require_opencode_binary(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from headroom.cli.wrap import wrap

        runner = CliRunner()
        with (
            patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None),
            patch("shutil.which", return_value=None),
        ):
            result = runner.invoke(
                wrap,
                ["opencode", "--prepare-only", "--no-proxy"],
                catch_exceptions=False,
            )
        # Should exit 0 because --prepare-only returns before checking for the binary.
        assert result.exit_code == 0

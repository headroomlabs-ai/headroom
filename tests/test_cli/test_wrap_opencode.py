"""Tests for `headroom wrap opencode` and `headroom unwrap opencode`.

OpenCode reads its provider configuration from a project-local ``opencode.json``
in the current working directory, so these tests ``chdir`` into a temp project.
The unit tests call the inject/restore helpers directly; the integration tests
drive the real Click commands the way a user would.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def project_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A clean working directory that stands in for an OpenCode project root."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests: inject / snapshot / restore helpers against a temp project dir
# ---------------------------------------------------------------------------


class TestInjectOpencodeConfig:
    def test_inject_creates_config_with_schema_and_overrides(self, project_dir: Path) -> None:
        wrap_mod._inject_opencode_provider_config(8787)

        config_file = project_dir / "opencode.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["$schema"] == "https://opencode.ai/config.json"
        assert data["provider"]["anthropic"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"
        assert data["provider"]["openai"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"

    def test_inject_applies_project_prefix(self, project_dir: Path) -> None:
        wrap_mod._inject_opencode_provider_config(8787, project="my-repo")

        data = json.loads((project_dir / "opencode.json").read_text())
        assert (
            data["provider"]["openai"]["options"]["baseURL"] == "http://127.0.0.1:8787/p/my-repo/v1"
        )

    def test_inject_backs_up_existing_config_and_preserves_keys(self, project_dir: Path) -> None:
        config_file = project_dir / "opencode.json"
        original = json.dumps(
            {
                "model": "openai/gpt-4o",
                "provider": {"openai": {"options": {"apiKey": "{env:OPENAI_API_KEY}"}}},
            },
            indent=2,
        )
        config_file.write_text(original)

        wrap_mod._inject_opencode_provider_config(8787)

        backup = project_dir / "opencode.json.headroom-backup"
        assert backup.exists()
        assert backup.read_text() == original

        data = json.loads(config_file.read_text())
        # User keys preserved; baseURL merged in alongside the apiKey.
        assert data["model"] == "openai/gpt-4o"
        assert data["provider"]["openai"]["options"]["apiKey"] == "{env:OPENAI_API_KEY}"
        assert data["provider"]["openai"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"
        # We do not add $schema to a pre-existing user config.
        assert "$schema" not in data

    def test_inject_is_idempotent_with_port_change(self, project_dir: Path) -> None:
        wrap_mod._inject_opencode_provider_config(8787)
        wrap_mod._inject_opencode_provider_config(8787)
        wrap_mod._inject_opencode_provider_config(9999)  # port change

        data = json.loads((project_dir / "opencode.json").read_text())
        assert data["provider"]["openai"]["options"]["baseURL"] == "http://127.0.0.1:9999/v1"
        assert data["provider"]["anthropic"]["options"]["baseURL"] == "http://127.0.0.1:9999/v1"
        # No duplicate providers, no stale 8787 entry anywhere.
        assert set(data["provider"]) == {"anthropic", "openai"}
        assert "8787" not in json.dumps(data)

    def test_inject_overwrites_malformed_json_after_backup(self, project_dir: Path) -> None:
        config_file = project_dir / "opencode.json"
        malformed = "{ this is not valid json "
        config_file.write_text(malformed)

        wrap_mod._inject_opencode_provider_config(8787)

        # Backup keeps the exact malformed bytes; the live file is now valid.
        backup = project_dir / "opencode.json.headroom-backup"
        assert backup.read_text() == malformed
        data = json.loads(config_file.read_text())
        assert data["provider"]["openai"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"


class TestSnapshotOpencodeConfig:
    def test_no_backup_when_already_wrapped(self, project_dir: Path) -> None:
        config_file = project_dir / "opencode.json"
        backup_file = project_dir / "opencode.json.headroom-backup"
        config_file.write_text(
            json.dumps(
                {"provider": {"openai": {"options": {"baseURL": "http://127.0.0.1:8787/v1"}}}}
            )
        )

        wrap_mod._snapshot_opencode_config_if_unwrapped(config_file, backup_file)
        assert not backup_file.exists()

    def test_no_backup_when_config_missing(self, project_dir: Path) -> None:
        config_file = project_dir / "opencode.json"
        backup_file = project_dir / "opencode.json.headroom-backup"

        wrap_mod._snapshot_opencode_config_if_unwrapped(config_file, backup_file)
        assert not backup_file.exists()

    def test_does_not_overwrite_existing_backup(self, project_dir: Path) -> None:
        config_file = project_dir / "opencode.json"
        backup_file = project_dir / "opencode.json.headroom-backup"
        config_file.write_text("{}")
        backup_file.write_text('{"original": true}')

        wrap_mod._snapshot_opencode_config_if_unwrapped(config_file, backup_file)
        assert backup_file.read_text() == '{"original": true}'


class TestRestoreOpencodeConfig:
    def test_restore_from_backup_round_trips(self, project_dir: Path) -> None:
        config_file = project_dir / "opencode.json"
        original = json.dumps({"model": "anthropic/claude-sonnet-4-5"}, indent=2)
        config_file.write_text(original)

        wrap_mod._inject_opencode_provider_config(8787)
        assert "127.0.0.1" in config_file.read_text()

        status, _ = wrap_mod._restore_opencode_provider_config()
        assert status == "restored"
        assert config_file.read_text() == original
        assert not (project_dir / "opencode.json.headroom-backup").exists()

    def test_unwrap_removes_headroom_only_file(self, project_dir: Path) -> None:
        wrap_mod._inject_opencode_provider_config(8787)
        config_file = project_dir / "opencode.json"
        assert config_file.exists()

        status, _ = wrap_mod._restore_opencode_provider_config()
        assert status == "removed"
        assert not config_file.exists()

    def test_unwrap_cleans_overrides_without_backup(self, project_dir: Path) -> None:
        config_file = project_dir / "opencode.json"
        config_file.write_text(
            json.dumps(
                {
                    "model": "openai/gpt-4o",
                    "provider": {
                        "openai": {
                            "options": {
                                "baseURL": "http://127.0.0.1:8787/v1",
                                "apiKey": "{env:OPENAI_API_KEY}",
                            }
                        }
                    },
                },
                indent=2,
            )
        )

        status, _ = wrap_mod._restore_opencode_provider_config()
        assert status == "cleaned"
        data = json.loads(config_file.read_text())
        assert data["model"] == "openai/gpt-4o"
        assert data["provider"]["openai"]["options"] == {"apiKey": "{env:OPENAI_API_KEY}"}
        assert "baseURL" not in data["provider"]["openai"]["options"]

    def test_unwrap_is_noop_when_never_wrapped(self, project_dir: Path) -> None:
        status, _ = wrap_mod._restore_opencode_provider_config()
        assert status == "noop"


# ---------------------------------------------------------------------------
# Integration tests: full `headroom wrap opencode` / `headroom unwrap opencode`
# ---------------------------------------------------------------------------


def test_wrap_opencode_prepare_only_injects_config_and_rtk(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    with monkeypatch.context() as m:
        m.setattr(wrap_mod, "_ensure_rtk_binary", lambda verbose=False: tmp_path / "rtk")
        result = runner.invoke(main, ["wrap", "opencode", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    config_file = tmp_path / "opencode.json"
    assert config_file.exists()
    data = json.loads(config_file.read_text())
    # The live command attributes savings to the launch directory, so the
    # baseURL carries a /p/<project> prefix derived from the cwd basename.
    base_url = data["provider"]["openai"]["options"]["baseURL"]
    assert base_url.startswith("http://127.0.0.1:8787/")
    assert base_url.endswith("/v1")
    assert wrap_mod._opencode_config_has_headroom_overrides(data)
    # rtk guidance injected into AGENTS.md (the file OpenCode reads).
    agents_md = tmp_path / "AGENTS.md"
    assert agents_md.exists()
    assert wrap_mod._RTK_MARKER in agents_md.read_text()


def test_wrap_opencode_prepare_only_no_context_tool_skips_agents_md(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        main, ["wrap", "opencode", "--prepare-only", "--no-context-tool", "--port", "8787"]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "opencode.json").exists()
    assert not (tmp_path / "AGENTS.md").exists()


def test_wrap_unwrap_opencode_round_trips_end_to_end(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "opencode.json"
    original = json.dumps({"model": "anthropic/claude-sonnet-4-5"}, indent=2) + "\n"
    config_file.write_text(original)

    with monkeypatch.context() as m:
        m.setattr(wrap_mod, "_ensure_rtk_binary", lambda verbose=False: None)
        wrap_result = runner.invoke(main, ["wrap", "opencode", "--prepare-only", "--port", "8787"])
    assert wrap_result.exit_code == 0, wrap_result.output
    assert "127.0.0.1" in config_file.read_text()

    stopped: list[int] = []
    with monkeypatch.context() as m:
        m.setattr(
            wrap_mod,
            "_stop_local_proxy_for_unwrap",
            lambda port: stopped.append(port) or "stopped",
        )
        unwrap_result = runner.invoke(main, ["unwrap", "opencode", "--port", "9999"])

    assert unwrap_result.exit_code == 0, unwrap_result.output
    assert config_file.read_text() == original
    assert not (tmp_path / "opencode.json.headroom-backup").exists()
    assert stopped == [9999]


def test_unwrap_opencode_noop_when_never_wrapped(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(main, ["unwrap", "opencode", "--no-stop-proxy"])
    assert result.exit_code == 0, result.output
    assert "Nothing to undo" in result.output


# ---------------------------------------------------------------------------
# MCP retrieve tool parity (mirrors `wrap codex` --no-mcp behaviour)
# ---------------------------------------------------------------------------


def test_wrap_opencode_registers_headroom_mcp(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(wrap_mod, "_ensure_rtk_binary", lambda verbose=False: None)
        result = runner.invoke(
            main, ["wrap", "opencode", "--prepare-only", "--no-serena", "--port", "8787"]
        )
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "opencode.json").read_text())
    entry = data["mcp"]["headroom"]
    assert entry["type"] == "local"
    assert entry["command"] == ["headroom", "mcp", "serve"]
    assert entry["enabled"] is True


def test_wrap_opencode_no_mcp_skips_headroom_server(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(wrap_mod, "_ensure_rtk_binary", lambda verbose=False: None)
        result = runner.invoke(
            main,
            ["wrap", "opencode", "--prepare-only", "--no-mcp", "--no-serena", "--no-context-tool"],
        )
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "opencode.json").read_text())
    assert "headroom" not in data.get("mcp", {})
    # Provider routing is still injected even without the MCP tool.
    assert "127.0.0.1" in data["provider"]["openai"]["options"]["baseURL"]


def test_wrap_unwrap_opencode_reverts_mcp_via_backup(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "opencode.json"
    original = json.dumps({"model": "anthropic/claude-sonnet-4-5"}, indent=2) + "\n"
    config_file.write_text(original)

    with monkeypatch.context() as m:
        m.setattr(wrap_mod, "_ensure_rtk_binary", lambda verbose=False: None)
        runner.invoke(main, ["wrap", "opencode", "--prepare-only", "--no-serena", "--port", "8787"])
    assert "headroom" in json.loads(config_file.read_text())["mcp"]

    with monkeypatch.context() as m:
        m.setattr(wrap_mod, "_stop_local_proxy_for_unwrap", lambda port: "stopped")
        result = runner.invoke(main, ["unwrap", "opencode", "--port", "9999"])
    assert result.exit_code == 0, result.output
    # Backup-restore reverts providers AND the MCP server byte-for-byte.
    assert config_file.read_text() == original


def test_restore_opencode_no_backup_strips_provider_and_mcp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "opencode.json"
    # Crash case: wrapped config with no pre-wrap backup.
    config_file.write_text(
        json.dumps(
            {
                "model": "openai/gpt-4o",
                "provider": {"openai": {"options": {"baseURL": "http://127.0.0.1:8787/v1"}}},
                "mcp": {
                    "headroom": {
                        "type": "local",
                        "command": ["headroom", "mcp", "serve"],
                        "enabled": True,
                    }
                },
            },
            indent=2,
        )
    )

    status, _ = wrap_mod._restore_opencode_provider_config()
    assert status == "cleaned"
    data = json.loads(config_file.read_text())
    assert data == {"model": "openai/gpt-4o"}


def test_restore_opencode_removes_headroom_only_file_with_mcp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "opencode.json"
    config_file.write_text(
        json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "provider": {"openai": {"options": {"baseURL": "http://127.0.0.1:8787/v1"}}},
                "mcp": {
                    "headroom": {
                        "type": "local",
                        "command": ["headroom", "mcp", "serve"],
                        "enabled": True,
                    }
                },
            }
        )
    )

    status, _ = wrap_mod._restore_opencode_provider_config()
    assert status == "removed"
    assert not config_file.exists()


# ---------------------------------------------------------------------------
# Savings parity: opencode joins the agent-90 high-savings profile set
# ---------------------------------------------------------------------------


def test_start_proxy_applies_agent_90_for_opencode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    popen_kwargs: dict[str, object] = {}

    class FakeProc:
        returncode = None

        def poll(self) -> None:
            return None

    def fake_popen(*args: object, **kwargs: object) -> FakeProc:
        popen_kwargs.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(wrap_mod, "_get_log_path", lambda: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_mod.subprocess, "Popen", fake_popen)

    wrap_mod._start_proxy(8787, agent_type="opencode")

    env = popen_kwargs["env"]
    assert isinstance(env, dict)
    assert env["HEADROOM_SAVINGS_PROFILE"] == "agent-90"
    assert env["HEADROOM_TARGET_RATIO"] == "0.10"

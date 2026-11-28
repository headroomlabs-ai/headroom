"""Tests for the consolidated ``headroom uninstall`` command.

The valuable behaviour here is orchestration: detect what's installed, reverse
each piece by reusing the existing per-tool commands, stop the proxy exactly
once, and never touch things that aren't there. These tests run against a temp
``$HOME`` and stub the proxy/MCP edges so nothing on the real machine is hit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli import uninstall as uninstall_mod
from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


def _set_test_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def isolate_edges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Neutralize the MCP / Claude / OpenClaw edges so a test can focus on Codex.

    Returns a list that records the port passed to the proxy-stop helper.
    """
    monkeypatch.setattr(uninstall_mod, "_claude_wrapped", lambda: False)
    monkeypatch.setattr(uninstall_mod, "_openclaw_wrapped", lambda: False)
    monkeypatch.setattr(uninstall_mod, "_deployment_profiles", lambda: [])
    # Keep `mcp uninstall` a real no-op instead of reaching the live machine.
    import headroom.mcp_registry as mcp_registry

    monkeypatch.setattr(mcp_registry, "get_all_registrars", lambda: [])

    stopped: list[int] = []
    monkeypatch.setattr(
        wrap_mod,
        "_stop_local_proxy_for_unwrap",
        lambda port: stopped.append(port) or "stopped",
    )
    return stopped


def test_uninstall_is_registered_with_help(runner: CliRunner) -> None:
    assert "uninstall" in main.commands
    result = runner.invoke(main, ["uninstall", "--help"])
    assert result.exit_code == 0, result.output
    assert "Reverse a Headroom install" in result.output


def test_dry_run_changes_nothing(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolate_edges: list
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    wrap_mod._inject_codex_provider_config(8787)
    wrapped = config_file.read_text()

    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: False)
    result = runner.invoke(main, ["uninstall", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "Codex wrap:          remove" in result.output
    # Nothing was touched.
    assert config_file.read_text() == wrapped
    assert isolate_edges == []


def test_unwraps_codex_and_stops_proxy_once(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolate_edges: list
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    original = '[profiles.default]\nmodel = "gpt-4o"\n'
    config_file.write_text(original)
    wrap_mod._inject_codex_provider_config(8787)
    assert 'model_provider = "headroom"' in config_file.read_text()

    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: True)
    result = runner.invoke(main, ["uninstall", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert "HEADROOM UNWRAP: CODEX" in result.output
    assert config_file.read_text() == original
    # Proxy stopped exactly once, on the requested port — not once per unwrap.
    assert isolate_edges == [9999]


def test_skips_codex_when_not_wrapped(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolate_edges: list
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: False)

    result = runner.invoke(main, ["uninstall"])

    assert result.exit_code == 0, result.output
    assert "HEADROOM UNWRAP: CODEX" not in result.output
    assert "Headroom integrations removed" in result.output


def test_keep_proxy_does_not_stop_proxy(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolate_edges: list
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: True)

    result = runner.invoke(main, ["uninstall", "--keep-proxy"])

    assert result.exit_code == 0, result.output
    assert isolate_edges == []


def test_claude_wrapped_detects_rtk_hooks_without_headroom_mcp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": str(claude_dir / "rtk-rewrite.sh"),
                                }
                            ],
                        }
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Registrar:
        def detect(self) -> bool:
            return True

        def get_server(self, server_name: str):
            return None

    monkeypatch.setattr("headroom.mcp_registry.ClaudeRegistrar", lambda: Registrar())

    assert uninstall_mod._claude_wrapped() is True


def test_purge_state_deletes_workspace(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolate_edges: list
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    workspace = tmp_path / ".headroom"
    workspace.mkdir()
    (workspace / "session_stats.jsonl").write_text("{}\n")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: False)

    result = runner.invoke(main, ["uninstall", "--purge-state"])

    assert result.exit_code == 0, result.output
    assert not workspace.exists()
    assert "Deleted workspace state" in result.output


def test_purge_state_reports_delete_failures(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolate_edges: list
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    workspace = tmp_path / ".headroom"
    workspace.mkdir()
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: False)

    def _raise_delete_error(_path: Path) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(uninstall_mod.shutil, "rmtree", _raise_delete_error)

    result = runner.invoke(main, ["uninstall", "--purge-state"])

    assert result.exit_code != 0
    assert workspace.exists()
    assert "could not delete workspace state" in result.output


def test_keeps_workspace_by_default(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolate_edges: list
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    workspace = tmp_path / ".headroom"
    workspace.mkdir()
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: False)

    result = runner.invoke(main, ["uninstall"])

    assert result.exit_code == 0, result.output
    assert workspace.exists()
    assert "use --purge-state to delete" in result.output

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom import paths
from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _no_persistent_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda _port: None)


def test_remove_claude_managed_hooks_preserves_unrelated_hooks(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "model": "opus",
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": (
                                        "headroom init hook ensure --marker headroom-init-claude"
                                    ),
                                },
                                {"type": "command", "command": "echo keep"},
                            ],
                        }
                    ],
                    "SessionStart": [
                        {"matcher": "startup", "hooks": [{"type": "command", "command": "keep"}]}
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert wrap_cli._remove_claude_managed_hooks(settings) is True

    payload = json.loads(settings.read_text(encoding="utf-8"))
    pre_tool_hooks = payload["hooks"]["PreToolUse"][0]["hooks"]
    assert pre_tool_hooks == [{"type": "command", "command": "echo keep"}]
    assert payload["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "keep"


def test_unwrap_claude_removes_mcp_purges_retired_hook_and_stops_proxy(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.delenv("HEADROOM_WORKSPACE_DIR", raising=False)
    bin_dir = paths.bin_dir()
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    hooks_dir = claude_dir / "hooks"
    hooks_dir.mkdir()
    hook_script = hooks_dir / "rtk-rewrite.sh"
    hook_script.write_text(f'#!/bin/sh\nexec {bin_dir / "rtk"} "$@"\n', encoding="utf-8")
    settings = claude_dir / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": str(hook_script)}],
                        }
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stopped: list[int] = []
    unregistered: list[str] = []

    class Registrar:
        name = "claude"

        def detect(self) -> bool:
            return True

        def unregister_server(self, server_name: str) -> bool:
            unregistered.append(server_name)
            return True

        def get_server(self, server_name: str):
            return None

    with (
        patch("headroom.mcp_registry.ClaudeRegistrar", return_value=Registrar()),
        patch(
            "headroom.cli.wrap._stop_local_proxy_for_unwrap",
            side_effect=lambda port: stopped.append(port) or "stopped",
        ),
    ):
        result = runner.invoke(main, ["unwrap", "claude", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert unregistered == ["headroom", "codebase-memory-mcp"]
    assert stopped == [9999]
    assert "Stopped local Headroom proxy on port 9999" in result.output
    # The leftover retired context-tool hook is purged end-to-end by unwrap
    # (via purge_context_tool_artifacts), leaving no hooks behind.
    assert "hooks" not in json.loads(settings.read_text(encoding="utf-8"))


def test_unwrap_claude_preserves_user_managed_serena(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    unregistered: list[str] = []

    class Registrar:
        name = "claude"

        def detect(self) -> bool:
            return True

        def unregister_server(self, server_name: str) -> bool:
            unregistered.append(server_name)
            return True

        def get_server(self, server_name: str):
            if server_name == "serena":
                from headroom.mcp_registry.base import ServerSpec

                return ServerSpec(name="serena", command="/usr/local/bin/custom-serena")
            return None

    with (
        patch("headroom.mcp_registry.ClaudeRegistrar", return_value=Registrar()),
        patch("headroom.cli.wrap._remove_claude_managed_hooks", return_value=False),
        patch("headroom.cli.wrap._stop_local_proxy_for_unwrap"),
    ):
        result = runner.invoke(main, ["unwrap", "claude"])

    assert result.exit_code == 0, result.output
    assert unregistered == ["headroom", "codebase-memory-mcp"]


def test_unwrap_claude_removes_headroom_installed_serena(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))

    from headroom.mcp_registry import build_serena_spec
    from headroom.mcp_registry.ledger import record_install

    serena_spec = build_serena_spec("claude-code")
    record_install("claude", serena_spec)
    unregistered: list[str] = []

    class Registrar:
        name = "claude"

        def detect(self) -> bool:
            return True

        def unregister_server(self, server_name: str) -> bool:
            unregistered.append(server_name)
            return True

        def get_server(self, server_name: str):
            if server_name == "serena":
                return serena_spec
            return None

    with (
        patch("headroom.mcp_registry.ClaudeRegistrar", return_value=Registrar()),
        patch("headroom.cli.wrap._remove_claude_managed_hooks", return_value=False),
        patch("headroom.cli.wrap._stop_local_proxy_for_unwrap"),
    ):
        result = runner.invoke(main, ["unwrap", "claude"])

    assert result.exit_code == 0, result.output
    assert unregistered == ["headroom", "codebase-memory-mcp", "serena"]
    assert "Removed Headroom-installed Serena MCP server" in result.output


def test_unwrap_claude_keep_flags_skip_cleanup(
    runner: CliRunner,
) -> None:
    with (
        patch("headroom.mcp_registry.ClaudeRegistrar") as registrar,
        patch("headroom.cli.wrap._remove_claude_managed_hooks", return_value=False),
        patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_proxy,
    ):
        result = runner.invoke(
            main,
            ["unwrap", "claude", "--keep-mcp", "--no-stop-proxy"],
        )

    assert result.exit_code == 0, result.output
    registrar.assert_not_called()
    stop_proxy.assert_not_called()


def test_unwrap_claude_restores_all_base_url_modes(runner: CliRunner) -> None:
    restore_calls: list[dict[str, object]] = []

    def restore_base_url(previous: str | None, **kwargs: object) -> None:
        restore_calls.append({"previous": previous, **kwargs})

    with patch("headroom.cli.wrap._restore_claude_wrap_base_url", side_effect=restore_base_url):
        result = runner.invoke(
            main,
            ["unwrap", "claude", "--keep-mcp", "--no-stop-proxy"],
        )

    assert result.exit_code == 0, result.output
    settings_path = Path.cwd() / ".claude" / "settings.local.json"
    assert restore_calls == [
        {
            "previous": None,
            "foundry_mode": False,
            "vertex_mode": False,
            "settings_path": settings_path,
            # unwrap is the user asking for their settings back, so it drops
            # every wrap session's ownership claim instead of deferring to a
            # live sibling and silently doing nothing (#3205).
            "force": True,
        },
        {
            "previous": None,
            "foundry_mode": True,
            "vertex_mode": False,
            "settings_path": settings_path,
            "force": True,
        },
        {
            "previous": None,
            "foundry_mode": False,
            "vertex_mode": True,
            "settings_path": settings_path,
            "force": True,
        },
    ]


def test_unwrap_claude_stops_claude_owned_persistent_deployment(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manifest:
        profile = "unwrap-2340"
        targets = ["claude"]
        tool_envs = {"claude": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}
        mutations: list[object] = []
        supervisor_kind = "service"

    stopped: list[str] = []
    deactivated: list[str] = []

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: Manifest())
    monkeypatch.setattr(
        "headroom.cli.install._deactivate_deployment_mutations",
        lambda manifest: deactivated.append(manifest.profile),
    )
    monkeypatch.setattr(
        "headroom.cli.install._stop_deployment",
        lambda manifest: stopped.append(manifest.profile),
    )

    with (
        patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_local,
    ):
        result = runner.invoke(
            main,
            ["unwrap", "claude", "--keep-mcp", "--port", "8787"],
        )

    assert result.exit_code == 0, result.output
    stop_local.assert_not_called()
    assert deactivated == ["unwrap-2340"]
    assert stopped == ["unwrap-2340"]
    assert "Stopped Claude-owned persistent deployment 'unwrap-2340' on port 8787." in result.output
    assert "Claude is no longer durably wrapped by Headroom." in result.output


def test_unwrap_claude_reports_ambiguous_same_port_persistent_deployment(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manifest:
        profile = "shared-proxy"
        targets = ["codex"]
        tool_envs = {"codex": {"OPENAI_BASE_URL": "http://127.0.0.1:8787"}}
        mutations: list[object] = []
        supervisor_kind = "service"

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: Manifest())

    with patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_local:
        result = runner.invoke(
            main,
            ["unwrap", "claude", "--keep-mcp", "--port", "8787"],
        )

    assert result.exit_code == 0, result.output
    stop_local.assert_not_called()
    assert "same-port persistent deployment 'shared-proxy' still owns port 8787" in result.output
    assert "headroom install stop --profile shared-proxy" in result.output
    assert "Claude is no longer durably wrapped by Headroom." not in result.output


def test_unwrap_claude_warns_about_same_port_inherited_env(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")

    with patch("headroom.cli.wrap._stop_local_proxy_for_unwrap", return_value="stopped"):
        result = runner.invoke(
            main,
            ["unwrap", "claude", "--keep-mcp", "--port", "8787"],
        )

    assert result.exit_code == 0, result.output
    assert "current shell still exports ANTHROPIC_BASE_URL for port 8787" in result.output
    assert "Claude is no longer durably wrapped by Headroom." not in result.output


def test_unwrap_claude_ignores_malformed_inherited_env_port(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:notaport")

    with patch("headroom.cli.wrap._stop_local_proxy_for_unwrap", return_value="stopped"):
        result = runner.invoke(
            main,
            ["unwrap", "claude", "--keep-mcp", "--port", "8787"],
        )

    assert result.exit_code == 0, result.output
    assert "current shell still exports ANTHROPIC_BASE_URL" not in result.output
    assert "Claude is no longer durably wrapped by Headroom." in result.output


def test_remove_claude_managed_hooks_removes_init_hooks_and_env(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "model": "opus",
                "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787", "FOO": "bar"},
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "startup|resume",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": (
                                        "/home/u/.local/bin/headroom init hook ensure "
                                        "--profile init-user --marker headroom-init-claude"
                                    ),
                                    "timeout": 15,
                                }
                            ],
                        }
                    ],
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "headroom init hook ensure --marker headroom-init-claude",
                                },
                                {"type": "command", "command": "echo keep-me"},
                            ],
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert wrap_cli._remove_claude_managed_hooks(settings) is True

    payload = json.loads(settings.read_text(encoding="utf-8"))
    # ANTHROPIC_BASE_URL stripped; unrelated env var preserved
    assert payload.get("env") == {"FOO": "bar"}
    # SessionStart removed entirely (its only hook was the init marker)
    assert "SessionStart" not in payload.get("hooks", {})
    # PreToolUse: init-marker hook gone, unrelated hook kept
    assert payload["hooks"]["PreToolUse"][0]["hooks"] == [
        {"type": "command", "command": "echo keep-me"}
    ]
    assert payload["model"] == "opus"


def test_remove_claude_managed_hooks_strips_env_without_hooks(tmp_path: Path) -> None:
    # Regression: unwrap previously returned early when no hooks existed,
    # leaving init's ANTHROPIC_BASE_URL behind in settings.json.
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}) + "\n",
        encoding="utf-8",
    )

    assert wrap_cli._remove_claude_managed_hooks(settings) is True

    payload = json.loads(settings.read_text(encoding="utf-8"))
    assert "env" not in payload  # emptied env dict is dropped


def test_remove_claude_managed_hooks_noop_when_nothing_managed(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    original = {
        "model": "opus",
        "env": {"FOO": "bar"},
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo hi"}]}
            ]
        },
    }
    settings.write_text(json.dumps(original) + "\n", encoding="utf-8")

    assert wrap_cli._remove_claude_managed_hooks(settings) is False
    # nothing managed -> file untouched
    assert json.loads(settings.read_text(encoding="utf-8")) == original


def test_remove_claude_managed_hooks_strips_enable_tool_search(tmp_path: Path) -> None:
    # unwrap must remove BOTH env vars init writes (ANTHROPIC_BASE_URL +
    # ENABLE_TOOL_SEARCH, GH #746), leaving user-set vars intact.
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
                    "ENABLE_TOOL_SEARCH": "true",
                    "KEEP": "1",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert wrap_cli._remove_claude_managed_hooks(settings) is True

    payload = json.loads(settings.read_text(encoding="utf-8"))
    assert payload["env"] == {"KEEP": "1"}

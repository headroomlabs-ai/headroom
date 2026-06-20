from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_remove_claude_rtk_hooks_preserves_unrelated_hooks(tmp_path: Path) -> None:
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
                                    "command": "/Users/test/.claude/hooks/rtk-rewrite.sh",
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

    assert wrap_cli._remove_claude_rtk_hooks(settings) is True

    payload = json.loads(settings.read_text(encoding="utf-8"))
    pre_tool_hooks = payload["hooks"]["PreToolUse"][0]["hooks"]
    assert pre_tool_hooks == [{"type": "command", "command": "echo keep"}]
    assert payload["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "keep"


def test_unwrap_claude_removes_mcp_rtk_and_stops_proxy(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {"type": "command", "command": str(claude_dir / "rtk-rewrite.sh")}
                            ],
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
        patch("headroom.cli.wrap._remove_claude_rtk_hooks", return_value=False),
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
        patch("headroom.cli.wrap._remove_claude_rtk_hooks", return_value=False),
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
        patch("headroom.cli.wrap._remove_claude_rtk_hooks") as remove_rtk,
        patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_proxy,
    ):
        result = runner.invoke(
            main,
            ["unwrap", "claude", "--keep-mcp", "--keep-rtk", "--no-stop-proxy"],
        )

    assert result.exit_code == 0, result.output
    registrar.assert_not_called()
    remove_rtk.assert_not_called()
    stop_proxy.assert_not_called()


def test_remove_claude_rtk_hooks_removes_init_hooks_and_env(tmp_path: Path) -> None:
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

    assert wrap_cli._remove_claude_rtk_hooks(settings) is True

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


def test_remove_claude_rtk_hooks_strips_env_without_hooks(tmp_path: Path) -> None:
    # Regression: unwrap previously returned early when no hooks existed,
    # leaving init's ANTHROPIC_BASE_URL behind in settings.json.
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}) + "\n",
        encoding="utf-8",
    )

    assert wrap_cli._remove_claude_rtk_hooks(settings) is True

    payload = json.loads(settings.read_text(encoding="utf-8"))
    assert "env" not in payload  # emptied env dict is dropped


def test_remove_claude_rtk_hooks_noop_when_nothing_managed(tmp_path: Path) -> None:
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

    assert wrap_cli._remove_claude_rtk_hooks(settings) is False
    # nothing managed -> file untouched
    assert json.loads(settings.read_text(encoding="utf-8")) == original


def test_remove_claude_rtk_hooks_strips_enable_tool_search(tmp_path: Path) -> None:
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

    assert wrap_cli._remove_claude_rtk_hooks(settings) is True

    payload = json.loads(settings.read_text(encoding="utf-8"))
    assert payload["env"] == {"KEEP": "1"}

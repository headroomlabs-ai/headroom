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


def test_remove_codebuddy_rtk_hooks_preserves_unrelated_hooks(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "model": "deepseek",
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "rtk hook codebuddy",
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

    assert wrap_cli._remove_codebuddy_rtk_hooks(settings) is True

    payload = json.loads(settings.read_text(encoding="utf-8"))
    pre_tool_hooks = payload["hooks"]["PreToolUse"][0]["hooks"]
    assert pre_tool_hooks == [{"type": "command", "command": "echo keep"}]
    assert payload["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "keep"


def test_remove_codebuddy_rtk_hooks_also_matches_legacy_claude_hook(tmp_path: Path) -> None:
    """Older rtk versions use 'rtk hook claude' in CodeBuddy settings."""
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "rtk hook claude",
                                },
                            ],
                        }
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert wrap_cli._remove_codebuddy_rtk_hooks(settings) is True

    payload = json.loads(settings.read_text(encoding="utf-8"))
    assert "hooks" not in payload


def test_unwrap_codebuddy_removes_mcp_rtk_and_stops_proxy(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    cb_dir = tmp_path / ".codebuddy"
    cb_dir.mkdir()
    settings = cb_dir / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "rtk hook codebuddy",
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
    stopped: list[int] = []
    unregistered: list[str] = []

    class Registrar:
        name = "codebuddy"

        def detect(self) -> bool:
            return True

        def unregister_server(self, server_name: str) -> bool:
            unregistered.append(server_name)
            return True

        def get_server(self, server_name: str):
            return None

    with (
        patch("headroom.mcp_registry.CodeBuddyRegistrar", return_value=Registrar()),
        patch(
            "headroom.cli.wrap._stop_local_proxy_for_unwrap",
            side_effect=lambda port: stopped.append(port) or "stopped",
        ),
    ):
        result = runner.invoke(main, ["unwrap", "codebuddy", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert unregistered == ["headroom"]
    assert stopped == [9999]
    assert "Stopped local Headroom proxy on port 9999" in result.output
    assert "hooks" not in json.loads(settings.read_text(encoding="utf-8"))


def test_unwrap_codebuddy_keep_flags_skip_cleanup(
    runner: CliRunner,
) -> None:
    with (
        patch("headroom.mcp_registry.CodeBuddyRegistrar") as registrar,
        patch("headroom.cli.wrap._remove_codebuddy_rtk_hooks") as remove_rtk,
        patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_proxy,
    ):
        result = runner.invoke(
            main,
            ["unwrap", "codebuddy", "--keep-mcp", "--no-stop-proxy"],
        )

    assert result.exit_code == 0, result.output
    registrar.assert_not_called()
    remove_rtk.assert_called_once()
    stop_proxy.assert_not_called()


def test_unwrap_codebuddy_preserves_user_managed_serena(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    unregistered: list[str] = []

    class Registrar:
        name = "codebuddy"

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
        patch("headroom.mcp_registry.CodeBuddyRegistrar", return_value=Registrar()),
        patch("headroom.cli.wrap._remove_codebuddy_rtk_hooks", return_value=False),
        patch("headroom.cli.wrap._stop_local_proxy_for_unwrap"),
    ):
        result = runner.invoke(main, ["unwrap", "codebuddy"])

    assert result.exit_code == 0, result.output
    assert unregistered == ["headroom"]

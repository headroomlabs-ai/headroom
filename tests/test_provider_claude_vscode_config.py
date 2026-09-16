"""Tests for reversible Claude Code VS Code configuration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import click
import pytest

from headroom.providers.claude.vscode import (
    claude_user_settings_path,
    configure_vscode_claude_settings,
    remove_vscode_claude_settings,
    vscode_claude_proxy_url,
)


def test_settings_path_honors_claude_config_dir(tmp_path: Path) -> None:
    assert claude_user_settings_path({"CLAUDE_CONFIG_DIR": str(tmp_path)}) == (
        tmp_path / "settings.json"
    )


def test_settings_path_uses_windows_profile() -> None:
    path = claude_user_settings_path(
        {"HOME": "/wrong", "USERPROFILE": r"C:\\Users\\claude"}, platform="win32"
    )
    assert path == Path(r"C:\\Users\\claude") / ".claude" / "settings.json"


def test_proxy_url_is_project_scoped() -> None:
    assert vscode_claude_proxy_url(8787, "my project").endswith("/p/my%20project")


def test_configure_and_remove_preserve_unrelated_and_previous_values(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Read"]},
                "env": {
                    "KEEP": "yes",
                    "ANTHROPIC_BASE_URL": "https://gateway.example",
                    "ENABLE_TOOL_SEARCH": "false",
                },
            }
        ),
        encoding="utf-8",
    )

    assert configure_vscode_claude_settings(path, "http://127.0.0.1:8787/p/demo") == "added"
    configured = json.loads(path.read_text(encoding="utf-8"))
    assert configured["env"] == {
        "KEEP": "yes",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787/p/demo",
        "ENABLE_TOOL_SEARCH": "false",
    }
    assert configured["permissions"] == {"allow": ["Read"]}

    assert remove_vscode_claude_settings(path)
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["env"] == {
        "KEEP": "yes",
        "ANTHROPIC_BASE_URL": "https://gateway.example",
        "ENABLE_TOOL_SEARCH": "false",
    }
    assert restored["permissions"] == {"allow": ["Read"]}
    assert not (tmp_path / ".headroom-vscode-claude.json").exists()


def test_reconfigure_updates_port_without_losing_original_values(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"env":{"ANTHROPIC_BASE_URL":"https://original.example"}}', encoding="utf-8")

    configure_vscode_claude_settings(path, "http://127.0.0.1:8787/p/demo")
    assert configure_vscode_claude_settings(path, "http://127.0.0.1:9999/p/demo") == "updated"
    assert remove_vscode_claude_settings(path)
    assert json.loads(path.read_text(encoding="utf-8"))["env"] == {
        "ANTHROPIC_BASE_URL": "https://original.example"
    }


def test_remove_deletes_settings_created_only_for_headroom(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    configure_vscode_claude_settings(path, "http://127.0.0.1:8787/p/demo")
    assert path.exists()
    assert remove_vscode_claude_settings(path)
    assert not path.exists()


def test_configure_refuses_malformed_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(click.ClickException, match="not valid JSON"):
        configure_vscode_claude_settings(path, "http://127.0.0.1:8787")
    assert path.read_text(encoding="utf-8") == "{broken"


@pytest.mark.parametrize("contents", ["[]", '{"env": []}'])
def test_configure_refuses_unsafe_settings_shapes(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "settings.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(click.ClickException, match="refusing to overwrite"):
        configure_vscode_claude_settings(path, "http://127.0.0.1:8787")
    assert path.read_text(encoding="utf-8") == contents


def test_configure_refuses_unreadable_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{}", encoding="utf-8")
    with (
        patch("headroom.providers.claude.vscode.fsutil.read_text", side_effect=OSError("denied")),
        pytest.raises(click.ClickException, match="Could not read Claude settings"),
    ):
        configure_vscode_claude_settings(path, "http://127.0.0.1:8787")


def test_empty_existing_settings_is_restored_as_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("", encoding="utf-8")
    configure_vscode_claude_settings(path, "http://127.0.0.1:8787")
    assert remove_vscode_claude_settings(path)
    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_remove_without_headroom_state_is_noop(tmp_path: Path) -> None:
    assert not remove_vscode_claude_settings(tmp_path / "settings.json")


@pytest.mark.parametrize(
    ("state_update", "message"),
    [
        ({"version": 2}, "unsupported or incomplete"),
        ({"managed": None}, "has no managed values"),
        ({"previous": {"ANTHROPIC_BASE_URL": None}}, "is incomplete"),
    ],
)
def test_remove_refuses_incomplete_state(
    tmp_path: Path, state_update: dict[str, object], message: str
) -> None:
    path = tmp_path / "settings.json"
    configure_vscode_claude_settings(path, "http://127.0.0.1:8787")
    state_path = tmp_path / ".headroom-vscode-claude.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(state_update)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(click.ClickException, match=message):
        remove_vscode_claude_settings(path)


def test_reconfigure_refuses_incomplete_or_conflicting_state(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    proxy_url = "http://127.0.0.1:8787"
    configure_vscode_claude_settings(path, proxy_url)
    state_path = tmp_path / ".headroom-vscode-claude.json"
    state_path.write_text("{}", encoding="utf-8")
    with pytest.raises(click.ClickException, match="unsupported or incomplete"):
        configure_vscode_claude_settings(path, proxy_url)

    state_path.unlink()
    configure_vscode_claude_settings(path, proxy_url)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["env"]["ENABLE_TOOL_SEARCH"] = "true"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(click.ClickException, match="managed values"):
        configure_vscode_claude_settings(path, proxy_url)


def test_remove_refuses_to_overwrite_changed_managed_value(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    configure_vscode_claude_settings(path, "http://127.0.0.1:8787/p/demo")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["env"]["ANTHROPIC_BASE_URL"] = "https://user-change.example"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(click.ClickException, match="refusing to overwrite"):
        remove_vscode_claude_settings(path)
    assert json.loads(path.read_text(encoding="utf-8"))["env"]["ANTHROPIC_BASE_URL"] == (
        "https://user-change.example"
    )

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
    resolve_vscode_claude_model,
    resolve_vscode_claude_model_for_instructions,
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


def test_configure_1m_snapshots_selected_model_and_restores_exact_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    original = {
        "model": " claude-sonnet-5 ",
        "permissions": {"allow": ["Read"]},
        "env": {"KEEP": "yes"},
    }
    path.write_text(json.dumps(original), encoding="utf-8")

    configure_vscode_claude_settings(path, "http://127.0.0.1:8787", context_1m=True)
    configured = json.loads(path.read_text(encoding="utf-8"))
    state = json.loads((tmp_path / ".headroom-vscode-claude.json").read_text(encoding="utf-8"))
    assert configured["model"] == "claude-sonnet-5[1m]"
    assert state["model"] == {
        "previous": {"present": True, "value": " claude-sonnet-5 "},
        "managed": "claude-sonnet-5[1m]",
    }

    assert remove_vscode_claude_settings(path)
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_configure_1m_is_idempotent_and_does_not_double_suffix(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"model":"claude-opus-5[1m]"}', encoding="utf-8")

    configure_vscode_claude_settings(path, "http://127.0.0.1:8787", context_1m=True)
    configure_vscode_claude_settings(path, "http://127.0.0.1:9999", context_1m=True)

    configured = json.loads(path.read_text(encoding="utf-8"))
    state = json.loads((tmp_path / ".headroom-vscode-claude.json").read_text(encoding="utf-8"))
    assert configured["model"] == "claude-opus-5[1m]"
    assert state["model"]["previous"] == {"present": True, "value": "claude-opus-5[1m]"}
    assert remove_vscode_claude_settings(path)
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "claude-opus-5[1m]"


def test_configure_1m_preserves_present_empty_model(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"model":""}', encoding="utf-8")

    configure_vscode_claude_settings(path, "http://127.0.0.1:8787", context_1m=True)
    state = json.loads((tmp_path / ".headroom-vscode-claude.json").read_text(encoding="utf-8"))
    assert state["model"]["previous"] == {"present": True, "value": ""}
    assert remove_vscode_claude_settings(path)
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == ""


def test_configure_1m_uses_fallback_and_disable_restores_missing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.json"
    monkeypatch.setenv("HEADROOM_1M_MODEL", "claude-opus-9")

    configure_vscode_claude_settings(path, "http://127.0.0.1:8787", context_1m=True)
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "claude-opus-9[1m]"

    assert configure_vscode_claude_settings(path, "http://127.0.0.1:9999") == "updated"
    configured = json.loads(path.read_text(encoding="utf-8"))
    state = json.loads((tmp_path / ".headroom-vscode-claude.json").read_text(encoding="utf-8"))
    assert "model" not in configured
    assert "model" not in state
    assert configured["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"

    assert remove_vscode_claude_settings(path)
    assert not path.exists()


def test_legacy_v1_sidecar_can_add_and_restore_model_state(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    configure_vscode_claude_settings(path, "http://127.0.0.1:8787")
    configure_vscode_claude_settings(path, "http://127.0.0.1:8787", context_1m=True)

    state = json.loads((tmp_path / ".headroom-vscode-claude.json").read_text(encoding="utf-8"))
    assert state["version"] == 1
    assert state["model"]["previous"] == {"present": False, "value": None}
    assert remove_vscode_claude_settings(path)
    assert not path.exists()


def test_configure_1m_rejects_non_string_model_without_writes(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = {"model": {"name": "opus"}, "env": {"KEEP": "yes"}}
    path.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(click.ClickException, match="non-string"):
        configure_vscode_claude_settings(path, "http://127.0.0.1:8787", context_1m=True)

    assert json.loads(path.read_text(encoding="utf-8")) == original
    assert not (tmp_path / ".headroom-vscode-claude.json").exists()


def test_configure_without_1m_preserves_unmanaged_model_and_nested_values(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = {
        "model": "claude-sonnet-5",
        "permissions": {"allow": ["Read"]},
        "custom": {"nested": {"value": True}},
    }
    path.write_text(json.dumps(original), encoding="utf-8")

    configure_vscode_claude_settings(path, "http://127.0.0.1:8787")
    configured = json.loads(path.read_text(encoding="utf-8"))
    assert configured["model"] == original["model"]
    assert configured["permissions"] == original["permissions"]
    assert configured["custom"] == original["custom"]


def test_model_conflicts_fail_closed_on_configure_and_remove(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"model":"claude-sonnet-5"}', encoding="utf-8")
    configure_vscode_claude_settings(path, "http://127.0.0.1:8787", context_1m=True)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"] = "user-selected-model"
    path.write_text(json.dumps(payload), encoding="utf-8")
    state_path = tmp_path / ".headroom-vscode-claude.json"
    state_before = state_path.read_text(encoding="utf-8")

    with pytest.raises(click.ClickException, match="managed model"):
        configure_vscode_claude_settings(path, "http://127.0.0.1:9999", context_1m=True)
    with pytest.raises(click.ClickException, match="managed model"):
        remove_vscode_claude_settings(path)

    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "user-selected-model"
    assert state_path.read_text(encoding="utf-8") == state_before


def test_resolve_vscode_claude_model_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"model":"claude-opus-5"}', encoding="utf-8")

    assert resolve_vscode_claude_model(path) == "claude-opus-5[1m]"
    assert path.read_text(encoding="utf-8") == '{"model":"claude-opus-5"}'
    assert not (tmp_path / ".headroom-vscode-claude.json").exists()


def test_resolve_vscode_claude_model_for_instructions_falls_back_for_non_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.json"
    original = '{"model":{"name":"opus"}}'
    path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("HEADROOM_1M_MODEL", "claude-opus-9")

    with pytest.raises(click.ClickException, match="non-string"):
        resolve_vscode_claude_model(path)

    assert resolve_vscode_claude_model_for_instructions(path) == "claude-opus-9[1m]"
    assert path.read_text(encoding="utf-8") == original
    assert not (tmp_path / ".headroom-vscode-claude.json").exists()


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
        configure_vscode_claude_settings(path, "http://127.0.0.1:8787", context_1m=True)
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
        ({"model": None}, "incomplete model record"),
        (
            {"model": {"previous": None, "managed": "claude-opus-5[1m]"}},
            "incomplete model record",
        ),
        (
            {
                "model": {
                    "previous": {"present": True},
                    "managed": "claude-opus-5[1m]",
                }
            },
            "incomplete model record",
        ),
        (
            {
                "model": {
                    "previous": {"present": False, "value": "unexpected"},
                    "managed": "claude-opus-5[1m]",
                }
            },
            "incomplete model record",
        ),
        (
            {
                "model": {
                    "previous": {"present": True, "value": 5},
                    "managed": "claude-opus-5[1m]",
                }
            },
            "incomplete model record",
        ),
        (
            {
                "model": {
                    "previous": {"present": False, "value": None},
                    "managed": None,
                }
            },
            "incomplete model record",
        ),
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

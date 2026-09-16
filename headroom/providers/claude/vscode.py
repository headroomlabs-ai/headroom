"""Persistent configuration for Claude Code's VS Code extension."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click

from headroom import fsutil
from headroom.proxy.project_context import with_project_prefix

_BASE_URL_KEY = "ANTHROPIC_BASE_URL"
_TOOL_SEARCH_KEY = "ENABLE_TOOL_SEARCH"
_MANAGED_KEYS = (_BASE_URL_KEY, _TOOL_SEARCH_KEY)
_STATE_FILENAME = ".headroom-vscode-claude.json"
_STATE_VERSION = 1


def claude_user_settings_path(
    environ: Mapping[str, str] | None = None, *, platform: str | None = None
) -> Path:
    """Return the Claude Code user settings path, respecting CLAUDE_CONFIG_DIR."""
    env = environ if environ is not None else os.environ
    config_dir = env.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir).expanduser() / "settings.json"
    current_platform = platform or sys.platform
    home_var = "USERPROFILE" if current_platform == "win32" else "HOME"
    home = Path(env.get(home_var) or Path.home())
    return home / ".claude" / "settings.json"


def vscode_claude_proxy_url(port: int, project: str | None = None) -> str:
    """Return the project-scoped Anthropic endpoint for Claude Code in VS Code."""
    return str(with_project_prefix(f"http://127.0.0.1:{port}", project))


def _state_path(settings_path: Path) -> Path:
    return settings_path.with_name(_STATE_FILENAME)


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = fsutil.read_text(path)
    except OSError as exc:
        raise click.ClickException(f"Could not read {label} {path}: {exc}") from exc
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"{label.capitalize()} {path} is not valid JSON ({exc}); refusing to overwrite it."
        ) from exc
    if not isinstance(payload, dict):
        raise click.ClickException(
            f"{label.capitalize()} {path} must contain a JSON object; refusing to overwrite it."
        )
    return payload


def _read_settings(path: Path) -> dict[str, Any]:
    return _read_object(path, label="Claude settings") if path.exists() else {}


def _env_map(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    env = payload.get("env")
    if env is None:
        return {}
    if not isinstance(env, dict):
        raise click.ClickException(
            f"Claude settings {path} has a non-object 'env' value; refusing to overwrite it."
        )
    return dict(env)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fsutil.write_text(path, json.dumps(payload, indent=2) + "\n")


def configure_vscode_claude_settings(path: Path, proxy_url: str) -> str:
    """Route Claude Code's VS Code process through Headroom, reversibly."""
    payload = _read_settings(path)
    env = _env_map(payload, path)
    state_path = _state_path(path)
    # Claude Code's VS Code webview cannot render the server_tool_use /
    # tool_search_tool_result blocks emitted by deferred tool search (#2028).
    # Keep it disabled for this surface; the standalone CLI retains its own
    # configurable/default-on policy.
    managed = {_BASE_URL_KEY: proxy_url, _TOOL_SEARCH_KEY: "false"}

    if state_path.exists():
        state = _read_object(state_path, label="Headroom state")
        if state.get("version") != _STATE_VERSION or not isinstance(state.get("previous"), dict):
            raise click.ClickException(
                f"Headroom state {state_path} is unsupported or incomplete; refusing to edit."
            )
        old_managed = state.get("managed")
        if not isinstance(old_managed, dict) or any(
            env.get(key) != old_managed.get(key) for key in _MANAGED_KEYS
        ):
            raise click.ClickException(
                "Claude settings changed one of Headroom's managed values. Run "
                "`headroom unwrap vscode-claude` or resolve the conflict before retrying."
            )
        action = "updated"
    else:
        state = {
            "version": _STATE_VERSION,
            "settings_existed": path.exists(),
            "previous": {
                key: {"present": key in env, "value": env.get(key)} for key in _MANAGED_KEYS
            },
        }
        action = "added"

    env.update(managed)
    payload["env"] = env
    state["managed"] = managed
    _write_json(path, payload)
    _write_json(state_path, state)
    return action


def remove_vscode_claude_settings(path: Path) -> bool:
    """Restore the values saved before Headroom configured the VS Code extension."""
    state_path = _state_path(path)
    if not state_path.exists():
        return False
    state = _read_object(state_path, label="Headroom state")
    previous = state.get("previous")
    managed = state.get("managed")
    if state.get("version") != _STATE_VERSION or not isinstance(previous, dict):
        raise click.ClickException(
            f"Headroom state {state_path} is unsupported or incomplete; refusing to edit."
        )
    if not isinstance(managed, dict):
        raise click.ClickException(f"Headroom state {state_path} has no managed values.")

    payload = _read_settings(path)
    env = _env_map(payload, path)
    if any(env.get(key) != managed.get(key) for key in _MANAGED_KEYS):
        raise click.ClickException(
            "Claude settings changed one of Headroom's managed values; refusing to overwrite "
            f"the user's change. Resolve the conflict in {path}, then retry."
        )

    for key in _MANAGED_KEYS:
        saved = previous.get(key)
        if not isinstance(saved, dict) or not isinstance(saved.get("present"), bool):
            raise click.ClickException(
                f"Headroom state {state_path} is incomplete; refusing to edit."
            )
        if saved["present"]:
            env[key] = saved.get("value")
        else:
            env.pop(key, None)
    if env:
        payload["env"] = env
    else:
        payload.pop("env", None)

    if payload or state.get("settings_existed"):
        _write_json(path, payload)
    elif path.exists():
        path.unlink()
    state_path.unlink()
    return True

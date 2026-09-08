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
from headroom.providers.claude.runtime import resolve_1m_model
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


def _model_value(payload: dict[str, Any], path: Path) -> str | None:
    if "model" not in payload:
        return None
    model = payload["model"]
    if not isinstance(model, str):
        raise click.ClickException(
            f"Claude settings {path} has a non-string top-level 'model' value; "
            "refusing to overwrite it."
        )
    return model


def resolve_vscode_claude_model(path: Path) -> str:
    """Resolve the 1M model from settings without writing settings or state."""
    payload = _read_settings(path)
    return resolve_1m_model(_model_value(payload, path))


def resolve_vscode_claude_model_for_instructions(path: Path) -> str:
    """Resolve a model for manual setup instructions, tolerating bad settings."""
    try:
        return resolve_vscode_claude_model(path)
    except click.ClickException:
        return resolve_1m_model(None)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fsutil.write_text(path, json.dumps(payload, indent=2) + "\n")


def _validate_previous(previous: dict[str, Any], state_path: Path) -> None:
    for key in _MANAGED_KEYS:
        saved = previous.get(key)
        if (
            not isinstance(saved, dict)
            or not isinstance(saved.get("present"), bool)
            or "value" not in saved
        ):
            raise click.ClickException(
                f"Headroom state {state_path} is incomplete; refusing to edit."
            )


def _validate_model_record(state: dict[str, Any], state_path: Path) -> dict[str, Any] | None:
    if "model" not in state:
        return None
    model_state = state["model"]
    if not isinstance(model_state, dict):
        raise click.ClickException(
            f"Headroom state {state_path} has an incomplete model record; refusing to edit."
        )
    previous = model_state.get("previous")
    managed = model_state.get("managed")
    if (
        not isinstance(previous, dict)
        or not isinstance(previous.get("present"), bool)
        or "value" not in previous
        or not isinstance(managed, str)
        or (not previous["present"] and previous.get("value") is not None)
        or (previous["present"] and not isinstance(previous.get("value"), str))
    ):
        raise click.ClickException(
            f"Headroom state {state_path} has an incomplete model record; refusing to edit."
        )
    return model_state


def _validate_state(state: dict[str, Any], state_path: Path) -> dict[str, Any] | None:
    if state.get("version") != _STATE_VERSION or not isinstance(state.get("previous"), dict):
        raise click.ClickException(
            f"Headroom state {state_path} is unsupported or incomplete; refusing to edit."
        )
    previous = state["previous"]
    _validate_previous(previous, state_path)
    managed = state.get("managed")
    if not isinstance(managed, dict):
        raise click.ClickException(f"Headroom state {state_path} has no managed values.")
    if any(key not in managed or not isinstance(managed[key], str) for key in _MANAGED_KEYS):
        raise click.ClickException(f"Headroom state {state_path} is incomplete; refusing to edit.")
    return _validate_model_record(state, state_path)


def _model_is_managed(payload: dict[str, Any], managed: str) -> bool:
    return "model" in payload and payload["model"] == managed


def _restore_model(payload: dict[str, Any], model_state: dict[str, Any]) -> None:
    previous = model_state["previous"]
    if previous["present"]:
        payload["model"] = previous["value"]
    else:
        payload.pop("model", None)


def configure_vscode_claude_settings(
    path: Path, proxy_url: str, *, context_1m: bool = False
) -> str:
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
        model_state = _validate_state(state, state_path)
        old_managed = state["managed"]
        if any(env.get(key) != old_managed[key] for key in _MANAGED_KEYS):
            raise click.ClickException(
                "Claude settings changed one of Headroom's managed values. Run "
                "`headroom unwrap vscode-claude` or resolve the conflict before retrying."
            )
        if model_state is not None and not _model_is_managed(payload, model_state["managed"]):
            raise click.ClickException(
                "Claude settings changed Headroom's managed model. Run "
                "`headroom unwrap vscode-claude` or resolve the conflict before retrying."
            )
        action = "updated"
    else:
        model_state = None
        state = {
            "version": _STATE_VERSION,
            "settings_existed": path.exists(),
            "previous": {
                key: {"present": key in env, "value": env.get(key)} for key in _MANAGED_KEYS
            },
        }
        action = "added"

    if context_1m:
        current_model = _model_value(payload, path)
        if model_state is None:
            resolved_model = resolve_1m_model(current_model)
            state["model"] = {
                "previous": {
                    "present": "model" in payload,
                    "value": current_model,
                },
                "managed": resolved_model,
            }
        else:
            resolved_model = resolve_1m_model(current_model)
            model_state["managed"] = resolved_model
            state["model"] = model_state
        payload["model"] = resolved_model
    elif model_state is not None:
        _restore_model(payload, model_state)
        state.pop("model", None)

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
    previous = state["previous"]
    managed = state["managed"]
    model_state = _validate_state(state, state_path)

    payload = _read_settings(path)
    env = _env_map(payload, path)
    if any(env.get(key) != managed[key] for key in _MANAGED_KEYS):
        raise click.ClickException(
            "Claude settings changed one of Headroom's managed values; refusing to overwrite "
            f"the user's change. Resolve the conflict in {path}, then retry."
        )
    if model_state is not None and not _model_is_managed(payload, model_state["managed"]):
        raise click.ClickException(
            "Claude settings changed Headroom's managed model; refusing to overwrite the user's "
            f"change. Resolve the conflict in {path}, then retry."
        )

    for key in _MANAGED_KEYS:
        saved = previous.get(key)
        if saved["present"]:
            env[key] = saved.get("value")
        else:
            env.pop(key, None)
    if env:
        payload["env"] = env
    else:
        payload.pop("env", None)

    if model_state is not None:
        _restore_model(payload, model_state)

    if payload or state.get("settings_existed"):
        _write_json(path, payload)
    elif path.exists():
        path.unlink()
    state_path.unlink()
    return True

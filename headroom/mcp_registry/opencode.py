"""OpenCode MCP registrar.

OpenCode stores MCP server configuration in ``~/.config/opencode/opencode.json``
under the ``mcp`` key.  Each entry has this shape::

    {
      "type": "local",
      "command": ["headroom", "mcp", "serve", ...],
      "enabled": true
    }

This registrar reads and writes that file directly — OpenCode has no CLI
equivalent of ``claude mcp add``.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

logger = logging.getLogger(__name__)

_OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"


class OpenCodeRegistrar(MCPRegistrar):
    """Register MCP servers with OpenCode via ``opencode.json``."""

    name = "opencode"
    display_name = "OpenCode"

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or _OPENCODE_CONFIG

    # ------------------------------------------------------------------
    # MCPRegistrar interface
    # ------------------------------------------------------------------

    def detect(self) -> bool:
        return bool(shutil.which("opencode")) or self._config_path.exists()

    def get_server(self, server_name: str) -> ServerSpec | None:
        try:
            config = _read_json(self._config_path)
        except MalformedConfigError:
            return None
        entry = config.get("mcp", {}).get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        existing = self.get_server(spec.name)
        if existing is not None:
            if _specs_equivalent(existing, spec):
                return RegisterResult(RegisterStatus.ALREADY, "matches current configuration")
            if not force:
                return RegisterResult(RegisterStatus.MISMATCH, _diff_specs(existing, spec))
            # force=True — overwrite below

        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            config = _read_json(self._config_path)
            mcp_section = config.setdefault("mcp", {})
            mcp_section[spec.name] = _spec_to_entry(spec)
            _write_json(self._config_path, config)
        except MalformedConfigError as exc:
            return RegisterResult(
                RegisterStatus.FAILED,
                f"refusing to overwrite malformed config at {exc.path}: {exc.cause}",
            )
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"could not write {self._config_path}: {exc}"
            )
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {self._config_path}")

    def unregister_server(self, server_name: str) -> bool:
        if not self._config_path.exists():
            return False
        try:
            config = _read_json(self._config_path)
        except (OSError, MalformedConfigError):
            return False
        mcp = config.get("mcp", {})
        if server_name not in mcp:
            return False
        del mcp[server_name]
        try:
            _write_json(self._config_path, config)
        except OSError:
            return False
        return True


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class MalformedConfigError(Exception):
    """Raised when the OpenCode config file exists but cannot be parsed."""

    def __init__(self, path: Path, cause: Exception) -> None:
        self.path = path
        self.cause = cause
        super().__init__(f"Cannot parse {path}: {cause}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise MalformedConfigError(path, exc) from exc
    if not isinstance(data, dict):
        raise MalformedConfigError(
            path, ValueError(f"Expected top-level object, got {type(data).__name__}")
        )
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    """Serialize a ServerSpec to the opencode.json mcp entry format."""
    entry: dict[str, Any] = {
        "type": "local",
        "command": [spec.command, *spec.args],
        "enabled": True,
    }
    if spec.env:
        entry["environment"] = dict(spec.env)
    return entry


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    """Deserialize an opencode.json mcp entry to a ServerSpec."""
    command_list = entry.get("command", [])
    if isinstance(command_list, list) and command_list:
        command = str(command_list[0])
        args = tuple(str(a) for a in command_list[1:])
    else:
        command = ""
        args = ()
    env_raw = entry.get("environment", entry.get("env", {}))
    env: dict[str, str] = (
        {str(k): str(v) for k, v in env_raw.items()} if isinstance(env_raw, dict) else {}
    )
    return ServerSpec(name=name, command=command, args=args, env=env)


def _specs_equivalent(a: ServerSpec, b: ServerSpec) -> bool:
    return (
        a.name == b.name
        and a.command == b.command
        and tuple(a.args) == tuple(b.args)
        and dict(a.env) == dict(b.env)
    )


def _diff_specs(existing: ServerSpec, requested: ServerSpec) -> str:
    parts: list[str] = []
    if existing.command != requested.command:
        parts.append(f"command {existing.command!r} -> {requested.command!r}")
    if tuple(existing.args) != tuple(requested.args):
        parts.append(f"args {list(existing.args)} -> {list(requested.args)}")
    if dict(existing.env) != dict(requested.env):
        parts.append(f"env {dict(existing.env)} -> {dict(requested.env)}")
    return "; ".join(parts) if parts else "spec differs in unidentified field(s)"

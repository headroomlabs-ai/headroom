"""OpenCode (sst/opencode) MCP registrar.

OpenCode stores MCP servers in its JSON config (project-local ``opencode.json``,
overriding the global ``~/.config/opencode/opencode.json``) under the ``mcp``
key. Each local (stdio) server is described as::

    "mcp": {
      "headroom": {
        "type": "local",
        "command": ["headroom", "mcp", "serve"],
        "enabled": true,
        "environment": { "HEADROOM_PROXY_URL": "http://127.0.0.1:8787" }
      }
    }

This registrar edits that JSON file in place, keyed on the same project-local
``opencode.json`` that ``headroom wrap opencode`` routes providers through, so
the retrieve tool and the proxy stay on the same config.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec


class OpenCodeRegistrar(MCPRegistrar):
    """Register MCP servers with OpenCode via its ``opencode.json`` config."""

    name = "opencode"
    display_name = "OpenCode"

    def __init__(self, *, config_path: Path | None = None) -> None:
        # OpenCode resolves project config from the working directory, so the
        # registrar targets ``$PWD/opencode.json`` unless told otherwise.
        self._config_file = config_path or (Path.cwd() / "opencode.json")

    # ------------------------------------------------------------------
    # MCPRegistrar interface
    # ------------------------------------------------------------------

    def detect(self) -> bool:
        if shutil.which("opencode"):
            return True
        global_dir = Path.home() / ".config" / "opencode"
        if global_dir.is_dir():
            return True
        return self._config_file.exists()

    def get_server(self, server_name: str) -> ServerSpec | None:
        servers = self._load_json().get("mcp")
        if not isinstance(servers, dict):
            return None
        entry = servers.get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        existing = self.get_server(spec.name)
        if existing is not None and _specs_equivalent(existing, spec):
            return RegisterResult(RegisterStatus.ALREADY, "matches current configuration")
        if existing is not None and not force:
            return RegisterResult(RegisterStatus.MISMATCH, _diff_specs(existing, spec))
        return self._write_entry(spec)

    def unregister_server(self, server_name: str) -> bool:
        data = self._load_json()
        servers = data.get("mcp")
        if not isinstance(servers, dict) or server_name not in servers:
            return False
        del servers[server_name]
        if not servers:
            data.pop("mcp", None)
        return self._dump_json(data)

    # ------------------------------------------------------------------
    # File IO
    # ------------------------------------------------------------------

    def _load_json(self) -> dict[str, Any]:
        if not self._config_file.exists():
            return {}
        try:
            data = json.loads(self._config_file.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _dump_json(self, data: dict[str, Any]) -> bool:
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            self._config_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError:
            return False
        return True

    def _write_entry(self, spec: ServerSpec) -> RegisterResult:
        data = self._load_json()
        servers = data.get("mcp")
        if not isinstance(servers, dict):
            servers = {}
            data["mcp"] = servers
        servers[spec.name] = _spec_to_entry(spec)
        if self._dump_json(data):
            return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {self._config_file}")
        return RegisterResult(RegisterStatus.FAILED, f"could not write {self._config_file}")


# ----------------------------------------------------------------------
# JSON <-> ServerSpec helpers (kept module-private)
# ----------------------------------------------------------------------


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    """Serialize a :class:`ServerSpec` into OpenCode's local-MCP JSON shape."""
    entry: dict[str, Any] = {
        "type": "local",
        "command": [spec.command, *spec.args],
        "enabled": True,
    }
    if spec.env:
        entry["environment"] = dict(spec.env)
    return entry


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    command_value = entry.get("command", [])
    if isinstance(command_value, list) and command_value:
        command = str(command_value[0])
        args = tuple(str(a) for a in command_value[1:])
    elif isinstance(command_value, str):
        command = command_value
        args = ()
    else:
        command = ""
        args = ()
    env_value = entry.get("environment")
    if not isinstance(env_value, dict):
        env_value = entry.get("env", {})
    env: dict[str, str] = {}
    if isinstance(env_value, dict):
        env = {str(k): str(v) for k, v in env_value.items()}
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
    return "; ".join(parts) or "spec differs in unidentified field(s)"

"""OpenCode MCP registrar.

OpenCode stores MCP server configuration under ``~/.config/opencode/opencode.json``
(or ``opencode.jsonc``, if present) under the top-level ``mcp`` key. This registrar
edits that JSON file directly.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from headroom.install.paths import opencode_config_path

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

logger = logging.getLogger(__name__)


def _strip_json_line_comments(text: str) -> str:
    """Strip ``//`` line comments from JSONC text.

    Tries standard JSON first (via the caller) so URLs containing ``//`` are
    never mangled; this is only invoked as a fallback once plain
    ``json.loads`` has already failed. Two-pass: (1) remove comment-only
    lines, (2) strip inline trailing comments that follow a comma.
    """
    cleaned = re.sub(r"^\s*//[^\n]*\n", "", text, flags=re.MULTILINE)
    return re.sub(r",\s*//[^\n]*", ",", cleaned)


def _parse_json_loose(raw: str) -> dict[str, Any] | None:
    """Parse JSON or JSONC text, returning ``None`` if it can't be parsed as an object."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(_strip_json_line_comments(raw))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON/JSONC file, returning empty dict if absent or unparseable.

    Safe for READ-ONLY callers only. Do NOT use before a full-file rewrite:
    an unparseable existing file returns ``{}`` here, and writing that back
    would destroy the user's other config. Use :func:`_read_json_for_write`
    on the write path instead.
    """
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    return _parse_json_loose(raw) or {}


class _MalformedConfigError(Exception):
    """The target config file exists but is not a parseable JSON object.

    Raised on the write path so we refuse to clobber a config we can't safely
    merge into, rather than silently overwriting the user's other settings.
    """


def _read_json_for_write(path: Path) -> dict[str, Any]:
    """Read a JSON/JSONC object for a subsequent full-file rewrite.

    Returns ``{}`` only when the file is absent or empty (safe to start fresh).
    ``//`` line comments are stripped before parsing, matching the loose JSONC
    handling used by the OpenCode provider-block writer — note the rewritten
    file loses any comments it had. If the file exists with content but still
    does not parse as a JSON object, raise :class:`_MalformedConfigError` so
    the caller aborts instead of overwriting unrelated user config — the
    OpenCode config file holds ``theme``/``model``/``provider``/other MCP
    servers alongside the ``mcp`` block.
    """
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")  # OSError propagates to the caller
    if not raw.strip():
        return {}
    data = _parse_json_loose(raw)
    if data is None:
        raise _MalformedConfigError("not valid JSON/JSONC")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    command_value = entry.get("command")
    if isinstance(command_value, list):
        args = tuple(str(x) for x in command_value[1:])
        command = str(command_value[0])
    else:
        command = str(command_value) if command_value else ""
        args = ()
    env_value = entry.get("environment", entry.get("env", {}))
    env: dict[str, str] = {}
    if isinstance(env_value, dict):
        env = {str(k): str(v) for k, v in env_value.items()}
    return ServerSpec(name=name, command=command, args=args, env=env)


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "local",
        "command": [spec.command, *spec.args],
        "enabled": True,
    }
    if spec.env:
        entry["environment"] = dict(spec.env)
    return entry


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
    if not parts:
        return "spec differs in unidentified field(s)"
    return "; ".join(parts)


class OpencodeRegistrar(MCPRegistrar):
    """Register MCP servers with OpenCode."""

    name = "opencode"
    display_name = "OpenCode"

    def __init__(self, *, config_path: Path | None = None) -> None:
        self._config_path = config_path or opencode_config_path()

    def detect(self) -> bool:
        if shutil.which("opencode"):
            return True
        return self._config_path.parent.is_dir()

    def get_server(self, server_name: str) -> ServerSpec | None:
        data = _read_json(self._config_path)
        mcp = data.get("mcp", {})
        if not isinstance(mcp, dict):
            return None
        entry = mcp.get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        existing = self.get_server(spec.name)

        if existing is not None and _specs_equivalent(existing, spec):
            return RegisterResult(RegisterStatus.ALREADY, "matches current configuration")

        if existing is not None and not force:
            return RegisterResult(
                RegisterStatus.MISMATCH,
                _diff_specs(existing, spec),
            )

        if existing is not None and force:
            # Remove the existing entry before rewriting.
            self.unregister_server(spec.name)

        return self._write_entry(spec)

    def unregister_server(self, server_name: str) -> bool:
        data = _read_json(self._config_path)
        mcp = data.get("mcp", {})
        if not isinstance(mcp, dict):
            return False
        if server_name not in mcp:
            return False
        del mcp[server_name]
        if not mcp:
            data.pop("mcp", None)
        try:
            _write_json(self._config_path, data)
        except OSError:
            return False
        return True

    def _write_entry(self, spec: ServerSpec) -> RegisterResult:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = _read_json_for_write(self._config_path)
            mcp = data.setdefault("mcp", {})
            if not isinstance(mcp, dict):
                mcp = {}
                data["mcp"] = mcp
            mcp[spec.name] = _spec_to_entry(spec)
            _write_json(self._config_path, data)
        except _MalformedConfigError as exc:
            # Refuse to overwrite: the config file holds theme/model/provider
            # and other MCP servers that a blind rewrite would wipe.
            return RegisterResult(
                RegisterStatus.FAILED,
                f"{self._config_path} exists but is not valid JSON ({exc}); "
                "refusing to overwrite. Fix or remove the file, then re-run.",
            )
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"could not write {self._config_path}: {exc}"
            )
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {self._config_path}")

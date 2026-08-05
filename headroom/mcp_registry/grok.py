"""Grok CLI MCP registrar.

Grok stores MCP server config in ``$GROK_HOME/config.toml`` (default
``~/.grok/config.toml``) as ``[mcp_servers.<name>]`` tables. There is no
general-purpose CLI for adding entries, so we edit the file in place using
marker-delimited blocks so we can idempotently inject, replace, and remove
our entry without disturbing anything else the user has configured.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from headroom import fsutil

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — exercised only on 3.10
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_MARKER_START = "# --- Headroom MCP server ---"
_MARKER_END = "# --- end Headroom MCP server ---"


def _marker_start(server_name: str) -> str:
    if server_name == "headroom":
        return _MARKER_START
    return f"# --- Headroom MCP server: {server_name} ---"


def _marker_end(server_name: str) -> str:
    if server_name == "headroom":
        return _MARKER_END
    return f"# --- end Headroom MCP server: {server_name} ---"


class GrokRegistrar(MCPRegistrar):
    """Register MCP servers with the Grok CLI."""

    name = "grok"
    display_name = "Grok CLI"

    def __init__(self, *, home_dir: Path | None = None) -> None:
        if home_dir is not None:
            self._grok_dir = home_dir / ".grok"
        elif os.environ.get("GROK_HOME"):
            self._grok_dir = Path(os.environ["GROK_HOME"]).expanduser()
        else:
            self._grok_dir = Path.home() / ".grok"
        self._config_file = self._grok_dir / "config.toml"

    def detect(self) -> bool:
        return self._grok_dir.is_dir()

    def get_server(self, server_name: str) -> ServerSpec | None:
        data = self._load_toml()
        servers = data.get("mcp_servers", {})
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
            content = self._read_text()
            if _marker_start(spec.name) not in content:
                return RegisterResult(
                    RegisterStatus.MISMATCH,
                    "user-managed [mcp_servers."
                    f"{spec.name}] entry outside Headroom markers; "
                    f"{_diff_specs(existing, spec)}",
                )
            return RegisterResult(RegisterStatus.MISMATCH, _diff_specs(existing, spec))

        if existing is not None and force:
            content = self._read_text()
            if _marker_start(spec.name) not in content:
                return RegisterResult(
                    RegisterStatus.MISMATCH,
                    "user-managed [mcp_servers."
                    f"{spec.name}] entry outside Headroom markers; "
                    f"{_diff_specs(existing, spec)}",
                )
            self.unregister_server(spec.name)

        return self._write_block(spec)

    def unregister_server(self, server_name: str) -> bool:
        if not self._config_file.exists():
            return False
        content = self._read_text()
        marker_start = _marker_start(server_name)
        marker_end = _marker_end(server_name)
        if marker_start not in content or marker_end not in content:
            return False
        try:
            start = content.index(marker_start)
            end = content.index(marker_end) + len(marker_end)
        except ValueError:
            return False
        before = content[:start].rstrip("\n")
        after = content[end:].lstrip("\n")
        if before and after:
            new_content = before + "\n\n" + after
        else:
            new_content = (before or after).rstrip("\n") + ("\n" if (before or after) else "")
        try:
            fsutil.write_text(self._config_file, new_content)
        except OSError:
            return False
        return True

    def _load_toml(self) -> dict[str, Any]:
        if not self._config_file.exists():
            return {}
        try:
            data = tomllib.loads(fsutil.read_text(self._config_file))
        except (tomllib.TOMLDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _read_text(self) -> str:
        return fsutil.read_text(self._config_file, default="")

    def _write_block(self, spec: ServerSpec) -> RegisterResult:
        block = _render_block(spec)
        try:
            self._grok_dir.mkdir(parents=True, exist_ok=True)
            content = self._read_text()
            marker_start = _marker_start(spec.name)
            marker_end = _marker_end(spec.name)
            if marker_start in content and marker_end in content:
                start = content.index(marker_start)
                end = content.index(marker_end) + len(marker_end)
                content = (
                    content[:start].rstrip("\n")
                    + ("\n\n" if content[:start].rstrip("\n") else "")
                    + block
                    + "\n"
                    + content[end:].lstrip("\n")
                )
            elif content.strip():
                content = content.rstrip("\n") + "\n\n" + block + "\n"
            else:
                content = block + "\n"
            fsutil.write_text(self._config_file, content)
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"could not write {self._config_file}: {exc}"
            )
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {self._config_file}")


def _render_block(spec: ServerSpec) -> str:
    lines: list[str] = [
        _marker_start(spec.name),
        f"[mcp_servers.{spec.name}]",
        f"command = {_toml_str(spec.command)}",
    ]
    if spec.args:
        items = ", ".join(_toml_str(a) for a in spec.args)
        lines.append(f"args = [{items}]")
    if spec.env:
        lines.append("")
        lines.append(f"[mcp_servers.{spec.name}.env]")
        for k, v in spec.env.items():
            lines.append(f"{k} = {_toml_str(v)}")
    lines.append(_marker_end(spec.name))
    return "\n".join(lines)


def _toml_str(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    args_value = entry.get("args", [])
    if isinstance(args_value, list):
        args = tuple(str(x) for x in args_value)
    else:
        args = ()
    env_value = entry.get("env", {})
    env: dict[str, str] = {}
    if isinstance(env_value, dict):
        env = {str(k): str(v) for k, v in env_value.items()}
    return ServerSpec(
        name=name,
        command=str(entry.get("command", "")),
        args=args,
        env=env,
    )


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

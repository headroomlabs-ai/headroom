"""OpenAI Codex CLI MCP registrar.

Codex stores MCP server config in ``$CODEX_HOME/config.toml`` when
``CODEX_HOME`` is set, otherwise ``~/.codex/config.toml``, as
``[mcp_servers.<name>]`` tables (with optional ``[mcp_servers.<name>.env]``
sub-tables). There is no general-purpose CLI for adding entries, so we
edit the file in place — using marker-delimited blocks so we can
idempotently inject, replace, and remove our entry without disturbing
anything else the user has configured.
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


class CodexRegistrar(MCPRegistrar):
    """Register MCP servers with the OpenAI Codex CLI."""

    name = "codex"
    display_name = "OpenAI Codex CLI"

    def __init__(self, *, home_dir: Path | None = None) -> None:
        if home_dir is not None:
            self._codex_dir = home_dir / ".codex"
        elif os.environ.get("CODEX_HOME"):
            self._codex_dir = Path(os.environ["CODEX_HOME"]).expanduser()
        else:
            self._codex_dir = Path.home() / ".codex"
        self._config_file = self._codex_dir / "config.toml"

    # ------------------------------------------------------------------
    # MCPRegistrar interface
    # ------------------------------------------------------------------

    def detect(self) -> bool:
        return self._codex_dir.is_dir()

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
                # Entry exists but wasn't written by us — refuse to clobber.
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
                # Even force=True is only allowed to replace blocks that
                # Headroom owns. Otherwise appending our table would create a
                # duplicate [mcp_servers.<name>] TOML section and may clobber a
                # user-managed integration.
                return RegisterResult(
                    RegisterStatus.MISMATCH,
                    "user-managed [mcp_servers."
                    f"{spec.name}] entry outside Headroom markers; "
                    f"{_diff_specs(existing, spec)}",
                )
            # Drop any prior Headroom block before re-writing.
            self.unregister_server(spec.name)

        return self._write_block(spec)

    def unregister_server(self, server_name: str) -> bool:
        # Only removes the marker-block we wrote. User-managed entries
        # outside markers are intentionally preserved.
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

    # ------------------------------------------------------------------
    # File IO
    # ------------------------------------------------------------------

    def _load_toml(self) -> dict[str, Any]:
        if not self._config_file.exists():
            return {}
        try:
            # Read via fsutil (UTF-8 with locale fallback) so a config that a
            # tool wrote in the system locale (e.g. GBK) still parses instead
            # of failing tomllib's UTF-8 requirement. See #733.
            data = tomllib.loads(fsutil.read_text(self._config_file))
        except (tomllib.TOMLDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _read_text(self) -> str:
        return fsutil.read_text(self._config_file, default="")

    def _write_block(self, spec: ServerSpec) -> RegisterResult:
        block = _render_block(spec)
        try:
            self._codex_dir.mkdir(parents=True, exist_ok=True)
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


# ----------------------------------------------------------------------
# TOML rendering / parsing helpers (kept module-private)
# ----------------------------------------------------------------------


def _render_block(spec: ServerSpec) -> str:
    """Render a Headroom-marked TOML block for ``spec``."""
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
    """Render a Python string as a TOML basic string literal."""
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

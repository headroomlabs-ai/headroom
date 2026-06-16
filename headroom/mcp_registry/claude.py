"""Claude Code MCP registrar.

Claude Code 2.x stores MCP server configuration in ``~/.claude/.claude.json``
and ships a CLI (``claude mcp add/remove/list/get``) that owns the file.
Older Claude Code releases (and the Claude Desktop app) read
``~/.claude/mcp.json``. This registrar prefers the CLI for writes when
available, and reads the underlying JSON files directly for compare /
``get_server`` so it is robust to CLI output format changes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

logger = logging.getLogger(__name__)


class ClaudeRegistrar(MCPRegistrar):
    """Register MCP servers with Claude Code."""

    name = "claude"
    display_name = "Claude Code"

    def __init__(
        self,
        *,
        claude_cli: str | None | object = ...,
        home_dir: Path | None = None,
        config_dir: Path | None = None,
    ) -> None:
        """Allow overrides for testing.

        ``claude_cli`` defaults to :func:`shutil.which` lookup. Pass
        ``None`` to force the file-based fallback path. Pass an explicit
        path to point at a specific binary. ``CLAUDE_CONFIG_DIR`` is honored
        for real user sessions; ``home_dir`` keeps tests isolated from the
        caller's environment unless ``config_dir`` is passed explicitly.
        """
        home = home_dir if home_dir is not None else Path.home()
        self._claude_dir = _resolve_claude_config_dir(home, config_dir, honor_env=home_dir is None)
        self._modern_config = self._claude_dir / ".claude.json"
        self._legacy_config = self._claude_dir / "mcp.json"
        if claude_cli is ...:
            self._claude_cli = shutil.which("claude")
        else:
            # ``...`` sentinel preserves "not set"; explicit None disables CLI.
            self._claude_cli = claude_cli  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # MCPRegistrar interface
    # ------------------------------------------------------------------

    def detect(self) -> bool:
        if self._claude_cli:
            return True
        return self._claude_dir.is_dir()

    def get_server(self, server_name: str) -> ServerSpec | None:
        # Read from disk regardless of whether the CLI is present — the file
        # format is stable and easier to compare than CLI output.
        for config_path in (self._modern_config, self._legacy_config):
            entry = self._read_server_entry(config_path, server_name)
            if entry is not None:
                return entry
        return None

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        existing = self.get_server(spec.name)
        if existing is not None:
            if _specs_equivalent(existing, spec):
                return RegisterResult(RegisterStatus.ALREADY, "matches current configuration")
            if not force:
                return RegisterResult(
                    RegisterStatus.MISMATCH,
                    _diff_specs(existing, spec),
                )
            # force=True: remove first, then write fresh below.
            self.unregister_server(spec.name)

        if self._claude_cli:
            return self._register_via_cli(spec)
        return self._register_via_file(spec)

    def unregister_server(self, server_name: str) -> bool:
        if self._claude_cli:
            result = subprocess.run(
                [str(self._claude_cli), "mcp", "remove", server_name, "-s", "user"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True
            logger.debug("claude mcp remove failed: %s", result.stderr.strip())
            # Fall through to file-based removal in case CLI didn't know
            # about the user-scope entry but the file still has it.
        removed = False
        for config_path in (self._modern_config, self._legacy_config):
            removed = self._remove_from_file(config_path, server_name) or removed
        return removed

    # ------------------------------------------------------------------
    # CLI-backed implementation
    # ------------------------------------------------------------------

    def _register_via_cli(self, spec: ServerSpec) -> RegisterResult:
        cmd = [str(self._claude_cli), "mcp", "add", spec.name, "-s", "user"]
        for k, v in spec.env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += ["--", spec.command, *spec.args]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return RegisterResult(RegisterStatus.REGISTERED, "via `claude mcp add` (scope: user)")
        # CLI failed — try the file fallback rather than giving up.
        logger.warning("claude mcp add failed: %s", result.stderr.strip())
        file_result = self._register_via_file(spec)
        if file_result.status == RegisterStatus.REGISTERED:
            return RegisterResult(
                RegisterStatus.REGISTERED,
                f"via file fallback after CLI failed: {result.stderr.strip()}",
            )
        return RegisterResult(
            RegisterStatus.FAILED,
            f"CLI: {result.stderr.strip()}; file: {file_result.detail}",
        )

    # ------------------------------------------------------------------
    # File-backed implementation (CLI absent / older clients)
    # ------------------------------------------------------------------

    def _register_via_file(self, spec: ServerSpec) -> RegisterResult:
        # Prefer the modern config path. If only the legacy file exists,
        # write to that to avoid surprising older clients.
        target = self._modern_config
        if not self._modern_config.exists() and self._legacy_config.exists():
            target = self._legacy_config

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            config = _read_json(target)
            servers = config.setdefault("mcpServers", {})
            servers[spec.name] = _spec_to_entry(spec)
            _write_json(target, config)
        except OSError as exc:
            return RegisterResult(RegisterStatus.FAILED, f"could not write {target}: {exc}")
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {target}")

    def _remove_from_file(self, path: Path, server_name: str) -> bool:
        if not path.exists():
            return False
        try:
            config = _read_json(path)
        except OSError:
            return False
        servers = config.get("mcpServers", {})
        if server_name not in servers:
            return False
        del servers[server_name]
        try:
            _write_json(path, config)
        except OSError:
            return False
        return True

    def _read_server_entry(self, path: Path, server_name: str) -> ServerSpec | None:
        if not path.exists():
            return None
        try:
            config = _read_json(path)
        except OSError:
            return None
        entry = config.get("mcpServers", {}).get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_claude_config_dir(
    home: Path,
    config_dir: Path | None,
    *,
    honor_env: bool,
) -> Path:
    if config_dir is not None:
        return config_dir
    if honor_env:
        env_dir = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
        if env_dir:
            return Path(env_dir).expanduser()
    return home / ".claude"


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning empty dict if absent or unparseable."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {"command": spec.command}
    if spec.args:
        entry["args"] = list(spec.args)
    if spec.env:
        entry["env"] = dict(spec.env)
    return entry


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
    """Two specs match when every field is equal."""
    return (
        a.name == b.name
        and a.command == b.command
        and tuple(a.args) == tuple(b.args)
        and dict(a.env) == dict(b.env)
    )


def _diff_specs(existing: ServerSpec, requested: ServerSpec) -> str:
    """Render the difference between two specs for human consumption."""
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

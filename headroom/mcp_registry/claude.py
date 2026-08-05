"""Claude Code MCP registrar.

Claude Code 2.x stores user-scope MCP server configuration in
``~/.claude.json``, directly under the home directory. The ``claude`` CLI
(``claude mcp add/remove/list/get``) owns this file; setting
``CLAUDE_CONFIG_DIR`` relocates it to ``$CLAUDE_CONFIG_DIR/.claude.json``.
Older Claude Code releases (and the Claude Desktop app) read
``~/.claude/mcp.json`` instead. This registrar prefers the CLI for writes
when available, and reads the underlying JSON files directly for compare /
``get_server`` so it is robust to CLI output format changes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from headroom._subprocess import run

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
        for real user sessions; ``home_dir`` isolates file-based reads and
        writes from the caller's real home directory, unless ``config_dir``
        is passed explicitly. It does not isolate CLI subprocess calls (see
        ``claude_cli``) — pass ``claude_cli=None`` alongside ``home_dir`` to
        keep a test fully off the real ``claude`` binary.
        """
        home = home_dir if home_dir is not None else Path.home()
        modern_dir = _resolve_claude_config_dir(home, config_dir, honor_env=home_dir is None)
        # Legacy config lives under the real ``.claude`` directory regardless
        # of where the modern config resolved to (CLAUDE_CONFIG_DIR only
        # relocates the modern file, per Claude Code's own behavior).
        self._claude_dir = home / ".claude"
        self._modern_dir = modern_dir
        self._isolated_cli_env = home_dir is not None or config_dir is not None
        self._modern_config = modern_dir / ".claude.json"
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
        return self._claude_dir.is_dir() or self._modern_config.exists()

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
        removed = False
        if self._claude_cli:
            result = run(
                [str(self._claude_cli), "mcp", "remove", server_name, "-s", "user"],
                capture_output=True,
                text=True,
                env=self._claude_cli_env(),
            )
            if result.returncode == 0:
                removed = True
            else:
                logger.debug("claude mcp remove failed: %s", result.stderr.strip())
        # Always clean up both files too — the CLI only touches the modern
        # config, so a legacy entry (or one it didn't know about) can remain.
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

        result = run(
            cmd,
            capture_output=True,
            text=True,
            env=self._claude_cli_env(),
        )
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
            config = _read_json_for_write(target)
            servers = config.get("mcpServers")
            if not isinstance(servers, dict):
                config["mcpServers"] = servers = {}
            servers[spec.name] = _spec_to_entry(spec)
            _write_json(target, config)
        except _MalformedConfigError as exc:
            # Refuse to overwrite: the file holds unrelated Claude state
            # (projects, oauthAccount, history) that a blind rewrite would wipe.
            return RegisterResult(
                RegisterStatus.FAILED,
                f"{target} exists but is not valid JSON ({exc}); refusing to overwrite. "
                "Fix or remove the file, then re-run.",
            )
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
        servers = config.get("mcpServers")
        if not isinstance(servers, dict) or server_name not in servers:
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
        servers = config.get("mcpServers")
        if not isinstance(servers, dict):
            return None
        entry = servers.get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _claude_cli_env(self) -> dict[str, str] | None:
        if not self._isolated_cli_env:
            return None
        env = os.environ.copy()
        # Point the CLI at the directory holding the modern ``.claude.json``
        # (CLAUDE_CONFIG_DIR relocates that file), not the legacy ``.claude`` dir.
        env["CLAUDE_CONFIG_DIR"] = str(self._modern_dir)
        return env


def _resolve_claude_config_dir(
    home: Path,
    config_dir: Path | None,
    *,
    honor_env: bool,
) -> Path:
    """Resolve the directory holding the *modern* ``.claude.json`` config.

    Defaults to ``home`` itself, since Claude Code's modern config lives at
    ``~/.claude.json`` directly under the home directory. ``CLAUDE_CONFIG_DIR``,
    when set, relocates it to ``$CLAUDE_CONFIG_DIR/.claude.json``.
    """
    if config_dir is not None:
        return config_dir
    if honor_env:
        env_dir = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
        if env_dir:
            return Path(env_dir).expanduser()
    return home


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning empty dict if absent or unparseable.

    Safe for READ-ONLY callers. Do NOT use before a full-file rewrite: an
    unparseable existing file returns ``{}`` here, and writing that back would
    destroy the user's other config. Use :func:`_read_json_for_write` on the
    write path instead.
    """
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


class _MalformedConfigError(Exception):
    """The target config file exists but is not a parseable JSON object.

    Raised on the write path so we refuse to clobber a file we can't safely
    merge into, rather than silently overwriting the user's state.
    """


def _read_json_for_write(path: Path) -> dict[str, Any]:
    """Read a JSON object for a subsequent full-file rewrite.

    Returns ``{}`` only when the file is absent or empty (safe to start fresh).
    If the file exists with content but does not parse as a JSON object, raise
    :class:`_MalformedConfigError` so the caller aborts instead of overwriting
    unrelated user config (e.g. ``~/.claude/.claude.json`` holds ``projects``,
    ``oauthAccount``, and session history alongside ``mcpServers``).
    """
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")  # OSError propagates to the caller
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _MalformedConfigError(str(exc)) from exc
    if not isinstance(data, dict):
        raise _MalformedConfigError("top-level JSON is not an object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
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

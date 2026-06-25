"""CodeBuddy MCP registrar.

CodeBuddy stores MCP server configuration in ``~/.codebuddy/.mcp.json``
and ships a CLI (``codebuddy mcp add/remove/list/get``) that owns the file.
This registrar prefers the CLI for writes when available, and reads the
underlying JSON files directly for compare / ``get_server`` so it is robust
to CLI output format changes.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from headroom._subprocess import run as _sp_run

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

logger = logging.getLogger(__name__)


class CodeBuddyRegistrar(MCPRegistrar):
    """Register MCP servers with CodeBuddy."""

    name = "codebuddy"
    display_name = "CodeBuddy"

    def __init__(
        self,
        *,
        codebuddy_cli: str | None | object = ...,
        home_dir: Path | None = None,
    ) -> None:
        """Allow overrides for testing.

        ``codebuddy_cli`` defaults to :func:`shutil.which` lookup. Pass
        ``None`` to force the file-based fallback path. Pass an explicit
        path to point at a specific binary.
        """
        home = home_dir if home_dir is not None else Path.home()
        self._codebuddy_dir = home / ".codebuddy"
        self._config = home / ".codebuddy" / ".mcp.json"
        if codebuddy_cli is ...:
            self._codebuddy_cli = shutil.which("codebuddy")
        else:
            self._codebuddy_cli = codebuddy_cli  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # MCPRegistrar interface
    # ------------------------------------------------------------------

    def detect(self) -> bool:
        if self._codebuddy_cli:
            return True
        return self._codebuddy_dir.is_dir()

    def get_server(self, server_name: str) -> ServerSpec | None:
        entry = self._read_server_entry(self._config, server_name)
        return entry

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
            self.unregister_server(spec.name)

        if self._codebuddy_cli:
            return self._register_via_cli(spec)
        return self._register_via_file(spec)

    def unregister_server(self, server_name: str) -> bool:
        if self._codebuddy_cli:
            result = _sp_run(
                [str(self._codebuddy_cli), "mcp", "remove", server_name, "-s", "user"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True
            logger.debug("codebuddy mcp remove failed: %s", result.stderr.strip())
        return self._remove_from_file(self._config, server_name)

    # ------------------------------------------------------------------
    # CLI-backed implementation
    # ------------------------------------------------------------------

    def _register_via_cli(self, spec: ServerSpec) -> RegisterResult:
        cmd = [str(self._codebuddy_cli), "mcp", "add", spec.name, "-s", "user"]
        for k, v in spec.env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += ["--", spec.command, *spec.args]

        result = _sp_run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return RegisterResult(
                RegisterStatus.REGISTERED, "via `codebuddy mcp add` (scope: user)"
            )
        logger.warning("codebuddy mcp add failed: %s", result.stderr.strip())
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
    # File-backed implementation
    # ------------------------------------------------------------------

    def _register_via_file(self, spec: ServerSpec) -> RegisterResult:
        try:
            self._config.parent.mkdir(parents=True, exist_ok=True)
            config = _read_json(self._config)
            servers = config.setdefault("mcpServers", {})
            servers[spec.name] = _spec_to_entry(spec)
            _write_json(self._config, config)
        except OSError as exc:
            return RegisterResult(RegisterStatus.FAILED, f"could not write {self._config}: {exc}")
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {self._config}")

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

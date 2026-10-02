"""Antigravity IDE MCP registrar.

Antigravity (Google's AI-first IDE) stores MCP server configuration in a
JSON file holding a top-level ``mcpServers`` object::

    {
      "mcpServers": {
        "headroom": {
          "command": "/path/to/headroom",
          "args": ["mcp", "serve"],
          "env": {"HEADROOM_PROXY_URL": "http://127.0.0.1:8787"}
        }
      }
    }

The on-disk location has moved between Antigravity releases — Google's
current Antigravity codelab identifies ``~/.gemini/config/mcp_config.json``
as the active Desktop config, while older guides cite
``~/.gemini/antigravity/mcp_config.json`` — so this registrar treats the
former as authoritative: it probes the known candidates in order and uses
the first one that exists, falling back to the current location when none
does yet. Uninstall removes the entry from every candidate that still holds
it, so a stale duplicate in the legacy location can never survive.

Only global configuration is supported: Antigravity does not read
per-project MCP files.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from headroom import fsutil

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

logger = logging.getLogger(__name__)


def _config_candidates(home: Path) -> list[Path]:
    """Known Antigravity MCP config locations, current first."""
    gemini_dir = home / ".gemini"
    return [
        gemini_dir / "config" / "mcp_config.json",
        gemini_dir / "antigravity" / "mcp_config.json",
    ]


def _mcp_cache_dirs(home: Path, server_name: str) -> list[Path]:
    """Local MCP server caches Antigravity keeps per install surface."""
    gemini_dir = home / ".gemini"
    return [
        gemini_dir / "antigravity-ide" / "mcp" / server_name,
        gemini_dir / "antigravity" / "mcp" / server_name,
        gemini_dir / "antigravity-cli" / "mcp" / server_name,
    ]


class AntigravityRegistrar(MCPRegistrar):
    """Register MCP servers with Antigravity IDE."""

    name = "antigravity"
    display_name = "Antigravity IDE"

    def __init__(self, *, home_dir: Path | None = None) -> None:
        """Allow ``home_dir`` override for testing."""
        self._home = home_dir if home_dir is not None else Path.home()

    def _resolve_config(self) -> Path:
        """Return the config file to use.

        The current Desktop location (``~/.gemini/config/mcp_config.json``)
        is authoritative: when its parent directory exists the IDE reads
        that location, so a stale legacy file must not capture the
        registration. Otherwise use the first existing candidate, falling
        back to the current location when none does yet.
        """
        candidates = _config_candidates(self._home)
        if candidates[0].parent.is_dir():
            return candidates[0]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def detect(self) -> bool:
        home = self._home
        gemini_dir = home / ".gemini"
        if (gemini_dir / "antigravity").is_dir():
            return True
        # The current Desktop config lives directly under ~/.gemini/config;
        # an install that has not created its MCP file yet still counts.
        if (gemini_dir / "config").is_dir():
            return True
        return any(candidate.exists() for candidate in _config_candidates(home))

    def get_server(self, server_name: str) -> ServerSpec | None:
        data = self._load_config()
        if data is None:
            return None
        servers = data.get("mcpServers", {})
        if not isinstance(servers, dict):
            return None
        entry = servers.get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)

    def register_server(
        self, spec: ServerSpec, *, force: bool = False
    ) -> RegisterResult:
        existing = self.get_server(spec.name)

        if existing is not None and _specs_equivalent(existing, spec):
            return RegisterResult(
                RegisterStatus.ALREADY, "matches current configuration"
            )

        if existing is not None and not force:
            return RegisterResult(RegisterStatus.MISMATCH, _diff_specs(existing, spec))

        config_path = self._resolve_config()
        data = self._load_config()
        if data is None and config_path.exists():
            # The file exists but would not parse: refuse to overwrite it and
            # turn a hand-edited typo into total loss of the user's MCP setup.
            return RegisterResult(
                RegisterStatus.FAILED,
                f"{config_path} is not valid JSON. Fix or move it, then re-run — "
                "refusing to overwrite it and lose your MCP servers.",
            )
        payload: dict[str, Any] = data if isinstance(data, dict) else {}
        servers = payload.get("mcpServers")
        if not isinstance(servers, dict):
            servers = {}
            payload["mcpServers"] = servers
        servers[spec.name] = _spec_to_entry(spec)

        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            fsutil.write_text(config_path, json.dumps(payload, indent=2) + "\n")
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"could not write {config_path}: {exc}"
            )
        verb = "updated" if existing is not None else "wrote"
        return RegisterResult(
            RegisterStatus.REGISTERED, f"{verb} {spec.name} in {config_path}"
        )

    def unregister_server(self, server_name: str) -> bool:
        removed = False
        remaining: list[Path] = []
        for config_path in _config_candidates(self._home):
            if not config_path.exists():
                continue
            data = self._load_config(config_path)
            if not isinstance(data, dict):
                # Refuse to touch a file that does not parse — the same rule
                # register_server follows — and keep checking the rest.
                continue
            servers = data.get("mcpServers")
            if not isinstance(servers, dict) or server_name not in servers:
                continue
            del servers[server_name]
            try:
                fsutil.write_text(config_path, json.dumps(data, indent=2) + "\n")
            except OSError:
                # The write failed, so this config still registers the
                # server. Record it and report failure instead of claiming a
                # removal that the untouched file contradicts.
                remaining.append(config_path)
                continue
            removed = True
        if not removed:
            return False
        # Antigravity caches each MCP server under ~/.gemini/<surface>/mcp/;
        # a removed entry keeps loading from cache until that directory goes.
        # Clear it whenever at least one config entry was removed, even if a
        # failed write means this still reports False below.
        for cache_dir in _mcp_cache_dirs(self._home, server_name):
            try:
                shutil.rmtree(cache_dir)
            except OSError:
                pass
        if remaining:
            logger.warning(
                "could not remove %r from %s; the server stays registered there",
                server_name,
                ", ".join(str(path) for path in remaining),
            )
            return False
        return True

    def _load_config(self, config_path: Path | None = None) -> dict[str, Any] | None:
        """Load a config file, defaulting to the resolved one.

        Returns ``None`` when the file does not exist or does not parse;
        callers distinguish those cases via :meth:`_resolve_config` + exists.
        """
        path = config_path if config_path is not None else self._resolve_config()
        if not path.exists():
            return None
        try:
            data = json.loads(fsutil.read_text(path))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    args_value = entry.get("args", [])
    args = tuple(str(x) for x in args_value) if isinstance(args_value, list) else ()
    env_value = entry.get("env", {})
    env = (
        {str(k): str(v) for k, v in env_value.items()}
        if isinstance(env_value, dict)
        else {}
    )
    return ServerSpec(
        name=name,
        command=str(entry.get("command", "")),
        args=args,
        env=env,
    )


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {"command": spec.command}
    if spec.args:
        entry["args"] = list(spec.args)
    if spec.env:
        entry["env"] = dict(spec.env)
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

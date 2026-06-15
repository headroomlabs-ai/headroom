"""OpenCode MCP registrar.

OpenCode stores its configuration in ``~/.config/opencode/opencode.jsonc``
as a JSONC (JSON with comments) file. MCP servers are configured in the
``"mcp"`` section, and providers are configured in the ``"provider"`` section.
This registrar edits the file in place using surgical JSONC modification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

logger = logging.getLogger(__name__)

_OPENCODE_CONFIG_DIR = Path.home() / ".config" / "opencode"
_OPENCODE_CONFIG_FILE = _OPENCODE_CONFIG_DIR / "opencode.jsonc"


def _strip_jsonc_comments(text: str) -> str:
    """Strip C-style comments from JSONC content.

    Handles both ``//`` line comments and ``/* */`` block comments,
    while preserving strings that contain comment-like sequences.
    """
    result: list[str] = []
    i = 0
    in_string = False
    escape_next = False

    while i < len(text):
        ch = text[i]

        if escape_next:
            result.append(ch)
            escape_next = False
            i += 1
            continue

        if in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            result.append(ch)
            i += 1
            continue

        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < len(text):
            next_ch = text[i + 1]
            if next_ch == "/":
                # Line comment — skip until newline
                while i < len(text) and text[i] != "\n":
                    i += 1
                continue
            if next_ch == "*":
                # Block comment — skip until */
                i += 2
                while i + 1 < len(text):
                    if text[i] == "*" and text[i + 1] == "/":
                        i += 2
                        break
                    i += 1
                continue

        result.append(ch)
        i += 1

    return "".join(result)


def _parse_jsonc(text: str) -> dict[str, Any]:
    """Parse a JSONC file, stripping comments before JSON parsing."""
    stripped = _strip_jsonc_comments(text)
    stripped = stripped.strip()
    if not stripped:
        return {}
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_jsonc(path: Path, data: dict[str, Any]) -> None:
    """Write a dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


class OpenCodeRegistrar(MCPRegistrar):
    """Register MCP servers with OpenCode."""

    name = "opencode"
    display_name = "OpenCode"

    def __init__(self, *, home_dir: Path | None = None) -> None:
        home = home_dir if home_dir is not None else Path.home()
        self._config_dir = home / ".config" / "opencode"
        self._config_file = self._config_dir / "opencode.jsonc"

    # ------------------------------------------------------------------
    # MCPRegistrar interface
    # ------------------------------------------------------------------

    def detect(self) -> bool:
        return self._config_dir.is_dir()

    def get_server(self, server_name: str) -> ServerSpec | None:
        config = self._read_config()
        servers = config.get("mcp", {})
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

        if existing is not None and force:
            self.unregister_server(spec.name)

        return self._write_server(spec)

    def unregister_server(self, server_name: str) -> bool:
        config = self._read_config()
        servers = config.get("mcp", {})
        if not isinstance(servers, dict) or server_name not in servers:
            return False
        del servers[server_name]
        if not servers:
            del config["mcp"]
        try:
            _write_jsonc(self._config_file, config)
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------
    # Config file IO
    # ------------------------------------------------------------------

    def _read_config(self) -> dict[str, Any]:
        if not self._config_file.exists():
            return {}
        try:
            text = self._config_file.read_text(encoding="utf-8")
        except OSError:
            return {}
        return _parse_jsonc(text)

    def _write_server(self, spec: ServerSpec) -> RegisterResult:
        config = self._read_config()
        mcp = config.setdefault("mcp", {})
        mcp[spec.name] = _spec_to_entry(spec)
        try:
            _write_jsonc(self._config_file, config)
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"could not write {self._config_file}: {exc}"
            )
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {self._config_file}")

    # ------------------------------------------------------------------
    # Provider config (used by wrap command)
    # ------------------------------------------------------------------

    def add_provider(
        self,
        name: str,
        *,
        base_url: str,
        models: dict[str, Any] | None = None,
    ) -> None:
        """Add a provider to the OpenCode config."""
        config = self._read_config()
        providers = config.setdefault("provider", {})
        provider_entry: dict[str, Any] = {
            "name": name.title(),
            "options": {
                "baseURL": base_url,
            },
        }
        if models:
            provider_entry["models"] = models
        providers[name] = provider_entry
        _write_jsonc(self._config_file, config)

    def remove_provider(self, name: str) -> None:
        """Remove a provider from the OpenCode config."""
        config = self._read_config()
        providers = config.get("provider", {})
        if isinstance(providers, dict) and name in providers:
            del providers[name]
            if not providers:
                del config["provider"]
            _write_jsonc(self._config_file, config)

    def override_provider_base_url(self, provider_name: str, base_url: str) -> None:
        """Override a built-in provider's baseURL to route through the proxy.

        This is the reliable way to route traffic through a proxy for OpenCode,
        since built-in providers take precedence over custom providers.
        """
        config = self._read_config()
        providers = config.setdefault("provider", {})
        provider_entry = providers.setdefault(provider_name, {})
        options = provider_entry.setdefault("options", {})
        options["baseURL"] = base_url
        _write_jsonc(self._config_file, config)

    def add_instruction(self, path: str) -> None:
        """Add an instruction path to the OpenCode config."""
        config = self._read_config()
        instructions = config.setdefault("instructions", [])
        if isinstance(instructions, list) and path not in instructions:
            instructions.append(path)
            _write_jsonc(self._config_file, config)

    def remove_instruction(self, path: str) -> None:
        """Remove an instruction path from the OpenCode config."""
        config = self._read_config()
        instructions = config.get("instructions", [])
        if isinstance(instructions, list) and path in instructions:
            instructions.remove(path)
            if not instructions:
                del config["instructions"]
            _write_jsonc(self._config_file, config)

    def is_wrapped(self) -> bool:
        """Check if the config has Headroom modifications.

        Detects both custom providers and built-in provider baseURL overrides.
        """
        config = self._read_config()
        providers = config.get("provider", {})

        # Check for custom headroom provider
        if isinstance(providers, dict) and "headroom" in providers:
            return True

        # Check for proxy baseURL override on built-in providers
        if isinstance(providers, dict):
            for _provider_name, provider_config in providers.items():
                if isinstance(provider_config, dict):
                    options = provider_config.get("options", {})
                    if isinstance(options, dict):
                        base_url = options.get("baseURL", "")
                        if isinstance(base_url, str) and "127.0.0.1:" in base_url:
                            return True

        return False

    def snapshot_config(self) -> Path | None:
        """Snapshot config before modification. Returns backup path."""
        if not self._config_file.exists():
            return None
        backup = self._config_file.with_suffix(".jsonc.headroom-backup")
        if backup.exists():
            return backup
        try:
            import shutil

            shutil.copy2(self._config_file, backup)
        except OSError:
            return None
        return backup

    def restore_config(self) -> bool:
        """Restore config from backup."""
        backup = self._config_file.with_suffix(".jsonc.headroom-backup")
        if not backup.exists():
            return False
        try:
            import shutil

            shutil.move(backup, self._config_file)
        except OSError:
            return False
        return True


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    """Convert a ServerSpec to an OpenCode MCP entry."""
    entry: dict[str, Any] = {
        "type": "local",
        "command": [spec.command, *spec.args] if spec.args else [spec.command],
        "enabled": True,
    }
    if spec.env:
        entry["environment"] = dict(spec.env)
    return entry


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    """Convert an OpenCode MCP entry to a ServerSpec."""
    command_list = entry.get("command", [])
    if isinstance(command_list, list) and command_list:
        command = str(command_list[0])
        args = tuple(str(x) for x in command_list[1:])
    else:
        command = str(entry.get("command", ""))
        args = ()

    env_value = entry.get("environment", {})
    env: dict[str, str] = {}
    if isinstance(env_value, dict):
        env = {str(k): str(v) for k, v in env_value.items()}

    return ServerSpec(
        name=name,
        command=command,
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

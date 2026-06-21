"""OpenCode MCP registrar.

OpenCode stores MCP server configuration in
``~/.config/opencode/opencode.json`` under the top-level ``mcp`` key.
Each MCP server is a dict with ``type``, ``url``, and optional
``enabled`` / ``headers`` fields. This registrar reads and writes
that key directly.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

logger = logging.getLogger(__name__)


def _opencode_config_path() -> Path:
    from headroom.providers.opencode.config import _opencode_config_path as _cfg
    return _cfg()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        import json as _json
        return _json.loads(text)
    except (ValueError, TypeError):
        pass
    try:
        import json as _json

        from headroom.providers.opencode.runtime import _strip_jsonc_comments

        return _json.loads(_strip_jsonc_comments(text))
    except (ValueError, TypeError, ImportError):
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    url = str(entry.get("url", ""))
    if name == "headroom":
        env: dict[str, str] = {}
        if url and url != "http://127.0.0.1:8787/mcp":
            env["HEADROOM_PROXY_URL"] = url
        return ServerSpec(
            name=name,
            command="headroom",
            args=("mcp", "serve"),
            env=env,
        )
    return ServerSpec(
        name=name,
        command="",
        args=(url,) if url else (),
        env={},
    )


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    url = spec.env.get(
        "HEADROOM_PROXY_URL",
        "http://127.0.0.1:8787/mcp",
    )
    return {
        "type": "remote",
        "url": url,
        "enabled": True,
    }


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
        self._config_path = config_path or _opencode_config_path()

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

    def register_server(
        self, spec: ServerSpec, *, force: bool = False
    ) -> RegisterResult:
        existing = self.get_server(spec.name)
        if existing is not None and _specs_equivalent(existing, spec):
            return RegisterResult(
                RegisterStatus.ALREADY, "matches current configuration"
            )
        if existing is not None and not force:
            return RegisterResult(
                RegisterStatus.MISMATCH, _diff_specs(existing, spec)
            )
        if existing is not None and force:
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
            data = _read_json(self._config_path)
            mcp = data.setdefault("mcp", {})
            if not isinstance(mcp, dict):
                mcp = {}
                data["mcp"] = mcp
            mcp[spec.name] = _spec_to_entry(spec)
            _write_json(self._config_path, data)
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED,
                f"could not write {self._config_path}: {exc}",
            )
        return RegisterResult(
            RegisterStatus.REGISTERED, f"wrote to {self._config_path}"
        )

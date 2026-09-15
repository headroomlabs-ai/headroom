"""DeepSeek Harness (dsh) MCP registrar."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

logger = logging.getLogger(__name__)

_MARKER_START = "# --- Headroom MCP server ---"
_MARKER_END = "# --- end Headroom MCP server ---"
_MCP_CLIENT_PLUGIN = "@deepseek-ai/dsh-mcp-client"


def _dsh_home() -> Path:
    """Return the dsh Harness home ($DSH_HOME, else ~/.dsh)."""
    return Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh")


def _entry_id(name: str) -> str:
    """Return the dsh patch ``insert`` id for a server name."""
    return f"mcp-{name}"


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    """Render a :class:`ServerSpec` as one dsh ``insert`` row."""
    config: dict[str, Any] = {
        "serverName": spec.name,
        "transport": "stdio",
        "command": spec.command,
        "args": spec.args,
    }
    if spec.env:
        config["env"] = spec.env
    return {"id": _entry_id(spec.name), "name": _MCP_CLIENT_PLUGIN, "config": config}


def _entry_to_spec(entry: dict[str, Any]) -> ServerSpec | None:
    """Parse one dsh ``insert`` row into a :class:`ServerSpec`; ``None`` if malformed."""
    config = entry.get("config")
    if not isinstance(config, dict):
        return None
    args = config.get("args")
    if args is not None and not isinstance(args, (list, tuple)):
        return None
    name = config.get("serverName", "")
    if not isinstance(name, str) or not name:
        return None
    env = config.get("env", {})
    if not isinstance(env, dict):
        env = {}
    return ServerSpec(
        name=name,
        command=config.get("command", ""),
        args=tuple(args or ()),
        env=env,
    )


def _render_block(entries: list[dict[str, Any]]) -> str:
    """Render the marker-fenced ``insert`` list holding all ``entries``."""
    body = yaml.safe_dump([{"insert": entries}], default_flow_style=False, sort_keys=False)
    return f"{_MARKER_START}\n{body}{_MARKER_END}\n"


def _block_to_entries(block: str) -> dict[str, ServerSpec] | None:
    """Parse a fenced block into ``{name: spec}``; ``None`` if unparsable."""
    body = block.replace(_MARKER_START, "").replace(_MARKER_END, "").strip()
    try:
        doc = yaml.safe_load(body)
    except yaml.YAMLError:
        return None
    if not isinstance(doc, list):
        return None
    entries: dict[str, ServerSpec] = {}
    for op in doc:
        if not isinstance(op, dict):
            continue
        for row in op.get("insert", []):
            if isinstance(row, dict) and "config" in row:
                spec = _entry_to_spec(row)
                if spec is None:
                    return None
                entries[spec.name] = spec
    return entries


@dataclass
class _ManagedBlock:
    """Read-back of the marker-fenced block in ``cordis.patch.yml``.

    ``entries`` set → block present and clean (may be empty); ``corrupt`` set →
    block present but broken; ``entries`` is ``{}`` and ``corrupt`` is ``None``
    when absent.
    """

    entries: dict[str, ServerSpec] | None
    corrupt: str | None


def _read_managed_block(config_file: Path) -> _ManagedBlock:
    """Read the managed block, distinguishing absent / clean / corrupt."""
    if not config_file.exists():
        return _ManagedBlock({}, None)
    text = config_file.read_text(encoding="utf-8")
    if _MARKER_START not in text:
        return _ManagedBlock({}, None)
    start = text.index(_MARKER_START)
    end_marker = text.find(_MARKER_END, start)
    if end_marker == -1:
        return _ManagedBlock(None, "unterminated start marker (missing end marker)")
    block = text[start : end_marker + len(_MARKER_END)]
    entries = _block_to_entries(block)
    if entries is None:
        return _ManagedBlock(None, "malformed YAML in managed block")
    return _ManagedBlock(entries, None)


def _remove_managed_block(text: str) -> str:
    """Return ``text`` with the marker-fenced block removed.

    Handles a truncated block (start marker with no end marker) by dropping
    everything from the start marker to end-of-file. Unrelated bytes before
    and after the block are preserved verbatim.
    """
    if _MARKER_START not in text:
        return text
    start = text.index(_MARKER_START)
    end_marker = text.find(_MARKER_END, start)
    if end_marker == -1:
        return text[:start]
    tail = text[end_marker + len(_MARKER_END) :]
    if tail.startswith("\n"):
        tail = tail[1:]
    return text[:start] + tail


def _atomic_write(config_file: Path, text: str) -> None:
    """Write ``text`` atomically via a temp file + ``os.replace``."""
    tmp = config_file.with_name(config_file.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, config_file)


def _specs_equivalent(a: ServerSpec, b: ServerSpec) -> bool:
    return (
        a.name == b.name
        and a.command == b.command
        and tuple(a.args) == tuple(b.args)
        and dict(a.env) == dict(b.env)
    )


def _diff_specs(existing: ServerSpec, requested: ServerSpec) -> str:
    return f"existing {existing.command} {existing.args}, requested {requested.command} {requested.args}"


class DshRegistrar(MCPRegistrar):
    """Register MCP servers with DeepSeek Harness (dsh)."""

    name = "dsh"
    display_name = "DeepSeek Harness"

    @property
    def _config_file(self) -> Path:
        return _dsh_home() / "cordis.patch.yml"

    def detect(self) -> bool:
        return _dsh_home().exists()

    def ensure_home(self) -> None:
        """Create the dsh config home if missing (explicit wrap path only).

        ``headroom wrap dsh`` targets a harness that is about to launch, so it
        must register before dsh creates its own home on first boot. Global
        ``headroom mcp install`` detection stays conservative: ``detect()``
        still requires an existing home.
        """
        _dsh_home().mkdir(parents=True, exist_ok=True)

    def get_server(self, server_name: str) -> ServerSpec | None:
        managed = _read_managed_block(self._config_file)
        if managed.entries is None:
            return None
        return managed.entries.get(server_name)

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        if not self.detect():
            return RegisterResult(
                RegisterStatus.NOT_DETECTED, f"dsh home not found at {_dsh_home()}"
            )
        managed = _read_managed_block(self._config_file)
        if managed.corrupt is not None:
            if not force:
                return RegisterResult(
                    RegisterStatus.FAILED,
                    f"corrupt managed block in {self._config_file}: {managed.corrupt}; "
                    "re-run with --force to overwrite",
                )
            managed = _ManagedBlock({}, None)
        entries = managed.entries or {}
        existing = entries.get(spec.name)
        if existing is not None:
            if _specs_equivalent(existing, spec):
                return RegisterResult(
                    RegisterStatus.ALREADY, f"already registered in {self._config_file}"
                )
            if not force:
                return RegisterResult(RegisterStatus.MISMATCH, _diff_specs(existing, spec))
        entries = dict(entries)
        entries[spec.name] = spec
        existing_text = (
            self._config_file.read_text(encoding="utf-8") if self._config_file.exists() else ""
        )
        base = _remove_managed_block(existing_text)
        if base and not base.endswith("\n"):
            base += "\n"
        new_text = base + _render_block([_spec_to_entry(s) for s in entries.values()])
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(self._config_file, new_text)
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"failed to write {self._config_file}: {exc}"
            )
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {self._config_file}")

    def unregister_server(self, server_name: str) -> bool:
        if not self._config_file.exists():
            return True
        text = self._config_file.read_text(encoding="utf-8")
        if _MARKER_START not in text:
            return True
        managed = _read_managed_block(self._config_file)
        if managed.corrupt is not None or managed.entries is None:
            _atomic_write(self._config_file, _remove_managed_block(text))
            return True
        if server_name not in managed.entries:
            return True
        entries = {k: v for k, v in managed.entries.items() if k != server_name}
        new_text = _remove_managed_block(text)
        if entries:
            if new_text and not new_text.endswith("\n"):
                new_text += "\n"
            new_text += _render_block([_spec_to_entry(s) for s in entries.values()])
        _atomic_write(self._config_file, new_text)
        return True

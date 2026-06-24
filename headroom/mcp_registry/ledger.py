"""Headroom-owned MCP install ledger.

The ledger tracks MCP servers that Headroom registered on the user's behalf
when the target agent config cannot carry Headroom-specific ownership markers.
It lets unwrap remove only entries still matching the spec Headroom installed,
preserving user-managed MCP servers with the same name.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from headroom import paths

from .base import ServerSpec

_LEDGER_FILE = "mcp_installs.json"


def ledger_path() -> Path:
    """Return the Headroom MCP install ledger path."""
    return paths.workspace_dir() / _LEDGER_FILE


def spec_fingerprint(spec: ServerSpec) -> str:
    """Stable fingerprint for a registered MCP server spec."""
    payload = {
        "name": spec.name,
        "command": spec.command,
        "args": list(spec.args),
        "env": dict(sorted(spec.env.items())),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def record_install(agent: str, spec: ServerSpec, *, path: Path | None = None) -> None:
    """Record that Headroom installed ``spec`` for ``agent``."""
    ledger_file = path or ledger_path()
    data = _read_ledger(ledger_file)
    agents = data.setdefault("agents", {})
    agent_entry = agents.setdefault(agent, {})
    agent_entry[spec.name] = {
        "fingerprint": spec_fingerprint(spec),
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_ledger(ledger_file, data)


def clear_install(agent: str, server_name: str, *, path: Path | None = None) -> None:
    """Remove one ledger entry if present."""
    ledger_file = path or ledger_path()
    data = _read_ledger(ledger_file)
    agents = data.get("agents")
    if not isinstance(agents, dict):
        return
    agent_entry = agents.get(agent)
    if not isinstance(agent_entry, dict) or server_name not in agent_entry:
        return
    del agent_entry[server_name]
    if not agent_entry:
        del agents[agent]
    if not agents:
        data.pop("agents", None)
    _write_ledger(ledger_file, data)


def headroom_installed_matching(
    agent: str,
    current_spec: ServerSpec | None,
    *,
    path: Path | None = None,
) -> bool:
    """Return True when the ledger says Headroom installed ``current_spec``."""
    if current_spec is None:
        return False
    ledger_file = path or ledger_path()
    data = _read_ledger(ledger_file)
    try:
        entry = data["agents"][agent][current_spec.name]
    except (KeyError, TypeError):
        return False
    if not isinstance(entry, dict):
        return False
    return entry.get("fingerprint") == spec_fingerprint(current_spec)


def _read_ledger(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_ledger(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

"""OpenCode config file helpers for wrap and persistent install.

Marker-based injection with backup/restore semantics.  Headroom
blocks are wrapped in ``// --- Headroom proxy provider ---`` /
``// --- end Headroom proxy provider ---`` comments so they can
be stripped safely during unwrap without touching user content.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .runtime import _strip_jsonc_comments


def _opencode_config_path() -> Path:
    env_path = os.environ.get("OPENCODE_CONFIG", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return _opencode_home_dir() / "opencode.json"

_PROVIDER_MARKER_START = "// --- Headroom proxy provider ---"
_PROVIDER_MARKER_END = "// --- end Headroom proxy provider ---"
_MCP_MARKER_START = "// --- Headroom MCP server ---"
_MCP_MARKER_END = "// --- end Headroom MCP server ---"

_PROVIDER_BLOCK_RE = re.compile(
    re.escape(_PROVIDER_MARKER_START)
    + r".*?"
    + re.escape(_PROVIDER_MARKER_END),
    re.DOTALL,
)
_MCP_BLOCK_RE = re.compile(
    re.escape(_MCP_MARKER_START)
    + r".*?"
    + re.escape(_MCP_MARKER_END),
    re.DOTALL,
)

HEADROOM_OPENCODE_PLUGIN = "headroom-opencode"


def _opencode_home_dir() -> Path:
    env_path = os.environ.get("OPENCODE_HOME", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".config" / "opencode"


def opencode_config_paths() -> tuple[Path, Path]:
    config_file = _opencode_config_path()
    backup_file = config_file.with_suffix(".json.headroom-backup")
    return config_file, backup_file


def snapshot_opencode_config_if_unwrapped(
    config_file: Path, backup_file: Path
) -> None:
    """Create a pre-wrap backup if no Headroom block is already present."""
    if backup_file.exists():
        return
    if not config_file.exists():
        return
    try:
        content = config_file.read_text(encoding="utf-8")
    except OSError:
        return
    if _PROVIDER_MARKER_START in content or _MCP_MARKER_START in content:
        return
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, backup_file)


def strip_opencode_headroom_blocks(
    content: str, *, remove_mcp: bool = True
) -> str:
    """Remove marker-wrapped Headroom blocks from *content*."""
    content = _PROVIDER_BLOCK_RE.sub("", content)
    if remove_mcp:
        content = _MCP_BLOCK_RE.sub("", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _parse_json_loose(text: str) -> dict[str, Any]:
    """Parse JSON/JSONC with comment stripping."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_strip_jsonc_comments(text))
    except (json.JSONDecodeError, ValueError):
        return {}


def _inject_key_into_json(
    data: dict[str, Any], key: str, value: Any
) -> dict[str, Any]:
    existing = data.get(key)
    if isinstance(existing, dict) and isinstance(value, dict):
        merged = {**existing, **value}
        data[key] = merged
    else:
        data[key] = value
    return data


def append_headroom_plugin(config: dict[str, object]) -> bool:
    plugin = config.get("plugin")
    if plugin is None:
        config["plugin"] = [HEADROOM_OPENCODE_PLUGIN]
        return True
    if not isinstance(plugin, list):
        return False
    for entry in plugin:
        if entry == HEADROOM_OPENCODE_PLUGIN:
            return False
        if isinstance(entry, list) and entry and entry[0] == HEADROOM_OPENCODE_PLUGIN:
            return False
    plugin.append(HEADROOM_OPENCODE_PLUGIN)
    return True

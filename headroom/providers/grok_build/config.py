"""Grok Build config.toml helpers for wrap and persistent install."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from headroom import fsutil

from .runtime import build_proxy_targets

_MARKER_START = "# --- headroom:grok-build:start ---"
_MARKER_END = "# --- headroom:grok-build:end ---"
_BLOCK_RE = re.compile(
    re.escape(_MARKER_START) + r".*?" + re.escape(_MARKER_END) + r"\n?",
    re.DOTALL,
)


def grok_home_dir() -> Path:
    """Return the Grok home/config directory."""
    env_path = os.environ.get("GROK_HOME", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".grok"


def grok_config_paths() -> tuple[Path, Path]:
    """Return ``(config_file, backup_file)`` for Grok Build."""
    config_file = grok_home_dir() / "config.toml"
    backup_file = config_file.with_suffix(".toml.headroom-backup")
    return config_file, backup_file


def snapshot_grok_config_if_unwrapped(config_file: Path, backup_file: Path) -> None:
    """Snapshot ``config.toml`` before the first Headroom injection."""
    if backup_file.exists():
        return
    if not config_file.exists():
        return
    try:
        content = fsutil.read_text(config_file)
    except OSError:
        return
    if _MARKER_START in content:
        return
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, backup_file)


def strip_grok_headroom_blocks(content: str) -> str:
    """Remove Headroom-managed Grok config blocks."""
    content = _BLOCK_RE.sub("", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def render_headroom_block(port: int, project: str | None = None) -> str:
    """Render the Headroom-managed ``[model.grok-build]`` override block."""
    target = build_proxy_targets(port, project)
    return (
        f"{_MARKER_START}\n"
        "[model.grok-build]\n"
        f'base_url = "{target.base_url}"\n'
        f"{_MARKER_END}\n"
    )


def inject_grok_provider_config(port: int, project: str | None = None) -> Path:
    """Inject or refresh the Headroom proxy override into Grok config."""
    config_file, backup_file = grok_config_paths()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    snapshot_grok_config_if_unwrapped(config_file, backup_file)

    if config_file.exists():
        content = strip_grok_headroom_blocks(fsutil.read_text(config_file))
    else:
        content = ""

    block = render_headroom_block(port, project)
    if content:
        content = content.rstrip() + "\n\n" + block
    else:
        content = block

    fsutil.write_text(config_file, content)
    return config_file


def restore_grok_provider_config() -> tuple[str, Path]:
    """Undo ``inject_grok_provider_config`` for the active Grok config file."""
    config_file, backup_file = grok_config_paths()
    if backup_file.exists():
        shutil.copy2(backup_file, config_file)
        backup_file.unlink()
        return "restored", config_file

    if not config_file.exists():
        return "noop", config_file

    content = strip_grok_headroom_blocks(fsutil.read_text(config_file))
    if content:
        fsutil.write_text(config_file, content)
        return "cleaned", config_file

    config_file.unlink(missing_ok=True)
    return "removed", config_file
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
_GROK_BUILD_TABLE_RE = re.compile(r"(?m)^\[model\.grok-build\]\s*$")
_NEXT_TABLE_RE = re.compile(r"(?m)^\[")
_BASE_URL_LINE_RE = re.compile(
    r'(?m)^(?P<indent>[ \t]*)base_url[ \t]*=[ \t]*"(?P<value>[^"\n]*)".*$'
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


def has_user_grok_build_model_table(content: str) -> bool:
    """Return True when ``content`` already declares ``[model.grok-build]``."""
    return _GROK_BUILD_TABLE_RE.search(content) is not None


def redirect_existing_grok_build_base_url(content: str, base_url: str) -> tuple[str, bool]:
    """Rewrite ``base_url`` inside an existing ``[model.grok-build]`` table.

    TOML rejects duplicate table headers, so when the user already owns
    ``[model.grok-build]`` we update that table in place instead of appending
    a second one. The previous ``base_url`` value is preserved in a trailing
    ``# was: …`` comment for visibility; the pre-wrap snapshot still enables
    byte-for-byte restore on ``headroom unwrap grok-build``.
    """
    match = _GROK_BUILD_TABLE_RE.search(content)
    if match is None:
        return content, False

    section_start = match.end()
    next_table = _NEXT_TABLE_RE.search(content, section_start)
    section_end = next_table.start() if next_table else len(content)
    section = content[section_start:section_end]

    if _BASE_URL_LINE_RE.search(section):

        def _replace(match_obj: re.Match[str]) -> str:
            original_value = match_obj.group("value")
            if original_value == base_url:
                return match_obj.group(0)
            indent = match_obj.group("indent")
            return f'{indent}base_url = "{base_url}"  # was: {original_value}'

        section = _BASE_URL_LINE_RE.sub(_replace, section, count=1)
    else:
        section = f'\nbase_url = "{base_url}"' + section

    updated = content[:section_start] + section + content[section_end:]
    return updated, updated != content


def render_headroom_block(port: int, project: str | None = None) -> str:
    """Render the Headroom-managed ``[model.grok-build]`` override block."""
    target = build_proxy_targets(port, project)
    return f'{_MARKER_START}\n[model.grok-build]\nbase_url = "{target.base_url}"\n{_MARKER_END}\n'


def inject_grok_provider_config(port: int, project: str | None = None) -> Path:
    """Inject or refresh the Headroom proxy override into Grok config."""
    config_file, backup_file = grok_config_paths()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    snapshot_grok_config_if_unwrapped(config_file, backup_file)

    if config_file.exists():
        content = strip_grok_headroom_blocks(fsutil.read_text(config_file))
    else:
        content = ""

    target = build_proxy_targets(port, project)
    if has_user_grok_build_model_table(content):
        content, _ = redirect_existing_grok_build_base_url(content, target.base_url)
    else:
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

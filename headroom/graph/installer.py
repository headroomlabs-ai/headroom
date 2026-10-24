"""Locate the codebase-memory-mcp binary used by the code-graph watcher.

The binary is looked up on PATH, then in ``~/.local/bin``.
"""

from __future__ import annotations

import platform
import shutil
from pathlib import Path

CBM_BIN_DIR = Path.home() / ".local" / "bin"
CBM_BIN_NAME = "codebase-memory-mcp"


def _installed_name(plat: str) -> str:
    """Return the binary's file name on ``plat``.

    On Windows, ``shutil.which`` (and so a PATH lookup) only finds a binary
    whose name ends in a PATHEXT extension such as ``.exe``.
    """
    return f"{CBM_BIN_NAME}.exe" if plat.startswith("windows") else CBM_BIN_NAME


def get_cbm_path() -> Path | None:
    """Find codebase-memory-mcp binary, return path or None."""
    # Check PATH first
    found = shutil.which(CBM_BIN_NAME)
    if found:
        return Path(found)

    # Check our install location
    installed = CBM_BIN_DIR / _installed_name(platform.system().lower())
    if installed.exists() and installed.is_file():
        return installed

    return None

"""rtk (Rust Token Killer) integration for Headroom.

rtk compresses CLI output (test results, git diffs, log dumps) before it
enters the LLM context window. Headroom downloads and manages the rtk binary.
"""

from __future__ import annotations

import platform
import shutil
from pathlib import Path

from headroom import paths as _paths

RTK_VERSION = "v0.28.2"
RTK_BIN_DIR = _paths.bin_dir()
_RTK_NAME = "rtk.exe" if platform.system() == "Windows" else "rtk"
RTK_BIN_PATH = RTK_BIN_DIR / _RTK_NAME


def _managed_rtk_candidates() -> list[Path]:
    """Return known Headroom-managed rtk binary paths."""
    candidates = [RTK_BIN_DIR / _RTK_NAME]
    for name in ("rtk", "rtk.exe"):
        path = RTK_BIN_DIR / name
        if path not in candidates:
            candidates.append(path)
    return candidates


def get_rtk_path() -> Path | None:
    """Get path to rtk binary — check PATH first, then ~/.headroom/bin/."""
    # Check if rtk is already in PATH (e.g., installed via brew)
    system_rtk = shutil.which("rtk")
    if system_rtk:
        return Path(system_rtk)

    # Check Headroom-managed install
    for candidate in _managed_rtk_candidates():
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def is_rtk_installed() -> bool:
    """Check if rtk is available."""
    return get_rtk_path() is not None

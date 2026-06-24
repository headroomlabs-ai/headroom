"""lean-ctx integration for Headroom.

lean-ctx configures supported coding agents to route tool output through its
context-filtering layer. Headroom downloads and manages the lean-ctx binary.
"""

from __future__ import annotations

import platform
import shutil
from pathlib import Path

from headroom import paths as _paths

LEAN_CTX_VERSION = "v3.4.7"
LEAN_CTX_BIN_DIR = _paths.bin_dir()
_LEAN_CTX_NAME = "lean-ctx.exe" if platform.system() == "Windows" else "lean-ctx"
LEAN_CTX_BIN_PATH = _paths.lean_ctx_path()


def _managed_lean_ctx_candidates() -> list[Path]:
    """Return known Headroom-managed lean-ctx binary paths."""
    candidates = [LEAN_CTX_BIN_DIR / _LEAN_CTX_NAME]
    for name in ("lean-ctx", "lean-ctx.exe"):
        path = LEAN_CTX_BIN_DIR / name
        if path not in candidates:
            candidates.append(path)
    return candidates


def get_lean_ctx_path() -> Path | None:
    """Get path to lean-ctx binary — check PATH first, then ~/.headroom/bin/."""
    system_lean_ctx = shutil.which("lean-ctx")
    if system_lean_ctx:
        return Path(system_lean_ctx)

    for candidate in _managed_lean_ctx_candidates():
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def is_lean_ctx_installed() -> bool:
    """Check if lean-ctx is available."""
    return get_lean_ctx_path() is not None

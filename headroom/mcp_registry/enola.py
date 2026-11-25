"""Resolve the pinned Enola installation without downloading at launch time."""

from __future__ import annotations

import os
import shutil
import sysconfig
from pathlib import Path

from .base import ServerSpec


def find_enola_binary() -> str:
    """Prefer an explicit override, then this Python environment, then PATH."""
    hint = "Install headroom-ai[enola] in the environment that runs Headroom."
    override = os.environ.get("ENOLA_PATH")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ValueError(f"ENOLA_PATH is not an executable file: {override}. {hint}")
        return str(candidate.resolve())

    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidate = Path(scripts) / ("enola.exe" if os.name == "nt" else "enola")
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    binary = shutil.which("enola")
    if binary:
        return str(Path(binary).resolve())
    raise ValueError(f"Enola executable not found. {hint}")


def build_enola_spec() -> ServerSpec:
    """Launch Enola's stdio MCP server; generation is requested over MCP."""
    return ServerSpec(
        name="enola",
        command=find_enola_binary(),
        env={"ENOLA_NO_UPDATE_CHECK": "1", "ENOLA_NO_PROMPTS": "1"},
    )

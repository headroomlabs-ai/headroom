"""Grok CLI MCP registrar.

Grok stores MCP server config in ``$GROK_HOME/config.toml`` (default
``~/.grok/config.toml``) as ``[mcp_servers.<name>]`` tables. There is no
general-purpose CLI for adding entries, so we edit the file in place using
marker-delimited blocks so we can idempotently inject, replace, and remove
our entry without disturbing anything else the user has configured. The
file format matches Codex's, so the marker-block logic lives in
:mod:`headroom.mcp_registry.codex`.
"""

from __future__ import annotations

import os
from pathlib import Path

from .codex import _TomlMarkerRegistrar


class GrokRegistrar(_TomlMarkerRegistrar):
    """Register MCP servers with the Grok CLI."""

    name = "grok"
    display_name = "Grok CLI"

    def __init__(self, *, home_dir: Path | None = None) -> None:
        if home_dir is not None:
            config_dir = home_dir / ".grok"
        elif os.environ.get("GROK_HOME"):
            config_dir = Path(os.environ["GROK_HOME"]).expanduser()
        else:
            config_dir = Path.home() / ".grok"
        super().__init__(config_dir)

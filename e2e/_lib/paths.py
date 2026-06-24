"""Per-agent settings-file locators for e2e assertions.

These paths mirror the logic in ``headroom.cli.init`` so e2e tests can
verify that the right file was written without importing private init
helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

Agent = Literal["claude", "codex", "copilot", "openclaw"]
Scope = Literal["user", "local"]


def agent_settings_path(agent: Agent, *, scope: Scope, home: Path, project: Path) -> Path:
    """Return the file that ``headroom init`` should have written for ``agent``.

    ``home`` is the test's simulated HOME directory and ``project`` is the cwd
    used when invoking ``headroom init``. For global (``-g``) invocations only
    ``home`` matters; for local invocations only ``project`` matters.
    """

    home = Path(home)
    project = Path(project)

    if agent == "claude":
        if scope == "user":
            return home / ".claude" / "settings.json"
        return project / ".claude" / "settings.local.json"

    if agent == "codex":
        if scope == "user":
            return home / ".codex" / "config.toml"
        return project / ".codex" / "config.toml"

    if agent == "copilot":
        # Copilot init requires -g; no local scope.
        return home / ".copilot" / "config.json"

    if agent == "openclaw":
        # OpenClaw init is delegated to `headroom wrap openclaw`; it writes
        # the openclaw json under $HOME.
        return home / ".openclaw" / "openclaw.json"

    raise ValueError(f"Unknown agent: {agent!r}")

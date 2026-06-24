"""Shared assertion helpers for e2e cases.

Assertions raise ``AssertionError`` with a descriptive message. The harness
catches them and attributes the failure to the owning ``Case``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import Agent, Scope, agent_settings_path


def assert_exit(actual: int, expected: int, *, context: str = "") -> None:
    if actual != expected:
        suffix = f" ({context})" if context else ""
        raise AssertionError(f"Expected exit code {expected}, got {actual}{suffix}")


def assert_stdout_contains(stdout: str, needle: str) -> None:
    if needle not in stdout:
        raise AssertionError(f"stdout missing {needle!r}:\n---\n{stdout}\n---")


def assert_stderr_contains(stderr: str, needle: str) -> None:
    if needle not in stderr:
        raise AssertionError(f"stderr missing {needle!r}:\n---\n{stderr}\n---")


def read_agent_settings(
    agent: Agent, *, scope: Scope, home: Path, project: Path
) -> dict[str, Any] | str:
    """Read an agent's settings file, returning dict for JSON and str for TOML/other."""

    path = agent_settings_path(agent, scope=scope, home=home, project=project)
    if not path.exists():
        raise AssertionError(f"Expected settings file at {path}, not found")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    return text

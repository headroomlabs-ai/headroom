"""Shared helpers for Docker / CI e2e tests.

This package centralizes utilities used by the per-command e2e harnesses
(`e2e/init/run.py`, future `e2e/install/run.py`, `e2e/wrap/run.py`, ...).
The goal is that each command test suite is a small declarative file that
imports from this package, so new commands can be covered with minimal
duplication.
"""

from __future__ import annotations

from .assertions import (
    assert_exit,
    assert_stderr_contains,
    assert_stdout_contains,
    read_agent_settings,
)
from .harness import Case, CaseContext, run_case_sequence, run_cases
from .path_env import with_clean_path
from .paths import agent_settings_path
from .shims import make_shim

__all__ = [
    "Case",
    "CaseContext",
    "agent_settings_path",
    "assert_exit",
    "assert_stderr_contains",
    "assert_stdout_contains",
    "make_shim",
    "read_agent_settings",
    "run_case_sequence",
    "run_cases",
    "with_clean_path",
]

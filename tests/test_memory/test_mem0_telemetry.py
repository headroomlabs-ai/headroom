"""Mem0's PostHog telemetry is off unless the caller turns it on."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from headroom.memory.backends.mem0 import Mem0Backend, Mem0Config


class _FakeMem0(types.ModuleType):
    """Stand-in ``mem0`` that records MEM0_TELEMETRY when ``Memory`` is imported.

    Real Mem0 reads the variable at that import, so the value seen here is the
    one it would use.
    """

    def __init__(self) -> None:
        super().__init__("mem0")
        self.telemetry_at_import: str | None = None

    def __getattr__(self, name: str) -> Any:
        if name != "Memory":
            raise AttributeError(name)
        self.telemetry_at_import = os.environ.get("MEM0_TELEMETRY")
        return MagicMock()


@pytest.mark.asyncio
async def test_backend_turns_mem0_telemetry_off_before_importing_mem0(monkeypatch) -> None:
    monkeypatch.delenv("MEM0_TELEMETRY", raising=False)
    fake = _FakeMem0()
    monkeypatch.setitem(sys.modules, "mem0", fake)

    await Mem0Backend(Mem0Config(enable_graph=False))._ensure_client()

    assert fake.telemetry_at_import == "False"


@pytest.mark.asyncio
async def test_caller_choice_of_mem0_telemetry_wins(monkeypatch) -> None:
    monkeypatch.setenv("MEM0_TELEMETRY", "True")
    fake = _FakeMem0()
    monkeypatch.setitem(sys.modules, "mem0", fake)

    await Mem0Backend(Mem0Config(enable_graph=False))._ensure_client()

    assert fake.telemetry_at_import == "True"


def test_real_mem0_reads_the_default_as_disabled() -> None:
    pytest.importorskip("mem0")
    env = {k: v for k, v in os.environ.items() if k != "MEM0_TELEMETRY"}
    script = (
        "from headroom.memory.backends.mem0 import disable_mem0_telemetry_by_default\n"
        "disable_mem0_telemetry_by_default()\n"
        "import mem0.memory.telemetry as t\n"
        "print(t.MEM0_TELEMETRY)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False"

"""Mem0SystemAdapter is deprecated in favour of DirectMem0Adapter."""

from __future__ import annotations

import pytest

from headroom.memory.backends import mem0_system_adapter


def test_mem0_system_adapter_warns_deprecated(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mem0Backend's constructor needs no live services, but stub it so the test
    # does not depend on the optional mem0 install.
    monkeypatch.setattr(mem0_system_adapter, "Mem0Backend", lambda config: object())
    with pytest.warns(DeprecationWarning, match="Mem0SystemAdapter is deprecated"):
        mem0_system_adapter.Mem0SystemAdapter()

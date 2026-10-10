"""Deprecated memory adapter methods still work and warn."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.memory.adapters.sqlite import SQLiteMemoryStore


def test_count_sync_warns_deprecated(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / "mem.db"))
    with pytest.warns(DeprecationWarning, match="SQLiteMemoryStore.count_sync is deprecated"):
        assert store.count_sync() == 0

"""Deprecated memory adapter methods still work and warn."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.memory.adapters.sqlite import SQLiteMemoryStore


def test_count_sync_warns_deprecated(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / "mem.db"))
    with pytest.warns(DeprecationWarning, match="SQLiteMemoryStore.count_sync is deprecated"):
        assert store.count_sync() == 0


def test_set_ef_search_warns_deprecated() -> None:
    from headroom.memory.adapters.hnsw import HNSWVectorIndex, _check_hnswlib_available

    if not _check_hnswlib_available():
        pytest.skip("hnswlib not installed")
    index = HNSWVectorIndex(dimension=8)
    with pytest.warns(DeprecationWarning, match="HNSWVectorIndex.set_ef_search is deprecated"):
        index.set_ef_search(64)
    assert index._ef_search == 64

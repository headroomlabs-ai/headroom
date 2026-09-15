"""The factory must reload the persisted HNSW index instead of starting empty.

``_create_vector_index`` builds ``HNSWVectorIndex`` with ``auto_save=True`` and
a ``save_path`` derived from the db path, so every mutation persists the index
next to the db. But nothing ever loaded those artifacts back: each fresh
process started with an EMPTY index, so

* vector search returned nothing until every row was re-indexed, and
* the first write from the fresh process overwrote the persisted artifacts
  with the near-empty in-memory index, discarding it for every later process.

These tests fail on that behavior and pass once the factory reloads the
persisted index when it exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from headroom.memory.config import MemoryConfig, VectorBackend
from headroom.memory.factory import _create_vector_index
from headroom.memory.models import Memory

try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not installed")


def _config(db_path: Path) -> MemoryConfig:
    return MemoryConfig(
        db_path=db_path,
        vector_backend=VectorBackend.HNSW,
        vector_dimension=8,
    )


def _mem(i: int) -> Memory:
    rng = np.random.default_rng(i)
    return Memory(
        content=f"persisted memory {i}",
        user_id="user",
        embedding=rng.standard_normal(8).astype(np.float32),
    )


@pytest.mark.asyncio
async def test_fresh_index_reloads_persisted_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    first = _create_vector_index(_config(db_path))
    await first.index(_mem(1))
    await first.index(_mem(2))

    # auto_save must have persisted the artifacts next to the db.
    assert (tmp_path / "memory_hnsw.hnsw").exists()
    assert (tmp_path / "memory_hnsw.meta").exists()

    # A fresh process must see the persisted entries immediately.
    second = _create_vector_index(_config(db_path))
    assert second.size == 2, (
        "fresh HNSWVectorIndex started empty: the factory did not reload the persisted index"
    )


@pytest.mark.asyncio
async def test_fresh_save_does_not_clobber_persisted_index(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    first = _create_vector_index(_config(db_path))
    await first.index(_mem(1))
    await first.index(_mem(2))
    assert first.size == 2

    # A fresh process adds one entry; with load-on-init the persisted index
    # keeps all previous entries instead of being rewritten from near-empty.
    second = _create_vector_index(_config(db_path))
    await second.index(_mem(3))

    third = _create_vector_index(_config(db_path))
    assert third.size == 3, "persisted HNSW index was clobbered by a write from a fresh process"

"""HNSW index_batch must resize based on the assigned-id high-water mark, not
the live entry count, so batch adds after eviction/deletion churn don't overflow
hnswlib's max_elements."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from headroom.memory.models import Memory

try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False


@pytest.fixture
def temp_hnsw_path():
    with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
        yield Path(f.name)


def _mem(i: int, dim: int = 8) -> Memory:
    rng = np.random.default_rng(i)
    return Memory(
        content=f"m{i}",
        user_id="u",
        embedding=rng.standard_normal(dim).astype(np.float32),
    )


@pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not installed")
@pytest.mark.asyncio
async def test_index_batch_after_deletion_churn_does_not_overflow(temp_hnsw_path):
    from headroom.memory.adapters.hnsw import HNSWVectorIndex

    # Small ceiling so we hit it quickly. mark_deleted (remove) never frees a
    # slot, so the assigned-id counter climbs toward max_elements while the live
    # count stays low.
    index = HNSWVectorIndex(dimension=8, max_elements=8, save_path=temp_hnsw_path)

    singles = [_mem(i) for i in range(6)]
    for m in singles:
        await index.index(m)  # assigned ids 0..5; next id high-water = 6

    # Delete 5 of them (mark_deleted; the 5 hnswlib slots are NOT reclaimed).
    for m in singles[:5]:
        await index.remove(m.id)

    # A batch of 3 now needs slots 6,7,8 -> hnswlib must hold 9 labels. The old
    # check used the live count (1) + 3 = 4 <= 8 and skipped the resize, so
    # add_items raised "number of elements exceeds the specified limit".
    added = await index.index_batch([_mem(100), _mem(101), _mem(102)])

    assert added == 3

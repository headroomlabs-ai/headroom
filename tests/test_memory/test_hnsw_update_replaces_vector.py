"""Re-indexing a memory on the HNSW backend must move it to its new vector.

`HierarchicalMemory.update`, `refresh` and `detach` re-index a memory under the
same id after its content (and so its embedding) changed. The HNSW adapter only
cached the new embedding; the graph kept the old vector, so search still found
the memory by its OLD content and not by the new one.
See https://github.com/headroomlabs-ai/headroom/issues/4362 (item 5).
"""

from __future__ import annotations

import numpy as np
import pytest

from headroom.memory.models import Memory
from headroom.memory.ports import VectorFilter

try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not installed"),
    pytest.mark.asyncio,
]

DIM = 8
OLD = np.eye(DIM, dtype=np.float32)[0]
NEW = np.eye(DIM, dtype=np.float32)[1]


def _index():
    from headroom.memory.adapters.hnsw import HNSWVectorIndex

    return HNSWVectorIndex(dimension=DIM, max_elements=16)


async def _top(index, vector: np.ndarray) -> tuple[str, float]:
    results = await index.search(VectorFilter(query_vector=vector, top_k=1))
    return results[0].memory.id, results[0].similarity


async def _seed(index) -> Memory:
    memory = Memory(id="m1", content="old", user_id="u", embedding=OLD.copy())
    await index.index(memory)
    # A second memory so a stale vector cannot win by being the only entry.
    await index.index(
        Memory(id="m2", content="other", user_id="u", embedding=np.ones(DIM, np.float32))
    )
    return memory


async def test_index_on_existing_id_replaces_the_vector() -> None:
    index = _index()
    memory = await _seed(index)

    memory.embedding = NEW.copy()
    await index.index(memory)

    assert index.size == 2
    found, similarity = await _top(index, NEW)
    assert found == "m1"
    assert similarity == pytest.approx(1.0, abs=1e-5)


async def test_update_embedding_replaces_the_vector() -> None:
    index = _index()
    await _seed(index)

    assert await index.update_embedding("m1", NEW.copy()) is True

    found, similarity = await _top(index, NEW)
    assert found == "m1"
    assert similarity == pytest.approx(1.0, abs=1e-5)


async def test_index_batch_on_existing_id_replaces_the_vector() -> None:
    index = _index()
    memory = await _seed(index)

    memory.embedding = NEW.copy()
    assert await index.index_batch([memory]) == 1

    assert index.size == 2
    found, similarity = await _top(index, NEW)
    assert found == "m1"
    assert similarity == pytest.approx(1.0, abs=1e-5)


async def test_index_batch_repeated_id_ends_on_the_last_vector() -> None:
    index = _index()
    await _seed(index)

    first = Memory(
        id="m1", content="first", user_id="u", embedding=np.eye(DIM, dtype=np.float32)[2]
    )
    last = Memory(id="m1", content="last", user_id="u", embedding=NEW.copy())
    await index.index_batch([first, last])

    assert index.size == 2
    found, similarity = await _top(index, NEW)
    assert found == "m1"
    assert similarity == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(index._embeddings["m1"], NEW)

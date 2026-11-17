"""Regression tests for primary-store and vector-metadata consistency."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from headroom.memory.core import HierarchicalMemory
from headroom.memory.models import Memory


def _memory_system(memory: Memory, *, cache=None):
    store = AsyncMock()
    store.get.return_value = memory
    vector_index = AsyncMock()
    text_index = AsyncMock()
    embedder = AsyncMock()
    system = HierarchicalMemory(store, vector_index, text_index, embedder, cache=cache)
    return system, store, vector_index, text_index


@pytest.mark.asyncio
async def test_metadata_only_update_refreshes_vector_metadata() -> None:
    memory = Memory(
        id="memory-1",
        content="stable content",
        user_id="default",
        embedding=np.array([0.1, 0.2], dtype=np.float32),
        metadata={"evidence_count": 5},
    )
    system, store, vector_index, text_index = _memory_system(memory)

    updated = await system.update(memory.id, metadata={"evidence_count": 7})

    assert updated is memory
    store.save.assert_awaited_once_with(memory)
    vector_index.index.assert_awaited_once_with(memory)
    text_index.index.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_memory_indexes_reloads_store_and_cache() -> None:
    memory = Memory(
        id="memory-1",
        content="stable content",
        user_id="default",
        embedding=np.array([0.1, 0.2], dtype=np.float32),
        metadata={"evidence_count": 7},
    )
    cache = AsyncMock()
    system, store, vector_index, _ = _memory_system(memory, cache=cache)

    refreshed = await system.refresh_memory_indexes(memory.id)

    assert refreshed is memory
    store.get.assert_awaited_once_with(memory.id)
    vector_index.index.assert_awaited_once_with(memory)
    cache.invalidate.assert_awaited_once_with(memory.id)
    cache.put.assert_awaited_once_with(memory)

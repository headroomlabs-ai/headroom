"""Reject incompatible embeddings before changing primary or cached memories."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from headroom.memory.adapters.cache import LRUMemoryCache
from headroom.memory.adapters.hnsw import HNSWVectorIndex
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.config import MemoryConfig
from headroom.memory.core import HierarchicalMemory
from headroom.memory.models import Memory
from headroom.memory.ports import MemoryFilter


@pytest.fixture
def components(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memories.db")
    vector = HNSWVectorIndex(dimension=4)
    text = AsyncMock()
    embedder = AsyncMock()
    cache = LRUMemoryCache()
    memory = HierarchicalMemory(
        store, vector, text, embedder, cache, MemoryConfig(auto_bubble=False)
    )
    return memory, store, vector, text, embedder, cache


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", [(8,), (2, 2), ()])
async def test_add_rejects_incompatible_shape_without_persisting(components, shape):
    memory, store, vector, text, embedder, cache = components
    embedder.embed.return_value = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError):
        await memory.add("Must not be half-written", user_id="alice")
    assert await store.query(MemoryFilter(user_id="alice")) == []
    assert vector.size == 0
    text.index_memory.assert_not_called()
    assert cache.size == 0


@pytest.mark.asyncio
async def test_batch_validates_every_embedding_before_saving_any_row(components):
    memory, store, vector, _, embedder, _ = components
    embedder.embed_batch.return_value = [np.ones(4), np.ones(8)]
    with pytest.raises(ValueError):
        await memory.add_batch(
            [
                {"content": "Valid first vector", "user_id": "alice"},
                {"content": "Invalid second vector", "user_id": "alice"},
            ]
        )
    assert await store.query(MemoryFilter(user_id="alice")) == []
    assert vector.size == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["update", "supersede"])
async def test_rejected_replacement_keeps_the_existing_memory_and_lineage(components, operation):
    memory, store, vector, _, embedder, cache = components
    embedder.embed.return_value = np.ones(4)
    original = await memory.add("Original fact", user_id="alice")
    embedder.embed.return_value = np.ones(8)
    with pytest.raises(ValueError):
        if operation == "update":
            await memory.update(
                original.id, content="Invalid replacement", metadata={"changed": True}
            )
        else:
            await memory.supersede(original.id, "Invalid replacement")
    stored = await store.get(original.id)
    assert stored.content == "Original fact"
    assert stored.metadata == {}
    assert stored.valid_until is None
    assert stored.superseded_by is None
    assert len(await store.query(MemoryFilter(user_id="alice", include_superseded=True))) == 1
    assert (await cache.get(original.id)).content == "Original fact"
    assert vector.size == 1


@pytest.mark.asyncio
async def test_failed_update_does_not_mutate_a_reference_returning_store(components):
    memory, _, _, _, embedder, _ = components
    original = Memory(content="Original", user_id="alice", embedding=np.ones(4))
    memory._store = AsyncMock()
    memory._store.get.return_value = original
    embedder.embed.return_value = np.ones(8)
    with pytest.raises(ValueError):
        await memory.update(original.id, content="Replacement", metadata={"changed": True})
    assert original.content == "Original"
    assert original.metadata == {}
    memory._store.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_matching_embeddings_and_explicit_no_embedding_still_work(components):
    memory, store, vector, _, embedder, _ = components
    embedder.embed.return_value = np.ones(4)
    with_vector = await memory.add("Indexed fact", user_id="alice")
    without_vector = await memory.add("Text-only fact", user_id="alice", auto_embed=False)
    assert (await store.get(with_vector.id)).embedding.shape == (4,)
    assert (await store.get(without_vector.id)).embedding is None
    assert vector.size == 1

"""Isolated HNSW tests - copy of relevant parts from test_hierarchical.py."""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tempfile
from pathlib import Path

import numpy as np
import pytest

from headroom.memory.models import Memory
from headroom.memory.ports import VectorFilter

# Check if hnswlib is available (use lazy check to avoid SIGILL on incompatible CPUs)
try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not installed")
class TestHNSWVectorIndex:
    """Tests for HNSWVectorIndex."""

    @pytest.fixture
    def vector_index(self, temp_db_path):
        """Create an HNSW vector index for testing."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        return HNSWVectorIndex(dimension=384, save_path=temp_db_path.with_suffix(".hnsw"))

    @pytest.mark.asyncio
    async def test_index_and_search(self, vector_index):
        """Test indexing and searching vectors."""
        print("\n[TEST] Starting test_index_and_search")

        # Create memories with random embeddings
        np.random.seed(42)
        memories = []
        for i in range(10):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Test content {i}",
                user_id="alice",
                embedding=embedding,
            )
            memories.append(memory)
        print(f"[TEST] Created {len(memories)} memories")

        # Index all memories
        print("[TEST] Indexing...")
        for memory in memories:
            await vector_index.index(memory)
        print("[TEST] All indexed!")

        # Search with first memory's embedding
        filter = VectorFilter(
            query_vector=memories[0].embedding,
            top_k=3,
            user_id="alice",
        )
        print("[TEST] Searching...")
        results = await vector_index.search(filter)
        print(f"[TEST] Found {len(results)} results")

        assert len(results) == 3
        assert results[0].memory.id == memories[0].id
        assert results[0].similarity > 0.99
        print("[TEST] PASSED!")

    @pytest.mark.asyncio
    async def test_bounded_index_eviction(self, temp_db_path):
        """Test that bounded index evicts low-importance entries."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        # Create bounded index with max 5 entries
        index = HNSWVectorIndex(
            dimension=384,
            max_entries=5,
            eviction_batch_size=2,
        )

        np.random.seed(42)

        # Add 5 memories with varying importance
        memories = []
        for i in range(5):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=embedding,
                importance=0.1 * (i + 1),  # 0.1, 0.2, 0.3, 0.4, 0.5
            )
            await index.index(memory)
            memories.append(memory)

        assert index.size == 5

        # Add one more - should trigger eviction of lowest importance
        new_embedding = np.random.randn(384).astype(np.float32)
        new_memory = Memory(
            content="New high importance",
            user_id="alice",
            embedding=new_embedding,
            importance=0.9,
        )
        await index.index(new_memory)

        # Should have evicted 2 entries (eviction_batch_size) then added 1
        # So size should be 5 - 2 + 1 = 4
        assert index.size == 4

        # The lowest importance entries (0.1, 0.2) should be gone
        stats = index.get_memory_stats()
        assert stats.evictions == 2

        # Search should not find the evicted memories
        filter = VectorFilter(
            query_vector=memories[0].embedding,  # Lowest importance, should be evicted
            top_k=10,
            user_id="alice",
        )
        results = await index.search(filter)

        # memories[0] and memories[1] should be evicted
        result_ids = {r.memory.id for r in results}
        assert memories[0].id not in result_ids
        assert memories[1].id not in result_ids

    @pytest.mark.asyncio
    async def test_bounded_index_stats(self, temp_db_path):
        """Test that bounded index reports correct stats."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        index = HNSWVectorIndex(
            dimension=384,
            max_entries=100,
        )

        stats = index.get_memory_stats()
        assert stats.name == "vector_index"
        assert stats.entry_count == 0
        assert stats.budget_bytes is not None  # Should have budget when max_entries set
        assert stats.evictions == 0

        # Add some entries
        np.random.seed(42)
        for i in range(10):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=embedding,
            )
            await index.index(memory)

        stats = index.get_memory_stats()
        assert stats.entry_count == 10
        assert stats.size_bytes > 0

    @pytest.mark.asyncio
    async def test_unbounded_index_no_eviction(self, temp_db_path):
        """Test that unbounded index doesn't evict."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        # Create unbounded index (max_entries=None)
        index = HNSWVectorIndex(dimension=384)

        np.random.seed(42)

        # Add many memories
        for i in range(20):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=embedding,
                importance=0.1,
            )
            await index.index(memory)

        # All should be present
        assert index.size == 20

        stats = index.get_memory_stats()
        assert stats.budget_bytes is None  # No budget when unbounded
        assert stats.evictions == 0

    @pytest.mark.asyncio
    async def test_eviction_prefers_low_importance_then_old(self, temp_db_path):
        """Test eviction order: lowest importance first, then oldest."""
        import time

        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        index = HNSWVectorIndex(
            dimension=384,
            max_entries=3,
            eviction_batch_size=1,
        )

        np.random.seed(42)

        # Add memories with same importance but different times
        memories = []
        for i in range(3):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=embedding,
                importance=0.5,  # Same importance
            )
            await index.index(memory)
            memories.append(memory)
            time.sleep(0.01)  # Small delay to ensure different created_at

        # Add one more to trigger eviction
        new_embedding = np.random.randn(384).astype(np.float32)
        await index.index(
            Memory(
                content="New",
                user_id="alice",
                embedding=new_embedding,
                importance=0.5,
            )
        )

        # Should have evicted the oldest (first) entry
        assert index.size == 3
        assert memories[0].id not in index._memory_to_hnsw

    @pytest.mark.asyncio
    async def test_save_load_preserves_eviction_settings(self, temp_db_path):
        """Test that save/load preserves eviction settings."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        index = HNSWVectorIndex(
            dimension=384,
            max_entries=50,
            eviction_batch_size=10,
            save_path=temp_db_path,
        )

        np.random.seed(42)

        # Add some entries
        for i in range(5):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=embedding,
            )
            await index.index(memory)

        # Save
        index.save_index(temp_db_path)

        # Create new index and load
        index2 = HNSWVectorIndex(dimension=384)
        index2.load_index(temp_db_path)

        assert index2._max_entries == 50
        assert index2._eviction_batch_size == 10
        assert index2.size == 5


class _StubHnswIndex:
    """Minimal stand-in for the hnswlib index used by _evict_entries."""

    def __init__(self) -> None:
        self.deleted: list[int] = []

    def mark_deleted(self, hnsw_id: int) -> None:
        self.deleted.append(hnsw_id)


def test_evict_entries_selects_lowest_importance_then_oldest() -> None:
    """_evict_entries must remove the `count` lowest-importance (then oldest)
    entries. This drives the selection logic directly (no hnswlib), locking in
    the contract preserved when the full sort was replaced with heapq.nsmallest.
    """
    import types
    from dataclasses import dataclass

    from headroom.memory.adapters.hnsw import HNSWVectorIndex

    @dataclass
    class _Meta:
        importance: float
        created_at: float

    # ids a..f: importance ascending; two ties (0.5) broken by created_at.
    meta = {
        "a": _Meta(0.1, 100.0),
        "b": _Meta(0.2, 100.0),
        "c": _Meta(0.5, 100.0),  # older of the 0.5 tie
        "d": _Meta(0.5, 200.0),  # newer of the 0.5 tie
        "e": _Meta(0.9, 100.0),
        "f": _Meta(0.95, 100.0),
    }
    self = types.SimpleNamespace(
        _metadata=dict(meta),
        _memory_to_hnsw={k: i for i, k in enumerate(meta)},
        _hnsw_to_memory=dict(enumerate(meta)),
        _embeddings=dict.fromkeys(meta),
        _index=_StubHnswIndex(),
        _eviction_count=0,
    )

    evicted = HNSWVectorIndex._evict_entries(self, 3)

    assert evicted == 3
    # The three lowest by (importance, created_at): a, b, then c (older tie).
    assert set(meta) - set(self._metadata) == {"a", "b", "c"}
    assert "d" in self._metadata  # newer of the 0.5 tie survives
    # Mappings and embeddings for evicted ids are cleaned up too.
    assert "a" not in self._memory_to_hnsw
    assert "a" not in self._embeddings
    assert self._eviction_count == 3

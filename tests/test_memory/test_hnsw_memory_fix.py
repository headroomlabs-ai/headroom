"""Tests for HNSW memory-bloat fixes.

Covers:
1. Default max_elements is <= 10_000 (was 100_000, ~144 MB wasted at rest).
2. No _embeddings dict -- duplicate float arrays removed.
3. Startup warning fires when max_entries is None (unbounded).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

# Skip entire module when hnswlib is not installed / compiled for this CPU.
try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(content: str = "test", user_id: str = "alice", importance: float = 0.5):
    from headroom.memory.models import Memory

    embedding = np.random.randn(384).astype(np.float32)
    return Memory(content=content, user_id=user_id, embedding=embedding, importance=importance)


# ---------------------------------------------------------------------------
# Problem 1 -- default max_elements capped at 10_000
# ---------------------------------------------------------------------------


class TestDefaultMaxElements:
    """HNSWVectorIndex default max_elements must be <= 10_000."""

    def test_default_max_elements_is_at_most_10k(self):
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        idx = HNSWVectorIndex(dimension=4, max_entries=1)  # suppress unbounded warning
        assert idx._max_elements <= 10_000, (
            f"Default max_elements {idx._max_elements} exceeds 10_000; "
            "this pre-allocates too much C++ heap at startup."
        )

    def test_explicit_max_elements_respected(self):
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        idx = HNSWVectorIndex(dimension=4, max_elements=500, max_entries=1)
        assert idx._max_elements == 500

    @pytest.mark.asyncio
    async def test_index_auto_resizes_beyond_initial_capacity(self):
        """Index must still work past its initial allocation."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        # Start very small (dim=384 to match _make_memory) so resize triggers quickly.
        idx = HNSWVectorIndex(dimension=384, max_elements=2, max_entries=None)
        np.random.seed(0)
        for i in range(5):
            await idx.index(_make_memory(content=f"m{i}", importance=0.5))
        assert idx.size == 5

    def test_factory_creates_hnsw_without_100k_default(self):
        """_create_vector_index must not silently pass max_elements=100_000."""
        import tempfile
        from pathlib import Path

        from headroom.memory.config import MemoryConfig, VectorBackend
        from headroom.memory.factory import _create_vector_index

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryConfig(
                vector_backend=VectorBackend.HNSW,
                db_path=Path(tmpdir) / "test.db",
                hnsw_max_entries=100,  # set to avoid unbounded warning in factory
            )
            idx = _create_vector_index(config)
            assert idx._max_elements <= 10_000


# ---------------------------------------------------------------------------
# Problem 2 -- no duplicate _embeddings dict
# ---------------------------------------------------------------------------


class TestNoEmbeddingsDict:
    """The _embeddings attribute must not exist on HNSWVectorIndex."""

    def test_no_embeddings_attribute_on_new_index(self):
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        idx = HNSWVectorIndex(dimension=4, max_entries=10)
        assert not hasattr(idx, "_embeddings"), (
            "_embeddings dict found -- duplicate float32 array copy not removed."
        )

    @pytest.mark.asyncio
    async def test_no_embeddings_after_index(self):
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        idx = HNSWVectorIndex(dimension=384, max_entries=10)
        np.random.seed(1)
        m = _make_memory()
        await idx.index(m)
        assert not hasattr(idx, "_embeddings")

    @pytest.mark.asyncio
    async def test_no_embeddings_after_index_batch(self):
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        idx = HNSWVectorIndex(dimension=384, max_entries=10)
        np.random.seed(2)
        memories = [_make_memory(content=f"m{i}") for i in range(3)]
        await idx.index_batch(memories)
        assert not hasattr(idx, "_embeddings")

    @pytest.mark.asyncio
    async def test_no_embeddings_after_remove(self):
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        idx = HNSWVectorIndex(dimension=384, max_entries=10)
        np.random.seed(3)
        m = _make_memory()
        await idx.index(m)
        await idx.remove(m.id)
        assert not hasattr(idx, "_embeddings")

    @pytest.mark.asyncio
    async def test_search_still_returns_embedding_from_hnsw(self):
        """Search results must include an embedding (from hnswlib.get_items)."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex
        from headroom.memory.ports import VectorFilter

        idx = HNSWVectorIndex(dimension=384, max_entries=10)
        np.random.seed(4)
        m = _make_memory()
        await idx.index(m)

        results = await idx.search(VectorFilter(query_vector=m.embedding, top_k=1, user_id="alice"))

        assert len(results) == 1
        # Embedding must be present and have the correct dimensionality.
        assert results[0].memory.embedding is not None
        assert len(results[0].memory.embedding) == 384

    @pytest.mark.asyncio
    async def test_eviction_does_not_fail_without_embeddings_dict(self):
        """Eviction must work without a _embeddings dict."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        idx = HNSWVectorIndex(dimension=384, max_entries=3, eviction_batch_size=1)
        np.random.seed(5)
        for i in range(3):
            await idx.index(_make_memory(content=f"m{i}", importance=0.1 * (i + 1)))

        # One more to trigger eviction
        await idx.index(_make_memory(content="new", importance=0.9))

        assert idx.size == 3  # 3 - 1 evicted + 1 added
        stats = idx.get_memory_stats()
        assert stats.evictions == 1


# ---------------------------------------------------------------------------
# Problem 3 -- startup warning when max_entries is None
# ---------------------------------------------------------------------------


class TestUnboundedWarning:
    """HNSWVectorIndex must emit a logger.warning when max_entries is None."""

    def test_warning_fires_when_max_entries_none(self, caplog):
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        with caplog.at_level(logging.WARNING, logger="headroom.memory.adapters.hnsw"):
            HNSWVectorIndex(dimension=4, max_entries=None)

        assert any(
            "max_entries" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), "Expected a WARNING about unbounded max_entries but none was found."

    def test_no_warning_when_max_entries_set(self, caplog):
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        with caplog.at_level(logging.WARNING, logger="headroom.memory.adapters.hnsw"):
            HNSWVectorIndex(dimension=4, max_entries=100)

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING and "max_entries" in r.message
        ]
        assert not warning_records, (
            f"Unexpected max_entries warning when max_entries=100: {warning_records}"
        )

    def test_warning_message_mentions_production(self, caplog):
        """Warning text should guide operators toward a fix."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        with caplog.at_level(logging.WARNING, logger="headroom.memory.adapters.hnsw"):
            HNSWVectorIndex(dimension=4, max_entries=None)

        combined = " ".join(r.message for r in caplog.records if r.levelno == logging.WARNING)
        assert "production" in combined.lower() or "max_entries" in combined.lower()

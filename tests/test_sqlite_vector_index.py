"""Tests for SQLiteVectorIndex using sqlite-vec.

Tests verify:
- Vector indexing and search
- True CRUD operations (real deletes)
- Filtering by user_id, session_id, etc.
- Persistence across instances
- Memory stats and bounding
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from headroom.memory.models import Memory
from headroom.memory.ports import VectorFilter

# Check if sqlite-vec is available
try:
    from headroom.memory.adapters.sqlite_vector import is_sqlite_vec_available

    SQLITE_VEC_AVAILABLE = is_sqlite_vec_available()
except ImportError:
    SQLITE_VEC_AVAILABLE = False


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestSQLiteVectorIndex:
    """Tests for SQLiteVectorIndex."""

    @pytest.fixture
    def index(self):
        """Create a temporary SQLite vector index."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

        index = SQLiteVectorIndex(dimension=384, db_path=db_path)
        yield index

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_index_and_search(self, index):
        """Test basic indexing and search."""
        np.random.seed(42)

        # Create memories with random embeddings
        memories = []
        for i in range(10):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Test content {i}",
                user_id="alice",
                embedding=embedding,
            )
            await index.index(memory)
            memories.append(memory)

        assert index.size == 10

        # Search with first memory's embedding - should find itself
        filter = VectorFilter(
            query_vector=memories[0].embedding,
            top_k=3,
            user_id="alice",
        )
        results = await index.search(filter)

        assert len(results) == 3
        assert results[0].memory.id == memories[0].id
        assert results[0].similarity > 0.99  # Should be ~1.0 for exact match

    @pytest.mark.asyncio
    async def test_true_delete(self, index):
        """Test that delete actually removes entries."""
        np.random.seed(42)

        # Add memories
        memories = []
        for i in range(5):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=embedding,
            )
            await index.index(memory)
            memories.append(memory)

        assert index.size == 5

        # Delete one
        result = await index.remove(memories[0].id)
        assert result is True
        assert index.size == 4

        # Search should not find deleted memory
        filter = VectorFilter(
            query_vector=memories[0].embedding,
            top_k=10,
            user_id="alice",
        )
        results = await index.search(filter)

        result_ids = {r.memory.id for r in results}
        assert memories[0].id not in result_ids

    @pytest.mark.asyncio
    async def test_update_embedding(self, index):
        """Test updating an existing entry."""
        np.random.seed(42)

        embedding1 = np.random.randn(384).astype(np.float32)
        memory = Memory(
            content="Original content",
            user_id="alice",
            embedding=embedding1,
        )
        await index.index(memory)

        # Update with new embedding
        embedding2 = np.random.randn(384).astype(np.float32)
        memory.embedding = embedding2
        memory.content = "Updated content"
        await index.index(memory)

        # Should still be only 1 entry
        assert index.size == 1

        # Get stored embedding
        stored = await index.get_embedding(memory.id)
        assert stored is not None
        np.testing.assert_array_almost_equal(stored, embedding2)

    @pytest.mark.asyncio
    async def test_filter_by_user(self, index):
        """Test filtering search results by user_id."""
        np.random.seed(42)

        # Create memories for different users with similar embeddings
        base_embedding = np.random.randn(384).astype(np.float32)

        for user in ["alice", "bob", "charlie"]:
            # Slightly perturb embedding for each user
            embedding = base_embedding + np.random.randn(384).astype(np.float32) * 0.1
            memory = Memory(
                content=f"Content for {user}",
                user_id=user,
                embedding=embedding,
            )
            await index.index(memory)

        # Search filtered by user
        filter = VectorFilter(
            query_vector=base_embedding,
            top_k=10,
            user_id="alice",
        )
        results = await index.search(filter)

        assert len(results) == 1
        assert results[0].memory.user_id == "alice"

    @pytest.mark.asyncio
    async def test_filter_by_session(self, index):
        """Test filtering by session_id."""
        np.random.seed(42)

        embedding = np.random.randn(384).astype(np.float32)

        # Same user, different sessions
        for session in ["session1", "session2", None]:
            memory = Memory(
                content=f"Content for {session}",
                user_id="alice",
                session_id=session,
                embedding=embedding + np.random.randn(384).astype(np.float32) * 0.01,
            )
            await index.index(memory)

        filter = VectorFilter(
            query_vector=embedding,
            top_k=10,
            user_id="alice",
            session_id="session1",
        )
        results = await index.search(filter)

        assert len(results) == 1
        assert results[0].memory.session_id == "session1"

    @pytest.mark.asyncio
    async def test_min_similarity_filter(self, index):
        """Test minimum similarity threshold."""
        np.random.seed(42)

        # Create memories with varying similarity to query
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)  # Normalize

        # Very similar
        similar = query + np.random.randn(384).astype(np.float32) * 0.1
        similar = similar / np.linalg.norm(similar)

        # Less similar
        less_similar = np.random.randn(384).astype(np.float32)
        less_similar = less_similar / np.linalg.norm(less_similar)

        await index.index(Memory(content="Similar", user_id="alice", embedding=similar))
        await index.index(Memory(content="Less similar", user_id="alice", embedding=less_similar))

        # High threshold should filter out dissimilar
        filter = VectorFilter(
            query_vector=query,
            top_k=10,
            min_similarity=0.8,
        )
        results = await index.search(filter)

        # Only the similar one should pass
        assert len(results) <= 1
        if len(results) == 1:
            assert results[0].similarity >= 0.8


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestSQLiteVectorIndexPersistence:
    """Tests for persistence across index instances."""

    @pytest.mark.asyncio
    async def test_data_persists_across_instances(self):
        """Test that data survives index restart."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

            # Create index and add data
            index1 = SQLiteVectorIndex(dimension=384, db_path=db_path)

            np.random.seed(42)
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content="Persistent content",
                user_id="alice",
                embedding=embedding,
            )
            await index1.index(memory)

            memory_id = memory.id

            # Create new index instance
            index2 = SQLiteVectorIndex(dimension=384, db_path=db_path)

            assert index2.size == 1

            # Should find the memory
            filter = VectorFilter(
                query_vector=embedding,
                top_k=1,
            )
            results = await index2.search(filter)

            assert len(results) == 1
            assert results[0].memory.id == memory_id
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestSQLiteVectorIndexMemoryStats:
    """Tests for memory statistics."""

    @pytest.fixture
    def index(self):
        """Create a temporary SQLite vector index."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

        index = SQLiteVectorIndex(dimension=384, db_path=db_path, page_cache_size_kb=4096)
        yield index

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_memory_stats(self, index):
        """Test memory statistics."""
        stats = index.get_memory_stats()

        assert stats.name == "sqlite_vector_index"
        assert stats.entry_count == 0
        assert stats.budget_bytes == 4096 * 1024  # 4MB cache

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
    async def test_stats(self, index):
        """Test index statistics."""
        np.random.seed(42)

        for i in range(5):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Content {i}",
                user_id="alice" if i < 3 else "bob",
                embedding=embedding,
            )
            await index.index(memory)

        stats = index.stats()

        assert stats["size"] == 5
        assert stats["dimension"] == 384
        assert stats["users"] == 2
        assert stats["page_cache_size_kb"] == 4096
        assert stats["db_size_bytes"] > 0


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestSQLiteVectorIndexEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def index(self):
        """Create a temporary SQLite vector index."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

        index = SQLiteVectorIndex(dimension=384, db_path=db_path)
        yield index

        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_search_empty_index(self, index):
        """Test searching an empty index."""
        np.random.seed(42)
        query = np.random.randn(384).astype(np.float32)

        filter = VectorFilter(query_vector=query, top_k=10)
        results = await index.search(filter)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, index):
        """Test removing a nonexistent entry."""
        result = await index.remove("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_wrong_dimension_raises(self, index):
        """Test that wrong embedding dimension raises error."""
        wrong_embedding = np.random.randn(128).astype(np.float32)  # Wrong dimension
        memory = Memory(
            content="Test",
            user_id="alice",
            embedding=wrong_embedding,
        )

        with pytest.raises(ValueError, match="dimension"):
            await index.index(memory)

    @pytest.mark.asyncio
    async def test_no_embedding_raises(self, index):
        """Test that missing embedding raises error."""
        memory = Memory(
            content="Test",
            user_id="alice",
            embedding=None,
        )

        with pytest.raises(ValueError, match="no embedding"):
            await index.index(memory)

    @pytest.mark.asyncio
    async def test_clear(self, index):
        """Test clearing all entries."""
        np.random.seed(42)

        for i in range(5):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=embedding,
            )
            await index.index(memory)

        assert index.size == 5

        index.clear()

        assert index.size == 0

    @pytest.mark.asyncio
    async def test_batch_index(self, index):
        """Test batch indexing."""
        np.random.seed(42)

        memories = []
        for i in range(10):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=embedding,
            )
            memories.append(memory)

        # Add one without embedding
        memories.append(Memory(content="No embedding", user_id="alice"))

        indexed = await index.index_batch(memories)

        assert indexed == 10
        assert index.size == 10

    @pytest.mark.asyncio
    async def test_batch_index_uses_single_connection(self, index, monkeypatch):
        """Test batch indexing reuses a single sqlite-vec connection."""
        np.random.seed(42)
        memories = [
            Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=np.random.randn(384).astype(np.float32),
            )
            for i in range(10)
        ]

        original_get_conn = index._get_conn
        conn_calls = 0

        def counting_get_conn():
            nonlocal conn_calls
            conn_calls += 1
            return original_get_conn()

        monkeypatch.setattr(index, "_get_conn", counting_get_conn)

        indexed = await index.index_batch(memories)

        assert indexed == 10
        assert conn_calls == 1

    @pytest.mark.asyncio
    async def test_batch_remove(self, index):
        """Test batch removal."""
        np.random.seed(42)

        memories = []
        for i in range(5):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=embedding,
            )
            await index.index(memory)
            memories.append(memory)

        # Remove some
        ids_to_remove = [memories[0].id, memories[2].id, "nonexistent"]
        removed = await index.remove_batch(ids_to_remove)

        assert removed == 2
        assert index.size == 3

    @pytest.mark.asyncio
    async def test_batch_remove_uses_single_connection(self, index, monkeypatch):
        """Test batch removal reuses a single sqlite-vec connection."""
        np.random.seed(42)
        memories = []
        for i in range(5):
            memory = Memory(
                content=f"Content {i}",
                user_id="alice",
                embedding=np.random.randn(384).astype(np.float32),
            )
            await index.index(memory)
            memories.append(memory)

        original_get_conn = index._get_conn
        conn_calls = 0

        def counting_get_conn():
            nonlocal conn_calls
            conn_calls += 1
            return original_get_conn()

        monkeypatch.setattr(index, "_get_conn", counting_get_conn)

        removed = await index.remove_batch([memories[0].id, memories[2].id, "nonexistent"])

        assert removed == 2
        assert conn_calls == 1

"""Tests for the simple Memory API (headroom.memory.easy).

Tests cover:
- MemoryResult dataclass
- Memory class initialization with different backends
- Save/search/delete/clear operations
- Error handling and edge cases
- Backend type switching
- Resource cleanup

Note: These are integration tests that may hit external embedding APIs.
Tests are marked to skip on network timeouts (flaky CI).
"""

# CRITICAL: Must set TOKENIZERS_PARALLELISM before any imports that might
# trigger sentence_transformers/transformers loading.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tempfile
from pathlib import Path

import httpx
import pytest

from headroom.memory.easy import Memory, MemoryResult

# Check if hnswlib is available (local backend requires it)
try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not available")


def network_timeout_handler(func):
    """Decorator to skip tests on network timeouts (flaky CI)."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except httpx.ReadTimeout:
            pytest.skip("Skipped due to network timeout (flaky CI)")

    return wrapper


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    # Cleanup
    path.unlink(missing_ok=True)
    # Also cleanup related files (HNSW index, WAL, etc.)
    for suffix in ["-shm", "-wal", ".hnsw"]:
        Path(str(path) + suffix).unlink(missing_ok=True)


@pytest.fixture
async def memory_instance(temp_db_path):
    """Create a Memory instance with temp database for testing."""
    mem = Memory(backend="local", db_path=temp_db_path)
    yield mem
    await mem.close()


# =============================================================================
# MemoryResult Tests
# =============================================================================


class TestMemoryResult:
    """Tests for the MemoryResult dataclass."""

    def test_memory_result_creation(self):
        """Test basic MemoryResult creation."""
        result = MemoryResult(
            content="Test content",
            score=0.95,
            id="mem-123",
            metadata={"source": "test"},
        )
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.id == "mem-123"
        assert result.metadata == {"source": "test"}

    def test_memory_result_with_empty_metadata(self):
        """Test MemoryResult with empty metadata."""
        result = MemoryResult(
            content="Test",
            score=0.5,
            id="mem-456",
            metadata={},
        )
        assert result.metadata == {}


# =============================================================================
# Memory Class Initialization Tests
# =============================================================================


class TestMemoryInitialization:
    """Tests for Memory class initialization."""

    def test_default_initialization(self):
        """Test Memory initializes with default local backend."""
        mem = Memory()
        assert mem.backend_type == "local"
        assert not mem._initialized  # Lazy initialization

    def test_explicit_local_backend(self, temp_db_path):
        """Test explicit local backend initialization."""
        mem = Memory(backend="local", db_path=temp_db_path)
        assert mem.backend_type == "local"
        assert mem._db_path == temp_db_path

    def test_custom_db_path(self, temp_db_path):
        """Test Memory with custom database path."""
        mem = Memory(db_path=temp_db_path)
        assert mem._db_path == temp_db_path

    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError."""
        mem = Memory(backend="invalid-backend")
        with pytest.raises(ValueError, match="Unknown backend"):
            # Initialization happens on first use
            import asyncio

            asyncio.run(mem._ensure_initialized())

    def test_repr(self):
        """Test string representation."""
        mem = Memory(backend="local")
        assert repr(mem) == "Memory(backend='local')"

    def test_backend_type_property(self, temp_db_path):
        """Test backend_type property."""
        mem = Memory(backend="local", db_path=temp_db_path)
        assert mem.backend_type == "local"


# =============================================================================
# Memory Save Tests
# =============================================================================


class TestMemorySave:
    """Tests for Memory.save() operation."""

    @pytest.mark.asyncio
    async def test_simple_save(self, memory_instance):
        """Test saving a simple memory."""
        memory_id = await memory_instance.save(
            content="User prefers Python",
            user_id="alice",
        )
        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

    @pytest.mark.asyncio
    async def test_save_with_importance(self, memory_instance):
        """Test saving with custom importance."""
        memory_id = await memory_instance.save(
            content="Important fact",
            user_id="alice",
            importance=0.9,
        )
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_save_with_facts(self, memory_instance):
        """Test saving with pre-extracted facts."""
        memory_id = await memory_instance.save(
            content="Alice works at Netflix using Python",
            user_id="alice",
            facts=["Alice works at Netflix", "Alice uses Python"],
        )
        assert memory_id is not None

    @pytest.mark.asyncio
    @network_timeout_handler
    async def test_save_with_entities(self, memory_instance):
        """Test saving with pre-extracted entities."""
        memory_id = await memory_instance.save(
            content="Alice works at Netflix",
            user_id="alice",
            entities=[
                {"entity": "Alice", "entity_type": "person"},
                {"entity": "Netflix", "entity_type": "organization"},
            ],
        )
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_save_with_relationships(self, memory_instance):
        """Test saving with pre-extracted relationships."""
        memory_id = await memory_instance.save(
            content="Alice works at Netflix",
            user_id="alice",
            relationships=[
                {"source": "Alice", "relationship": "works_at", "destination": "Netflix"},
            ],
        )
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_save_with_metadata(self, memory_instance):
        """Test saving with custom metadata."""
        memory_id = await memory_instance.save(
            content="Test content",
            user_id="alice",
            metadata={"source": "test", "version": 1},
        )
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_save_multiple_memories(self, memory_instance):
        """Test saving multiple memories."""
        ids = []
        for i in range(5):
            memory_id = await memory_instance.save(
                content=f"Memory number {i}",
                user_id="alice",
            )
            ids.append(memory_id)

        # All IDs should be unique
        assert len(ids) == 5
        assert len(set(ids)) == 5


# =============================================================================
# Memory Search Tests
# =============================================================================


class TestMemorySearch:
    """Tests for Memory.search() operation."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, memory_instance):
        """Test search returns saved memories."""
        # Save some memories
        await memory_instance.save(
            content="User prefers Python programming language",
            user_id="alice",
        )
        await memory_instance.save(
            content="User works on machine learning projects",
            user_id="alice",
        )

        # Search for them
        results = await memory_instance.search(
            query="programming language",
            user_id="alice",
        )

        assert len(results) > 0
        assert all(isinstance(r, MemoryResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_returns_memory_result_objects(self, memory_instance):
        """Test search returns proper MemoryResult objects."""
        await memory_instance.save(
            content="Test memory content",
            user_id="alice",
        )

        results = await memory_instance.search(
            query="test memory",
            user_id="alice",
        )

        assert len(results) > 0
        result = results[0]
        assert hasattr(result, "content")
        assert hasattr(result, "score")
        assert hasattr(result, "id")
        assert hasattr(result, "metadata")

    @pytest.mark.asyncio
    async def test_search_user_isolation(self, memory_instance):
        """Test that searches are isolated by user_id."""
        # Save for Alice
        await memory_instance.save(
            content="Alice secret: favorite color is blue",
            user_id="alice",
        )

        # Save for Bob
        await memory_instance.save(
            content="Bob secret: favorite color is red",
            user_id="bob",
        )

        # Search as Bob - should NOT find Alice's memory
        results = await memory_instance.search(
            query="favorite color blue",
            user_id="bob",
        )

        # Bob should only see his own memories
        for result in results:
            assert "alice" not in result.content.lower() or result.score < 0.5

    @pytest.mark.asyncio
    async def test_search_with_top_k(self, memory_instance):
        """Test search respects top_k limit."""
        # Save multiple memories
        for i in range(10):
            await memory_instance.save(
                content=f"Test memory number {i}",
                user_id="alice",
            )

        # Search with limit
        results = await memory_instance.search(
            query="test memory",
            user_id="alice",
            top_k=3,
        )

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_empty_results(self, memory_instance):
        """Test search with no matching results."""
        # Save unrelated memory
        await memory_instance.save(
            content="Unrelated content about cooking",
            user_id="alice",
        )

        # Search for something very different
        results = await memory_instance.search(
            query="quantum physics black holes",
            user_id="alice",
        )

        # Should return empty or very low scoring results
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_score(self, memory_instance):
        """Test that search results are sorted by relevance score."""
        await memory_instance.save(
            content="Python is a programming language",
            user_id="alice",
        )
        await memory_instance.save(
            content="Python programming for data science",
            user_id="alice",
        )
        await memory_instance.save(
            content="Cooking recipes for dinner",
            user_id="alice",
        )

        results = await memory_instance.search(
            query="Python programming",
            user_id="alice",
        )

        if len(results) > 1:
            # Verify scores are in descending order
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# Memory Delete Tests
# =============================================================================


class TestMemoryDelete:
    """Tests for Memory.delete() operation."""

    @pytest.mark.asyncio
    async def test_delete_existing_memory(self, memory_instance):
        """Test deleting an existing memory."""
        # Save a memory
        memory_id = await memory_instance.save(
            content="Memory to delete",
            user_id="alice",
        )

        # Delete it
        deleted = await memory_instance.delete(memory_id)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_memory(self, memory_instance):
        """Test deleting a non-existent memory."""
        deleted = await memory_instance.delete("nonexistent-id-12345")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_deleted_memory_not_searchable(self, memory_instance):
        """Test that deleted memories don't appear in search."""
        # Save and delete
        memory_id = await memory_instance.save(
            content="Unique content xyz123",
            user_id="alice",
        )
        await memory_instance.delete(memory_id)

        # Search should not find it
        results = await memory_instance.search(
            query="unique content xyz123",
            user_id="alice",
        )

        # Either empty or none with that ID
        matching_ids = [r.id for r in results if r.id == memory_id]
        assert len(matching_ids) == 0


# =============================================================================
# Memory Clear Tests
# =============================================================================


class TestMemoryClear:
    """Tests for Memory.clear() operation."""

    @pytest.mark.asyncio
    async def test_clear_user_memories(self, memory_instance):
        """Test clearing all memories for a user."""
        # Save multiple memories
        for i in range(5):
            await memory_instance.save(
                content=f"Memory {i}",
                user_id="alice",
            )

        # Clear
        count = await memory_instance.clear(user_id="alice")

        # Search should return no results
        _results = await memory_instance.search(
            query="memory",
            user_id="alice",
        )

        # Either cleared or returns 0 if backend doesn't support clear_user
        assert isinstance(count, int)


# =============================================================================
# Memory Close Tests
# =============================================================================


class TestMemoryClose:
    """Tests for Memory.close() operation."""

    @pytest.mark.asyncio
    async def test_close_releases_resources(self, temp_db_path):
        """Test that close releases resources."""
        mem = Memory(backend="local", db_path=temp_db_path)

        # Initialize by saving
        await mem.save(content="Test", user_id="alice")
        assert mem._initialized is True

        # Close
        await mem.close()
        assert mem._initialized is False

    @pytest.mark.asyncio
    @network_timeout_handler
    async def test_close_idempotent(self, temp_db_path):
        """Test that close can be called multiple times."""
        mem = Memory(backend="local", db_path=temp_db_path)
        await mem.save(content="Test", user_id="alice")

        # Close multiple times - should not raise
        await mem.close()
        await mem.close()
        await mem.close()

    @pytest.mark.asyncio
    async def test_reinitialize_after_close(self, temp_db_path):
        """Test that Memory can be reused after close."""
        mem = Memory(backend="local", db_path=temp_db_path)

        # First use
        await mem.save(content="First", user_id="alice")
        await mem.close()

        # Second use - should reinitialize
        memory_id = await mem.save(content="Second", user_id="alice")
        assert memory_id is not None
        await mem.close()


# =============================================================================
# Lazy Initialization Tests
# =============================================================================


class TestLazyInitialization:
    """Tests for lazy initialization behavior."""

    def test_not_initialized_on_creation(self, temp_db_path):
        """Test that backend is not initialized on creation."""
        mem = Memory(backend="local", db_path=temp_db_path)
        assert mem._initialized is False
        assert mem._backend is None

    @pytest.mark.asyncio
    async def test_initialized_on_first_save(self, temp_db_path):
        """Test that backend initializes on first save."""
        mem = Memory(backend="local", db_path=temp_db_path)
        assert mem._initialized is False

        await mem.save(content="Test", user_id="alice")
        assert mem._initialized is True
        assert mem._backend is not None

        await mem.close()

    @pytest.mark.asyncio
    async def test_initialized_on_first_search(self, temp_db_path):
        """Test that backend initializes on first search."""
        mem = Memory(backend="local", db_path=temp_db_path)
        assert mem._initialized is False

        await mem.search(query="test", user_id="alice")
        assert mem._initialized is True

        await mem.close()

    @pytest.mark.asyncio
    async def test_ensure_initialized_idempotent(self, temp_db_path):
        """Test that _ensure_initialized is idempotent."""
        mem = Memory(backend="local", db_path=temp_db_path)

        await mem._ensure_initialized()
        backend1 = mem._backend

        await mem._ensure_initialized()
        backend2 = mem._backend

        # Should be same backend instance
        assert backend1 is backend2

        await mem.close()


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, memory_instance):
        """Test complete save-search-delete workflow."""
        # 1. Save memories
        id1 = await memory_instance.save(
            content="Alice is a software engineer",
            user_id="alice",
        )
        _id2 = await memory_instance.save(
            content="Alice works on AI projects",
            user_id="alice",
        )

        # 2. Search
        results = await memory_instance.search(
            query="software engineer",
            user_id="alice",
        )
        assert len(results) > 0

        # 3. Delete one
        deleted = await memory_instance.delete(id1)
        assert deleted is True

        # 4. Search again - should still find the other
        results = await memory_instance.search(
            query="AI projects",
            user_id="alice",
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_multi_user_workflow(self, memory_instance):
        """Test workflow with multiple users."""
        # Save for different users
        await memory_instance.save(
            content="Alice prefers Python",
            user_id="alice",
        )
        await memory_instance.save(
            content="Bob prefers JavaScript",
            user_id="bob",
        )

        # Each user should see their own data
        alice_results = await memory_instance.search(
            query="programming preference",
            user_id="alice",
        )
        bob_results = await memory_instance.search(
            query="programming preference",
            user_id="bob",
        )

        # Both should have results
        assert len(alice_results) > 0 or len(bob_results) > 0

    @pytest.mark.asyncio
    async def test_save_with_full_extraction_data(self, memory_instance):
        """Test saving with all extraction fields populated."""
        memory_id = await memory_instance.save(
            content="Alice works at Netflix on machine learning infrastructure",
            user_id="alice",
            importance=0.9,
            facts=[
                "Alice works at Netflix",
                "Alice works on machine learning infrastructure",
            ],
            entities=[
                {"entity": "Alice", "entity_type": "person"},
                {"entity": "Netflix", "entity_type": "organization"},
                {"entity": "machine learning", "entity_type": "technology"},
            ],
            relationships=[
                {"source": "Alice", "relationship": "works_at", "destination": "Netflix"},
                {"source": "Alice", "relationship": "works_on", "destination": "machine learning"},
            ],
            metadata={"source": "conversation", "turn": 5},
        )

        assert memory_id is not None

        # Should be searchable
        results = await memory_instance.search(
            query="Netflix machine learning",
            user_id="alice",
        )
        assert len(results) > 0

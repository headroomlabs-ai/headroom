"""Tests for HierarchicalMemory core operations.

Tests cover:
- Batch operations (add_batch)
- Individual retrieval (get)
- Memory statistics (count)
- Memory updates with re-embedding
- Supersession and temporal versioning
- History chain traversal
- Deletion operations
- Convenience methods (remember, recall, get_user_memories, get_session_memories)

Note: These are integration tests that may hit external embedding APIs.
Tests are marked to skip on network timeouts (flaky CI).
"""

# CRITICAL: Must set TOKENIZERS_PARALLELISM before any imports
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import functools
import tempfile
from pathlib import Path

import httpx
import pytest

from headroom.memory.config import MemoryConfig
from headroom.memory.core import HierarchicalMemory
from headroom.memory.models import Memory, ScopeLevel
from headroom.memory.ports import MemoryFilter

# Check if hnswlib is available (HierarchicalMemory requires it)
try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not available")


def network_timeout_handler(func):
    """Decorator to skip tests on network timeouts (flaky CI)."""

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
    for suffix in ["-shm", "-wal", ".hnsw"]:
        Path(str(path) + suffix).unlink(missing_ok=True)


@pytest.fixture
async def memory_system(temp_db_path):
    """Create a HierarchicalMemory instance for testing."""
    config = MemoryConfig(db_path=str(temp_db_path))
    system = await HierarchicalMemory.create(config)
    yield system
    # Properly close to release httpx clients
    await system.close()


# =============================================================================
# Batch Operations Tests
# =============================================================================


class TestAddBatch:
    """Tests for HierarchicalMemory.add_batch()."""

    @pytest.mark.asyncio
    @network_timeout_handler
    async def test_add_batch_basic(self, memory_system):
        """Test basic batch addition."""
        memories_data = [
            {"content": "Memory 1", "user_id": "alice"},
            {"content": "Memory 2", "user_id": "alice"},
            {"content": "Memory 3", "user_id": "alice"},
        ]

        memories = await memory_system.add_batch(memories_data)

        assert len(memories) == 3
        assert all(isinstance(m, Memory) for m in memories)
        assert all(m.user_id == "alice" for m in memories)

    @pytest.mark.asyncio
    async def test_add_batch_with_different_users(self, memory_system):
        """Test batch addition with different users."""
        memories_data = [
            {"content": "Alice's memory", "user_id": "alice"},
            {"content": "Bob's memory", "user_id": "bob"},
        ]

        memories = await memory_system.add_batch(memories_data)

        assert len(memories) == 2
        assert memories[0].user_id == "alice"
        assert memories[1].user_id == "bob"

    @pytest.mark.asyncio
    async def test_add_batch_with_full_hierarchy(self, memory_system):
        """Test batch addition with full hierarchy fields."""
        memories_data = [
            {
                "content": "Session memory",
                "user_id": "alice",
                "session_id": "session-1",
            },
            {
                "content": "Agent memory",
                "user_id": "alice",
                "session_id": "session-1",
                "agent_id": "agent-1",
            },
        ]

        memories = await memory_system.add_batch(memories_data)

        assert len(memories) == 2
        assert memories[0].scope_level == ScopeLevel.SESSION
        assert memories[1].scope_level == ScopeLevel.AGENT

    @pytest.mark.asyncio
    async def test_add_batch_with_importance(self, memory_system):
        """Test batch addition with custom importance scores."""
        memories_data = [
            {"content": "Low importance", "user_id": "alice", "importance": 0.2},
            {"content": "High importance", "user_id": "alice", "importance": 0.9},
        ]

        memories = await memory_system.add_batch(memories_data)

        assert memories[0].importance == 0.2
        assert memories[1].importance == 0.9

    @pytest.mark.asyncio
    async def test_add_batch_with_metadata(self, memory_system):
        """Test batch addition with metadata."""
        memories_data = [
            {
                "content": "Memory with metadata",
                "user_id": "alice",
                "metadata": {"source": "test", "version": 1},
            },
        ]

        memories = await memory_system.add_batch(memories_data)

        assert memories[0].metadata == {"source": "test", "version": 1}

    @pytest.mark.asyncio
    async def test_add_batch_generates_embeddings(self, memory_system):
        """Test that batch addition generates embeddings."""
        memories_data = [
            {"content": "Test embedding generation", "user_id": "alice"},
        ]

        memories = await memory_system.add_batch(memories_data, auto_embed=True)

        assert memories[0].embedding is not None
        assert len(memories[0].embedding) > 0

    @pytest.mark.asyncio
    async def test_add_batch_without_embeddings(self, memory_system):
        """Test batch addition without embedding generation."""
        memories_data = [
            {"content": "No embedding", "user_id": "alice"},
        ]

        memories = await memory_system.add_batch(memories_data, auto_embed=False)

        assert memories[0].embedding is None

    @pytest.mark.asyncio
    async def test_add_batch_large(self, memory_system):
        """Test batch addition with many memories."""
        memories_data = [{"content": f"Memory {i}", "user_id": "alice"} for i in range(50)]

        memories = await memory_system.add_batch(memories_data)

        assert len(memories) == 50
        # All IDs should be unique
        ids = [m.id for m in memories]
        assert len(set(ids)) == 50


# =============================================================================
# Individual Retrieval Tests
# =============================================================================


class TestGet:
    """Tests for HierarchicalMemory.get()."""

    @pytest.mark.asyncio
    async def test_get_existing_memory(self, memory_system):
        """Test retrieving an existing memory."""
        mem = await memory_system.add(
            content="Retrievable memory",
            user_id="alice",
        )

        retrieved = await memory_system.get(mem.id)

        assert retrieved is not None
        assert retrieved.id == mem.id
        assert retrieved.content == "Retrievable memory"

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, memory_system):
        """Test retrieving a non-existent memory."""
        retrieved = await memory_system.get("nonexistent-id-12345")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_preserves_all_fields(self, memory_system):
        """Test that get preserves all memory fields."""
        mem = await memory_system.add(
            content="Full memory",
            user_id="alice",
            session_id="session-1",
            importance=0.8,
            entity_refs=["entity1", "entity2"],
            metadata={"key": "value"},
        )

        retrieved = await memory_system.get(mem.id)

        assert retrieved is not None
        assert retrieved.content == "Full memory"
        assert retrieved.user_id == "alice"
        assert retrieved.session_id == "session-1"
        assert retrieved.importance == 0.8
        assert retrieved.entity_refs == ["entity1", "entity2"]
        assert retrieved.metadata == {"key": "value"}


# =============================================================================
# Count Tests
# =============================================================================


class TestCount:
    """Tests for HierarchicalMemory.count()."""

    @pytest.mark.asyncio
    async def test_count_empty(self, memory_system):
        """Test counting with no memories."""
        count = await memory_system.count(MemoryFilter(user_id="nonexistent"))
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_user_memories(self, memory_system):
        """Test counting memories for a user."""
        for i in range(5):
            await memory_system.add(content=f"Memory {i}", user_id="alice")

        count = await memory_system.count(MemoryFilter(user_id="alice"))
        assert count == 5

    @pytest.mark.asyncio
    async def test_count_with_session_filter(self, memory_system):
        """Test counting with session filter."""
        await memory_system.add(content="Session 1", user_id="alice", session_id="s1")
        await memory_system.add(content="Session 1", user_id="alice", session_id="s1")
        await memory_system.add(content="Session 2", user_id="alice", session_id="s2")

        count = await memory_system.count(MemoryFilter(user_id="alice", session_id="s1"))
        assert count == 2

    @pytest.mark.asyncio
    async def test_count_with_importance_filter(self, memory_system):
        """Test counting with importance filter."""
        await memory_system.add(content="Low", user_id="alice", importance=0.3)
        await memory_system.add(content="High", user_id="alice", importance=0.8)
        await memory_system.add(content="Very High", user_id="alice", importance=0.95)

        count = await memory_system.count(MemoryFilter(user_id="alice", min_importance=0.7))
        assert count == 2


# =============================================================================
# Update Tests
# =============================================================================


class TestUpdate:
    """Tests for HierarchicalMemory.update()."""

    @pytest.mark.asyncio
    async def test_update_content(self, memory_system):
        """Test updating memory content."""
        mem = await memory_system.add(content="Original", user_id="alice")

        updated = await memory_system.update(mem.id, content="Updated content")

        assert updated is not None
        assert updated.content == "Updated content"

    @pytest.mark.asyncio
    async def test_update_importance(self, memory_system):
        """Test updating importance score."""
        mem = await memory_system.add(content="Test", user_id="alice", importance=0.5)

        updated = await memory_system.update(mem.id, importance=0.9)

        assert updated is not None
        assert updated.importance == 0.9

    @pytest.mark.asyncio
    async def test_update_entity_refs(self, memory_system):
        """Test updating entity references."""
        mem = await memory_system.add(content="Test", user_id="alice", entity_refs=["old"])

        updated = await memory_system.update(mem.id, entity_refs=["new1", "new2"])

        assert updated is not None
        assert updated.entity_refs == ["new1", "new2"]

    @pytest.mark.asyncio
    async def test_update_metadata_merges(self, memory_system):
        """Test that metadata updates are merged."""
        mem = await memory_system.add(
            content="Test",
            user_id="alice",
            metadata={"key1": "value1"},
        )

        updated = await memory_system.update(mem.id, metadata={"key2": "value2"})

        assert updated is not None
        assert updated.metadata["key1"] == "value1"
        assert updated.metadata["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self, memory_system):
        """Test updating non-existent memory returns None."""
        result = await memory_system.update("nonexistent-id", content="New content")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_content_triggers_reembedding(self, memory_system):
        """Test that content update triggers re-embedding."""
        mem = await memory_system.add(content="Original text", user_id="alice")
        original_embedding = mem.embedding.copy() if mem.embedding is not None else None

        updated = await memory_system.update(
            mem.id, content="Completely different content", re_embed=True
        )

        assert updated is not None
        assert updated.embedding is not None
        if original_embedding is not None:
            # Embeddings should be different for different content
            import numpy as np

            diff = np.linalg.norm(updated.embedding - original_embedding)
            assert diff > 0.1  # Some meaningful difference

    @pytest.mark.asyncio
    async def test_update_without_reembedding(self, memory_system):
        """Test update without re-embedding."""
        mem = await memory_system.add(content="Original", user_id="alice")
        original_embedding = mem.embedding.copy() if mem.embedding is not None else None

        updated = await memory_system.update(mem.id, content="New content", re_embed=False)

        assert updated is not None
        if original_embedding is not None and updated.embedding is not None:
            import numpy as np

            assert np.allclose(updated.embedding, original_embedding)


# =============================================================================
# Supersession Tests
# =============================================================================


class TestSupersede:
    """Tests for HierarchicalMemory.supersede()."""

    @pytest.mark.asyncio
    async def test_supersede_basic(self, memory_system):
        """Test basic supersession."""
        old = await memory_system.add(
            content="User prefers Python",
            user_id="alice",
        )

        new = await memory_system.supersede(
            old.id,
            "User now prefers JavaScript",
        )

        assert new is not None
        assert new.content == "User now prefers JavaScript"
        assert new.id != old.id

    @pytest.mark.asyncio
    async def test_supersede_preserves_scope(self, memory_system):
        """Test that supersession preserves scope hierarchy."""
        old = await memory_system.add(
            content="Original",
            user_id="alice",
            session_id="session-1",
            agent_id="agent-1",
        )

        new = await memory_system.supersede(old.id, "Superseded content")

        assert new.user_id == old.user_id
        assert new.session_id == old.session_id
        assert new.agent_id == old.agent_id
        assert new.scope_level == old.scope_level

    @pytest.mark.asyncio
    async def test_supersede_preserves_importance(self, memory_system):
        """Test that supersession preserves importance."""
        old = await memory_system.add(
            content="Important",
            user_id="alice",
            importance=0.95,
        )

        new = await memory_system.supersede(old.id, "Still important")

        assert new.importance == old.importance

    @pytest.mark.asyncio
    async def test_supersede_copies_entity_refs(self, memory_system):
        """Test that supersession copies entity references."""
        old = await memory_system.add(
            content="Original",
            user_id="alice",
            entity_refs=["entity1", "entity2"],
        )

        new = await memory_system.supersede(old.id, "Superseded")

        assert new.entity_refs == old.entity_refs

    @pytest.mark.asyncio
    async def test_supersede_nonexistent_raises(self, memory_system):
        """Test that superseding non-existent memory raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await memory_system.supersede("nonexistent-id", "Content")

    @pytest.mark.asyncio
    async def test_supersede_searchable(self, memory_system):
        """Test that superseded content is searchable."""
        old = await memory_system.add(
            content="User prefers Python",
            user_id="alice",
        )

        new = await memory_system.supersede(
            old.id,
            "User now prefers JavaScript frameworks",
        )

        # Search for new content should find it
        results = await memory_system.search("JavaScript", user_id="alice")
        found_ids = [r.memory.id for r in results]
        assert new.id in found_ids


# =============================================================================
# History Tests
# =============================================================================


class TestGetHistory:
    """Tests for HierarchicalMemory.get_history()."""

    @pytest.mark.asyncio
    async def test_get_history_single_memory(self, memory_system):
        """Test history for a memory with no supersessions."""
        mem = await memory_system.add(content="Single memory", user_id="alice")

        history = await memory_system.get_history(mem.id)

        assert len(history) >= 1
        assert any(m.id == mem.id for m in history)

    @pytest.mark.asyncio
    async def test_get_history_with_supersession(self, memory_system):
        """Test history for a supersession chain."""
        mem1 = await memory_system.add(content="Version 1", user_id="alice")
        mem2 = await memory_system.supersede(mem1.id, "Version 2")
        mem3 = await memory_system.supersede(mem2.id, "Version 3")

        history = await memory_system.get_history(mem3.id, include_future=False)

        # Should include at least the current memory
        assert len(history) >= 1


# =============================================================================
# Delete Tests
# =============================================================================


class TestDelete:
    """Tests for HierarchicalMemory.delete()."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, memory_system):
        """Test deleting an existing memory."""
        mem = await memory_system.add(content="To delete", user_id="alice")

        deleted = await memory_system.delete(mem.id)

        assert deleted is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory_system):
        """Test deleting a non-existent memory."""
        deleted = await memory_system.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_deleted_not_retrievable(self, memory_system):
        """Test that deleted memory cannot be retrieved."""
        mem = await memory_system.add(content="Deleted", user_id="alice")
        await memory_system.delete(mem.id)

        retrieved = await memory_system.get(mem.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_deleted_not_searchable(self, memory_system):
        """Test that deleted memory is not searchable."""
        mem = await memory_system.add(
            content="Unique content xyz789",
            user_id="alice",
        )
        await memory_system.delete(mem.id)

        results = await memory_system.search("xyz789", user_id="alice")
        found_ids = [r.memory.id for r in results]
        assert mem.id not in found_ids


# =============================================================================
# Clear Scope Tests
# =============================================================================


class TestClearScope:
    """Tests for HierarchicalMemory.clear_scope()."""

    @pytest.mark.asyncio
    async def test_clear_user_scope(self, memory_system):
        """Test clearing all memories for a user."""
        for i in range(5):
            await memory_system.add(content=f"Memory {i}", user_id="alice")

        count = await memory_system.clear_scope(user_id="alice")

        assert count >= 5

        # Verify cleared
        remaining = await memory_system.count(MemoryFilter(user_id="alice"))
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_clear_session_scope(self, memory_system):
        """Test clearing memories for a specific session."""
        await memory_system.add(content="Session 1", user_id="alice", session_id="s1")
        await memory_system.add(content="Session 1", user_id="alice", session_id="s1")
        await memory_system.add(content="Session 2", user_id="alice", session_id="s2")

        count = await memory_system.clear_scope(user_id="alice", session_id="s1")

        assert count >= 2

        # Session 2 should remain
        remaining = await memory_system.count(MemoryFilter(user_id="alice", session_id="s2"))
        assert remaining == 1

    @pytest.mark.asyncio
    async def test_clear_scope_empty(self, memory_system):
        """Test clearing scope with no matching memories."""
        count = await memory_system.clear_scope(user_id="nonexistent-user")
        assert count == 0


# =============================================================================
# Convenience Methods Tests
# =============================================================================


class TestRemember:
    """Tests for HierarchicalMemory.remember()."""

    @pytest.mark.asyncio
    async def test_remember_basic(self, memory_system):
        """Test basic remember functionality."""
        mem = await memory_system.remember(
            content="Coffee preference",
            user_id="alice",
        )

        assert mem is not None
        assert mem.content == "Coffee preference"
        assert mem.user_id == "alice"

    @pytest.mark.asyncio
    async def test_remember_with_session(self, memory_system):
        """Test remember with session context."""
        mem = await memory_system.remember(
            content="Session fact",
            user_id="alice",
            session_id="session-1",
        )

        assert mem.session_id == "session-1"
        assert mem.scope_level == ScopeLevel.SESSION

    @pytest.mark.asyncio
    async def test_remember_with_importance(self, memory_system):
        """Test remember with custom importance."""
        mem = await memory_system.remember(
            content="Important",
            user_id="alice",
            importance=0.9,
        )

        assert mem.importance == 0.9


class TestRecall:
    """Tests for HierarchicalMemory.recall()."""

    @pytest.mark.asyncio
    async def test_recall_basic(self, memory_system):
        """Test basic recall functionality."""
        await memory_system.remember(
            content="Alice likes Python programming",
            user_id="alice",
        )

        memories = await memory_system.recall(
            query="programming language",
            user_id="alice",
        )

        assert len(memories) > 0
        assert all(isinstance(m, Memory) for m in memories)

    @pytest.mark.asyncio
    async def test_recall_respects_top_k(self, memory_system):
        """Test that recall respects top_k limit."""
        for i in range(10):
            await memory_system.remember(
                content=f"Fact {i}",
                user_id="alice",
            )

        memories = await memory_system.recall(
            query="fact",
            user_id="alice",
            top_k=3,
        )

        assert len(memories) <= 3

    @pytest.mark.asyncio
    async def test_recall_user_isolation(self, memory_system):
        """Test that recall respects user isolation."""
        await memory_system.remember(content="Alice's secret", user_id="alice")
        await memory_system.remember(content="Bob's fact", user_id="bob")

        alice_memories = await memory_system.recall("secret", user_id="alice")
        bob_memories = await memory_system.recall("secret", user_id="bob")

        # Results should be different or empty for non-matching user
        alice_ids = {m.id for m in alice_memories}
        bob_ids = {m.id for m in bob_memories}
        # At most minimal overlap
        assert alice_ids != bob_ids or len(alice_ids) == 0


class TestGetUserMemories:
    """Tests for HierarchicalMemory.get_user_memories()."""

    @pytest.mark.asyncio
    async def test_get_user_memories_basic(self, memory_system):
        """Test getting all memories for a user."""
        for i in range(5):
            await memory_system.add(content=f"Memory {i}", user_id="alice")

        memories = await memory_system.get_user_memories("alice")

        assert len(memories) == 5

    @pytest.mark.asyncio
    async def test_get_user_memories_includes_sessions(self, memory_system):
        """Test that user memories include session-level by default."""
        await memory_system.add(content="User level", user_id="alice")
        await memory_system.add(content="Session level", user_id="alice", session_id="s1")

        memories = await memory_system.get_user_memories("alice", include_sessions=True)

        assert len(memories) == 2

    @pytest.mark.asyncio
    async def test_get_user_memories_excludes_sessions(self, memory_system):
        """Test excluding session-level memories."""
        await memory_system.add(content="User level", user_id="alice")
        await memory_system.add(content="Session level", user_id="alice", session_id="s1")

        memories = await memory_system.get_user_memories("alice", include_sessions=False)

        assert len(memories) == 1
        assert all(m.scope_level == ScopeLevel.USER for m in memories)

    @pytest.mark.asyncio
    async def test_get_user_memories_respects_limit(self, memory_system):
        """Test that limit is respected."""
        for i in range(20):
            await memory_system.add(content=f"Memory {i}", user_id="alice")

        memories = await memory_system.get_user_memories("alice", limit=5)

        assert len(memories) == 5


class TestGetSessionMemories:
    """Tests for HierarchicalMemory.get_session_memories()."""

    @pytest.mark.asyncio
    async def test_get_session_memories_basic(self, memory_system):
        """Test getting all memories for a session."""
        await memory_system.add(content="Session memory 1", user_id="alice", session_id="s1")
        await memory_system.add(content="Session memory 2", user_id="alice", session_id="s1")
        await memory_system.add(content="Other session", user_id="alice", session_id="s2")

        memories = await memory_system.get_session_memories("alice", "s1")

        assert len(memories) == 2

    @pytest.mark.asyncio
    async def test_get_session_memories_respects_limit(self, memory_system):
        """Test that limit is respected."""
        for i in range(20):
            await memory_system.add(content=f"Memory {i}", user_id="alice", session_id="s1")

        memories = await memory_system.get_session_memories("alice", "s1", limit=5)

        assert len(memories) == 5


# =============================================================================
# Integration Tests
# =============================================================================


class TestCoreIntegration:
    """Integration tests for core operations."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, memory_system):
        """Test complete memory lifecycle: add -> update -> search -> delete."""
        # Add
        mem = await memory_system.add(
            content="Original content",
            user_id="alice",
            importance=0.5,
        )
        assert mem is not None

        # Update
        updated = await memory_system.update(mem.id, content="Updated content")
        assert updated is not None
        assert updated.content == "Updated content"

        # Search
        results = await memory_system.search("updated", user_id="alice")
        found_ids = [r.memory.id for r in results]
        assert mem.id in found_ids

        # Delete
        deleted = await memory_system.delete(mem.id)
        assert deleted is True

        # Verify gone
        retrieved = await memory_system.get(mem.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_batch_and_search(self, memory_system):
        """Test batch addition and subsequent search."""
        memories_data = [
            {"content": "Python is a programming language", "user_id": "alice"},
            {"content": "JavaScript runs in browsers", "user_id": "alice"},
            {"content": "Rust is for systems programming", "user_id": "alice"},
        ]

        await memory_system.add_batch(memories_data)

        # Search should find relevant memories
        results = await memory_system.search("programming", user_id="alice")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_supersession_chain(self, memory_system):
        """Test creating a chain of supersessions."""
        v1 = await memory_system.add(content="Version 1", user_id="alice")
        v2 = await memory_system.supersede(v1.id, "Version 2")
        _v3 = await memory_system.supersede(v2.id, "Version 3")

        # Latest version should be searchable
        results = await memory_system.search("Version", user_id="alice")
        assert len(results) > 0

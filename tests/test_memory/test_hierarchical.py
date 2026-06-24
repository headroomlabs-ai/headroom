"""Tests for the hierarchical memory system.

Tests cover:
- Memory models (Memory, ScopeLevel)
- SQLite memory store
- HNSW vector index
- FTS5 text index
- LRU cache
- HierarchicalMemory orchestrator
- Memory bubbling
- Temporal versioning (supersession)
"""

# CRITICAL: Must set TOKENIZERS_PARALLELISM before any imports that might
# trigger sentence_transformers/transformers loading. The Rust tokenizers
# use parallelism that conflicts with Python's forking model, causing
# deadlocks when combined with asyncio/pytest.
# See: https://github.com/huggingface/transformers/issues/5486
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from headroom.memory.adapters.cache import LRUMemoryCache
from headroom.memory.adapters.fts5 import FTS5TextIndex
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.models import Memory, ScopeLevel
from headroom.memory.ports import MemoryFilter, TextFilter, VectorFilter

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        content="User prefers Python over JavaScript",
        user_id="alice",
        session_id="session-123",
        importance=0.8,
        entity_refs=["Python", "JavaScript"],
        metadata={"source": "conversation"},
    )


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return np.random.randn(384).astype(np.float32)


# =============================================================================
# Memory Model Tests
# =============================================================================


class TestMemoryModel:
    """Tests for the Memory dataclass."""

    def test_memory_creation(self):
        """Test basic memory creation."""
        memory = Memory(
            content="Test content",
            user_id="test-user",
        )
        assert memory.content == "Test content"
        assert memory.user_id == "test-user"
        assert memory.id is not None  # Auto-generated UUID
        assert memory.importance == 0.5  # Default

    def test_scope_level_computation(self):
        """Test scope level is correctly computed from hierarchy fields."""
        # USER level - only user_id
        user_mem = Memory(content="test", user_id="alice")
        assert user_mem.scope_level == ScopeLevel.USER

        # SESSION level - user_id + session_id
        session_mem = Memory(content="test", user_id="alice", session_id="sess-1")
        assert session_mem.scope_level == ScopeLevel.SESSION

        # AGENT level - user_id + session_id + agent_id
        agent_mem = Memory(content="test", user_id="alice", session_id="sess-1", agent_id="agent-1")
        assert agent_mem.scope_level == ScopeLevel.AGENT

        # TURN level - all four
        turn_mem = Memory(
            content="test",
            user_id="alice",
            session_id="sess-1",
            agent_id="agent-1",
            turn_id="turn-1",
        )
        assert turn_mem.scope_level == ScopeLevel.TURN

    def test_is_current_property(self):
        """Test is_current property for supersession detection."""
        current = Memory(content="test", user_id="alice")
        assert current.is_current is True

        superseded = Memory(
            content="test",
            user_id="alice",
            valid_until=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        assert superseded.is_current is False

    def test_memory_serialization(self, sample_embedding):
        """Test Memory to_dict and from_dict."""
        memory = Memory(
            content="Test content",
            user_id="alice",
            session_id="sess-1",
            importance=0.9,
            entity_refs=["entity1"],
            metadata={"key": "value"},
            embedding=sample_embedding,
        )

        # Serialize
        data = memory.to_dict()
        assert data["content"] == "Test content"
        assert data["user_id"] == "alice"
        assert data["embedding"] is not None

        # Deserialize
        restored = Memory.from_dict(data)
        assert restored.content == memory.content
        assert restored.user_id == memory.user_id
        assert restored.importance == memory.importance
        assert np.allclose(restored.embedding, memory.embedding)


# =============================================================================
# SQLite Store Tests
# =============================================================================


class TestSQLiteMemoryStore:
    """Tests for SQLiteMemoryStore."""

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a SQLite store for testing."""
        return SQLiteMemoryStore(temp_db_path)

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_memory):
        """Test saving and retrieving a memory."""
        await store.save(sample_memory)

        retrieved = await store.get(sample_memory.id)
        assert retrieved is not None
        assert retrieved.id == sample_memory.id
        assert retrieved.content == sample_memory.content
        assert retrieved.user_id == sample_memory.user_id

    @pytest.mark.asyncio
    async def test_save_batch(self, store):
        """Test batch saving memories."""
        memories = [Memory(content=f"Memory {i}", user_id="alice") for i in range(10)]

        await store.save_batch(memories)

        for memory in memories:
            retrieved = await store.get(memory.id)
            assert retrieved is not None
            assert retrieved.content == memory.content

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_memory):
        """Test deleting a memory."""
        await store.save(sample_memory)

        deleted = await store.delete(sample_memory.id)
        assert deleted is True

        retrieved = await store.get(sample_memory.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_query_by_user(self, store):
        """Test querying memories by user_id."""
        # Create memories for different users
        alice_memories = [Memory(content=f"Alice {i}", user_id="alice") for i in range(5)]
        bob_memories = [Memory(content=f"Bob {i}", user_id="bob") for i in range(3)]

        await store.save_batch(alice_memories + bob_memories)

        # Query Alice's memories
        results = await store.query(MemoryFilter(user_id="alice"))
        assert len(results) == 5

        # Query Bob's memories
        results = await store.query(MemoryFilter(user_id="bob"))
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_by_importance_range(self, store):
        """Test querying memories by importance range."""
        memories = [
            Memory(content="Low importance", user_id="alice", importance=0.2),
            Memory(content="Medium importance", user_id="alice", importance=0.5),
            Memory(content="High importance", user_id="alice", importance=0.9),
        ]

        await store.save_batch(memories)

        # Query high importance only
        results = await store.query(MemoryFilter(user_id="alice", min_importance=0.8))
        assert len(results) == 1
        assert results[0].content == "High importance"

    @pytest.mark.asyncio
    async def test_query_by_importance(self, store):
        """Test querying memories by importance range."""
        memories = [
            Memory(content="Low", user_id="alice", importance=0.3),
            Memory(content="Medium", user_id="alice", importance=0.5),
            Memory(content="High", user_id="alice", importance=0.9),
        ]

        await store.save_batch(memories)

        # Query high importance only
        results = await store.query(MemoryFilter(user_id="alice", min_importance=0.8))
        assert len(results) == 1
        assert results[0].content == "High"

    @pytest.mark.asyncio
    async def test_query_by_scope_level(self, store):
        """Test querying by explicit scope level."""
        memories = [
            Memory(content="User level", user_id="alice"),
            Memory(content="Session level", user_id="alice", session_id="sess-1"),
            Memory(content="Agent level", user_id="alice", session_id="sess-1", agent_id="agent-1"),
        ]

        await store.save_batch(memories)

        # Query only USER level
        results = await store.query(MemoryFilter(user_id="alice", scope_levels=[ScopeLevel.USER]))
        assert len(results) == 1
        assert results[0].content == "User level"

        # Query SESSION level
        results = await store.query(
            MemoryFilter(user_id="alice", scope_levels=[ScopeLevel.SESSION])
        )
        assert len(results) == 1
        assert results[0].content == "Session level"

    @pytest.mark.asyncio
    async def test_supersession(self, store):
        """Test memory supersession."""
        original = Memory(
            content="User prefers Python",
            user_id="alice",
        )
        await store.save(original)

        # Supersede with new preference
        new_memory = Memory(
            content="User now prefers Rust",
            user_id="alice",
        )

        superseded = await store.supersede(original.id, new_memory)

        # New memory should be linked to old
        assert superseded.supersedes == original.id

        # Old memory should be marked as superseded
        old_retrieved = await store.get(original.id)
        assert old_retrieved.superseded_by == superseded.id
        assert old_retrieved.valid_until is not None
        assert old_retrieved.is_current is False

        # New memory should be current
        assert superseded.is_current is True

    @pytest.mark.asyncio
    async def test_get_history(self, store):
        """Test getting supersession chain history."""
        # Create a chain: v1 -> v2 -> v3
        v1 = Memory(content="Version 1", user_id="alice")
        await store.save(v1)

        v2 = Memory(content="Version 2", user_id="alice")
        v2 = await store.supersede(v1.id, v2)

        v3 = Memory(content="Version 3", user_id="alice")
        v3 = await store.supersede(v2.id, v3)

        # Get history from middle
        history = await store.get_history(v2.id, include_future=True)
        assert len(history) == 3
        assert history[0].content == "Version 1"
        assert history[1].content == "Version 2"
        assert history[2].content == "Version 3"

    @pytest.mark.asyncio
    async def test_clear_scope(self, store):
        """Test clearing memories at a scope level."""
        # Create memories at different scopes
        memories = [
            Memory(content="User 1", user_id="alice"),
            Memory(content="User 2", user_id="alice"),
            Memory(content="Session 1", user_id="alice", session_id="sess-1"),
            Memory(content="Other user", user_id="bob"),
        ]
        await store.save_batch(memories)

        # Clear Alice's session
        deleted = await store.clear_scope("alice", session_id="sess-1")
        assert deleted == 1

        # Alice's user-level memories should remain
        remaining = await store.query(MemoryFilter(user_id="alice"))
        assert len(remaining) == 2


# =============================================================================
# LRU Cache Tests
# =============================================================================


class TestLRUMemoryCache:
    """Tests for LRUMemoryCache."""

    @pytest.fixture
    def cache(self):
        """Create a cache for testing."""
        return LRUMemoryCache(max_size=5)

    async def test_set_and_get(self, cache, sample_memory):
        """Test basic cache put and get."""
        await cache.put(sample_memory)

        retrieved = await cache.get(sample_memory.id)
        assert retrieved is not None
        assert retrieved.id == sample_memory.id

    async def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache with 5 memories
        memories = [Memory(content=f"Mem {i}", user_id="alice") for i in range(5)]
        for m in memories:
            await cache.put(m)

        assert cache.size == 5

        # Add one more - should evict the first
        new_mem = Memory(content="New", user_id="alice")
        await cache.put(new_mem)

        assert cache.size == 5
        assert await cache.get(memories[0].id) is None  # First was evicted
        assert await cache.get(new_mem.id) is not None

    async def test_access_updates_lru_order(self, cache):
        """Test that accessing a key moves it to end of LRU."""
        memories = [Memory(content=f"Mem {i}", user_id="alice") for i in range(5)]
        for m in memories:
            await cache.put(m)

        # Access the first memory (makes it most recently used)
        await cache.get(memories[0].id)

        # Add new memory - should evict second (now oldest)
        new_mem = Memory(content="New", user_id="alice")
        await cache.put(new_mem)

        assert await cache.get(memories[0].id) is not None  # Still present
        assert await cache.get(memories[1].id) is None  # Evicted

    async def test_delete(self, cache, sample_memory):
        """Test deleting from cache."""
        await cache.put(sample_memory)
        assert cache.size == 1

        deleted = await cache.invalidate(sample_memory.id)
        assert deleted is True
        assert cache.size == 0
        assert await cache.get(sample_memory.id) is None

    async def test_clear(self, cache):
        """Test clearing the cache."""
        memories = [Memory(content=f"Mem {i}", user_id="alice") for i in range(3)]
        for m in memories:
            await cache.put(m)

        await cache.clear()
        assert cache.size == 0


# =============================================================================
# FTS5 Text Index Tests
# =============================================================================


class TestFTS5TextIndex:
    """Tests for FTS5TextIndex."""

    @pytest.fixture
    def text_index(self, temp_db_path):
        """Create a FTS5 text index for testing."""
        return FTS5TextIndex(temp_db_path)

    def test_index_and_search(self, text_index):
        """Test indexing and searching text."""
        # Index some memories
        text_index.index("mem-1", "User prefers Python programming", {"user_id": "alice"})
        text_index.index("mem-2", "JavaScript is also popular", {"user_id": "alice"})
        text_index.index("mem-3", "Python is great for data science", {"user_id": "alice"})

        # Search for Python
        results = text_index.search("Python", k=10)
        assert len(results) == 2

        # Results should include memory IDs
        result_ids = [r.memory_id for r in results]
        assert "mem-1" in result_ids
        assert "mem-3" in result_ids

    def test_search_with_user_filter(self, text_index):
        """Test searching with user filter."""
        text_index.index("mem-1", "Python programming", {"user_id": "alice"})
        text_index.index("mem-2", "Python scripting", {"user_id": "bob"})

        # Search only Alice's memories
        filter = TextFilter(user_id="alice")
        results = text_index.search("Python", k=10, filter=filter)

        assert len(results) == 1
        assert results[0].memory_id == "mem-1"

    def test_search_with_session_filter(self, text_index):
        """Test searching with session filter."""
        text_index.index("mem-1", "Prefers Python", {"user_id": "alice", "session_id": "sess-1"})
        text_index.index(
            "mem-2", "Python is installed", {"user_id": "alice", "session_id": "sess-2"}
        )

        # Search only session-1
        filter = TextFilter(user_id="alice", session_id="sess-1")
        results = text_index.search("Python", k=10, filter=filter)

        assert len(results) == 1
        assert results[0].memory_id == "mem-1"

    def test_delete(self, text_index):
        """Test deleting from text index."""
        text_index.index("mem-1", "Test content", {"user_id": "alice"})

        deleted = text_index.delete("mem-1")
        assert deleted is True

        results = text_index.search("Test", k=10)
        assert len(results) == 0

    def test_batch_index(self, text_index):
        """Test batch indexing."""
        memory_ids = ["mem-1", "mem-2", "mem-3"]
        texts = ["Python code", "JavaScript code", "Rust code"]
        metadata = [{"user_id": "alice"} for _ in range(3)]

        text_index.index_batch(memory_ids, texts, metadata)

        assert text_index.count() == 3


# =============================================================================
# Memory Config Tests
# =============================================================================


class TestMemoryConfig:
    """Tests for MemoryConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        from headroom.memory.config import MemoryConfig

        config = MemoryConfig()
        assert config.vector_dimension == 384
        assert config.cache_enabled is True
        assert config.auto_bubble is True

    def test_invalid_dimension(self):
        """Test that invalid dimension raises error."""
        from headroom.memory.config import MemoryConfig

        with pytest.raises(ValueError):
            MemoryConfig(vector_dimension=0)

    def test_openai_requires_api_key(self):
        """Test that OpenAI backend requires API key."""
        from headroom.memory.config import EmbedderBackend, MemoryConfig

        with pytest.raises(ValueError, match="openai_api_key"):
            MemoryConfig(embedder_backend=EmbedderBackend.OPENAI)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests that test multiple components together."""

    @pytest.mark.asyncio
    async def test_store_with_embeddings(self, temp_db_path, sample_embedding):
        """Test storing and retrieving memories with embeddings."""
        store = SQLiteMemoryStore(temp_db_path)

        memory = Memory(
            content="Test content",
            user_id="alice",
            embedding=sample_embedding,
        )

        await store.save(memory)

        retrieved = await store.get(memory.id)
        assert retrieved.embedding is not None
        assert np.allclose(retrieved.embedding, sample_embedding)

    @pytest.mark.asyncio
    async def test_temporal_query(self, temp_db_path):
        """Test point-in-time temporal queries."""
        store = SQLiteMemoryStore(temp_db_path)

        # Create a supersession chain
        original = Memory(content="Original preference", user_id="alice")
        await store.save(original)

        # Capture time after original was created (valid_from is set at Memory creation)
        time_when_original_valid = original.valid_from + timedelta(milliseconds=1)

        # Wait a bit for time difference
        await asyncio.sleep(0.01)

        # Supersede
        new_memory = Memory(content="New preference", user_id="alice")
        supersede_time = datetime.now(timezone.utc).replace(tzinfo=None)
        await store.supersede(original.id, new_memory, supersede_time)

        # Query at a point when original was valid (after its valid_from, before supersession)
        # The past_time must be >= original.valid_from and < supersede_time
        results = await store.query(
            MemoryFilter(
                user_id="alice", valid_at=time_when_original_valid, include_superseded=True
            )
        )
        assert len(results) == 1
        assert results[0].content == "Original preference"

        # Query current - should return new
        results = await store.query(MemoryFilter(user_id="alice"))
        assert len(results) == 1
        assert results[0].content == "New preference"

    @pytest.mark.asyncio
    async def test_hierarchical_scope_query(self, temp_db_path):
        """Test hierarchical scope filtering."""
        store = SQLiteMemoryStore(temp_db_path)

        # Create memories at different scopes
        user_mem = Memory(content="User pref", user_id="alice")
        session_mem = Memory(content="Session context", user_id="alice", session_id="sess-1")
        agent_mem = Memory(
            content="Agent decision",
            user_id="alice",
            session_id="sess-1",
            agent_id="agent-1",
        )

        await store.save_batch([user_mem, session_mem, agent_mem])

        # Query user scope only - should get just user_mem
        user_only = await store.query(MemoryFilter(user_id="alice", scope_levels=[ScopeLevel.USER]))
        assert len(user_only) == 1
        assert user_only[0].content == "User pref"

        # Query all scopes for this user
        all_memories = await store.query(MemoryFilter(user_id="alice"))
        assert len(all_memories) == 3

        # Query specific session
        session_memories = await store.query(MemoryFilter(user_id="alice", session_id="sess-1"))
        assert len(session_memories) == 2  # session and agent level


# =============================================================================
# HNSW Vector Index Tests
# =============================================================================

# Check if hnswlib is available (use lazy check to avoid SIGILL on incompatible CPUs)
try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False


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

        # Index all memories
        for memory in memories:
            await vector_index.index(memory)

        # Search with first memory's embedding - should find itself as most similar
        filter = VectorFilter(
            query_vector=memories[0].embedding,
            top_k=3,
            user_id="alice",
        )
        results = await vector_index.search(filter)
        assert len(results) == 3
        assert results[0].memory.id == memories[0].id
        assert results[0].similarity > 0.99  # Should be very close to 1.0

    @pytest.mark.asyncio
    async def test_batch_index(self, vector_index):
        """Test batch indexing."""

        np.random.seed(42)
        memories = []
        for i in range(100):
            embedding = np.random.randn(384).astype(np.float32)
            memory = Memory(
                content=f"Test content {i}",
                user_id="alice",
                embedding=embedding,
            )
            memories.append(memory)

        count = await vector_index.index_batch(memories)

        # Verify count
        assert count == 100
        assert vector_index.size == 100

        # Search should work
        filter = VectorFilter(
            query_vector=memories[50].embedding,
            top_k=5,
            user_id="alice",
        )
        results = await vector_index.search(filter)
        assert len(results) == 5
        assert results[0].memory.id == memories[50].id

    @pytest.mark.asyncio
    async def test_remove(self, vector_index):
        """Test removing from index."""
        np.random.seed(42)
        embedding = np.random.randn(384).astype(np.float32)
        memory = Memory(
            content="Test content",
            user_id="alice",
            embedding=embedding,
        )
        await vector_index.index(memory)

        # HNSW doesn't support true deletion, but marks as deleted
        removed = await vector_index.remove(memory.id)
        assert removed is True

    @pytest.mark.asyncio
    async def test_persistence(self, temp_db_path):
        """Test that index persists to disk."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        save_path = temp_db_path.with_suffix(".hnsw")
        np.random.seed(42)
        embedding = np.random.randn(384).astype(np.float32)
        memory = Memory(
            content="Test content",
            user_id="alice",
            embedding=embedding,
        )

        # Create and populate index
        index1 = HNSWVectorIndex(dimension=384, save_path=save_path)
        await index1.index(memory)
        index1.save_index(save_path)

        # Create new index and load from same path
        index2 = HNSWVectorIndex(dimension=384, save_path=save_path)
        index2.load_index(save_path)
        assert index2.size == 1

        filter = VectorFilter(
            query_vector=embedding,
            top_k=1,
            user_id="alice",
        )
        results = await index2.search(filter)
        assert results[0].memory.id == memory.id


# =============================================================================
# LocalEmbedder Tests
# =============================================================================


class TestLocalEmbedder:
    """Tests for LocalEmbedder (sentence-transformers)."""

    @pytest.fixture
    def embedder(self):
        """Create a local embedder for testing."""
        pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
        from headroom.memory.adapters.embedders import LocalEmbedder

        return LocalEmbedder()

    @pytest.mark.asyncio
    async def test_embed_single(self, embedder):
        """Test embedding a single text."""
        text = "User prefers Python programming"
        embedding = await embedder.embed(text)

        assert embedding is not None
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder):
        """Test embedding multiple texts."""
        texts = [
            "Python programming",
            "JavaScript development",
            "Rust systems programming",
        ]
        embeddings = await embedder.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.shape == (384,)

    @pytest.mark.asyncio
    async def test_similar_texts_have_high_similarity(self, embedder):
        """Test that semantically similar texts have similar embeddings."""
        text1 = "The user prefers Python for data analysis"
        text2 = "Python is the user's preferred language for data science"
        text3 = "The weather is sunny today"

        emb1 = await embedder.embed(text1)
        emb2 = await embedder.embed(text2)
        emb3 = await embedder.embed(text3)

        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Similar texts should have high similarity
        sim_related = cosine_sim(emb1, emb2)
        sim_unrelated = cosine_sim(emb1, emb3)

        assert sim_related > 0.7  # Related texts
        assert sim_unrelated < 0.5  # Unrelated texts
        assert sim_related > sim_unrelated

    def test_dimension_property(self, embedder):
        """Test that dimension property returns correct value."""
        assert embedder.dimension == 384


class TestOnnxLocalEmbedder:
    """Tests for OnnxLocalEmbedder batching behavior."""

    @pytest.mark.asyncio
    async def test_embed_batch_uses_batched_onnx_inference(self):
        """Test that non-empty inputs share ONNX batch inference."""
        from headroom.memory.adapters.embedders import OnnxLocalEmbedder

        class FakeEncoding:
            def __init__(self, ids: list[int], attention_mask: list[int]) -> None:
                self.ids = ids
                self.attention_mask = attention_mask

        class FakeTokenizer:
            def encode_batch(self, texts: list[str]) -> list[FakeEncoding]:
                encodings = []
                for i, text in enumerate(texts, start=1):
                    token = len(text) + i
                    encodings.append(FakeEncoding([token, token + 1, 0], [1, 1, 0]))
                return encodings

        class FakeSession:
            def __init__(self) -> None:
                self.run_calls = 0

            def run(self, _output_names, feeds):
                self.run_calls += 1
                input_ids = feeds["input_ids"]
                batch_size, seq_len = input_ids.shape
                token_embeddings = np.zeros((batch_size, seq_len, 384), dtype=np.float32)
                token_embeddings[:, :, 0] = input_ids
                token_embeddings[:, :, 1] = input_ids * 0.5
                return [token_embeddings]

        embedder = OnnxLocalEmbedder()
        embedder.MAX_BATCH_SIZE = 8
        embedder._session = FakeSession()
        embedder._tokenizer = FakeTokenizer()
        embedder._input_names = ["input_ids", "attention_mask", "token_type_ids"]

        embeddings = await embedder.embed_batch(["alpha", "   ", "beta", "gamma"])

        assert len(embeddings) == 4
        assert embedder._session.run_calls == 1
        assert np.array_equal(embeddings[1], np.zeros(384, dtype=np.float32))
        assert embeddings[0].shape == (384,)
        assert embeddings[2].shape == (384,)
        assert embeddings[3].shape == (384,)
        assert not np.allclose(embeddings[0], 0.0)
        assert not np.allclose(embeddings[2], 0.0)
        assert not np.allclose(embeddings[3], 0.0)

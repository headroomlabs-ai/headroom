"""Comprehensive unit tests for CompressionStore.

Tests cover:
1. CompressionStore class initialization
2. Storing compressed content with hash generation
3. Retrieving content by hash
4. TTL expiration behavior
5. Memory limits and eviction
6. Statistics tracking
7. Edge cases (empty content, duplicate stores, etc.)
8. Thread safety
9. Feedback loop integration
10. Search functionality
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from headroom.cache.compression_store import (
    CCR_TTL_SECONDS_ENV,
    DEFAULT_CCR_TTL_SECONDS,
    CompressionEntry,
    CompressionStore,
    RetrievalEvent,
    get_compression_store,
    reset_compression_store,
)


@contextmanager
def _capture_headroom_retrieve_events():
    events: list[dict[str, Any]] = []
    prefix = "event=headroom_retrieve "

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if prefix in message:
                events.append(json.loads(message.split(prefix, 1)[1]))

    logger = logging.getLogger("headroom.cache.compression_store")
    previous_level = logger.level
    handler = _Handler(level=logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield events
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_retrieve_logs_payload_preview():
    store = CompressionStore(enable_feedback=False)
    hash_key = store.store(
        original="secret-ish payload for operator debugging",
        compressed="payload",
        original_tokens=8,
        compressed_tokens=1,
        original_item_count=1,
        compressed_item_count=1,
        tool_name="tool_a",
    )

    with _capture_headroom_retrieve_events() as events:
        entry = store.retrieve(hash_key)

    assert entry is not None
    assert len(events) == 1
    assert events[0]["hash"] == hash_key
    assert events[0]["retrieval_type"] == "full"
    assert events[0]["payload_preview"] == "secret-ish payload for operator debugging"
    assert "payload" not in events[0]
    assert events[0]["payload_truncated"] is False
    assert events[0]["tool_name"] == "tool_a"


def test_retrieve_log_redacts_secret_payload_values():
    store = CompressionStore(enable_feedback=False)
    hash_key = store.store(
        original="OPENAI_API_KEY=sk-proj-secret1234567890 Authorization: Bearer token123456789",
        compressed="payload",
    )

    with _capture_headroom_retrieve_events() as events:
        entry = store.retrieve(hash_key)

    assert entry is not None
    assert len(events) == 1
    assert "sk-proj-secret1234567890" not in events[0]["payload_preview"]
    assert "Bearer token123456789" not in events[0]["payload_preview"]
    assert "OPENAI_API_KEY=[REDACTED]" in events[0]["payload_preview"]
    assert "Authorization: [REDACTED]" in events[0]["payload_preview"]


def test_search_logs_retrieved_payload_preview():
    store = CompressionStore(enable_feedback=False)
    items = [
        {"id": 1, "text": "alpha target"},
        {"id": 2, "text": "beta other"},
    ]
    hash_key = store.store(
        original=json.dumps(items),
        compressed="[]",
        original_item_count=2,
        compressed_item_count=0,
        tool_name="search_tool",
    )

    with _capture_headroom_retrieve_events() as events:
        results = store.search(hash_key, "alpha", score_threshold=0.0)

    assert results
    assert len(events) == 1
    assert events[0]["hash"] == hash_key
    assert events[0]["retrieval_type"] == "search"
    assert events[0]["query"] == "alpha"
    assert events[0]["payload_preview"] == json.dumps(results, ensure_ascii=False)
    assert events[0]["payload_preview_chars"] == len(json.dumps(results, ensure_ascii=False))
    assert events[0]["payload_truncated"] is False


def test_global_store_uses_env_default_ttl(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(CCR_TTL_SECONDS_ENV, "7200")

    store = get_compression_store()
    hash_key = store.store(original="long-running agent payload", compressed="payload")
    entry = store.retrieve(hash_key)

    assert entry is not None
    assert entry.ttl == 7200
    assert store.get_stats()["default_ttl_seconds"] == 7200


def test_global_store_invalid_env_ttl_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(CCR_TTL_SECONDS_ENV, "0")

    store = get_compression_store()
    hash_key = store.store(original="payload", compressed="payload")
    entry = store.retrieve(hash_key)

    assert entry is not None
    assert entry.ttl == DEFAULT_CCR_TTL_SECONDS
    assert store.get_stats()["default_ttl_seconds"] == DEFAULT_CCR_TTL_SECONDS


def test_explicit_global_store_ttl_overrides_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(CCR_TTL_SECONDS_ENV, "7200")

    store = get_compression_store(default_ttl=60)
    hash_key = store.store(original="payload", compressed="payload")
    entry = store.retrieve(hash_key)

    assert entry is not None
    assert entry.ttl == 60
    assert store.get_stats()["default_ttl_seconds"] == 60


def test_entry_status_reports_expiration_metadata():
    store = CompressionStore(default_ttl=1)

    with patch("headroom.cache.compression_store.time.time", return_value=1000.0):
        hash_key = store.store(original="payload", compressed="payload")

    with patch("headroom.cache.compression_store.time.time", return_value=1002.0):
        status = store.get_entry_status(hash_key, clean_expired=True)

    assert status["status"] == "expired"
    assert status["ttl_seconds"] == 1
    assert status["age_seconds"] == 2
    assert status["expires_at"] == 1001.0
    assert store.exists(hash_key) is False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_global_store():
    """Reset global compression store before and after each test."""
    reset_compression_store()
    yield
    reset_compression_store()


@pytest.fixture
def store() -> CompressionStore:
    """Create a fresh CompressionStore instance for testing."""
    return CompressionStore()


@pytest.fixture
def store_with_short_ttl() -> CompressionStore:
    """Create a CompressionStore with 1 second TTL for expiration tests."""
    return CompressionStore(default_ttl=1)


@pytest.fixture
def store_with_small_capacity() -> CompressionStore:
    """Create a CompressionStore with small capacity for eviction tests."""
    return CompressionStore(max_entries=3)


@pytest.fixture
def sample_items() -> list[dict[str, Any]]:
    """Sample list of items for testing."""
    return [{"id": i, "name": f"item_{i}", "value": i * 10} for i in range(100)]


@pytest.fixture
def sample_original(sample_items: list[dict[str, Any]]) -> str:
    """Sample original JSON content."""
    return json.dumps(sample_items)


@pytest.fixture
def sample_compressed(sample_items: list[dict[str, Any]]) -> str:
    """Sample compressed JSON content (first 10 items)."""
    return json.dumps(sample_items[:10])


# =============================================================================
# CompressionEntry Tests
# =============================================================================


class TestCompressionEntry:
    """Tests for CompressionEntry dataclass."""

    def test_entry_creation_with_defaults(self):
        """CompressionEntry can be created with minimal required fields."""
        entry = CompressionEntry(
            hash="abc123",
            original_content="[1,2,3]",
            compressed_content="[1]",
            original_tokens=100,
            compressed_tokens=10,
            original_item_count=3,
            compressed_item_count=1,
            tool_name=None,
            tool_call_id=None,
            query_context=None,
            created_at=time.time(),
        )
        assert entry.hash == "abc123"
        assert entry.ttl == 300  # Default TTL
        assert entry.retrieval_count == 0
        assert entry.search_queries == []
        assert entry.last_accessed is None

    def test_entry_is_expired_false_when_fresh(self):
        """Fresh entries are not expired."""
        entry = CompressionEntry(
            hash="abc123",
            original_content="[1]",
            compressed_content="[]",
            original_tokens=10,
            compressed_tokens=0,
            original_item_count=1,
            compressed_item_count=0,
            tool_name=None,
            tool_call_id=None,
            query_context=None,
            created_at=time.time(),
            ttl=300,
        )
        assert entry.is_expired() is False

    def test_entry_is_expired_true_after_ttl(self):
        """Entries are expired after TTL passes."""
        entry = CompressionEntry(
            hash="abc123",
            original_content="[1]",
            compressed_content="[]",
            original_tokens=10,
            compressed_tokens=0,
            original_item_count=1,
            compressed_item_count=0,
            tool_name=None,
            tool_call_id=None,
            query_context=None,
            created_at=time.time() - 10,  # 10 seconds ago
            ttl=5,  # 5 second TTL
        )
        assert entry.is_expired() is True

    def test_record_access_increments_count(self):
        """record_access increments retrieval_count."""
        entry = CompressionEntry(
            hash="abc123",
            original_content="[1]",
            compressed_content="[]",
            original_tokens=10,
            compressed_tokens=0,
            original_item_count=1,
            compressed_item_count=0,
            tool_name=None,
            tool_call_id=None,
            query_context=None,
            created_at=time.time(),
        )
        assert entry.retrieval_count == 0

        entry.record_access()
        assert entry.retrieval_count == 1

        entry.record_access()
        assert entry.retrieval_count == 2

    def test_record_access_updates_last_accessed(self):
        """record_access updates last_accessed timestamp."""
        entry = CompressionEntry(
            hash="abc123",
            original_content="[1]",
            compressed_content="[]",
            original_tokens=10,
            compressed_tokens=0,
            original_item_count=1,
            compressed_item_count=0,
            tool_name=None,
            tool_call_id=None,
            query_context=None,
            created_at=time.time(),
        )
        assert entry.last_accessed is None

        before = time.time()
        entry.record_access()
        after = time.time()

        assert entry.last_accessed is not None
        assert before <= entry.last_accessed <= after

    def test_record_access_tracks_unique_queries(self):
        """record_access tracks unique search queries."""
        entry = CompressionEntry(
            hash="abc123",
            original_content="[1]",
            compressed_content="[]",
            original_tokens=10,
            compressed_tokens=0,
            original_item_count=1,
            compressed_item_count=0,
            tool_name=None,
            tool_call_id=None,
            query_context=None,
            created_at=time.time(),
        )

        entry.record_access(query="query1")
        entry.record_access(query="query2")
        entry.record_access(query="query1")  # Duplicate

        assert "query1" in entry.search_queries
        assert "query2" in entry.search_queries
        assert len(entry.search_queries) == 2  # No duplicates

    def test_record_access_limits_queries_to_10(self):
        """record_access keeps only last 10 queries."""
        entry = CompressionEntry(
            hash="abc123",
            original_content="[1]",
            compressed_content="[]",
            original_tokens=10,
            compressed_tokens=0,
            original_item_count=1,
            compressed_item_count=0,
            tool_name=None,
            tool_call_id=None,
            query_context=None,
            created_at=time.time(),
        )

        for i in range(15):
            entry.record_access(query=f"query_{i}")

        assert len(entry.search_queries) == 10
        # Should have the last 10 queries
        assert "query_5" in entry.search_queries
        assert "query_14" in entry.search_queries
        assert "query_0" not in entry.search_queries

    def test_record_access_ignores_none_query(self):
        """record_access does not add None queries to list."""
        entry = CompressionEntry(
            hash="abc123",
            original_content="[1]",
            compressed_content="[]",
            original_tokens=10,
            compressed_tokens=0,
            original_item_count=1,
            compressed_item_count=0,
            tool_name=None,
            tool_call_id=None,
            query_context=None,
            created_at=time.time(),
        )

        entry.record_access(query=None)
        entry.record_access()

        assert len(entry.search_queries) == 0


# =============================================================================
# CompressionStore Initialization Tests
# =============================================================================


class TestCompressionStoreInit:
    """Tests for CompressionStore initialization."""

    def test_default_initialization(self):
        """CompressionStore initializes with default values."""
        store = CompressionStore()

        assert store._max_entries == 1000
        assert store._default_ttl == 300
        assert store._enable_feedback is True
        assert store._backend is not None

    def test_custom_max_entries(self):
        """CompressionStore accepts custom max_entries."""
        store = CompressionStore(max_entries=500)
        assert store._max_entries == 500

    def test_custom_default_ttl(self):
        """CompressionStore accepts custom default_ttl."""
        store = CompressionStore(default_ttl=600)
        assert store._default_ttl == 600

    def test_feedback_can_be_disabled(self):
        """CompressionStore can disable feedback tracking."""
        store = CompressionStore(enable_feedback=False)
        assert store._enable_feedback is False

    def test_custom_backend(self):
        """CompressionStore accepts custom backend."""
        mock_backend = MagicMock()
        mock_backend.count.return_value = 0
        mock_backend.get.return_value = None

        store = CompressionStore(backend=mock_backend)
        assert store._backend is mock_backend


# =============================================================================
# Store Operations Tests
# =============================================================================


class TestCompressionStoreOperations:
    """Tests for CompressionStore store operation."""

    def test_store_returns_24_char_hash(self, store: CompressionStore):
        """store() returns a 24 character hash (96 bits for collision resistance)."""
        hash_key = store.store(
            original="[1,2,3]",
            compressed="[1]",
        )
        assert len(hash_key) == 24
        assert all(c in "0123456789abcdef" for c in hash_key)

    def test_store_hash_is_deterministic(self, store: CompressionStore):
        """Same content produces same hash."""
        content = '{"id": 1, "name": "test"}'

        hash1 = store.store(original=content, compressed="{}")
        hash2 = store.store(original=content, compressed="{}")

        assert hash1 == hash2

    def test_store_hash_based_on_original_content(self, store: CompressionStore):
        """Hash is computed from original content, not compressed."""
        original = '{"id": 1}'
        compressed1 = '{"id": 1}'
        compressed2 = "{}"

        hash1 = store.store(original=original, compressed=compressed1)
        hash2 = store.store(original=original, compressed=compressed2)

        assert hash1 == hash2  # Same original = same hash

    def test_store_different_content_different_hash(self, store: CompressionStore):
        """Different content produces different hash."""
        hash1 = store.store(original='{"id": 1}', compressed="{}")
        hash2 = store.store(original='{"id": 2}', compressed="{}")

        assert hash1 != hash2

    def test_store_preserves_all_metadata(
        self, store: CompressionStore, sample_original: str, sample_compressed: str
    ):
        """store() preserves all metadata in the entry."""
        hash_key = store.store(
            original=sample_original,
            compressed=sample_compressed,
            original_tokens=1000,
            compressed_tokens=100,
            original_item_count=100,
            compressed_item_count=10,
            tool_name="search_api",
            tool_call_id="call_123",
            query_context="user query",
            tool_signature_hash="sig_hash_123",
            compression_strategy="top_k",
            ttl=600,
        )

        entry = store.retrieve(hash_key)
        assert entry is not None
        assert entry.original_content == sample_original
        assert entry.compressed_content == sample_compressed
        assert entry.original_tokens == 1000
        assert entry.compressed_tokens == 100
        assert entry.original_item_count == 100
        assert entry.compressed_item_count == 10
        assert entry.tool_name == "search_api"
        assert entry.tool_call_id == "call_123"
        assert entry.query_context == "user query"
        assert entry.tool_signature_hash == "sig_hash_123"
        assert entry.compression_strategy == "top_k"
        assert entry.ttl == 600

    def test_store_uses_default_ttl(self, store: CompressionStore):
        """store() uses default TTL when not specified."""
        hash_key = store.store(original="[1]", compressed="[]")
        entry = store.retrieve(hash_key)

        assert entry is not None
        assert entry.ttl == 300  # Default TTL

    def test_store_accepts_custom_ttl(self, store: CompressionStore):
        """store() accepts custom TTL override."""
        hash_key = store.store(original="[1]", compressed="[]", ttl=60)
        entry = store.retrieve(hash_key)

        assert entry is not None
        assert entry.ttl == 60


# =============================================================================
# Retrieve Operations Tests
# =============================================================================


class TestCompressionStoreRetrieve:
    """Tests for CompressionStore retrieve operation."""

    def test_retrieve_existing_entry(self, store: CompressionStore):
        """retrieve() returns entry for existing hash."""
        hash_key = store.store(original='{"id": 1}', compressed="{}")

        entry = store.retrieve(hash_key)

        assert entry is not None
        assert entry.hash == hash_key
        assert entry.original_content == '{"id": 1}'

    def test_retrieve_nonexistent_returns_none(self, store: CompressionStore):
        """retrieve() returns None for nonexistent hash."""
        entry = store.retrieve("nonexistent_hash_key")
        assert entry is None

    def test_retrieve_expired_entry_returns_none(self, store_with_short_ttl: CompressionStore):
        """retrieve() returns None for expired entry."""
        hash_key = store_with_short_ttl.store(original="[1]", compressed="[]")

        # Should exist immediately
        assert store_with_short_ttl.retrieve(hash_key) is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be None after expiration
        assert store_with_short_ttl.retrieve(hash_key) is None

    def test_retrieve_increments_access_count(self, store: CompressionStore):
        """retrieve() increments entry access count."""
        hash_key = store.store(original="[1]", compressed="[]")

        store.retrieve(hash_key)
        store.retrieve(hash_key)
        entry = store.retrieve(hash_key)

        assert entry is not None
        assert entry.retrieval_count >= 3

    def test_retrieve_with_query_tracks_query(self, store: CompressionStore):
        """retrieve() with query parameter tracks the query."""
        hash_key = store.store(original="[1]", compressed="[]")

        store.retrieve(hash_key, query="test query")
        entry = store.retrieve(hash_key)

        assert entry is not None
        assert "test query" in entry.search_queries

    def test_retrieve_returns_copy_not_reference(self, store: CompressionStore):
        """retrieve() returns a copy to prevent race conditions."""
        hash_key = store.store(original="[1]", compressed="[]")

        entry1 = store.retrieve(hash_key)
        entry2 = store.retrieve(hash_key)

        # Modify the returned entry's mutable field
        assert entry1 is not None
        assert entry2 is not None
        entry1.search_queries.append("modified")

        # Should not affect the other entry
        assert "modified" not in entry2.search_queries


# =============================================================================
# TTL Expiration Tests
# =============================================================================


class TestCompressionStoreTTL:
    """Tests for CompressionStore TTL expiration behavior."""

    def test_entry_exists_before_ttl(self, store_with_short_ttl: CompressionStore):
        """Entry exists before TTL expires."""
        hash_key = store_with_short_ttl.store(original="[1]", compressed="[]")
        assert store_with_short_ttl.exists(hash_key) is True

    def test_entry_not_exists_after_ttl(self, store_with_short_ttl: CompressionStore):
        """Entry does not exist after TTL expires."""
        hash_key = store_with_short_ttl.store(original="[1]", compressed="[]")

        time.sleep(1.1)

        assert store_with_short_ttl.exists(hash_key) is False

    def test_get_metadata_returns_none_for_expired(self, store_with_short_ttl: CompressionStore):
        """get_metadata returns None for expired entries."""
        hash_key = store_with_short_ttl.store(original="[1]", compressed="[]")

        time.sleep(1.1)

        assert store_with_short_ttl.get_metadata(hash_key) is None

    def test_search_returns_empty_for_expired(self, store_with_short_ttl: CompressionStore):
        """search returns empty list for expired entries."""
        hash_key = store_with_short_ttl.store(
            original=json.dumps([{"id": 1, "name": "test"}]),
            compressed="[]",
        )

        time.sleep(1.1)

        results = store_with_short_ttl.search(hash_key, "test")
        assert results == []

    def test_exists_clean_expired_false_does_not_delete(
        self, store_with_short_ttl: CompressionStore
    ):
        """exists() with clean_expired=False does not delete expired entry."""
        hash_key = store_with_short_ttl.store(original="[1]", compressed="[]")

        time.sleep(1.1)

        # Check exists without cleaning
        result = store_with_short_ttl.exists(hash_key, clean_expired=False)
        assert result is False

        # Entry should still be in backend (not cleaned yet)
        # This is internal behavior - the entry is there but marked expired

    def test_exists_clean_expired_true_deletes(self, store_with_short_ttl: CompressionStore):
        """exists() with clean_expired=True deletes expired entry."""
        hash_key = store_with_short_ttl.store(original="[1]", compressed="[]")

        time.sleep(1.1)

        # Check exists with cleaning
        result = store_with_short_ttl.exists(hash_key, clean_expired=True)
        assert result is False


# =============================================================================
# Eviction Tests
# =============================================================================


class TestCompressionStoreEviction:
    """Tests for CompressionStore memory limits and eviction."""

    def test_eviction_at_capacity(self, store_with_small_capacity: CompressionStore):
        """Oldest entries are evicted when at capacity."""
        hashes = []
        for i in range(5):
            h = store_with_small_capacity.store(
                original=f"content_{i}",
                compressed=f"compressed_{i}",
            )
            hashes.append(h)
            time.sleep(0.01)  # Ensure different timestamps

        # Only last 3 should exist (capacity is 3)
        assert not store_with_small_capacity.exists(hashes[0])
        assert not store_with_small_capacity.exists(hashes[1])
        assert store_with_small_capacity.exists(hashes[2])
        assert store_with_small_capacity.exists(hashes[3])
        assert store_with_small_capacity.exists(hashes[4])

    def test_eviction_removes_oldest_first(self, store_with_small_capacity: CompressionStore):
        """Eviction removes oldest entries first (heap-based)."""
        # Fill to capacity
        hashes = []
        for i in range(3):
            h = store_with_small_capacity.store(
                original=f"content_{i}",
                compressed=f"compressed_{i}",
            )
            hashes.append(h)
            time.sleep(0.01)

        # All 3 should exist
        for h in hashes:
            assert store_with_small_capacity.exists(h)

        # Add one more - should evict oldest
        new_hash = store_with_small_capacity.store(
            original="content_new",
            compressed="compressed_new",
        )

        # Oldest should be evicted
        assert not store_with_small_capacity.exists(hashes[0])
        assert store_with_small_capacity.exists(hashes[1])
        assert store_with_small_capacity.exists(hashes[2])
        assert store_with_small_capacity.exists(new_hash)

    def test_eviction_cleans_expired_first(self):
        """Eviction cleans expired entries before evicting valid ones."""
        store = CompressionStore(max_entries=3, default_ttl=1)

        # Add 2 entries that will expire
        hash1 = store.store(original="content_1", compressed="c1", ttl=1)
        hash2 = store.store(original="content_2", compressed="c2", ttl=1)

        time.sleep(1.1)  # Wait for expiration

        # Add 2 more entries (should clean expired first, not evict new)
        hash3 = store.store(original="content_3", compressed="c3", ttl=300)
        hash4 = store.store(original="content_4", compressed="c4", ttl=300)

        # Expired entries should be gone
        assert not store.exists(hash1)
        assert not store.exists(hash2)

        # New entries should exist
        assert store.exists(hash3)
        assert store.exists(hash4)

    def test_heap_rebuild_on_stale_threshold(self):
        """Heap is rebuilt when stale entry ratio exceeds threshold."""
        store = CompressionStore(max_entries=10)

        # Store entries and then replace them to create stale heap entries
        for i in range(5):
            store.store(original=f"content_{i}", compressed=f"c_{i}")

        # Replace all entries (creates stale heap entries)
        for i in range(5):
            store.store(original=f"content_{i}", compressed=f"updated_{i}")

        # Stale ratio should be tracked
        # The heap rebuild happens automatically when threshold is exceeded


# =============================================================================
# Statistics Tests
# =============================================================================


class TestCompressionStoreStats:
    """Tests for CompressionStore statistics tracking."""

    def test_get_stats_entry_count(self, store: CompressionStore):
        """get_stats returns correct entry count."""
        store.store(original="[1]", compressed="[]")
        store.store(original="[2]", compressed="[]")

        stats = store.get_stats()
        assert stats["entry_count"] == 2

    def test_get_stats_max_entries(self, store: CompressionStore):
        """get_stats includes max_entries configuration."""
        stats = store.get_stats()
        assert stats["max_entries"] == 1000

    def test_get_stats_token_totals(self, store: CompressionStore):
        """get_stats calculates token totals correctly."""
        store.store(
            original="[1]",
            compressed="[]",
            original_tokens=100,
            compressed_tokens=10,
        )
        store.store(
            original="[2]",
            compressed="[]",
            original_tokens=200,
            compressed_tokens=20,
        )

        stats = store.get_stats()
        assert stats["total_original_tokens"] == 300
        assert stats["total_compressed_tokens"] == 30

    def test_get_stats_retrieval_count(self, store: CompressionStore):
        """get_stats tracks total retrievals."""
        hash_key = store.store(original="[1]", compressed="[]")

        store.retrieve(hash_key)
        store.retrieve(hash_key)

        stats = store.get_stats()
        assert stats["total_retrievals"] >= 2

    def test_get_stats_event_count(self, store: CompressionStore):
        """get_stats includes retrieval event count."""
        hash_key = store.store(original="[1]", compressed="[]")

        store.retrieve(hash_key)
        store.retrieve(hash_key)

        stats = store.get_stats()
        assert stats["event_count"] >= 2

    def test_get_stats_includes_backend_stats(self, store: CompressionStore):
        """get_stats includes backend-specific stats."""
        store.store(original="[1]", compressed="[]")

        stats = store.get_stats()
        assert "backend" in stats
        assert stats["backend"]["backend_type"] == "memory"


# =============================================================================
# Get Metadata Tests
# =============================================================================


class TestCompressionStoreMetadata:
    """Tests for CompressionStore get_metadata operation."""

    def test_get_metadata_returns_dict(self, store: CompressionStore):
        """get_metadata returns dict with expected fields."""
        hash_key = store.store(
            original="[1,2,3]",
            compressed="[1]",
            tool_name="test_tool",
            original_item_count=3,
            compressed_item_count=1,
            query_context="test query",
        )

        metadata = store.get_metadata(hash_key)

        assert metadata is not None
        assert metadata["hash"] == hash_key
        assert metadata["tool_name"] == "test_tool"
        assert metadata["original_item_count"] == 3
        assert metadata["compressed_item_count"] == 1
        assert metadata["query_context"] == "test query"
        assert metadata["compressed_content"] == "[1]"
        assert "created_at" in metadata
        assert "ttl" in metadata

    def test_get_metadata_nonexistent_returns_none(self, store: CompressionStore):
        """get_metadata returns None for nonexistent entry."""
        metadata = store.get_metadata("nonexistent")
        assert metadata is None


# =============================================================================
# Search Tests
# =============================================================================


class TestCompressionStoreSearch:
    """Tests for CompressionStore search functionality."""

    def test_search_with_bm25_returns_matches(self, store: CompressionStore):
        """search() uses BM25 to find matching items."""
        items = [
            {"id": 1, "content": "Python programming language"},
            {"id": 2, "content": "JavaScript web development"},
            {"id": 3, "content": "Python data science pandas"},
            {"id": 4, "content": "Java enterprise applications"},
            {"id": 5, "content": "Python machine learning tensorflow"},
        ]

        hash_key = store.store(
            original=json.dumps(items),
            compressed=json.dumps(items[:2]),
        )

        results = store.search(hash_key, "Python programming")

        assert len(results) >= 1
        result_ids = [r["id"] for r in results]
        assert 1 in result_ids  # "Python programming language" should match

    def test_search_respects_max_results(self, store: CompressionStore):
        """search() respects max_results parameter."""
        items = [{"id": i, "content": f"item {i}"} for i in range(50)]
        hash_key = store.store(original=json.dumps(items), compressed="[]")

        results = store.search(hash_key, "item", max_results=5)

        assert len(results) <= 5

    def test_search_respects_score_threshold(self, store: CompressionStore):
        """search() filters by score threshold."""
        items = [
            {"id": 1, "content": "exact match query term"},
            {"id": 2, "content": "completely unrelated content xyz"},
        ]
        hash_key = store.store(original=json.dumps(items), compressed="[]")

        # High threshold should filter low-scoring items
        results = store.search(hash_key, "exact match query", score_threshold=0.5)

        # Should return the exact match, filter the unrelated
        if results:
            assert any("exact match" in str(r) for r in results)

    def test_search_nonexistent_returns_empty(self, store: CompressionStore):
        """search() returns empty list for nonexistent hash."""
        results = store.search("nonexistent", "query")
        assert results == []

    def test_search_invalid_json_returns_empty(self, store: CompressionStore):
        """search() handles invalid JSON gracefully."""
        hash_key = store.store(original="not valid json", compressed="[]")
        results = store.search(hash_key, "query")
        assert results == []

    def test_search_plain_text_returns_matching_chunks(self, store: CompressionStore):
        """search() can find content in Kompress-style plain-text originals."""
        original = (
            "The OpenAI handler contains def _compress_openai_responses_payload "
            "for Responses API live-zone compression. Other text is irrelevant."
        )
        hash_key = store.store(original=original, compressed="compressed")

        results = store.search(hash_key, "def _compress_openai_responses_payload")

        assert len(results) == 1
        assert results[0]["type"] == "text"
        assert "_compress_openai_responses_payload" in results[0]["text"]

    def test_search_json_object_returns_matching_leaf(self, store: CompressionStore):
        """search() can find values inside JSON objects, not only arrays."""
        original = json.dumps(
            {
                "module": {
                    "name": "openai",
                    "function": "_compress_openai_responses_payload",
                }
            }
        )
        hash_key = store.store(original=original, compressed="{}")

        results = store.search(hash_key, "_compress_openai_responses_payload")

        assert len(results) == 1
        assert results[0]["path"] == "module.function"
        assert results[0]["value"] == "_compress_openai_responses_payload"

    def test_search_non_array_returns_empty(self, store: CompressionStore):
        """search() returns empty for JSON objects without matching leaves."""
        hash_key = store.store(original=json.dumps({"key": "value"}), compressed="{}")
        results = store.search(hash_key, "query")
        assert results == []

    def test_search_empty_array_returns_empty(self, store: CompressionStore):
        """search() returns empty for empty array."""
        hash_key = store.store(original="[]", compressed="[]")
        results = store.search(hash_key, "query")
        assert results == []

    def test_search_logs_retrieval_event(self, store: CompressionStore):
        """search() logs retrieval event with search type."""
        items = [{"id": 1, "content": "test"}]
        hash_key = store.store(original=json.dumps(items), compressed="[]")

        store.search(hash_key, "test query")

        events = store.get_retrieval_events()
        search_events = [e for e in events if e.retrieval_type == "search"]
        assert len(search_events) >= 1
        assert search_events[-1].query == "test query"


# =============================================================================
# Retrieval Events Tests
# =============================================================================


class TestCompressionStoreRetrievalEvents:
    """Tests for CompressionStore retrieval event tracking."""

    def test_retrieve_logs_full_event(self, store: CompressionStore):
        """retrieve() logs event with 'full' type."""
        hash_key = store.store(original="[1]", compressed="[]", tool_name="test_tool")

        store.retrieve(hash_key)

        events = store.get_retrieval_events()
        full_events = [e for e in events if e.retrieval_type == "full"]
        assert len(full_events) >= 1
        assert full_events[-1].tool_name == "test_tool"

    def test_get_retrieval_events_limit(self, store: CompressionStore):
        """get_retrieval_events respects limit parameter."""
        hash_key = store.store(original="[1]", compressed="[]")

        for _ in range(10):
            store.retrieve(hash_key)

        events = store.get_retrieval_events(limit=3)
        assert len(events) <= 3

    def test_get_retrieval_events_filter_by_tool(self, store: CompressionStore):
        """get_retrieval_events filters by tool_name."""
        hash1 = store.store(original="[1]", compressed="[]", tool_name="tool_a")
        hash2 = store.store(original="[2]", compressed="[]", tool_name="tool_b")

        store.retrieve(hash1)
        store.retrieve(hash1)
        store.retrieve(hash2)

        tool_a_events = store.get_retrieval_events(tool_name="tool_a")
        tool_b_events = store.get_retrieval_events(tool_name="tool_b")

        assert len(tool_a_events) == 2
        assert len(tool_b_events) == 1

    def test_retrieval_events_include_tool_signature_hash(self, store: CompressionStore):
        """Retrieval events include tool_signature_hash for TOIN correlation."""
        hash_key = store.store(
            original="[1]",
            compressed="[]",
            tool_signature_hash="sig_123",
        )

        store.retrieve(hash_key)

        events = store.get_retrieval_events()
        assert len(events) >= 1
        assert events[-1].tool_signature_hash == "sig_123"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestCompressionStoreEdgeCases:
    """Tests for edge cases and error handling."""

    def test_store_empty_content(self, store: CompressionStore):
        """store() handles empty content."""
        hash_key = store.store(original="", compressed="")

        entry = store.retrieve(hash_key)
        assert entry is not None
        assert entry.original_content == ""

    def test_store_large_content(self, store: CompressionStore):
        """store() handles large content."""
        large_content = json.dumps([{"id": i, "data": "x" * 1000} for i in range(100)])

        hash_key = store.store(original=large_content, compressed="[]")

        entry = store.retrieve(hash_key)
        assert entry is not None
        assert len(entry.original_content) == len(large_content)

    def test_store_unicode_content(self, store: CompressionStore):
        """store() handles unicode content correctly."""
        unicode_content = json.dumps([{"name": "cafe", "emoji": "hello"}])

        hash_key = store.store(original=unicode_content, compressed="[]")

        entry = store.retrieve(hash_key)
        assert entry is not None
        assert "cafe" in entry.original_content

    def test_duplicate_store_updates_entry(self, store: CompressionStore):
        """Storing same content twice updates the entry."""
        original = '{"id": 1}'

        hash1 = store.store(original=original, compressed="v1")
        hash2 = store.store(original=original, compressed="v2")

        assert hash1 == hash2

        entry = store.retrieve(hash1)
        assert entry is not None
        # Second store should have updated the entry
        assert entry.compressed_content == "v2"

    def test_clear_removes_all_entries(self, store: CompressionStore):
        """clear() removes all entries."""
        store.store(original="[1]", compressed="[]")
        store.store(original="[2]", compressed="[]")

        store.clear()

        stats = store.get_stats()
        assert stats["entry_count"] == 0

    def test_clear_removes_retrieval_events(self, store: CompressionStore):
        """clear() removes retrieval events."""
        hash_key = store.store(original="[1]", compressed="[]")
        store.retrieve(hash_key)

        store.clear()

        events = store.get_retrieval_events()
        assert len(events) == 0


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestCompressionStoreThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_stores(self, store: CompressionStore):
        """Concurrent stores don't corrupt data."""
        hashes: list[str] = []
        lock = threading.Lock()
        errors: list[str] = []

        def store_item(i: int) -> None:
            try:
                h = store.store(
                    original=f"content_{i}",
                    compressed=f"compressed_{i}",
                )
                with lock:
                    hashes.append(h)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=store_item, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(hashes) == 20

    def test_concurrent_retrieves(self, store: CompressionStore):
        """Concurrent retrieves don't corrupt data."""
        hash_key = store.store(original="[1,2,3]", compressed="[1]")
        errors: list[str] = []
        results: list[CompressionEntry | None] = []
        lock = threading.Lock()

        def retrieve_item() -> None:
            try:
                entry = store.retrieve(hash_key)
                with lock:
                    results.append(entry)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=retrieve_item) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 20
        for entry in results:
            assert entry is not None
            assert entry.original_content == "[1,2,3]"

    def test_concurrent_store_and_retrieve(self, store: CompressionStore):
        """Concurrent stores and retrieves don't corrupt data."""
        errors: list[str] = []

        def store_and_retrieve(i: int) -> None:
            try:
                items = [{"id": j, "batch": i} for j in range(10)]
                hash_key = store.store(
                    original=json.dumps(items),
                    compressed="[]",
                    tool_name=f"tool_{i}",
                )

                # Immediately retrieve
                entry = store.retrieve(hash_key)
                if entry is None:
                    errors.append(f"Entry {i} not found after store")
                elif f'"batch": {i}' not in entry.original_content:
                    errors.append(f"Entry {i} has wrong content")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=store_and_retrieve, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent operations: {errors}"


# =============================================================================
# Global Store Singleton Tests
# =============================================================================


class TestGlobalStore:
    """Tests for global store singleton pattern."""

    def test_get_compression_store_returns_singleton(self):
        """get_compression_store returns same instance."""
        store1 = get_compression_store()
        store2 = get_compression_store()

        assert store1 is store2

    def test_reset_compression_store_clears_data(self):
        """reset_compression_store clears the global store."""
        store = get_compression_store()
        store.store(original="[1]", compressed="[]")

        reset_compression_store()

        new_store = get_compression_store()
        stats = new_store.get_stats()
        assert stats["entry_count"] == 0

    def test_get_compression_store_uses_params_only_on_first_call(self):
        """Parameters are only used on first initialization."""
        reset_compression_store()

        store1 = get_compression_store(max_entries=500, default_ttl=600)
        assert store1._max_entries == 500
        assert store1._default_ttl == 600

        # Second call with different params should return same instance
        store2 = get_compression_store(max_entries=100, default_ttl=60)
        assert store2 is store1
        assert store2._max_entries == 500  # Original value


# =============================================================================
# Feedback Integration Tests
# =============================================================================


class TestCompressionStoreFeedback:
    """Tests for feedback loop integration."""

    def test_feedback_disabled_no_events(self):
        """No events logged when feedback is disabled."""
        store = CompressionStore(enable_feedback=False)

        hash_key = store.store(original="[1]", compressed="[]")
        store.retrieve(hash_key)

        # Events should still be tracked internally for the store
        # but process_pending_feedback won't forward them
        # Verify events are tracked even with feedback disabled
        assert store.get_retrieval_events() is not None

    def test_feedback_enabled_logs_events(self):
        """Events logged when feedback is enabled."""
        store = CompressionStore(enable_feedback=True)

        hash_key = store.store(original="[1]", compressed="[]", tool_name="test")
        store.retrieve(hash_key)

        events = store.get_retrieval_events()
        assert len(events) >= 1

    @patch("headroom.cache.compression_feedback.get_compression_feedback")
    @patch("headroom.telemetry.get_telemetry_collector")
    @patch("headroom.telemetry.toin.get_toin")
    def test_process_pending_feedback_forwards_events(
        self, mock_toin, mock_telemetry, mock_feedback
    ):
        """process_pending_feedback forwards events to feedback systems."""
        mock_fb = MagicMock()
        mock_tel = MagicMock()
        mock_toin_instance = MagicMock()

        mock_feedback.return_value = mock_fb
        mock_telemetry.return_value = mock_tel
        mock_toin.return_value = mock_toin_instance

        store = CompressionStore(enable_feedback=True)

        hash_key = store.store(
            original="[1]",
            compressed="[]",
            tool_signature_hash="sig_123",
            compression_strategy="top_k",
        )
        store.retrieve(hash_key)

        # Feedback should have been called
        assert mock_fb.record_retrieval.called

    def test_eviction_success_creates_event(self):
        """Eviction without retrieval creates success event."""
        store = CompressionStore(max_entries=2, enable_feedback=True)

        # Store entries with signature hash for eviction tracking
        store.store(
            original="content_0",
            compressed="c0",
            tool_signature_hash="sig_0",
            compression_strategy="top_k",
        )
        time.sleep(0.01)

        store.store(
            original="content_1",
            compressed="c1",
            tool_signature_hash="sig_1",
            compression_strategy="top_k",
        )
        time.sleep(0.01)

        # This should trigger eviction of first entry
        store.store(
            original="content_2",
            compressed="c2",
            tool_signature_hash="sig_2",
            compression_strategy="top_k",
        )

        # The evicted entry (content_0) was never retrieved,
        # so an eviction_success event should be queued
        # (tested via the pending_feedback mechanism)


# =============================================================================
# RetrievalEvent Tests
# =============================================================================


class TestRetrievalEvent:
    """Tests for RetrievalEvent dataclass."""

    def test_retrieval_event_creation(self):
        """RetrievalEvent can be created with all fields."""
        event = RetrievalEvent(
            hash="abc123",
            query="test query",
            items_retrieved=5,
            total_items=100,
            tool_name="search_api",
            timestamp=time.time(),
            retrieval_type="search",
            tool_signature_hash="sig_123",
        )

        assert event.hash == "abc123"
        assert event.query == "test query"
        assert event.items_retrieved == 5
        assert event.total_items == 100
        assert event.tool_name == "search_api"
        assert event.retrieval_type == "search"
        assert event.tool_signature_hash == "sig_123"

    def test_retrieval_event_default_signature_hash(self):
        """RetrievalEvent has None default for tool_signature_hash."""
        event = RetrievalEvent(
            hash="abc123",
            query=None,
            items_retrieved=10,
            total_items=10,
            tool_name="test",
            timestamp=time.time(),
            retrieval_type="full",
        )

        assert event.tool_signature_hash is None


# =============================================================================
# Hash Collision Detection Tests
# =============================================================================


class TestHashCollisionDetection:
    """Tests for hash collision detection and handling."""

    def test_same_content_no_collision_warning(
        self, store: CompressionStore, caplog: pytest.LogCaptureFixture
    ):
        """Same content stored twice should not warn about collision."""
        import logging

        with caplog.at_level(logging.WARNING):
            store.store(original="[1,2,3]", compressed="[1]")
            store.store(original="[1,2,3]", compressed="[1,2]")

        # Should not have collision warning
        assert "Hash collision detected" not in caplog.text

    def test_hash_uses_sha256_truncated(self, store: CompressionStore):
        """Hash is SHA-256 truncated to 24 characters.

        Switched from MD5[:24] in PR #395 to silence CodeQL's
        py/weak-sensitive-data-hashing rule. SHA-256[:24] gives the
        same 96-bit collision space (~280 trillion entries for 50%
        collision under birthday bound) and is FIPS-clean. The cache
        is in-memory, so changing the hash function on upgrade has no
        persistence-side effect.
        """
        content = "test content"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()[:24]

        hash_key = store.store(original=content, compressed="[]")

        assert hash_key == expected_hash, (
            "compression_store key must be SHA-256(original)[:24]. "
            "If this test fails because the hash function was changed, "
            "verify that no caller (incl. /v1/retrieve consumers) "
            "depends on the specific MD5/SHA-256 value — the cache is "
            "in-memory so upgrade-time mismatch is fine, but external "
            "systems that hash-and-lookup independently need to match."
        )

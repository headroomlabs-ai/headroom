"""Tests for Compress-Cache-Retrieve (CCR) architecture.

These tests verify that:
1. CompressionStore correctly caches compressed content
2. SmartCrusher integrates with CompressionStore
3. Retrieval works correctly (full and search)
4. Feedback tracking works
5. TTL expiration works
"""

import json
import time

import pytest

from headroom.cache.compression_store import (
    CompressionStore,
    get_compression_store,
    reset_compression_store,
)
from headroom.config import CCRConfig
from headroom.transforms.smart_crusher import (
    SmartCrusherConfig,
    smart_crush_tool_output,
)


class TestCompressionStore:
    """Test CompressionStore functionality."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_compression_store()
        yield
        reset_compression_store()

    def test_store_and_retrieve(self):
        """Basic store and retrieve flow."""
        store = CompressionStore()

        original = json.dumps([{"id": i} for i in range(100)])
        compressed = json.dumps([{"id": i} for i in range(10)])

        hash_key = store.store(
            original=original,
            compressed=compressed,
            original_tokens=1000,
            compressed_tokens=100,
            original_item_count=100,
            compressed_item_count=10,
        )

        assert (
            len(hash_key) == 24
        )  # SHA256 truncated to 24 chars (96 bits for collision resistance)

        entry = store.retrieve(hash_key)
        assert entry is not None
        assert entry.original_content == original
        assert entry.compressed_content == compressed
        assert entry.original_tokens == 1000
        assert entry.compressed_tokens == 100

    def test_retrieve_nonexistent(self):
        """Retrieve returns None for nonexistent hash."""
        store = CompressionStore()
        entry = store.retrieve("nonexistent1234")
        assert entry is None

    def test_ttl_expiration(self):
        """Entries expire after TTL."""
        store = CompressionStore(default_ttl=1)  # 1 second TTL

        hash_key = store.store(
            original="[1,2,3]",
            compressed="[1]",
            ttl=1,
        )

        # Should exist immediately
        assert store.exists(hash_key)

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert not store.exists(hash_key)
        entry = store.retrieve(hash_key)
        assert entry is None

    def test_eviction_at_capacity(self):
        """Oldest entries evicted when at capacity."""
        store = CompressionStore(max_entries=3)

        hashes = []
        for i in range(5):
            h = store.store(
                original=f"original_{i}",
                compressed=f"compressed_{i}",
            )
            hashes.append(h)
            time.sleep(0.01)  # Ensure different timestamps

        # Only last 3 should exist
        assert not store.exists(hashes[0])
        assert not store.exists(hashes[1])
        assert store.exists(hashes[2])
        assert store.exists(hashes[3])
        assert store.exists(hashes[4])

    def test_search_with_bm25(self):
        """Search within cached content using BM25."""
        store = CompressionStore()

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
            original_item_count=5,
            compressed_item_count=2,
        )

        # Search for Python items
        results = store.search(hash_key, "Python programming")

        assert len(results) >= 1
        # Should prioritize Python items
        result_ids = [r["id"] for r in results]
        assert 1 in result_ids  # "Python programming language"

    def test_retrieval_tracking(self):
        """Retrieval events are tracked for feedback."""
        store = CompressionStore(enable_feedback=True)

        hash_key = store.store(
            original="[1,2,3]",
            compressed="[1]",
            tool_name="test_tool",
        )

        # Retrieve multiple times
        store.retrieve(hash_key)
        store.retrieve(hash_key, query="test query")
        store.search(hash_key, "another query")

        events = store.get_retrieval_events(limit=10)
        assert len(events) >= 2

        # Check event details
        assert any(e.retrieval_type == "full" for e in events)
        assert any(e.retrieval_type == "search" for e in events)

    def test_access_tracking_on_entry(self):
        """Entry tracks access count and queries."""
        store = CompressionStore()

        hash_key = store.store(
            original=json.dumps([{"id": i} for i in range(10)]),
            compressed="[]",
        )

        # Access multiple times with queries
        store.retrieve(hash_key, query="first query")
        store.retrieve(hash_key, query="second query")
        store.retrieve(hash_key, query="first query")  # Duplicate

        entry = store.retrieve(hash_key)
        assert entry.retrieval_count >= 3
        assert "first query" in entry.search_queries
        assert "second query" in entry.search_queries

    def test_stats(self):
        """Store statistics are accurate."""
        store = CompressionStore()

        store.store(
            original="x" * 100,
            compressed="x" * 10,
            original_tokens=100,
            compressed_tokens=10,
        )
        store.store(
            original="y" * 200,
            compressed="y" * 20,
            original_tokens=200,
            compressed_tokens=20,
        )

        stats = store.get_stats()
        assert stats["entry_count"] == 2
        assert stats["total_original_tokens"] == 300
        assert stats["total_compressed_tokens"] == 30

    def test_global_store_singleton(self):
        """Global store uses singleton pattern."""
        reset_compression_store()

        store1 = get_compression_store()
        store2 = get_compression_store()

        assert store1 is store2

    def test_thread_safety(self):
        """Store is thread-safe."""
        import threading

        store = CompressionStore()
        hashes = []
        lock = threading.Lock()

        def store_item(i):
            h = store.store(
                original=f"original_{i}",
                compressed=f"compressed_{i}",
            )
            with lock:
                hashes.append(h)

        threads = [threading.Thread(target=store_item, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(hashes) == 10
        for h in hashes:
            assert store.exists(h)


class TestCCRConfig:
    """Test CCR configuration options."""

    def test_default_config(self):
        """Default CCR config values."""
        config = CCRConfig()
        assert config.enabled is True
        assert config.store_max_entries == 1000
        assert config.store_ttl_seconds == 1800  # session-scale (was 300)
        assert config.inject_retrieval_marker is True
        assert config.feedback_enabled is True
        assert config.min_items_to_cache == 20


class TestCCRFeedbackLoop:
    """Test CCR feedback tracking for learning."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_compression_store()
        yield
        reset_compression_store()

    def test_retrieval_events_logged(self):
        """Retrieval events are logged for feedback."""
        store = CompressionStore(enable_feedback=True)

        items = [{"id": i, "data": f"item_{i}"} for i in range(50)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed=json.dumps(items[:10]),
            original_item_count=50,
            compressed_item_count=10,
            tool_name="search_api",
        )

        # Simulate retrievals
        store.retrieve(hash_key)
        store.search(hash_key, "specific query")
        store.search(hash_key, "another query")

        events = store.get_retrieval_events(limit=10)

        # Should have logged all retrievals
        assert len(events) >= 3

        # Check event types
        full_events = [e for e in events if e.retrieval_type == "full"]
        search_events = [e for e in events if e.retrieval_type == "search"]

        assert len(full_events) >= 1
        assert len(search_events) >= 2

    def test_tool_name_in_events(self):
        """Tool name is preserved in retrieval events."""
        store = CompressionStore(enable_feedback=True)

        hash_key = store.store(
            original="[1,2,3]",
            compressed="[1]",
            tool_name="github_search",
        )

        store.retrieve(hash_key)

        events = store.get_retrieval_events(tool_name="github_search")
        assert len(events) >= 1
        assert all(e.tool_name == "github_search" for e in events)

    def test_event_filtering_by_tool(self):
        """Events can be filtered by tool name."""
        store = CompressionStore(enable_feedback=True)

        hash1 = store.store(
            original="[1]",
            compressed="[1]",
            tool_name="tool_a",
        )
        hash2 = store.store(
            original="[2]",
            compressed="[2]",
            tool_name="tool_b",
        )

        store.retrieve(hash1)
        store.retrieve(hash1)
        store.retrieve(hash2)

        tool_a_events = store.get_retrieval_events(tool_name="tool_a")
        tool_b_events = store.get_retrieval_events(tool_name="tool_b")

        assert len(tool_a_events) == 2
        assert len(tool_b_events) == 1


class TestCCREdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset global store before each test."""
        reset_compression_store()
        yield
        reset_compression_store()

    def test_search_expired_entry(self):
        """Search on expired entry returns empty."""
        store = CompressionStore(default_ttl=1)

        hash_key = store.store(
            original=json.dumps([{"id": 1}]),
            compressed="[]",
        )

        time.sleep(1.1)

        results = store.search(hash_key, "query")
        assert results == []

    def test_search_invalid_json(self):
        """Search handles invalid JSON gracefully."""
        store = CompressionStore()

        hash_key = store.store(
            original="not valid json",
            compressed="[]",
        )

        results = store.search(hash_key, "query")
        assert results == []

    def test_search_non_array(self):
        """Search handles non-array content gracefully."""
        store = CompressionStore()

        hash_key = store.store(
            original=json.dumps({"key": "value"}),
            compressed="{}",
        )

        results = store.search(hash_key, "query")
        assert results == []

    def test_empty_query_search(self):
        """Search with empty query returns empty or all."""
        store = CompressionStore()

        items = [{"id": i} for i in range(10)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
        )

        # Empty query should return something (BM25 handles this)
        results = store.search(hash_key, "")
        # Behavior depends on BM25 implementation
        assert isinstance(results, list)

    def test_ccr_disabled_no_caching(self):
        """When CCR disabled, no caching occurs."""
        reset_compression_store()

        items = [{"id": i, "score": 100 - i} for i in range(100)]
        content = json.dumps(items)

        config = SmartCrusherConfig(max_items_after_crush=15)
        ccr_config = CCRConfig(enabled=False)  # Disabled

        smart_crush_tool_output(content, config, ccr_config)

        store = get_compression_store()
        stats = store.get_stats()
        assert stats["entry_count"] == 0

    def test_concurrent_store_and_retrieve(self):
        """Concurrent operations don't corrupt data."""
        import threading

        store = CompressionStore()
        errors = []

        def store_and_retrieve(i):
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

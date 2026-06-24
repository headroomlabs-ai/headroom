"""Tests for CompressionStore storage backends.

These tests define the contract that all backends must fulfill.
Each backend implementation should pass all these tests.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

from headroom.cache.backends import CompressionStoreBackend, InMemoryBackend
from headroom.cache.compression_store import CompressionEntry

if TYPE_CHECKING:
    from collections.abc import Callable


def make_entry(
    hash_key: str = "test_hash",
    original: str = "original content",
    compressed: str = "compressed",
    original_tokens: int = 100,
    compressed_tokens: int = 10,
) -> CompressionEntry:
    """Create a test CompressionEntry."""
    return CompressionEntry(
        hash=hash_key,
        original_content=original,
        compressed_content=compressed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        original_item_count=5,
        compressed_item_count=2,
        tool_name="test_tool",
        tool_call_id="call_123",
        query_context="test query",
        created_at=time.time(),
        ttl=300,
    )


class TestCompressionStoreBackendProtocol:
    """Test that InMemoryBackend implements the protocol correctly."""

    def test_inmemory_backend_implements_protocol(self) -> None:
        """InMemoryBackend should implement CompressionStoreBackend protocol."""
        backend = InMemoryBackend()
        assert isinstance(backend, CompressionStoreBackend)

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol should be runtime checkable."""

        class NotABackend:
            pass

        assert not isinstance(NotABackend(), CompressionStoreBackend)


class TestInMemoryBackend:
    """Test suite for InMemoryBackend.

    These tests define the contract for all backends.
    """

    @pytest.fixture
    def backend(self) -> InMemoryBackend:
        """Create a fresh backend for each test."""
        return InMemoryBackend()

    # --- Basic CRUD operations ---

    def test_get_returns_none_for_missing_key(self, backend: InMemoryBackend) -> None:
        """get() should return None for keys that don't exist."""
        assert backend.get("nonexistent") is None

    def test_set_and_get_roundtrip(self, backend: InMemoryBackend) -> None:
        """set() followed by get() should return the same entry."""
        entry = make_entry(hash_key="abc123")
        backend.set("abc123", entry)

        retrieved = backend.get("abc123")
        assert retrieved is not None
        assert retrieved.hash == "abc123"
        assert retrieved.original_content == "original content"
        assert retrieved.compressed_content == "compressed"
        assert retrieved.original_tokens == 100
        assert retrieved.compressed_tokens == 10

    def test_set_overwrites_existing(self, backend: InMemoryBackend) -> None:
        """set() should overwrite existing entries with the same key."""
        entry1 = make_entry(hash_key="abc123", original="first")
        entry2 = make_entry(hash_key="abc123", original="second")

        backend.set("abc123", entry1)
        backend.set("abc123", entry2)

        retrieved = backend.get("abc123")
        assert retrieved is not None
        assert retrieved.original_content == "second"

    def test_delete_removes_entry(self, backend: InMemoryBackend) -> None:
        """delete() should remove the entry and return True."""
        entry = make_entry(hash_key="abc123")
        backend.set("abc123", entry)

        result = backend.delete("abc123")
        assert result is True
        assert backend.get("abc123") is None

    def test_delete_returns_false_for_missing(self, backend: InMemoryBackend) -> None:
        """delete() should return False for keys that don't exist."""
        result = backend.delete("nonexistent")
        assert result is False

    def test_exists_returns_true_for_stored_entry(self, backend: InMemoryBackend) -> None:
        """exists() should return True for stored entries."""
        entry = make_entry(hash_key="abc123")
        backend.set("abc123", entry)

        assert backend.exists("abc123") is True

    def test_exists_returns_false_for_missing(self, backend: InMemoryBackend) -> None:
        """exists() should return False for missing entries."""
        assert backend.exists("nonexistent") is False

    def test_clear_removes_all_entries(self, backend: InMemoryBackend) -> None:
        """clear() should remove all entries."""
        backend.set("key1", make_entry(hash_key="key1"))
        backend.set("key2", make_entry(hash_key="key2"))
        backend.set("key3", make_entry(hash_key="key3"))

        backend.clear()

        assert backend.count() == 0
        assert backend.get("key1") is None
        assert backend.get("key2") is None
        assert backend.get("key3") is None

    # --- Enumeration methods ---

    def test_count_returns_zero_for_empty(self, backend: InMemoryBackend) -> None:
        """count() should return 0 for empty backend."""
        assert backend.count() == 0

    def test_count_returns_correct_count(self, backend: InMemoryBackend) -> None:
        """count() should return the number of entries."""
        backend.set("key1", make_entry(hash_key="key1"))
        backend.set("key2", make_entry(hash_key="key2"))
        backend.set("key3", make_entry(hash_key="key3"))

        assert backend.count() == 3

    def test_keys_returns_empty_list_for_empty(self, backend: InMemoryBackend) -> None:
        """keys() should return empty list for empty backend."""
        assert backend.keys() == []

    def test_keys_returns_all_keys(self, backend: InMemoryBackend) -> None:
        """keys() should return all stored keys."""
        backend.set("key1", make_entry(hash_key="key1"))
        backend.set("key2", make_entry(hash_key="key2"))
        backend.set("key3", make_entry(hash_key="key3"))

        keys = backend.keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_items_returns_empty_list_for_empty(self, backend: InMemoryBackend) -> None:
        """items() should return empty list for empty backend."""
        assert backend.items() == []

    def test_items_returns_all_entries(self, backend: InMemoryBackend) -> None:
        """items() should return all (key, entry) pairs."""
        entry1 = make_entry(hash_key="key1", original="content1")
        entry2 = make_entry(hash_key="key2", original="content2")

        backend.set("key1", entry1)
        backend.set("key2", entry2)

        items = backend.items()
        assert len(items) == 2

        items_dict = dict(items)
        assert items_dict["key1"].original_content == "content1"
        assert items_dict["key2"].original_content == "content2"

    # --- Statistics ---

    def test_get_stats_returns_required_fields(self, backend: InMemoryBackend) -> None:
        """get_stats() should return required fields."""
        stats = backend.get_stats()

        assert "backend_type" in stats
        assert "entry_count" in stats
        assert stats["backend_type"] == "memory"
        assert stats["entry_count"] == 0

    def test_get_stats_entry_count_accurate(self, backend: InMemoryBackend) -> None:
        """get_stats() entry_count should match actual count."""
        backend.set("key1", make_entry(hash_key="key1"))
        backend.set("key2", make_entry(hash_key="key2"))

        stats = backend.get_stats()
        assert stats["entry_count"] == 2

    def test_get_stats_bytes_used_increases(self, backend: InMemoryBackend) -> None:
        """get_stats() bytes_used should increase with entries."""
        stats_empty = backend.get_stats()

        backend.set(
            "key1",
            make_entry(hash_key="key1", original="x" * 1000),
        )
        stats_one = backend.get_stats()

        backend.set(
            "key2",
            make_entry(hash_key="key2", original="y" * 1000),
        )
        stats_two = backend.get_stats()

        assert stats_one["bytes_used"] > stats_empty["bytes_used"]
        assert stats_two["bytes_used"] > stats_one["bytes_used"]

    # --- Thread safety ---

    def test_concurrent_set_operations(self, backend: InMemoryBackend) -> None:
        """Backend should handle concurrent set operations safely."""
        num_threads = 10
        entries_per_thread = 100
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(entries_per_thread):
                    key = f"thread{thread_id}_entry{i}"
                    entry = make_entry(hash_key=key)
                    backend.set(key, entry)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert backend.count() == num_threads * entries_per_thread

    def test_concurrent_get_set_delete(self, backend: InMemoryBackend) -> None:
        """Backend should handle mixed concurrent operations safely."""
        num_iterations = 100
        errors: list[Exception] = []

        # Pre-populate some entries
        for i in range(50):
            backend.set(f"key{i}", make_entry(hash_key=f"key{i}"))

        def setter() -> None:
            try:
                for i in range(num_iterations):
                    backend.set(f"new_key{i}", make_entry(hash_key=f"new_key{i}"))
            except Exception as e:
                errors.append(e)

        def getter() -> None:
            try:
                for i in range(num_iterations):
                    backend.get(f"key{i % 50}")
            except Exception as e:
                errors.append(e)

        def deleter() -> None:
            try:
                for i in range(num_iterations):
                    backend.delete(f"key{i % 50}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=setter),
            threading.Thread(target=setter),
            threading.Thread(target=getter),
            threading.Thread(target=getter),
            threading.Thread(target=deleter),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    # --- Edge cases ---

    def test_empty_string_key(self, backend: InMemoryBackend) -> None:
        """Backend should handle empty string as key."""
        entry = make_entry(hash_key="")
        backend.set("", entry)

        retrieved = backend.get("")
        assert retrieved is not None
        assert retrieved.hash == ""

    def test_unicode_content(self, backend: InMemoryBackend) -> None:
        """Backend should handle unicode content correctly."""
        entry = make_entry(
            hash_key="unicode",
            original="日本語テスト 🎉 émojis",
            compressed="日本語",
        )
        backend.set("unicode", entry)

        retrieved = backend.get("unicode")
        assert retrieved is not None
        assert retrieved.original_content == "日本語テスト 🎉 émojis"
        assert retrieved.compressed_content == "日本語"

    def test_large_content(self, backend: InMemoryBackend) -> None:
        """Backend should handle large content."""
        large_content = "x" * 10_000_000  # 10MB
        entry = make_entry(hash_key="large", original=large_content)
        backend.set("large", entry)

        retrieved = backend.get("large")
        assert retrieved is not None
        assert len(retrieved.original_content) == 10_000_000


# --- Parameterized tests for all backend implementations ---


def all_backends() -> list[Callable[[], CompressionStoreBackend]]:
    """Return factory functions for all backend implementations."""
    return [
        InMemoryBackend,
        # Add more backends here as they're implemented:
        # MongoDBBackend,
        # RedisBackend,
    ]


@pytest.mark.parametrize("backend_factory", all_backends())
class TestBackendContract:
    """Contract tests that ALL backends must pass.

    These tests are parameterized to run against every backend implementation.
    Add new backends to all_backends() to include them in these tests.
    """

    def test_implements_protocol(
        self, backend_factory: Callable[[], CompressionStoreBackend]
    ) -> None:
        """All backends must implement CompressionStoreBackend protocol."""
        backend = backend_factory()
        assert isinstance(backend, CompressionStoreBackend)

    def test_basic_crud_cycle(self, backend_factory: Callable[[], CompressionStoreBackend]) -> None:
        """All backends must support basic CRUD operations."""
        backend = backend_factory()

        # Create
        entry = make_entry(hash_key="test")
        backend.set("test", entry)
        assert backend.exists("test")

        # Read
        retrieved = backend.get("test")
        assert retrieved is not None
        assert retrieved.original_content == entry.original_content

        # Update (overwrite)
        entry2 = make_entry(hash_key="test", original="updated")
        backend.set("test", entry2)
        retrieved2 = backend.get("test")
        assert retrieved2 is not None
        assert retrieved2.original_content == "updated"

        # Delete
        assert backend.delete("test") is True
        assert backend.exists("test") is False
        assert backend.get("test") is None

    def test_clear_works(self, backend_factory: Callable[[], CompressionStoreBackend]) -> None:
        """All backends must support clear()."""
        backend = backend_factory()

        backend.set("key1", make_entry(hash_key="key1"))
        backend.set("key2", make_entry(hash_key="key2"))
        assert backend.count() == 2

        backend.clear()
        assert backend.count() == 0

    def test_stats_has_required_fields(
        self, backend_factory: Callable[[], CompressionStoreBackend]
    ) -> None:
        """All backends must return required stats fields."""
        backend = backend_factory()
        stats = backend.get_stats()

        assert "backend_type" in stats
        assert "entry_count" in stats
        assert isinstance(stats["backend_type"], str)
        assert isinstance(stats["entry_count"], int)

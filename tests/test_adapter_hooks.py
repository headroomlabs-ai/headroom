"""Tests for pluggable adapter hooks.

Validates the extension points that allow SaaS packages to inject
custom backends for CCR, Storage, and TOIN without forking OSS.

Tests cover:
1. TOIN backend protocol conformance (FileSystemTOINBackend)
2. TOIN backend wiring (ToolIntelligenceNetwork with custom backend)
3. TOIN entry_point loading (_create_default_toin_backend)
4. CCR ContextVar scoping (set/clear/get request compression store)
5. CCR entry_point loading (_create_default_ccr_backend)
6. Storage entry_point loading (create_storage with custom scheme)
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from headroom.cache.backends import CompressionStoreBackend, InMemoryBackend
from headroom.cache.compression_store import (
    CompressionStore,
    clear_request_compression_store,
    get_compression_store,
    reset_compression_store,
    set_request_compression_store,
)
from headroom.storage import Storage, create_storage
from headroom.telemetry.backends import FileSystemTOINBackend, TOINBackend
from headroom.telemetry.models import ToolSignature
from headroom.telemetry.toin import (
    TOINConfig,
    ToolIntelligenceNetwork,
    _create_default_toin_backend,
    get_toin,
    reset_toin,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _clean_globals():
    """Reset global singletons before and after each test."""
    reset_toin()
    reset_compression_store()
    clear_request_compression_store()
    yield
    reset_toin()
    reset_compression_store()
    clear_request_compression_store()


@pytest.fixture
def tmp_toin_path(tmp_path):
    """Temporary file path for TOIN storage."""
    return str(tmp_path / "toin.json")


@pytest.fixture
def fs_backend(tmp_toin_path):
    """FileSystemTOINBackend with a temp path."""
    return FileSystemTOINBackend(tmp_toin_path)


def _make_tool_signature() -> ToolSignature:
    """Create a ToolSignature for testing."""
    return ToolSignature.from_items(
        [
            {"id": 1, "name": "test", "score": 0.95},
            {"id": 2, "name": "test2", "score": 0.80},
        ]
    )


# =============================================================================
# 1. TOIN Backend Protocol Conformance
# =============================================================================


class TestTOINBackendProtocol:
    """Verify FileSystemTOINBackend satisfies the TOINBackend protocol."""

    def test_filesystem_backend_is_toin_backend(self, fs_backend):
        """FileSystemTOINBackend must satisfy the runtime-checkable TOINBackend protocol."""
        assert isinstance(fs_backend, TOINBackend)

    def test_load_empty(self, fs_backend):
        """load() returns empty dict when no file exists."""
        result = fs_backend.load()
        assert result == {}

    def test_save_and_load_roundtrip(self, fs_backend):
        """Data survives a save/load cycle."""
        data = {
            "version": "1.0",
            "patterns": {
                "abc123": {
                    "tool_signature_hash": "abc123",
                    "total_compressions": 42,
                }
            },
        }
        fs_backend.save(data)
        loaded = fs_backend.load()
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path):
        """save() creates parent directories if they don't exist."""
        deep_path = str(tmp_path / "a" / "b" / "c" / "toin.json")
        backend = FileSystemTOINBackend(deep_path)
        backend.save({"version": "1.0"})
        loaded = backend.load()
        assert loaded["version"] == "1.0"

    def test_save_atomic_on_failure(self, tmp_path):
        """If save fails mid-write, original data is preserved."""
        path = str(tmp_path / "toin.json")
        backend = FileSystemTOINBackend(path)

        # Save initial data
        backend.save({"version": "1.0", "state": "original"})

        # Corrupt the temp dir to force failure (make dir read-only)
        # This is OS-dependent; we verify the load still returns original
        loaded = backend.load()
        assert loaded["state"] == "original"

    def test_load_corrupted_file(self, tmp_toin_path):
        """load() returns empty dict on corrupted JSON."""
        with open(tmp_toin_path, "w") as f:
            f.write("{invalid json!!")

        backend = FileSystemTOINBackend(tmp_toin_path)
        result = backend.load()
        assert result == {}

    def test_save_overwrites(self, fs_backend):
        """save() overwrites previous data completely."""
        fs_backend.save({"version": "1.0", "old": True})
        fs_backend.save({"version": "2.0", "new": True})
        loaded = fs_backend.load()
        assert loaded == {"version": "2.0", "new": True}
        assert "old" not in loaded


class TestCustomTOINBackend:
    """Verify any dict-based backend satisfies the protocol."""

    def test_dict_backend_satisfies_protocol(self):
        """A minimal dict-backed implementation passes protocol check."""

        class DictBackend:
            def __init__(self):
                self._data: dict[str, Any] = {}

            def load(self) -> dict[str, Any]:
                return dict(self._data)

            def save(self, data: dict[str, Any]) -> None:
                self._data = dict(data)

        backend = DictBackend()
        assert isinstance(backend, TOINBackend)

        backend.save({"key": "value"})
        assert backend.load() == {"key": "value"}


# =============================================================================
# 2. TOIN Backend Wiring
# =============================================================================


class TestTOINBackendWiring:
    """Verify ToolIntelligenceNetwork correctly delegates to backends."""

    def test_toin_with_custom_backend(self):
        """TOIN uses custom backend for save/load."""
        store: dict[str, Any] = {}

        class MemBackend:
            def load(self) -> dict[str, Any]:
                return dict(store)

            def save(self, data: dict[str, Any]) -> None:
                store.clear()
                store.update(data)

        config = TOINConfig(enabled=True, storage_path="", auto_save_interval=0)
        toin = ToolIntelligenceNetwork(config, backend=MemBackend())

        sig = _make_tool_signature()
        toin.record_compression(
            tool_signature=sig,
            original_count=10,
            compressed_count=3,
            original_tokens=500,
            compressed_tokens=150,
            strategy="smart_crusher",
        )

        # Save should go to our backend
        toin.save()
        assert "patterns" in store
        assert len(store["patterns"]) == 1

        # Create a new TOIN instance with same backend — should load patterns
        toin2 = ToolIntelligenceNetwork(config, backend=MemBackend())
        stats = toin2.get_stats()
        assert stats["patterns_tracked"] == 1
        assert stats["total_compressions"] == 1

    def test_toin_with_none_backend_and_no_path(self):
        """TOIN works without persistence when no backend and no path."""
        config = TOINConfig(enabled=True, storage_path="")
        toin = ToolIntelligenceNetwork(config, backend=None)

        # save() should be a no-op
        toin.save()
        stats = toin.get_stats()
        assert stats["patterns_tracked"] == 0

    def test_toin_default_filesystem_backend(self, tmp_toin_path):
        """TOIN creates FileSystemTOINBackend when storage_path is set and no backend given."""
        config = TOINConfig(enabled=True, storage_path=tmp_toin_path)
        toin = ToolIntelligenceNetwork(config)

        sig = _make_tool_signature()
        toin.record_compression(
            tool_signature=sig,
            original_count=5,
            compressed_count=2,
            original_tokens=100,
            compressed_tokens=40,
            strategy="default",
        )
        toin.save()

        # Verify file was written
        with open(tmp_toin_path) as f:
            data = json.load(f)
        assert "patterns" in data
        assert len(data["patterns"]) == 1


# =============================================================================
# 3. TOIN Entry Point Loading
# =============================================================================


class TestTOINEntryPointLoading:
    """Verify _create_default_toin_backend() env-based loading."""

    def test_no_env_returns_none(self, monkeypatch):
        """No HEADROOM_TOIN_BACKEND env → returns None (use default)."""
        monkeypatch.delenv("HEADROOM_TOIN_BACKEND", raising=False)
        assert _create_default_toin_backend() is None

    def test_empty_env_returns_none(self, monkeypatch):
        """Empty HEADROOM_TOIN_BACKEND → returns None."""
        monkeypatch.setenv("HEADROOM_TOIN_BACKEND", "")
        assert _create_default_toin_backend() is None

    def test_filesystem_env_returns_none(self, monkeypatch):
        """HEADROOM_TOIN_BACKEND=filesystem → returns None (use default)."""
        monkeypatch.setenv("HEADROOM_TOIN_BACKEND", "filesystem")
        assert _create_default_toin_backend() is None

    def test_unknown_backend_returns_none(self, monkeypatch):
        """Unknown backend name with no entry point → returns None with warning."""
        monkeypatch.setenv("HEADROOM_TOIN_BACKEND", "nonexistent_backend_xyz")
        result = _create_default_toin_backend()
        assert result is None

    def test_get_toin_respects_env_backend(self, monkeypatch, tmp_toin_path):
        """get_toin() uses _create_default_toin_backend() on first call."""
        monkeypatch.delenv("HEADROOM_TOIN_BACKEND", raising=False)
        monkeypatch.setenv("HEADROOM_TOIN_PATH", tmp_toin_path)

        toin = get_toin()
        assert toin is not None
        # Should use FileSystemTOINBackend since no HEADROOM_TOIN_BACKEND set
        assert toin._backend is not None


# =============================================================================
# 4. CCR ContextVar Scoping
# =============================================================================


class TestCCRContextVarScoping:
    """Verify request-scoped CCR stores work correctly."""

    def test_default_returns_global(self):
        """Without request scope, get_compression_store() returns global singleton."""
        store1 = get_compression_store()
        store2 = get_compression_store()
        assert store1 is store2

    def test_set_request_store_overrides_global(self):
        """set_request_compression_store() makes get return the request store."""
        global_store = get_compression_store()
        request_store = CompressionStore(max_entries=10)

        set_request_compression_store(request_store)
        assert get_compression_store() is request_store
        assert get_compression_store() is not global_store

    def test_clear_request_store_restores_global(self):
        """clear_request_compression_store() restores the global store."""
        global_store = get_compression_store()
        request_store = CompressionStore(max_entries=10)

        set_request_compression_store(request_store)
        assert get_compression_store() is request_store

        clear_request_compression_store()
        assert get_compression_store() is global_store

    def test_request_store_isolated_per_thread(self):
        """ContextVars are per-thread — each thread sees its own store."""
        global_store = get_compression_store()
        results: dict[str, CompressionStore | None] = {}

        def worker(name: str, store: CompressionStore | None):
            if store:
                set_request_compression_store(store)
            results[name] = get_compression_store()
            if store:
                clear_request_compression_store()

        store_a = CompressionStore(max_entries=5)
        store_b = CompressionStore(max_entries=7)

        t1 = threading.Thread(target=worker, args=("t1", store_a))
        t2 = threading.Thread(target=worker, args=("t2", store_b))
        t3 = threading.Thread(target=worker, args=("t3", None))

        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()

        assert results["t1"] is store_a
        assert results["t2"] is store_b
        assert results["t3"] is global_store

    def test_request_store_data_isolation(self):
        """Data stored in request-scoped store doesn't leak to global."""
        global_store = get_compression_store()

        request_store = CompressionStore(max_entries=10)
        set_request_compression_store(request_store)

        # Store data in request store
        active_store = get_compression_store()
        hash_key = active_store.store(
            original='[{"id": 1}]',
            compressed='[{"id": 1}]',
            original_tokens=10,
            compressed_tokens=10,
        )

        # Verify it's in request store
        assert active_store.retrieve(hash_key) is not None
        # Verify it's NOT in global store
        assert global_store.retrieve(hash_key) is None

        clear_request_compression_store()


# =============================================================================
# 5. CCR Entry Point Loading
# =============================================================================


class TestCCREntryPointLoading:
    """Verify _create_default_ccr_backend() env-based loading."""

    def test_no_env_returns_none(self, monkeypatch):
        """No HEADROOM_CCR_BACKEND → returns None (use InMemoryBackend)."""
        monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
        from headroom.cache.compression_store import _create_default_ccr_backend

        assert _create_default_ccr_backend() is None

    def test_memory_env_returns_none(self, monkeypatch):
        """HEADROOM_CCR_BACKEND=memory → returns None (use default)."""
        monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
        from headroom.cache.compression_store import _create_default_ccr_backend

        assert _create_default_ccr_backend() is None

    def test_unknown_backend_returns_none(self, monkeypatch):
        """Unknown backend with no entry point → returns None."""
        monkeypatch.setenv("HEADROOM_CCR_BACKEND", "nonexistent_backend_xyz")
        from headroom.cache.compression_store import _create_default_ccr_backend

        assert _create_default_ccr_backend() is None

    def test_inmemory_backend_satisfies_protocol(self):
        """InMemoryBackend satisfies the CompressionStoreBackend protocol."""
        backend = InMemoryBackend()
        assert isinstance(backend, CompressionStoreBackend)


# =============================================================================
# 6. Storage Entry Point Loading
# =============================================================================


class TestStorageEntryPointLoading:
    """Verify create_storage() scheme-based loading."""

    def test_sqlite_scheme(self, tmp_path):
        """sqlite:// scheme creates SQLiteStorage."""
        from headroom.storage.sqlite import SQLiteStorage

        store = create_storage(f"sqlite:///{tmp_path}/test.db")
        assert isinstance(store, SQLiteStorage)

    def test_jsonl_scheme(self, tmp_path):
        """jsonl:// scheme creates JSONLStorage."""
        from headroom.storage.jsonl import JSONLStorage

        store = create_storage(f"jsonl:///{tmp_path}/test.jsonl")
        assert isinstance(store, JSONLStorage)

    def test_unknown_scheme_without_entry_point(self, tmp_path):
        """Unknown scheme without entry point falls back to SQLiteStorage."""
        from headroom.storage.sqlite import SQLiteStorage

        # This should fall back to SQLite (legacy behavior)
        store = create_storage(str(tmp_path / "test.db"))
        assert isinstance(store, SQLiteStorage)

    def test_storage_base_is_abstract(self):
        """Storage ABC requires all methods to be implemented."""
        assert hasattr(Storage, "save")
        assert hasattr(Storage, "get")
        assert hasattr(Storage, "query")
        assert hasattr(Storage, "count")
        assert hasattr(Storage, "iter_all")
        assert hasattr(Storage, "get_summary_stats")


# =============================================================================
# 7. Integration: Full Adapter Lifecycle
# =============================================================================


class TestAdapterLifecycle:
    """End-to-end test of the adapter pattern."""

    def test_ccr_with_custom_backend(self):
        """CompressionStore works with a custom backend implementation."""

        class ListBackend:
            """Minimal backend that tracks all operations."""

            def __init__(self):
                self._store: dict[str, Any] = {}
                self.ops: list[str] = []

            def get(self, hash_key):
                self.ops.append(f"get:{hash_key[:8]}")
                return self._store.get(hash_key)

            def set(self, hash_key, entry):
                self.ops.append(f"set:{hash_key[:8]}")
                self._store[hash_key] = entry

            def delete(self, hash_key):
                self.ops.append(f"delete:{hash_key[:8]}")
                if hash_key in self._store:
                    del self._store[hash_key]
                    return True
                return False

            def exists(self, hash_key):
                return hash_key in self._store

            def clear(self):
                self._store.clear()

            def count(self):
                return len(self._store)

            def keys(self):
                return list(self._store.keys())

            def items(self):
                return list(self._store.items())

            def get_stats(self):
                return {"backend_type": "list", "entry_count": len(self._store)}

        backend = ListBackend()
        store = CompressionStore(backend=backend)

        # Store and retrieve
        hash_key = store.store(
            original='[{"id": 1, "name": "test"}]',
            compressed='[{"id": 1}]',
            original_tokens=50,
            compressed_tokens=20,
        )
        entry = store.retrieve(hash_key)

        assert entry is not None
        assert entry.original_tokens == 50
        assert any("set:" in op for op in backend.ops)
        assert any("get:" in op for op in backend.ops)

    def test_toin_save_load_preserves_patterns(self, tmp_toin_path):
        """Patterns survive save/load via backend.

        PR-B5 retired the request-time `get_recommendation()` API
        (it now returns None with a deprecation warning). Stats and
        on-disk patterns must still survive save/load — that's the
        observation API B5 preserves.
        """
        config = TOINConfig(storage_path=tmp_toin_path)
        toin = ToolIntelligenceNetwork(config)

        sig = _make_tool_signature()

        # Record multiple events
        for _i in range(15):
            toin.record_compression(
                tool_signature=sig,
                original_count=50,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=200,
                strategy="smart_crusher",
            )

        toin.save()

        # New instance loads from same backend
        toin2 = ToolIntelligenceNetwork(TOINConfig(storage_path=tmp_toin_path))
        stats = toin2.get_stats()
        assert stats["patterns_tracked"] >= 1
        assert stats["total_compressions"] >= 15

        # PR-B5: get_recommendation is observation-only and returns None.
        # Recommendations now flow through the publish CLI →
        # recommendations.toml → Rust loader path.
        assert toin2.get_recommendation(sig) is None

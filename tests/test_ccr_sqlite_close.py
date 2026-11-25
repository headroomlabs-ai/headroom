"""Closing the public CCR store releases its SQLite backend."""

import sqlite3
from pathlib import Path

import pytest

from headroom.cache.backends.sqlite import SQLiteBackend
from headroom.cache.compression_store import CompressionStore
from headroom.exceptions import StorageError


def test_store_close_releases_connection_and_database_file(tmp_path: Path) -> None:
    path = tmp_path / "ccr.db"
    backend = SQLiteBackend(path)
    store = CompressionStore(backend=backend, enable_feedback=False)
    store.store("Original result", "Short result")
    connection = backend._conn
    try:
        store.close()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        path.rename(tmp_path / "renamed.db")
    finally:
        connection.close()


def test_close_is_idempotent_and_does_not_reopen_on_later_use(tmp_path: Path) -> None:
    path = tmp_path / "ccr.db"
    backend = SQLiteBackend(path)
    store = CompressionStore(backend=backend, enable_feedback=False)
    key = store.store("Original result", "Short result")
    try:
        store.close()
        store.close()
        assert store.retrieve(key) is None
        with pytest.raises(StorageError):
            store.store("Another result", "Another short result")
        assert backend._conn is None
    finally:
        if backend._conn is not None:
            backend._conn.close()


def test_closing_one_store_preserves_sibling_and_persisted_entries(tmp_path: Path) -> None:
    path = tmp_path / "ccr.db"
    backend = SQLiteBackend(path)
    sibling_backend = SQLiteBackend(path)
    store = CompressionStore(backend=backend, enable_feedback=False)
    sibling = CompressionStore(backend=sibling_backend, enable_feedback=False)
    key = store.store("Original result", "Short result")
    try:
        store.close()
        assert sibling.retrieve(key).original_content == "Original result"
        sibling.close()
        replacement_backend = SQLiteBackend(path)
        try:
            replacement = CompressionStore(backend=replacement_backend, enable_feedback=False)
            assert replacement.retrieve(key).original_content == "Original result"
        finally:
            replacement_backend._conn.close()
    finally:
        for item in (backend, sibling_backend):
            if item._conn is not None:
                item._conn.close()

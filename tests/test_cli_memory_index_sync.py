"""Tests for memory CLI index synchronization (issue #2856).

Verifies that headroom memory delete/prune/purge/edit remove stale entries
from the FTS5 and vector search indexes, not just from the primary store.

Vector index tests require sqlite-vec and are skipped when it is not installed.
They exercise the real SQLiteVectorIndex schema (vec0 virtual table) so that
the extension-aware connection path in _remove_from_search_indexes and
_clear_all_search_indexes is exercised rather than a plain-table stand-in.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

import headroom.cli.memory as memory_cli
from headroom.cli.main import main
from headroom.cli.memory import (
    _clear_all_search_indexes,
    _remove_from_search_indexes,
)
from headroom.memory.adapters.fts5 import FTS5TextIndex
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.models import Memory

# ---------------------------------------------------------------------------
# sqlite-vec availability guard
# ---------------------------------------------------------------------------

try:
    from headroom.memory.adapters.sqlite_vector import (
        SQLiteVectorIndex,
        is_sqlite_vec_available,
    )

    SQLITE_VEC_AVAILABLE = is_sqlite_vec_available()
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    SQLiteVectorIndex = None  # type: ignore[assignment,misc]

requires_sqlite_vec = pytest.mark.skipif(
    not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available"
)

_VEC_DIM = 4  # small dimension keeps test seeding fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(memory_id: str, content: str = "test content") -> Memory:
    return Memory(
        id=memory_id,
        content=content,
        user_id="test-user",
    )


def _seed_fts(db_path: Path, memories: list[Memory]) -> None:
    """Index memories into the FTS5 table."""
    fts = FTS5TextIndex(db_path=str(db_path))
    for mem in memories:
        asyncio.run(fts.index_memory(mem))


def _seed_vector(db_path: Path, memory_ids: list[str]) -> None:
    """Seed the vector DB using the real SQLiteVectorIndex (requires sqlite-vec).

    Creates the true vec0 virtual-table schema so the helpers under test
    exercise the extension-aware connection path.
    """
    vector_db = db_path.parent / f"{db_path.stem}_vectors.db"
    index = SQLiteVectorIndex(dimension=_VEC_DIM, db_path=str(vector_db))
    for mid in memory_ids:
        embedding = list(
            np.random.default_rng(abs(hash(mid))).standard_normal(_VEC_DIM).astype(float)
        )
        mem = Memory(id=mid, content="test", user_id="u", embedding=embedding)
        asyncio.run(index.index(mem))


def _fts_count(db_path: Path) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]


def _fts_ids(db_path: Path) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT memory_id FROM memory_fts").fetchall()
    return {r[0] for r in rows}


def _vector_ids(db_path: Path) -> set[str]:
    """Read surviving memory_ids from the metadata table (regular, no extension needed)."""
    vector_db = db_path.parent / f"{db_path.stem}_vectors.db"
    if not vector_db.exists():
        return set()
    with sqlite3.connect(str(vector_db)) as conn:
        rows = conn.execute("SELECT memory_id FROM vec_metadata").fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# _remove_from_search_indexes — FTS5 (no sqlite-vec required)
# ---------------------------------------------------------------------------


def test_remove_from_search_indexes_clears_fts_entries(tmp_path):
    db_path = tmp_path / "memory.db"
    mems = [_make_memory(f"id-{i}") for i in range(3)]
    _seed_fts(db_path, mems)
    assert _fts_count(db_path) == 3

    _remove_from_search_indexes(str(db_path), ["id-0", "id-2"])

    assert _fts_ids(db_path) == {"id-1"}


def test_remove_from_search_indexes_no_vector_db_is_noop(tmp_path):
    db_path = tmp_path / "memory.db"
    mems = [_make_memory("id-0")]
    _seed_fts(db_path, mems)

    # No vector DB → should not raise
    _remove_from_search_indexes(str(db_path), ["id-0"])

    assert _fts_count(db_path) == 0


def test_remove_from_search_indexes_empty_list_is_noop(tmp_path):
    db_path = tmp_path / "memory.db"
    _seed_fts(db_path, [_make_memory("id-0")])
    assert _fts_count(db_path) == 1

    _remove_from_search_indexes(str(db_path), [])

    assert _fts_count(db_path) == 1


def test_remove_from_search_indexes_absent_optional_indexes_is_noop(tmp_path):
    """A primary-only store must not fail after its mutation already succeeded."""
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    asyncio.run(store.save(_make_memory("id-0")))

    assert _remove_from_search_indexes(str(db_path), ["id-0"]) is True
    assert _clear_all_search_indexes(str(db_path)) is True


def test_empty_uninitialized_vector_database_is_noop_without_sqlite_vec(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    asyncio.run(store.save(_make_memory("id-0")))
    (tmp_path / "memory_vectors.db").touch()
    monkeypatch.setitem(sys.modules, "sqlite_vec", None)

    assert _remove_from_search_indexes(str(db_path), ["id-0"]) is True
    assert _clear_all_search_indexes(str(db_path)) is True


# ---------------------------------------------------------------------------
# _remove_from_search_indexes — vector index (real vec0 schema, requires sqlite-vec)
# ---------------------------------------------------------------------------


@requires_sqlite_vec
def test_remove_from_search_indexes_clears_vector_entries(tmp_path):
    """Exercise the real vec0 virtual-table schema so the extension-aware
    connection path in _remove_from_search_indexes is covered."""
    db_path = tmp_path / "memory.db"
    _seed_fts(db_path, [])  # ensure memory.db exists
    _seed_vector(db_path, ["id-0", "id-1", "id-2"])
    assert _vector_ids(db_path) == {"id-0", "id-1", "id-2"}

    _remove_from_search_indexes(str(db_path), ["id-0", "id-2"])

    assert _vector_ids(db_path) == {"id-1"}


@requires_sqlite_vec
def test_remove_from_search_indexes_no_vector_rows_to_delete_is_noop(tmp_path):
    """IDs not present in the vector index must be silently skipped."""
    db_path = tmp_path / "memory.db"
    _seed_fts(db_path, [])
    _seed_vector(db_path, ["id-0"])
    assert _vector_ids(db_path) == {"id-0"}

    _remove_from_search_indexes(str(db_path), ["id-99"])  # not in index

    assert _vector_ids(db_path) == {"id-0"}


# ---------------------------------------------------------------------------
# _clear_all_search_indexes — FTS5 (no sqlite-vec required)
# ---------------------------------------------------------------------------


def test_clear_all_search_indexes_removes_all_fts_entries(tmp_path):
    db_path = tmp_path / "memory.db"
    _seed_fts(db_path, [_make_memory(f"id-{i}") for i in range(5)])
    assert _fts_count(db_path) == 5

    _clear_all_search_indexes(str(db_path))

    assert _fts_count(db_path) == 0


def test_clear_all_search_indexes_no_vector_db_is_noop(tmp_path):
    db_path = tmp_path / "memory.db"
    _seed_fts(db_path, [_make_memory("id-0")])

    _clear_all_search_indexes(str(db_path))

    assert _fts_count(db_path) == 0  # FTS cleared; no vector DB is fine


# ---------------------------------------------------------------------------
# _clear_all_search_indexes — vector index (real vec0 schema, requires sqlite-vec)
# ---------------------------------------------------------------------------


@requires_sqlite_vec
def test_clear_all_search_indexes_removes_all_vector_entries(tmp_path):
    """Exercise the real vec0 virtual-table schema so the extension-aware
    connection path in _clear_all_search_indexes is covered."""
    db_path = tmp_path / "memory.db"
    _seed_fts(db_path, [])
    _seed_vector(db_path, ["id-0", "id-1"])
    assert _vector_ids(db_path) == {"id-0", "id-1"}

    _clear_all_search_indexes(str(db_path))

    assert _vector_ids(db_path) == set()


# ---------------------------------------------------------------------------
# Integration: CLI commands wire up index sync correctly
# ---------------------------------------------------------------------------


def test_delete_command_removes_from_fts(tmp_path):
    """Simulate delete command: delete_batch then _remove_from_search_indexes."""
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    mem = _make_memory("abc123")
    asyncio.run(store.save(mem))
    _seed_fts(db_path, [mem])
    assert _fts_count(db_path) == 1

    asyncio.run(store.delete_batch(["abc123"]))
    _remove_from_search_indexes(str(db_path), ["abc123"])

    assert _fts_count(db_path) == 0


@requires_sqlite_vec
def test_delete_command_removes_from_vector_index(tmp_path):
    """Simulate delete command end-to-end with the real vec0 schema."""
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    mem = _make_memory("abc123")
    asyncio.run(store.save(mem))
    _seed_fts(db_path, [mem])
    _seed_vector(db_path, ["abc123"])
    assert _vector_ids(db_path) == {"abc123"}

    asyncio.run(store.delete_batch(["abc123"]))
    _remove_from_search_indexes(str(db_path), ["abc123"])

    assert _fts_count(db_path) == 0
    assert _vector_ids(db_path) == set()


def test_purge_command_clears_fts(tmp_path):
    """Simulate purge command: clear_all then _clear_all_search_indexes."""
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    for i in range(3):
        asyncio.run(store.save(_make_memory(f"id-{i}")))
    _seed_fts(db_path, [_make_memory(f"id-{i}") for i in range(3)])
    assert _fts_count(db_path) == 3

    asyncio.run(store.clear_all())
    _clear_all_search_indexes(str(db_path))

    assert _fts_count(db_path) == 0


@requires_sqlite_vec
def test_purge_command_clears_vector_index(tmp_path):
    """Simulate purge command end-to-end with the real vec0 schema."""
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    for i in range(3):
        asyncio.run(store.save(_make_memory(f"id-{i}")))
    _seed_fts(db_path, [_make_memory(f"id-{i}") for i in range(3)])
    _seed_vector(db_path, [f"id-{i}" for i in range(3)])
    assert _vector_ids(db_path) == {"id-0", "id-1", "id-2"}

    asyncio.run(store.clear_all())
    _clear_all_search_indexes(str(db_path))

    assert _fts_count(db_path) == 0
    assert _vector_ids(db_path) == set()


# ---------------------------------------------------------------------------
# Failure-path: return value and exit-code impact
# ---------------------------------------------------------------------------


def test_remove_from_search_indexes_fts_failure_returns_false(tmp_path):
    """When FTS5 delete raises, the function returns False (not True)."""
    db_path = tmp_path / "memory.db"
    _seed_fts(db_path, [_make_memory("id-0")])

    # sqlite3 is imported locally inside the helper so we patch the global module.
    original_connect = sqlite3.connect
    call_count = [0]

    def failing_connect(path, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:  # first call is the FTS5 db open
            raise sqlite3.OperationalError("simulated FTS5 failure")
        return original_connect(path, **kwargs)

    with patch("sqlite3.connect", side_effect=failing_connect):
        result = _remove_from_search_indexes(str(db_path), ["id-0"])

    assert result is False


def test_remove_from_search_indexes_sqlite_vec_missing_returns_false(tmp_path):
    """When sqlite_vec is absent and a vector DB exists, returns False."""
    db_path = tmp_path / "memory.db"
    _seed_fts(db_path, [])
    # Create a non-empty vector DB file so the code doesn't short-circuit.
    vector_db = tmp_path / "memory_vectors.db"
    vector_db.write_bytes(b"placeholder")

    # Remove sqlite_vec from sys.modules so `import sqlite_vec` raises ImportError.
    with patch.dict(sys.modules, {"sqlite_vec": None}):
        result = _remove_from_search_indexes(str(db_path), ["id-0"])

    assert result is False


def test_clear_all_search_indexes_fts_failure_returns_false(tmp_path):
    """When FTS5 DELETE raises, _clear_all_search_indexes returns False."""
    db_path = tmp_path / "memory.db"
    _seed_fts(db_path, [_make_memory("id-0")])

    original_connect = sqlite3.connect
    call_count = [0]

    def failing_connect(path, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise sqlite3.OperationalError("simulated FTS5 failure")
        return original_connect(path, **kwargs)

    with patch("sqlite3.connect", side_effect=failing_connect):
        result = _clear_all_search_indexes(str(db_path))

    assert result is False


def test_clear_all_search_indexes_sqlite_vec_missing_returns_false(tmp_path):
    """When sqlite_vec is absent and a vector DB exists, returns False."""
    db_path = tmp_path / "memory.db"
    _seed_fts(db_path, [])
    vector_db = tmp_path / "memory_vectors.db"
    vector_db.write_bytes(b"placeholder")

    with patch.dict(sys.modules, {"sqlite_vec": None}):
        result = _clear_all_search_indexes(str(db_path))

    assert result is False


def test_reindex_pages_through_complete_store(tmp_path, monkeypatch):
    """Records beyond the first page remain represented in rebuilt FTS."""
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    memories = [_make_memory(f"id-{i}", f"content {i}") for i in range(5)]
    for memory in memories:
        asyncio.run(store.save(memory))
    _seed_fts(db_path, memories[:2])
    monkeypatch.setattr(memory_cli, "_REINDEX_PAGE_SIZE", 2)

    result = CliRunner().invoke(main, ["memory", "reindex", "--db-path", str(db_path)])

    assert result.exit_code == 0, result.output
    assert _fts_ids(db_path) == {memory.id for memory in memories}
    assert "Re-indexed 5/5 memories" in result.output


@requires_sqlite_vec
def test_reindex_keeps_valid_vectors_beyond_first_page(tmp_path, monkeypatch):
    """Complete primary IDs, not one page, determine vector orphans."""
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    memories = [_make_memory(f"id-{i}", f"content {i}") for i in range(5)]
    for memory in memories:
        asyncio.run(store.save(memory))
    _seed_vector(db_path, [memory.id for memory in memories] + ["orphan"])
    monkeypatch.setattr(memory_cli, "_REINDEX_PAGE_SIZE", 2)

    result = CliRunner().invoke(main, ["memory", "reindex", "--db-path", str(db_path)])

    assert result.exit_code == 0, result.output
    assert _vector_ids(db_path) == {memory.id for memory in memories}


def test_delete_command_exits_nonzero_when_index_sync_fails(tmp_path, monkeypatch):
    """The real Click command must not report a partially synced delete as success."""
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    memory = _make_memory("abc123")
    asyncio.run(store.save(memory))
    monkeypatch.setattr(memory_cli, "_remove_from_search_indexes", lambda *_args: False)

    result = CliRunner().invoke(
        main,
        ["memory", "delete", memory.id, "--force", "--db-path", str(db_path)],
    )

    assert result.exit_code == 1
    assert asyncio.run(store.get(memory.id)) is None
    assert "index sync incomplete" in result.output


def test_edit_command_exits_nonzero_when_index_sync_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    memory = _make_memory("abc123", "before")
    asyncio.run(store.save(memory))
    monkeypatch.setattr(memory_cli, "_remove_from_search_indexes", lambda *_args: False)

    result = CliRunner().invoke(
        main,
        [
            "memory",
            "edit",
            memory.id,
            "--content",
            "after",
            "--db-path",
            str(db_path),
        ],
    )

    assert result.exit_code == 1
    assert asyncio.run(store.get(memory.id)).content == "after"
    assert "index sync incomplete" in result.output


def test_prune_command_exits_nonzero_when_index_sync_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    memory = _make_memory("abc123")
    asyncio.run(store.save(memory))
    monkeypatch.setattr(memory_cli, "_remove_from_search_indexes", lambda *_args: False)

    result = CliRunner().invoke(
        main,
        [
            "memory",
            "prune",
            "--low-importance",
            "1.0",
            "--force",
            "--db-path",
            str(db_path),
        ],
    )

    assert result.exit_code == 1
    assert asyncio.run(store.get(memory.id)) is None
    assert "index sync incomplete" in result.output


def test_purge_command_exits_nonzero_when_index_sync_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(db_path))
    memory = _make_memory("abc123")
    asyncio.run(store.save(memory))
    monkeypatch.setattr(memory_cli, "_clear_all_search_indexes", lambda *_args: False)

    result = CliRunner().invoke(
        main,
        ["memory", "purge", "--confirm", "--db-path", str(db_path)],
        input="y\n",
    )

    assert result.exit_code == 1
    assert asyncio.run(store.get(memory.id)) is None
    assert "index sync incomplete" in result.output

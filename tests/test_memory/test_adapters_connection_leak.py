"""Regression tests: SQLite adapters must close every connection they open.

``_get_conn()`` in each adapter opens a brand-new ``sqlite3.Connection`` per
call. ``with conn:`` only commits/rolls back on exit -- it does not close the
connection -- so without an explicit ``.close()`` every operation leaks a
file descriptor.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from headroom.memory.adapters.fts5 import FTS5TextIndex
from headroom.memory.adapters.graph_models import Entity
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.adapters.sqlite_graph import SQLiteGraphStore
from headroom.memory.models import Memory


def _track_connections(monkeypatch: pytest.MonkeyPatch) -> list[sqlite3.Connection]:
    """Wrap ``sqlite3.connect`` to record every connection it creates."""
    created: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        created.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    return created


def _assert_all_closed(connections: list[sqlite3.Connection]) -> None:
    assert connections, "expected at least one sqlite3.Connection to be created"
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 1")


async def test_sqlite_memory_store_closes_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _track_connections(monkeypatch)

    store = SQLiteMemoryStore(str(tmp_path / "mem.db"))
    await store.save(Memory(content="hello", user_id="alice"))

    _assert_all_closed(created)


def test_fts5_text_index_closes_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _track_connections(monkeypatch)

    index = FTS5TextIndex(str(tmp_path / "fts.db"))
    index.index_raw("mem-1", "hello world", {"user_id": "alice"})

    _assert_all_closed(created)


async def test_sqlite_graph_store_closes_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _track_connections(monkeypatch)

    store = SQLiteGraphStore(str(tmp_path / "graph.db"))
    await store.add_entity(Entity(user_id="alice", name="Project X", entity_type="project"))

    _assert_all_closed(created)

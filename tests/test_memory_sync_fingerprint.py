"""Incremental sync must observe edits without new rows or IDs."""

from dataclasses import replace

import numpy as np
import pytest

from headroom.memory.adapters.fts5 import FTS5TextIndex
from headroom.memory.adapters.graph import InMemoryGraphStore
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex, is_sqlite_vec_available
from headroom.memory.backends.local import LocalBackend
from headroom.memory.config import MemoryConfig
from headroom.memory.core import HierarchicalMemory
from headroom.memory.models import Memory
from headroom.memory.sync import _db_fingerprint, sync
from headroom.memory.sync_adapters.codex_agent import CodexAdapter


class FixedEmbedder:
    async def embed(self, text):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


@pytest.mark.parametrize("position", [0, 6])
def test_fingerprint_observes_content_edits_at_every_position(position):
    rows = [Memory(id=f"memory-{i}", content=f"fact {i}", user_id="alice") for i in range(7)]
    changed = rows.copy()
    changed[position] = replace(rows[position], content="corrected fact")
    assert _db_fingerprint(rows) != _db_fingerprint(changed)


def test_fingerprint_uses_complete_ids():
    original = Memory(id="12345678-old", content="fact", user_id="alice")
    assert _db_fingerprint([original]) != _db_fingerprint([replace(original, id="12345678-new")])


def test_fingerprint_observes_export_eligibility():
    original = Memory(content="fact", user_id="alice")
    changed = replace(original, metadata={"source_agent": "codex", "sync_direction": "import"})
    assert _db_fingerprint([original]) != _db_fingerprint([changed])


def test_fingerprint_stable_for_equivalent_rows():
    rows = [Memory(content="fact", user_id="alice")]
    assert _db_fingerprint(rows) == _db_fingerprint([replace(rows[0])])
    assert _db_fingerprint([]) == _db_fingerprint([])


@pytest.mark.asyncio
@pytest.mark.skipif(not is_sqlite_vec_available(), reason="sqlite-vec unavailable")
async def test_incremental_sync_exports_in_place_sqlite_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_STATELESS", "0")
    memory = HierarchicalMemory(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        SQLiteVectorIndex(dimension=3, db_path=tmp_path / "vectors.db"),
        FTS5TextIndex(tmp_path / "fts.db"),
        FixedEmbedder(),
        config=MemoryConfig(auto_bubble=False, vector_dimension=3),
    )
    backend = LocalBackend()
    backend._hierarchical_memory = memory
    backend._graph = InMemoryGraphStore()
    backend._initialized = True
    adapter = CodexAdapter(tmp_path / "AGENTS.md")
    state_path = tmp_path / "state.json"
    try:
        saved = await memory.add("Original fact", user_id="alice")
        first = await sync(backend, adapter, "alice", state_path)
        assert first.exported == 1
        unchanged = await sync(backend, adapter, "alice", state_path)
        assert unchanged.imported == unchanged.exported == 0
        await memory.update(saved.id, content="Corrected fact")
        updated = await sync(backend, adapter, "alice", state_path)
        assert updated.exported == 1
        assert "Corrected fact" in adapter._path.read_text(encoding="utf-8")
        repeated = await sync(backend, adapter, "alice", state_path)
        assert repeated.imported == repeated.exported == 0
    finally:
        await memory.close()

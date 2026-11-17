"""Agent synchronization must include memories older than the newest 500."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import numpy as np
import pytest

from headroom.memory.adapters.graph import InMemoryGraphStore
from headroom.memory.adapters.hnsw import HNSWVectorIndex
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.backends.local import LocalBackend, LocalBackendConfig
from headroom.memory.config import MemoryConfig
from headroom.memory.core import HierarchicalMemory
from headroom.memory.models import Memory
from headroom.memory.ports import MemoryFilter
from headroom.memory.sync import sync, sync_export, sync_import
from headroom.memory.sync_adapters.codex_agent import CodexAdapter


@pytest.fixture
async def backend(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memories.db")
    embedder = AsyncMock()
    embedder.embed.return_value = np.ones(4)
    memory = HierarchicalMemory(
        store,
        HNSWVectorIndex(dimension=4),
        AsyncMock(),
        embedder,
        config=MemoryConfig(auto_bubble=False),
    )
    backend = LocalBackend(LocalBackendConfig(db_path=str(tmp_path / "memories.db")))
    backend._hierarchical_memory = memory
    backend._graph = InMemoryGraphStore()
    backend._initialized = True
    now = datetime.now()
    memories = [
        Memory(
            id=f"alice-{i:04}",
            content=f"Distinct memory {i}",
            user_id="alice",
            created_at=now - timedelta(seconds=i),
        )
        for i in range(501)
    ]
    memories.append(Memory(content="Other user's private fact", user_id="bob"))
    await store.save_batch(memories)
    return backend


@pytest.fixture
def adapter(tmp_path):
    return CodexAdapter(tmp_path / "AGENTS.md")


@pytest.mark.asyncio
async def test_import_deduplicates_a_memory_older_than_the_latest_500(backend, adapter):
    await adapter.write_memories([{"content": "Distinct memory 500", "headroom_id": "alice-0500"}])
    assert await sync_import(backend, adapter, "alice") == 0
    rows = await backend._hierarchical_memory._store.query(MemoryFilter(user_id="alice"))
    assert len(rows) == 501


@pytest.mark.asyncio
async def test_export_includes_the_oldest_memory_and_keeps_user_scope(backend, adapter):
    assert await sync_export(backend, adapter, "alice") == 501
    exported = await adapter.read_memories()
    assert "Distinct memory 500" in {m.content for m in exported}
    assert "Other user's private fact" not in {m.content for m in exported}


@pytest.mark.asyncio
@pytest.mark.parametrize("force", [False, True])
async def test_full_sync_exports_every_memory_and_is_idempotent(backend, adapter, tmp_path, force):
    state = tmp_path / "state.json"
    first = await sync(backend, adapter, "alice", state_path=state, force=force)
    assert first.imported == 0
    assert first.exported == 501
    second = await sync(backend, adapter, "alice", state_path=state, force=True)
    assert second.imported == 0
    assert second.exported == 0


@pytest.mark.asyncio
async def test_sync_refetch_after_import_does_not_truncate_exports(backend, adapter, tmp_path):
    await adapter.write_memories([{"content": "New agent fact", "headroom_id": "new-agent-fact"}])
    result = await sync(backend, adapter, "alice", state_path=tmp_path / "state.json", force=True)
    assert result.imported == 1
    assert result.exported == 501
    assert len(await adapter.read_memories()) == 502


@pytest.mark.asyncio
async def test_explicit_existing_snapshot_is_not_requeried(backend, adapter):
    snapshot = [Memory(content="Explicit snapshot", user_id="alice")]
    backend.get_user_memories = AsyncMock(side_effect=AssertionError("snapshot already provided"))
    assert await sync_export(backend, adapter, "alice", existing_memories=snapshot) == 1
    assert await sync_import(backend, adapter, "alice", existing_memories=snapshot) == 0
    backend.get_user_memories.assert_not_awaited()


@pytest.mark.asyncio
async def test_optional_unbounded_query_preserves_the_default_limit(backend):
    assert len(await backend.get_user_memories("alice")) == 100
    assert len(await backend.get_user_memories("alice", limit=3)) == 3
    assert len(await backend.get_user_memories("alice", limit=None)) == 501

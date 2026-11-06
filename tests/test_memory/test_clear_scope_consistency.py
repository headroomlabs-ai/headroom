"""Scoped clears must preserve sibling memories and their search indexes."""

from contextlib import asynccontextmanager

import numpy as np
import pytest

from headroom.memory.adapters.cache import LRUMemoryCache
from headroom.memory.adapters.fts5 import FTS5TextIndex
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex, is_sqlite_vec_available
from headroom.memory.config import MemoryConfig
from headroom.memory.core import HierarchicalMemory
from headroom.memory.ports import MemoryFilter, TextFilter


class FixedEmbedder:
    async def embed(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


@asynccontextmanager
async def memory_system(path):
    memory = HierarchicalMemory(
        SQLiteMemoryStore(path / "memory.db"),
        SQLiteVectorIndex(dimension=3, db_path=path / "vectors.db"),
        FTS5TextIndex(path / "memory.db"),
        FixedEmbedder(),
        LRUMemoryCache(),
        MemoryConfig(auto_bubble=False, vector_dimension=3),
    )
    try:
        yield memory
    finally:
        await memory.close()


async def assert_remaining(memory, ids, remaining):
    expected = {ids[label] for label in remaining}
    rows = await memory.query(MemoryFilter(include_superseded=True))
    assert {row.id for row in rows} == expected
    vector_hits = await memory.search("orchard", top_k=10)
    assert {hit.memory.id for hit in vector_hits} == expected
    text_hits = await memory.text_index.search_memories(TextFilter(query="orchard"))
    assert {hit.memory.id for hit in text_hits} == expected
    for label, memory_id in ids.items():
        assert (await memory.get(memory_id) is not None) == (label in remaining)


@pytest.mark.skipif(not is_sqlite_vec_available(), reason="sqlite-vec unavailable")
@pytest.mark.parametrize(
    ("scope", "removed"),
    [
        ({}, {"target", "other-turn", "other-agent", "other-session"}),
        ({"session_id": "s1"}, {"target", "other-turn", "other-agent"}),
        ({"agent_id": "planner"}, {"target", "other-turn", "other-session"}),
        ({"turn_id": "t1"}, {"target", "other-agent", "other-session"}),
        ({"session_id": "s1", "agent_id": "planner"}, {"target", "other-turn"}),
        ({"session_id": "s1", "turn_id": "t1"}, {"target", "other-agent"}),
        ({"agent_id": "planner", "turn_id": "t1"}, {"target", "other-session"}),
        ({"session_id": "s1", "agent_id": "planner", "turn_id": "t1"}, {"target"}),
    ],
    ids=["user", "session", "agent", "turn", "session-agent", "session-turn", "agent-turn", "full"],
)
async def test_clear_scope_preserves_siblings_and_indexes_after_reopen(tmp_path, scope, removed):
    rows = [
        ("target", "alice", "s1", "planner", "t1"),
        ("other-turn", "alice", "s1", "planner", "t2"),
        ("other-agent", "alice", "s1", "reviewer", "t1"),
        ("other-session", "alice", "s2", "planner", "t1"),
        ("other-user", "bob", "s1", "planner", "t1"),
    ]
    ids = {}
    async with memory_system(tmp_path) as memory:
        for label, user, session, agent, turn in rows:
            entry = await memory.add(
                "orchard " + label,
                user_id=user,
                session_id=session,
                agent_id=agent,
                turn_id=turn,
            )
            ids[label] = entry.id
        await assert_remaining(memory, ids, set(ids))
        assert await memory.clear_scope(user_id="alice", **scope) == len(removed)
        remaining = set(ids) - removed
        await assert_remaining(memory, ids, remaining)

    async with memory_system(tmp_path) as memory:
        await assert_remaining(memory, ids, remaining)

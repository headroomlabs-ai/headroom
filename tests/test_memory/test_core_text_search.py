"""Core text search must work with the factory's FTS5 and external adapters."""

import pytest

from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.config import MemoryConfig
from headroom.memory.core import HierarchicalMemory
from headroom.memory.factory import _create_text_index
from headroom.memory.models import Memory
from headroom.memory.ports import TextSearchResult


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "session_id", "limit", "expected_ids"),
    [
        ("Python", "first", 10, {"alice-first"}),
        ("Python", None, 10, {"alice-first", "alice-second"}),
        ("Python", None, 0, set()),
        ("unmatched", None, 10, set()),
    ],
)
async def test_text_search_uses_factory_fts5_adapter(
    tmp_path, query, session_id, limit, expected_ids
):
    path = tmp_path / "memory.db"
    text_index = _create_text_index(MemoryConfig(db_path=path))
    store = SQLiteMemoryStore(path)
    for user, session in [("alice", "first"), ("alice", "second"), ("bob", "first")]:
        memory = Memory(
            id=f"{user}-{session}",
            content="Python programming preference",
            user_id=user,
            session_id=session,
        )
        await store.save(memory)
        await text_index.index_memory(memory)
    # Vector search and embeddings are not involved in a text-only request.
    system = HierarchicalMemory(store, None, text_index, None)

    results = await system.text_search(query, user_id="alice", session_id=session_id, limit=limit)

    assert {result.memory.id for result in results} == expected_ids
    assert all(isinstance(result, TextSearchResult) for result in results)
    assert all(result.memory.user_id == "alice" and result.score > 0 for result in results)
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    # Keep the pre-existing synchronous string-search API working too.
    assert len(text_index.search("Python", k=10)) == 3


@pytest.mark.asyncio
async def test_text_search_preserves_external_text_index_protocol():
    result = TextSearchResult(
        memory=Memory(content="external result", user_id="alice"), score=1.0, rank=1
    )

    class ExternalTextIndex:
        async def search(self, filter):
            self.seen_filter = filter
            return [result]

    text_index = ExternalTextIndex()
    system = HierarchicalMemory(None, None, text_index, None)

    assert await system.text_search("Python", "alice", "session", 7) == [result]
    assert text_index.seen_filter.query == "Python"
    assert text_index.seen_filter.user_id == "alice"
    assert text_index.seen_filter.session_id == "session"
    assert text_index.seen_filter.limit == 7

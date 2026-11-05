"""Core text search must work with the factory's FTS5 and external adapters."""

from types import SimpleNamespace

import pytest

from headroom.memory.adapters.fts5 import FTS5TextIndex
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.config import MemoryConfig, TextBackend
from headroom.memory.core import HierarchicalMemory
from headroom.memory.factory import _create_text_index
from headroom.memory.models import Memory
from headroom.memory.ports import TextIndex, TextSearchResult


@pytest.mark.asyncio
@pytest.mark.parametrize("use_factory", [True, False], ids=["factory", "direct"])
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
    tmp_path, query, session_id, limit, expected_ids, use_factory
):
    path = tmp_path / "memory.db"
    text_index = (
        _create_text_index(MemoryConfig(db_path=path)) if use_factory else FTS5TextIndex(path)
    )
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
    assert system.text_index is text_index

    results = await system.text_search(query, user_id="alice", session_id=session_id, limit=limit)

    assert {result.memory.id for result in results} == expected_ids
    assert all(isinstance(result, TextSearchResult) for result in results)
    assert all(result.memory.user_id == "alice" and result.score > 0 for result in results)
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    # Keep the pre-existing synchronous string-search API working too.
    assert len(text_index.search("Python", k=10)) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_helper", ["absent", "sync", "async", "noncallable"])
async def test_text_search_preserves_external_text_index_protocol(extra_helper):
    result = TextSearchResult(
        memory=Memory(content="external result", user_id="alice"), score=1.0, rank=1
    )

    class ExternalTextIndex:
        async def index(self, memory):
            pass

        async def index_batch(self, memories):
            return len(memories)

        async def remove(self, memory_id):
            return False

        async def remove_batch(self, memory_ids):
            return 0

        async def update_content(self, memory_id, content):
            return False

        async def search(self, filter):
            self.seen_filter = filter
            return [result]

    text_index = ExternalTextIndex()
    # Extra names are not part of TextIndex and must not affect dispatch.
    if extra_helper == "sync":
        text_index.search_memories = lambda query: ["unrelated helper"]
    elif extra_helper == "async":

        async def unrelated_helper(query):
            return []

        text_index.search_memories = unrelated_helper
    elif extra_helper == "noncallable":
        text_index.search_memories = None
    assert isinstance(text_index, TextIndex)
    system = HierarchicalMemory(None, None, text_index, None)

    assert await system.text_search("Python", "alice", "session", 7) == [result]
    assert text_index.seen_filter.query == "Python"
    assert text_index.seen_filter.user_id == "alice"
    assert text_index.seen_filter.session_id == "session"
    assert text_index.seen_filter.limit == 7


@pytest.mark.asyncio
async def test_text_search_keeps_direct_fts5_indexing_and_lifecycle(tmp_path):
    class ManagedFTS5TextIndex(FTS5TextIndex):
        indexed_ids = None
        closed = False

        async def index_memory(self, memory):
            self.indexed_ids = [memory.id]
            await super().index_memory(memory)

        def search(self, query, k=10, filter=None):
            return super().search(query, k=k, filter=filter)

        async def close(self):
            self.closed = True

    path = tmp_path / "memory.db"
    text_index = ManagedFTS5TextIndex(path)
    store = SQLiteMemoryStore(path)
    async with HierarchicalMemory(store, None, text_index, None) as system:
        memory = await system.add("Python preference", "alice", auto_embed=False, auto_bubble=False)
        assert system.text_index is text_index
        assert text_index.indexed_ids == [memory.id]
        assert [result.memory.id for result in await system.text_search("Python", "alice")] == [
            memory.id
        ]
        assert [result.memory_id for result in text_index.search("Python")] == [memory.id]
        assert not text_index.closed
    assert text_index.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("use_fts5", [True, False], ids=["fts5", "external"])
async def test_text_search_observes_replaced_adapter_method(tmp_path, monkeypatch, use_fts5):
    if use_fts5:
        text_index = FTS5TextIndex(tmp_path / "memory.db")
        method_name = "search_memories"
    else:

        class ExternalTextIndex:
            async def search(self, filter):
                raise AssertionError("the replaced method must not be called")

        text_index = ExternalTextIndex()
        method_name = "search"
    system = HierarchicalMemory(None, None, text_index, None)
    seen = []

    async def replacement(filter):
        seen.append(filter.query)
        return []

    monkeypatch.setattr(text_index, method_name, replacement)
    assert await system.text_search("Python") == []
    assert seen == ["Python"]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_factory", [True, False], ids=["external-factory", "direct"])
async def test_text_search_preserves_async_fts5_subclass(tmp_path, monkeypatch, use_factory):
    class ProtocolFTS5TextIndex(FTS5TextIndex):
        async def search(self, filter):
            self.seen_filter = filter
            matches = super().search(filter.query, k=filter.limit, filter=filter)
            return [
                TextSearchResult(
                    memory=Memory(id=match.memory_id, content=match.content, user_id="alice"),
                    score=match.score,
                    rank=rank,
                )
                for rank, match in enumerate(matches, start=1)
            ]

    path = tmp_path / "memory.db"
    if use_factory:
        entry_point = SimpleNamespace(
            name="protocol-fts5", load=lambda: lambda config: ProtocolFTS5TextIndex(config.db_path)
        )
        monkeypatch.setattr(
            "headroom.memory.factory.entry_points",
            lambda *, group: [entry_point] if group == "headroom.memory_text" else [],
        )
        text_index = _create_text_index(
            MemoryConfig(
                db_path=path, text_backend=TextBackend.EXTERNAL, text_backend_name="protocol-fts5"
            )
        )
    else:
        text_index = ProtocolFTS5TextIndex(path)
    memory = Memory(content="Python preference", user_id="alice", session_id="session")
    await text_index.index_memory(memory)
    system = HierarchicalMemory(None, None, text_index, None)

    results = await system.text_search("Python", "alice", "session", 7)

    assert system.text_index is text_index
    assert [result.memory.id for result in results] == [memory.id]
    assert text_index.seen_filter.query == "Python"
    assert text_index.seen_filter.user_id == "alice"
    assert text_index.seen_filter.session_id == "session"
    assert text_index.seen_filter.limit == 7


@pytest.mark.asyncio
async def test_text_search_observes_fts5_search_api_replacement(tmp_path, monkeypatch):
    text_index = FTS5TextIndex(tmp_path / "memory.db")
    memory = Memory(content="Python preference", user_id="alice")
    await text_index.index_memory(memory)
    system = HierarchicalMemory(None, None, text_index, None)
    original_search = text_index.search
    seen = []

    async def protocol_search(filter):
        seen.append(filter.query)
        return []

    monkeypatch.setattr(text_index, "search", protocol_search)
    assert await system.text_search("Python", "alice") == []
    assert seen == ["Python"]

    monkeypatch.setattr(text_index, "search", original_search)
    assert [result.memory.id for result in await system.text_search("Python", "alice")] == [
        memory.id
    ]


@pytest.mark.asyncio
async def test_text_search_is_not_required_until_called(monkeypatch):
    class IndexOnly:
        async def index_memory(self, memory):
            self.indexed_id = memory.id

    text_index = IndexOnly()
    system = HierarchicalMemory(None, None, text_index, None)
    memory = Memory(content="Python preference", user_id="alice")
    await system.text_index.index_memory(memory)
    assert text_index.indexed_id == memory.id
    with pytest.raises(AttributeError, match="search"):
        await system.text_search("Python")

    async def protocol_search(filter):
        return []

    monkeypatch.setattr(text_index, "search", protocol_search, raising=False)
    assert await system.text_search("Python") == []

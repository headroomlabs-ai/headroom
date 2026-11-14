"""The advertised listing tool must reach the scoped backend without a query."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.backends.local import LocalBackend, LocalBackendConfig
from headroom.memory.core import HierarchicalMemory
from headroom.memory.models import Memory
from headroom.memory.system import MemorySystem


@pytest.mark.asyncio
@pytest.mark.parametrize("optimized", [False, True])
async def test_advertised_list_tool_dispatches_without_semantic_search(optimized):
    backend = AsyncMock()
    newest = Memory(id="new", content="Newest fact", user_id="alice")
    older = Memory(
        id="old",
        content="Older fact",
        user_id="alice",
        created_at=datetime.now() - timedelta(days=1),
    )
    backend.get_user_memories.return_value = [newest, older]
    system = MemorySystem(backend, user_id="alice")
    assert "memory_list" in [t["function"]["name"] for t in system.get_tools(optimized)]

    result = await system.process_tool_call("memory_list", {})

    assert result["success"] is True
    assert result["count"] == 2
    assert [m["memory_id"] for m in result["memories"]] == ["new", "old"]
    assert result["memories"][0]["content"] == "Newest fact"
    assert "embedding" not in result["memories"][0]
    backend.get_user_memories.assert_awaited_once_with(user_id="alice", limit=10)
    backend.search_memories.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(("requested", "effective"), [(1, 1), (100, 100), (0, 1), (200, 100)])
async def test_list_limit_matches_the_advertised_bounds(requested, effective):
    backend = AsyncMock()
    backend.get_user_memories.return_value = []
    result = await MemorySystem(backend, "alice").process_tool_call(
        "memory_list", {"limit": requested}
    )
    assert result["success"] is True
    assert result["count"] == 0
    assert result["memories"] == []
    backend.get_user_memories.assert_awaited_once_with(user_id="alice", limit=effective)


@pytest.mark.asyncio
async def test_backend_without_listing_reports_an_unsupported_operation():
    result = await MemorySystem(object(), "alice").process_tool_call("memory_list", {})
    assert result["success"] is False
    assert "does not support listing" in result["error"]


@pytest.mark.asyncio
async def test_list_backend_error_uses_the_tool_error_response():
    backend = AsyncMock()
    backend.get_user_memories.side_effect = RuntimeError("listing unavailable")
    result = await MemorySystem(backend, "alice").process_tool_call("memory_list", {})
    assert result["success"] is False
    assert result["error"] == "listing unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("session_id", "expected"),
    [("session-a", ["alice-a"]), ("missing", []), ("", []), (None, ["alice-b"])],
)
async def test_listing_applies_user_and_session_scope_before_the_limit(
    tmp_path, session_id, expected
):
    store = SQLiteMemoryStore(tmp_path / "memories.db")
    now = datetime.now()
    await store.save_batch(
        [
            Memory(
                id="alice-a",
                content="Current session",
                user_id="alice",
                session_id="session-a",
                created_at=now - timedelta(days=3),
            ),
            Memory(
                id="alice-global",
                content="Unscoped fact",
                user_id="alice",
                created_at=now - timedelta(days=2),
            ),
            Memory(
                id="alice-b",
                content="Other session",
                user_id="alice",
                session_id="session-b",
                created_at=now - timedelta(days=1),
            ),
            Memory(
                id="bob-a",
                content="Other user",
                user_id="bob",
                session_id="session-a",
                created_at=now,
            ),
        ]
    )
    backend = LocalBackend(LocalBackendConfig(db_path=str(tmp_path / "memories.db")))
    backend._hierarchical_memory = HierarchicalMemory(store, AsyncMock(), AsyncMock(), AsyncMock())
    backend._initialized = True
    result = await MemorySystem(backend, "alice", session_id=session_id).process_tool_call(
        "memory_list", {"limit": 1}
    )
    assert result["success"] is True
    assert [m["memory_id"] for m in result["memories"]] == expected


@pytest.mark.asyncio
async def test_session_listing_does_not_fall_back_to_a_user_only_backend():
    class UserOnlyBackend:
        get_user_memories = AsyncMock(return_value=[])

    backend = UserOnlyBackend()
    result = await MemorySystem(backend, "alice", session_id="session-a").process_tool_call(
        "memory_list", {}
    )
    assert result["success"] is False
    assert "does not support listing" in result["error"]
    backend.get_user_memories.assert_not_awaited()

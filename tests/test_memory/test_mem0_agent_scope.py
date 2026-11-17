"""Agent scope must survive Mem0's reserved-metadata normalization."""

from unittest.mock import MagicMock

import pytest

from headroom.memory.backends.mem0 import Mem0Backend
from headroom.memory.models import Memory
from headroom.memory.ports import MemoryFilter, VectorFilter


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_id", ["agent-a", "agent-b", None])
async def test_save_passes_agent_scope_as_mem0_argument(agent_id):
    backend = Mem0Backend()
    backend._client = MagicMock()
    backend._initialized = True
    backend._client.add.return_value = {"results": [{"id": "saved"}]}
    memory = Memory(content="prefers tea", user_id="user", agent_id=agent_id)

    assert await backend.save_memory(memory) == "saved"

    kwargs = backend._client.add.call_args.kwargs
    assert kwargs.get("agent_id") == agent_id
    assert kwargs["user_id"] == "user"
    assert kwargs["metadata"]["headroom_id"] == memory.id


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", ["explicit-user", None])
async def test_search_passes_scope_in_filters(user_id):
    backend = Mem0Backend()
    backend._client = MagicMock()
    backend._initialized = True
    backend._client.search.return_value = {"results": []}

    await backend.search_memories(
        "tea",
        user_id=user_id,
        filter=VectorFilter(user_id="filter-user", agent_id="agent-a"),
    )

    kwargs = backend._client.search.call_args.kwargs
    assert "user_id" not in kwargs
    assert kwargs["filters"] == {
        "user_id": user_id or "filter-user",
        "agent_id": "agent-a",
    }


@pytest.mark.parametrize("top_level", [False, True])
def test_result_restores_identity_scope(top_level):
    scope = {"user_id": "user", "agent_id": "agent-a"}
    result = {"id": "mem0-id", "memory": "tea", "metadata": {"headroom_id": "hr-id"}}
    if top_level:
        result.update(scope)
    else:
        result["metadata"].update(scope)

    memory = Mem0Backend()._mem0_result_to_memory(result)

    assert memory.id == "hr-id"
    assert memory.user_id == "user"
    assert memory.agent_id == "agent-a"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_all_memories", "query"])
async def test_listing_uses_mem0_filters_and_top_k(method):
    def get_all(*, filters, top_k):
        assert filters == {"user_id": "user"}
        assert top_k == 3
        return {
            "results": [
                {
                    "id": "mem0-id",
                    "memory": "tea",
                    "user_id": "user",
                    "agent_id": "agent-a",
                    "metadata": {"headroom_id": "hr-id"},
                }
            ]
        }

    backend = Mem0Backend()
    backend._client = MagicMock()
    backend._initialized = True
    backend._client.get_all.side_effect = get_all

    if method == "query":
        memories = await backend.query(MemoryFilter(user_id="user", agent_id="agent-a", limit=3))
    else:
        memories = await backend.get_all_memories("user", limit=3)

    assert [memory.id for memory in memories] == ["hr-id"]
    assert memories[0].user_id == "user"
    assert memories[0].agent_id == "agent-a"

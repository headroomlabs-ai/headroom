"""Background DirectMem0Adapter saves return the stored id (#4362 item 6)."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from headroom.memory.backends.direct_mem0 import DirectMem0Adapter, Mem0Config
from headroom.memory.system import MemorySystem


@dataclass
class _PointStruct:
    id: str
    vector: list[float]
    payload: dict[str, Any]


class _RecordingQdrant:
    def __init__(self) -> None:
        self.points: list[_PointStruct] = []

    def upsert(self, collection_name: str, points: list[_PointStruct]) -> None:
        self.points.extend(points)


class _RecordingMem0:
    def __init__(self) -> None:
        self.calls = 0

    def add(self, content: str, user_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {"results": [{"id": "mem0-assigned-id", "memory": content}]}


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> DirectMem0Adapter:
    models = types.ModuleType("qdrant_client.models")
    models.PointStruct = _PointStruct  # type: ignore[attr-defined]
    package = types.ModuleType("qdrant_client")
    package.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", package)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)

    instance = DirectMem0Adapter(Mem0Config(enable_graph=False))
    instance._initialized = True
    instance._qdrant_client = _RecordingQdrant()
    instance._mem0_client = _RecordingMem0()
    monkeypatch.setattr(instance, "_embed", lambda text: [0.0, 1.0])
    return instance


@pytest.mark.asyncio
async def test_background_save_returns_stored_fact_id(adapter: DirectMem0Adapter) -> None:
    returned = await adapter.save_memory(
        content="ignored when facts are given",
        user_id="alice",
        importance=0.7,
        facts=["Prefers Python", "Works at Acme"],
        background=True,
    )
    await adapter.flush_pending()

    stored_ids = [point.id for point in adapter._qdrant_client.points]
    assert stored_ids, "background task wrote nothing"
    assert returned.id == stored_ids[0]
    assert returned.metadata["_all_memory_ids"] == stored_ids

    status = adapter.get_task_status(returned.metadata["_task_id"])
    assert status["status"] == "completed"
    assert status["result"].id == returned.id


@pytest.mark.asyncio
async def test_mutating_lists_after_queueing_does_not_break_the_save(
    adapter: DirectMem0Adapter,
) -> None:
    facts = ["A"]
    returned = await adapter.save_memory(
        content="x", user_id="alice", importance=0.5, facts=facts, background=True
    )
    facts.append("B")
    returned.metadata["_all_memory_ids"].append("bogus")
    await adapter.flush_pending()

    status = adapter.get_task_status(returned.metadata["_task_id"])
    assert status["status"] == "completed", status
    assert [p.payload["memory"] for p in adapter._qdrant_client.points] == ["A"]
    assert [p.id for p in adapter._qdrant_client.points] == [returned.id]


@pytest.mark.asyncio
async def test_background_mem0_extraction_does_not_invent_an_id(
    adapter: DirectMem0Adapter,
) -> None:
    """Without pre-extraction, Mem0 assigns ids itself; none can be known up front."""
    returned = await adapter.save_memory(
        content="Likes tea",
        user_id="alice",
        importance=0.5,
        background=True,
    )

    assert returned.metadata["_id_pending"] is True
    assert returned.id == returned.metadata["_task_id"]

    await adapter.flush_pending()
    status = adapter.get_task_status(returned.metadata["_task_id"])
    assert status["result"].id == "mem0-assigned-id"


@pytest.mark.asyncio
async def test_memory_save_tool_withholds_a_pending_id(adapter: DirectMem0Adapter) -> None:
    system = MemorySystem(adapter, user_id="alice")

    result = await system.process_tool_call(
        "memory_save", {"content": "Likes tea", "importance": 0.6, "background": True}
    )
    await adapter.flush_pending()

    assert result["async"] is True
    assert result["memory_id"] is None
    assert result["task_id"]

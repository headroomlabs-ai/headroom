"""Tests for the Direct Mem0 adapter lifecycle."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from headroom.memory.backends import direct_mem0
from headroom.memory.backends.direct_mem0 import DirectMem0Adapter, Mem0Config
from headroom.memory.models import Memory


def _adapter() -> DirectMem0Adapter:
    return DirectMem0Adapter(Mem0Config(enable_graph=True))


async def _start_background_save(adapter: DirectMem0Adapter, content: str = "fact") -> str:
    result = await adapter.save_memory(
        content=content,
        user_id="alice",
        importance=0.8,
        background=True,
    )
    assert isinstance(result, Memory)
    return str(result.metadata["_task_id"])


async def _wait_for_completion_callback(adapter: DirectMem0Adapter, task_id: str) -> None:
    for _ in range(10):
        if task_id not in adapter._background_tasks:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"Completion callback did not run for {task_id}")


@pytest.fixture
def initialized_adapter() -> DirectMem0Adapter:
    adapter = DirectMem0Adapter(Mem0Config(enable_graph=False))
    adapter._initialized = True
    return adapter


@pytest.mark.asyncio
async def test_background_completion_is_captured_without_polling(
    initialized_adapter: DirectMem0Adapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_memory = Memory(content="stored", user_id="alice")

    async def save(**kwargs: Any) -> Memory:
        return stored_memory

    monkeypatch.setattr(initialized_adapter, "_save_memory_internal", save)
    task_id = await _start_background_save(initialized_adapter)
    await _wait_for_completion_callback(initialized_adapter, task_id)

    assert initialized_adapter.get_pending_tasks() == []
    assert initialized_adapter.get_task_status(task_id) == {
        "status": "completed",
        "result": stored_memory,
    }


@pytest.mark.asyncio
async def test_background_exception_is_captured_without_polling(
    initialized_adapter: DirectMem0Adapter,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fail(**kwargs: Any) -> Memory:
        raise RuntimeError("save failed")

    monkeypatch.setattr(initialized_adapter, "_save_memory_internal", fail)
    task_id = await _start_background_save(initialized_adapter)
    await _wait_for_completion_callback(initialized_adapter, task_id)

    assert initialized_adapter.get_task_status(task_id) == {
        "status": "failed",
        "error": "save failed",
    }
    assert f"task_id={task_id}" in caplog.text


@pytest.mark.asyncio
async def test_completed_task_results_are_bounded_and_expire(
    initialized_adapter: DirectMem0Adapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0

    async def save(**kwargs: Any) -> Memory:
        return Memory(content="stored", user_id="alice")

    monkeypatch.setattr(initialized_adapter, "_save_memory_internal", save)
    monkeypatch.setattr(direct_mem0, "_MAX_TASK_RESULTS", 2)
    monkeypatch.setattr(direct_mem0, "_TASK_RESULT_TTL_SECONDS", 10)
    monkeypatch.setattr(direct_mem0, "monotonic", lambda: now)

    task_ids: list[str] = []
    for index in range(3):
        now = float(index)
        task_id = await _start_background_save(initialized_adapter, f"fact-{index}")
        await _wait_for_completion_callback(initialized_adapter, task_id)
        task_ids.append(task_id)

    assert initialized_adapter.get_task_status(task_ids[0])["status"] == "not_found"
    assert initialized_adapter.get_task_status(task_ids[1])["status"] == "completed"
    now = 12.0
    assert initialized_adapter.get_task_status(task_ids[1])["status"] == "not_found"
    assert initialized_adapter.get_task_status(task_ids[2])["status"] == "not_found"


@pytest.mark.asyncio
async def test_close_drains_tasks_and_closes_initialized_resources() -> None:
    adapter = _adapter()
    resources = {
        "_mem0_client": MagicMock(),
        "_openai_client": MagicMock(),
        "_qdrant_client": MagicMock(),
        "_neo4j_driver": MagicMock(),
    }
    resources["_mem0_client"].close = AsyncMock()
    for name, resource in resources.items():
        setattr(adapter, name, resource)

    task = asyncio.create_task(asyncio.sleep(0, result="saved"))
    adapter._background_tasks["task_1"] = task

    await adapter.close(timeout=1.0)

    assert adapter.get_pending_tasks() == []
    assert adapter.get_task_status("task_1") == {
        "status": "completed",
        "result": "saved",
    }
    resources["_mem0_client"].close.assert_awaited_once_with()
    for resource in resources.values():
        resource.close.assert_called_once_with()
    assert adapter._initialized is False
    assert adapter._mem0_client is None
    assert adapter._openai_client is None
    assert adapter._qdrant_client is None
    assert adapter._neo4j_driver is None

    await adapter.close(timeout=0.01)
    resources["_mem0_client"].close.assert_awaited_once_with()
    for resource in resources.values():
        resource.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_close_timeout_keeps_tasks_and_resources_attached() -> None:
    adapter = _adapter()
    release = asyncio.Event()
    resource = MagicMock()
    adapter._mem0_client = resource
    task = asyncio.create_task(release.wait())
    adapter._background_tasks["task_1"] = task
    await asyncio.sleep(0)

    with pytest.raises(TimeoutError, match="task_1"):
        await adapter.close(timeout=0.01)

    assert not task.done()
    assert adapter.get_pending_tasks() == ["task_1"]
    assert adapter._mem0_client is resource
    resource.close.assert_not_called()

    release.set()
    await adapter.close(timeout=1.0)

    assert adapter.get_pending_tasks() == []
    assert adapter.get_task_status("task_1") == {
        "status": "completed",
        "result": True,
    }
    resource.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_close_does_not_close_resources_while_sync_worker_is_running() -> None:
    adapter = _adapter()
    worker_started = threading.Event()
    release_worker = threading.Event()

    class BlockingMem0Client:
        def __init__(self) -> None:
            self.close_calls = 0

        def add(self, *_args: object, **_kwargs: object) -> dict[str, list[dict[str, str]]]:
            worker_started.set()
            assert release_worker.wait(timeout=5.0), "test did not release worker"
            return {"results": [{"id": "memory-1", "memory": "saved"}]}

        def close(self) -> None:
            self.close_calls += 1

    client = BlockingMem0Client()
    adapter._mem0_client = client
    adapter._initialized = True

    memory = await adapter.save_memory(
        content="saved",
        user_id="user-1",
        importance=0.5,
        background=True,
    )
    task_id = memory.metadata["_task_id"]

    for _ in range(100):
        if worker_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert worker_started.is_set()

    with pytest.raises(TimeoutError, match=task_id):
        await adapter.close(timeout=0.01)

    assert client.close_calls == 0
    assert adapter._mem0_client is client
    assert adapter.get_pending_tasks() == [task_id]

    release_worker.set()
    await adapter.close(timeout=1.0)

    assert client.close_calls == 1
    assert adapter._mem0_client is None
    assert adapter.get_pending_tasks() == []
    assert adapter.get_task_status(task_id)["status"] == "completed"


@pytest.mark.asyncio
async def test_concurrent_close_calls_serialize_resource_cleanup() -> None:
    adapter = _adapter()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    resource = MagicMock()

    async def slow_close() -> None:
        close_started.set()
        await release_close.wait()

    resource.close = AsyncMock(side_effect=slow_close)
    adapter._mem0_client = resource

    first = asyncio.create_task(adapter.close())
    await close_started.wait()
    second = asyncio.create_task(adapter.close())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="adapter is closing"):
        await adapter.save_memory("too late", "user-1", 0.5, background=True)

    resource.close.assert_awaited_once_with()
    assert not first.done()
    assert not second.done()

    release_close.set()
    await asyncio.gather(first, second)

    resource.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_wait_for_task_timeout_does_not_cancel_background_write() -> None:
    adapter = _adapter()
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    adapter._background_tasks["task_1"] = task

    assert await adapter.wait_for_task("task_1", timeout=0.01) == {
        "status": "timeout",
        "task_id": "task_1",
    }
    assert not task.done()
    assert adapter.get_pending_tasks() == ["task_1"]

    release.set()
    assert await adapter.wait_for_task("task_1", timeout=1.0) == {
        "status": "completed",
        "result": True,
    }

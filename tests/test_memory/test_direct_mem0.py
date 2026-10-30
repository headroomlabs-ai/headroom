"""Tests for the Direct Mem0 adapter lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from headroom.memory.backends.direct_mem0 import DirectMem0Adapter, Mem0Config


def _adapter() -> DirectMem0Adapter:
    return DirectMem0Adapter(Mem0Config(enable_graph=True))


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


class _RaisingClient:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def delete(self, memory_id: str) -> None:
        raise self._exc

    def get(self, memory_id: str) -> None:
        raise self._exc


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["get_memory", "delete_memory"])
@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_malformed_exception_names_cannot_leak_from_memory_operations(
    op, level, monkeypatch, caplog
) -> None:
    error_type = type("PRIVATE_CREDENTIAL_CANARY\nFORGED_LOG_ENTRY", (RuntimeError,), {})
    adapter = _adapter()
    adapter._ensure_initialized = AsyncMock()
    adapter._mem0_client = _RaisingClient(error_type("backend content"))
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    with caplog.at_level(level, logger="headroom.memory.backends.direct_mem0"):
        result = await getattr(adapter, op)("m1")
    assert result is (None if op == "get_memory" else False)
    assert len([record for record in caplog.records if record.levelno == logging.WARNING]) == 1
    for record in caplog.records:
        rendered = logging.Formatter().format(record)
        assert "PRIVATE_CREDENTIAL_CANARY" not in rendered
        assert "FORGED_LOG_ENTRY" not in rendered
        assert "\n" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op", "exc", "warns"),
    [
        ("delete_memory", ValueError("Memory with id m1\nFAKE LOG LINE not found"), False),
        ("delete_memory", ValueError("payload user:secret rejected"), True),
        ("delete_memory", ValueError("Collection mem0 not found"), True),
        ("delete_memory", ValueError("Memory with id other-id not found"), True),
        ("delete_memory", RuntimeError("qdrant at user:secret@host down"), True),
        ("get_memory", RuntimeError("qdrant at user:secret@host down"), True),
    ],
)
async def test_backend_failures_warn_without_exception_text(
    op: str, exc: Exception, warns: bool
) -> None:
    """A missing id on delete is expected (DEBUG); other failures warn with the
    exception type only, never its text."""
    adapter = _adapter()
    adapter._ensure_initialized = AsyncMock()  # type: ignore[method-assign]
    adapter._mem0_client = _RaisingClient(exc)

    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("headroom.memory.backends.direct_mem0")
    handler = _Collect(level=logging.DEBUG)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        # Model-supplied id with a newline: must not split the log line.
        result = await getattr(adapter, op)("m1\nFAKE LOG LINE")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    assert not result
    # No level may carry the exception text, rendered as a handler would.
    for record in records:
        assert "secret" not in logging.Formatter().format(record)
    warnings = [r for r in records if r.levelname == "WARNING"]
    if warns:
        assert len(warnings) == 1
        assert type(exc).__name__ in warnings[0].getMessage()
        assert "secret" not in warnings[0].getMessage()
        assert warnings[0].exc_info is None
        assert "\n" not in warnings[0].getMessage()
    else:
        assert warnings == []
        assert any("not found" in r.getMessage() for r in records)


@pytest.mark.asyncio
async def test_huge_model_supplied_id_is_capped_in_the_warning() -> None:
    """A model can invent an id of any size; the warning must stay bounded."""
    adapter = _adapter()
    adapter._ensure_initialized = AsyncMock()  # type: ignore[method-assign]
    adapter._mem0_client = _RaisingClient(RuntimeError("backend down"))

    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("headroom.memory.backends.direct_mem0")
    handler = _Collect(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        assert await adapter.delete_memory("x" * 100_000) is False
        assert await adapter.delete_memory("short-id") is False
    finally:
        logger.removeHandler(handler)

    huge, short = (r.getMessage() for r in records)
    assert len(huge) < 200
    assert "…(+" in huge
    assert "'short-id'" in short and "…" not in short


class _Mem0V2SearchClient:
    """Minimal stand-in for the mem0ai 2.x ``Memory.search`` contract.

    Keyword-only signature with ``top_k``/``filters`` and no top-level entity
    parameters; any ``user_id``/``agent_id``/``run_id`` in ``**kwargs`` raises,
    and an unknown ``limit`` would be silently swallowed by ``**kwargs``.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._hits: object = [{"id": "m1", "memory": "Works at Netflix", "score": 0.9}]

    def search(self, query, *, top_k=20, filters=None, **kwargs):
        for name in ("user_id", "agent_id", "run_id"):
            if name in kwargs:
                raise ValueError(
                    f"Top-level entity parameters {{{name!r}}} are not supported in "
                    f"search(). Use filters={{{name!r}: '...'}} instead."
                )
        self.calls.append({"query": query, "top_k": top_k, "filters": filters, **kwargs})
        return self._hits


class _Mem0NullableMetadataClient(_Mem0V2SearchClient):
    """Real mem0ai 2.x hit shape: MemoryItem.model_dump() emits metadata=None."""

    def __init__(self) -> None:
        super().__init__()
        self._hits = {
            "results": [
                {
                    "id": "m1",
                    "memory": "Works at Netflix",
                    "hash": None,
                    "metadata": None,
                    "score": 0.9,
                    "created_at": None,
                    "updated_at": None,
                    "user_id": "u1",
                }
            ]
        }


@pytest.mark.asyncio
async def test_search_memories_uses_mem0_v2_filters_and_top_k() -> None:
    """search_memories must call mem0 2.x search with filters + top_k, not the
    removed top-level user_id/limit kwargs (issue #4246)."""
    adapter = _adapter()
    adapter._ensure_initialized = AsyncMock()  # type: ignore[method-assign]
    client = _Mem0V2SearchClient()
    adapter._mem0_client = client

    results = await adapter.search_memories("anything", user_id="u1", top_k=3)

    assert client.calls == [{"query": "anything", "top_k": 3, "filters": {"user_id": "u1"}}]
    assert [r.memory.content for r in results] == ["Works at Netflix"]


@pytest.mark.asyncio
async def test_search_memories_accepts_nullable_mem0_metadata() -> None:
    """Mem0 2.x dumps metadata=None when a hit has no extra payload keys.

    ``result.get("metadata", {})`` still returns None, which used to crash
    entity/session filtering and user_id construction (issue #4246 follow-up).
    """
    adapter = _adapter()
    adapter._ensure_initialized = AsyncMock()  # type: ignore[method-assign]
    adapter._mem0_client = _Mem0NullableMetadataClient()

    results = await adapter.search_memories("anything", user_id="u1", entities=["Netflix"], top_k=3)

    assert [r.memory.content for r in results] == ["Works at Netflix"]
    assert results[0].memory.user_id == "u1"
    assert results[0].memory.metadata == {}

    missed = await adapter.search_memories("anything", user_id="u1", session_id="s1")
    assert missed == []


def test_direct_qdrant_payload_includes_data_for_mem0_search() -> None:
    """Direct Qdrant writes must store text under ``data`` so mem0 2.x search
    does not skip the point (``if not payload.get("data"): continue``)."""
    from headroom.memory.backends.direct_mem0 import _direct_qdrant_payload

    payload = _direct_qdrant_payload("direct adapter fact", "tenant-a", 0.5)
    assert payload["data"] == "direct adapter fact"
    assert payload["memory"] == "direct adapter fact"
    assert payload["user_id"] == "tenant-a"
    # Caller metadata cannot strip the field mem0 search requires.
    overwritten = _direct_qdrant_payload(
        "fact", "u1", 0.1, metadata={"data": "nope", "memory": "nope"}
    )
    assert overwritten["data"] == "fact"
    assert overwritten["memory"] == "fact"


@pytest.mark.asyncio
@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_get_memory_survives_an_out_of_range_errno(level: int) -> None:
    """An OSError whose errno does not fit a C int must not escape get_memory,
    whether or not DEBUG (which builds the exception description) is on."""
    errno_value = 10 ** len("x" * 10000)  # built at runtime, far past any C int
    adapter = _adapter()
    adapter._ensure_initialized = AsyncMock()  # type: ignore[method-assign]
    adapter._mem0_client = _RaisingClient(OSError(errno_value, "backend down"))

    logger = logging.getLogger("headroom.memory.backends.direct_mem0")
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect(level=logging.DEBUG)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        assert await adapter.get_memory("m1") is None
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)

    assert [r.levelno for r in records if r.levelno == logging.WARNING] == [logging.WARNING]
    assert any(r.levelno == logging.DEBUG for r in records) is (level == logging.DEBUG)
    for record in records:
        logging.Formatter().format(record)


def _vec3(text: str) -> list[float]:
    """Deterministic 3-d unit vector so identical strings cosine-match at 1.0."""
    digest = hashlib.sha256(text.encode()).digest()
    raw = [((digest[i] / 255.0) * 2 - 1) for i in range(3)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


class _Deterministic3DEmbedder:
    """Stand-in for a Mem0 embedder: ``embed`` / ``embed_batch``, 3 dimensions."""

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return _vec3(text)

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return [_vec3(text) for text in texts]


def _inmemory_mem0(tmp_path: Any, collection_name: str = "headroom_memories"):
    """Real mem0ai Memory + in-memory Qdrant, no OpenAI/graph/network.

    Mirrors the maintainer reproduction: Memory.search against the real Qdrant
    store, deterministic 3-d embeddings, collection created by Mem0 itself.
    """
    pytest.importorskip("mem0")
    qdrant_client_mod = pytest.importorskip("qdrant_client")

    from mem0.memory import main as mem0_main

    client = qdrant_client_mod.QdrantClient(":memory:")
    embedder = _Deterministic3DEmbedder()

    def _embedder_factory(*_args: object, **_kwargs: object) -> _Deterministic3DEmbedder:
        return embedder

    def _llm_factory(*_args: object, **_kwargs: object) -> MagicMock:
        return MagicMock()

    with (
        patch.object(mem0_main, "MEM0_TELEMETRY", False),
        patch.object(mem0_main.EmbedderFactory, "create", side_effect=_embedder_factory),
        patch.object(mem0_main.LlmFactory, "create", side_effect=_llm_factory),
        patch.object(mem0_main, "capture_event", lambda *a, **k: None),
    ):
        memory = mem0_main.Memory.from_config(
            {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": collection_name,
                        "embedding_model_dims": 3,
                        "client": client,
                    },
                },
                "history_db_path": str(tmp_path / "mem0-history.db"),
            }
        )
    memory.embedding_model = embedder
    # Keep this test on dense cosine search. The BM25 sparse slot would load
    # fastembed's Qdrant/bm25 model, which is out of scope here.
    if hasattr(memory.vector_store, "_has_bm25_slot"):
        memory.vector_store._has_bm25_slot = False
    return memory, client


@pytest.mark.asyncio
async def test_direct_write_then_mem0_search_round_trip(tmp_path) -> None:
    """Write through the adapter, then retrieve through real mem0ai 2.x search.

    Covers the payload contract (``data`` vs ``memory``), nullable metadata on
    a Mem0-native hit, and the adapter's search decoding of both.
    """
    mem0_memory, qdrant = _inmemory_mem0(tmp_path)
    from qdrant_client.models import PointStruct

    adapter = DirectMem0Adapter(Mem0Config(enable_graph=False, collection_name="headroom_memories"))
    adapter._ensure_initialized = AsyncMock()  # type: ignore[method-assign]
    adapter._initialized = True
    adapter._qdrant_client = qdrant
    adapter._mem0_client = mem0_memory
    adapter._embed = lambda text: _vec3(text)  # type: ignore[method-assign]

    fact = "direct adapter fact"
    user_id = "tenant-a"
    await adapter._write_facts_to_qdrant(facts=[fact], user_id=user_id, importance=0.5)

    found = await adapter.search_memories(fact, user_id=user_id, top_k=5)
    assert [r.memory.content for r in found] == [fact]
    assert found[0].memory.user_id == user_id

    # Legacy payload shape: text under ``memory`` only. Mem0 2.x search skips it.
    qdrant.upsert(
        collection_name="headroom_memories",
        points=[
            PointStruct(
                id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                vector=_vec3("legacy memory only"),
                payload={
                    "memory": "legacy memory only",
                    "user_id": user_id,
                    "importance": 0.5,
                },
            )
        ],
    )
    legacy = await adapter.search_memories("legacy memory only", user_id=user_id, top_k=5)
    assert "legacy memory only" not in [r.memory.content for r in legacy]

    # Mem0-native payload (data + user_id, no extra keys) dumps metadata=None.
    qdrant.upsert(
        collection_name="headroom_memories",
        points=[
            PointStruct(
                id="11111111-2222-3333-4444-555555555555",
                vector=_vec3("native mem0 fact"),
                payload={"data": "native mem0 fact", "user_id": user_id},
            )
        ],
    )
    native = await adapter.search_memories("native mem0 fact", user_id=user_id, top_k=5)
    native_hit = next(r for r in native if r.memory.content == "native mem0 fact")
    assert native_hit.memory.metadata == {}
    assert native_hit.memory.user_id == user_id

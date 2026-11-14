"""Background IDs against embedded Qdrant with deterministic synthetic embeddings."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.memory.backends.direct_mem0 import DirectMem0Adapter, Mem0Config
from tests.test_memory.test_direct_mem0 import _inmemory_mem0, _vec3


@pytest.mark.asyncio
async def test_queued_id_supports_persistent_qdrant_read_update_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qdrant = pytest.importorskip("qdrant_client")
    from qdrant_client.models import Distance, PointIdsList, VectorParams

    storage = tmp_path / "qdrant"
    config = Mem0Config(enable_graph=False)
    client = qdrant.QdrantClient(path=str(storage))
    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
    )
    adapter = DirectMem0Adapter(config)
    adapter._initialized = True
    adapter._qdrant_client = client
    monkeypatch.setattr(adapter, "_embed", lambda text: [0.0, 1.0])
    try:
        facts = ["Prefers Python"]
        returned = await adapter.save_memory(
            content="fallback",
            user_id="alice",
            importance=0.7,
            facts=facts,
            background=True,
        )
        facts.append("Later caller edit")
        await adapter.flush_pending()
        status = adapter.get_task_status(returned.metadata["_task_id"])
        assert status["status"] == "completed"
        assert status["result"].id == returned.id
    finally:
        client.close()

    # Reopening exercises disk persistence, using the ID returned before the write.
    client = qdrant.QdrantClient(path=str(storage))
    try:
        points = client.retrieve(collection_name=config.collection_name, ids=[returned.id])
        assert len(points) == 1
        assert points[0].payload["data"] == "Prefers Python"
        assert points[0].payload["user_id"] == "alice"
        assert client.count(collection_name=config.collection_name).count == 1

        client.set_payload(
            collection_name=config.collection_name,
            points=[returned.id],
            payload={"data": "Prefers Rust", "memory": "Prefers Rust"},
            wait=True,
        )
        updated = client.retrieve(collection_name=config.collection_name, ids=[returned.id])
        assert updated[0].payload["data"] == "Prefers Rust"

        client.delete(
            collection_name=config.collection_name,
            points_selector=PointIdsList(points=[returned.id]),
            wait=True,
        )
        assert client.retrieve(collection_name=config.collection_name, ids=[returned.id]) == []
    finally:
        client.close()


@pytest.mark.asyncio
async def test_background_id_supports_real_mem0_read_update_delete(tmp_path) -> None:
    """Use real Mem0/Qdrant APIs with the existing deterministic embedder."""
    mem0_memory, qdrant = _inmemory_mem0(tmp_path)
    adapter = DirectMem0Adapter(Mem0Config(enable_graph=False))
    adapter._initialized = True
    adapter._qdrant_client = qdrant
    adapter._mem0_client = mem0_memory
    adapter._embed = _vec3
    try:
        facts = ["Prefers Python"]
        returned = await adapter.save_memory(
            content="fallback",
            user_id="alice",
            importance=0.7,
            facts=facts,
            background=True,
        )
        facts.append("Later caller edit")
        await adapter.flush_pending()
        status = adapter.get_task_status(returned.metadata["_task_id"])
        assert status["status"] == "completed"
        assert status["result"].id == returned.id

        stored = await adapter.get_memory(returned.id)
        assert stored is not None
        assert stored.content == "Prefers Python"
        assert stored.user_id == "alice"
        await adapter.update_memory(returned.id, "Prefers Rust", user_id="alice")
        updated = await adapter.get_memory(returned.id)
        assert updated is not None
        assert updated.content == "Prefers Rust"
        assert await adapter.delete_memory(returned.id, user_id="alice")
        assert await adapter.get_memory(returned.id) is None
        mem0_memory.llm.generate_response.assert_not_called()
    finally:
        await adapter.close()

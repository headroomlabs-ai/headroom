"""memory_update / memory_delete only touch the calling user's memories.

With one shared store (``global`` storage), a memory id that belongs to
another user is treated as not found: it is neither changed nor removed.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest

from headroom.memory.adapters.cache import LRUMemoryCache
from headroom.memory.adapters.fts5 import FTS5TextIndex
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex, is_sqlite_vec_available
from headroom.memory.backends.local import LocalBackend, LocalBackendConfig
from headroom.memory.config import MemoryConfig
from headroom.memory.core import HierarchicalMemory
from headroom.proxy.server import ProxyConfig, create_app

pytestmark = pytest.mark.skipif(not is_sqlite_vec_available(), reason="sqlite-vec unavailable")


class _FixedEmbedder:
    async def embed(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [await self.embed(t) for t in texts]


@pytest.fixture
def backend(tmp_path, monkeypatch) -> LocalBackend:
    async def create(cls: type, config: MemoryConfig | None = None) -> HierarchicalMemory:
        return HierarchicalMemory(
            SQLiteMemoryStore(tmp_path / "memory.db"),
            SQLiteVectorIndex(dimension=3, db_path=tmp_path / "vectors.db"),
            FTS5TextIndex(tmp_path / "memory.db"),
            _FixedEmbedder(),
            LRUMemoryCache(),
            MemoryConfig(auto_bubble=False, vector_dimension=3),
        )

    monkeypatch.setattr(HierarchicalMemory, "create", classmethod(create))
    return LocalBackend(
        LocalBackendConfig(db_path=str(tmp_path / "memory.db"), graph_persist=False)
    )


def _anthropic_turn(content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


async def _call_tool_as(
    tmp_path, backend: LocalBackend, user_id: str, tool: str, args: dict[str, Any]
) -> None:
    app = create_app(
        ProxyConfig(
            anthropic_api_url="https://anthropic.invalid",
            optimize=False,
            image_optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            discover_pipeline_extensions=False,
            compression_max_workers=1,
            memory_enabled=True,
            memory_storage_mode="global",
            memory_inject_tools=True,
            memory_inject_context=False,
            memory_db_path=str(tmp_path / "memory.db"),
        )
    )
    handler = app.state.proxy.memory_handler
    handler._backend = backend
    handler._initialized = True
    # Every scope resolves to the one shared fixture store, as in global mode.
    router = handler._router
    if router is not None:
        router.backend_for = lambda ctx: (backend, router.scope_for(ctx))

    turns = [
        _anthropic_turn(
            [{"type": "tool_use", "id": "toolu_1", "name": tool, "input": args}], "tool_use"
        ),
        _anthropic_turn([{"type": "text", "text": "done"}], "end_turn"),
    ]
    sent: list[dict[str, Any]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=turns[len(sent) - 1])

    mock = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app.state.proxy.http_client = mock
    app.state.proxy.http_client_h1 = mock
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
            },
            headers={
                "x-api-key": "FAKE-KEY",
                "anthropic-version": "2023-06-01",
                "x-headroom-user-id": user_id,
            },
            timeout=30,
        )
    await mock.aclose()
    assert response.status_code == 200
    # The proxy ran the tool and continued the turn.
    assert len(sent) == 2


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("memory_delete", {}),
        ("memory_update", {"new_content": "overwritten"}),
    ],
)
async def test_memory_owned_by_another_user_is_left_alone(
    tmp_path, monkeypatch, backend: LocalBackend, tool: str, args: dict[str, Any]
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    alice = await backend.save_memory(content="alice prefers tabs", user_id="alice")

    await _call_tool_as(tmp_path, backend, "bob", tool, {"memory_id": alice.id, **args})

    kept = await backend.get_memory(alice.id)
    assert kept is not None
    assert kept.content == "alice prefers tabs"
    assert kept.superseded_by is None


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("memory_delete", {}),
        ("memory_update", {"new_content": "bob prefers tabs"}),
    ],
)
async def test_user_can_still_change_own_memory(
    tmp_path, monkeypatch, backend: LocalBackend, tool: str, args: dict[str, Any]
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    bob = await backend.save_memory(content="bob prefers spaces", user_id="bob")

    await _call_tool_as(tmp_path, backend, "bob", tool, {"memory_id": bob.id, **args})

    old = await backend.get_memory(bob.id)
    if tool == "memory_delete":
        assert old is None
    else:
        assert old is not None
        assert old.superseded_by is not None

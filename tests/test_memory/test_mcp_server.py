import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from headroom.memory.models import Memory
from tests._mcp_stub import import_module_with_mcp_stub

mcp_server_mod = import_module_with_mcp_stub("headroom.memory.mcp_server")


class _CapturingServer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.list_tools_handler = None
        self.call_tool_handler = None

    def list_tools(self):
        def decorator(handler):
            self.list_tools_handler = handler
            return handler

        return decorator

    def call_tool(self):
        def decorator(handler):
            self.call_tool_handler = handler
            return handler

        return decorator


def test_warm_up_backend_batches_embedding_and_indexing() -> None:
    """Warm-up should batch missing embeddings and vector indexing."""
    warmup_embedding = np.ones(384, dtype=np.float32)
    batch_embeddings = [
        np.full(384, 2.0, dtype=np.float32),
        np.full(384, 3.0, dtype=np.float32),
    ]

    embedder = SimpleNamespace(
        embed=AsyncMock(return_value=warmup_embedding),
        embed_batch=AsyncMock(return_value=batch_embeddings),
    )
    store = SimpleNamespace(save_batch=AsyncMock())
    vector_index = SimpleNamespace(index_batch=AsyncMock(return_value=3))

    memory_without_embedding_a = Memory(content="First", user_id="alice")
    memory_with_embedding = Memory(
        content="Second",
        user_id="alice",
        embedding=np.full(384, 5.0, dtype=np.float32),
    )
    memory_without_embedding_b = Memory(content="Third", user_id="alice")
    memories = [
        memory_without_embedding_a,
        memory_with_embedding,
        memory_without_embedding_b,
    ]

    backend = SimpleNamespace(
        _ensure_initialized=AsyncMock(),
        _hierarchical_memory=SimpleNamespace(
            _embedder=embedder,
            _store=store,
            _vector_index=vector_index,
        ),
        get_user_memories=AsyncMock(return_value=memories),
    )

    asyncio.run(mcp_server_mod._warm_up_backend(backend, "alice"))

    backend._ensure_initialized.assert_awaited_once()
    backend.get_user_memories.assert_awaited_once_with("alice", limit=500)
    embedder.embed.assert_awaited_once_with("warmup")
    embedder.embed_batch.assert_awaited_once_with(["First", "Third"])
    store.save_batch.assert_awaited_once_with(
        [memory_without_embedding_a, memory_without_embedding_b]
    )
    vector_index.index_batch.assert_awaited_once_with(memories)
    assert np.array_equal(memory_without_embedding_a.embedding, batch_embeddings[0])
    assert np.array_equal(memory_without_embedding_b.embedding, batch_embeddings[1])


def test_tool_call_waits_for_handshake_warm_up(monkeypatch) -> None:
    async def scenario() -> None:
        warm_up_started = asyncio.Event()
        release_warm_up = asyncio.Event()
        backend = SimpleNamespace()

        async def warm_up(candidate, user_id: str) -> None:
            assert candidate is backend
            assert user_id == "alice"
            warm_up_started.set()
            await release_warm_up.wait()

        handle_search = AsyncMock(return_value=["search result"])
        monkeypatch.setattr(mcp_server_mod, "Server", _CapturingServer)
        monkeypatch.setattr(mcp_server_mod, "LocalBackend", lambda config: backend)
        monkeypatch.setattr(mcp_server_mod, "_warm_up_backend", warm_up)
        monkeypatch.setattr(mcp_server_mod, "_handle_search", handle_search)

        server = mcp_server_mod.create_memory_server("memory.db", user_id="alice")
        await server.list_tools_handler()
        await warm_up_started.wait()

        tool_call = asyncio.create_task(
            server.call_tool_handler("memory_search", {"query": "preferences"})
        )
        await asyncio.sleep(0)

        handle_search.assert_not_awaited()
        assert not tool_call.done()

        release_warm_up.set()
        assert await tool_call == ["search result"]
        handle_search.assert_awaited_once_with(
            backend,
            {"query": "preferences"},
            "alice",
        )

    asyncio.run(scenario())


def test_failed_handshake_init_is_discarded_and_retried(monkeypatch) -> None:
    async def scenario() -> None:
        first_warm_up_started = asyncio.Event()
        fail_first_warm_up = asyncio.Event()
        failed_backend_closed = asyncio.Event()

        async def close_failed_backend() -> None:
            failed_backend_closed.set()

        failed_backend = SimpleNamespace(close=AsyncMock(side_effect=close_failed_backend))
        ready_backend = SimpleNamespace(close=AsyncMock())
        backends = iter([failed_backend, ready_backend])

        async def warm_up(candidate, user_id: str) -> None:
            assert user_id == "alice"
            if candidate is failed_backend:
                first_warm_up_started.set()
                await fail_first_warm_up.wait()
                raise RuntimeError("warm-up failed")

        handle_search = AsyncMock(return_value=["search result"])
        monkeypatch.setattr(mcp_server_mod, "Server", _CapturingServer)
        monkeypatch.setattr(mcp_server_mod, "LocalBackend", lambda config: next(backends))
        monkeypatch.setattr(mcp_server_mod, "_warm_up_backend", warm_up)
        monkeypatch.setattr(mcp_server_mod, "_handle_search", handle_search)

        server = mcp_server_mod.create_memory_server("memory.db", user_id="alice")
        await server.list_tools_handler()
        await first_warm_up_started.wait()

        fail_first_warm_up.set()
        await failed_backend_closed.wait()
        await asyncio.sleep(0)

        failed_backend.close.assert_awaited_once()
        handle_search.assert_not_awaited()

        assert await server.call_tool_handler("memory_search", {"query": "preferences"}) == [
            "search result"
        ]
        handle_search.assert_awaited_once_with(
            ready_backend,
            {"query": "preferences"},
            "alice",
        )
        ready_backend.close.assert_not_awaited()

    asyncio.run(scenario())


def test_concurrent_tool_calls_share_backend_initialization(monkeypatch) -> None:
    async def scenario() -> None:
        warm_up_started = asyncio.Event()
        release_warm_up = asyncio.Event()
        backend = SimpleNamespace()
        created_backends = 0

        def create_backend(config):
            nonlocal created_backends
            created_backends += 1
            return backend

        async def warm_up(candidate, user_id: str) -> None:
            assert candidate is backend
            assert user_id == "alice"
            warm_up_started.set()
            await release_warm_up.wait()

        handle_search = AsyncMock(return_value=["search result"])
        monkeypatch.setattr(mcp_server_mod, "Server", _CapturingServer)
        monkeypatch.setattr(mcp_server_mod, "LocalBackend", create_backend)
        monkeypatch.setattr(mcp_server_mod, "_warm_up_backend", warm_up)
        monkeypatch.setattr(mcp_server_mod, "_handle_search", handle_search)

        server = mcp_server_mod.create_memory_server("memory.db", user_id="alice")
        calls = [
            asyncio.create_task(
                server.call_tool_handler("memory_search", {"query": f"query-{index}"})
            )
            for index in range(2)
        ]
        await warm_up_started.wait()
        await asyncio.sleep(0)

        assert created_backends == 1
        handle_search.assert_not_awaited()

        release_warm_up.set()
        assert await asyncio.gather(*calls) == [["search result"], ["search result"]]
        assert handle_search.await_count == 2
        assert all(call.args[0] is backend for call in handle_search.await_args_list)

    asyncio.run(scenario())


def test_memory_mcp_startup_context_reports_dynamic_project_db(tmp_path) -> None:
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    configured_db = str(project_dir / ".headroom" / "memory.db")

    context = mcp_server_mod._memory_mcp_startup_context(
        configured_db,
        project_dir,
        db_flag_present=False,
    )

    assert context == {
        "configured_db": configured_db,
        "resolved_db": configured_db,
        "config_source": "cwd-default",
        "cwd": str(project_dir),
        "project_root": str(project_dir),
        "storage_scope": "active-project",
        "path_exists": False,
        "path_readable": False,
        "resolution": "dynamic-cwd",
    }


def test_memory_mcp_startup_context_reports_static_external_db(tmp_path) -> None:
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    external_db = tmp_path / "shared-memory" / "memory.db"
    external_db.parent.mkdir()
    external_db.write_text("sqlite placeholder")

    context = mcp_server_mod._memory_mcp_startup_context(
        str(external_db),
        project_dir,
        db_flag_present=True,
    )

    assert context == {
        "configured_db": str(external_db),
        "resolved_db": str(external_db.resolve(strict=False)),
        "config_source": "cli-flag",
        "cwd": str(project_dir),
        "project_root": str(project_dir),
        "storage_scope": "external-memory-db",
        "path_exists": True,
        "path_readable": True,
        "resolution": "static-cli",
    }


def test_memory_mcp_startup_context_reports_custom_db_path(tmp_path) -> None:
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    custom_db = tmp_path / "queries.db"

    context = mcp_server_mod._memory_mcp_startup_context(
        str(custom_db),
        project_dir,
        db_flag_present=True,
    )

    assert context["storage_scope"] == "custom-db-path"
    assert context["config_source"] == "cli-flag"
    assert context["path_exists"] is False
    assert context["path_readable"] is False


def test_main_logs_memory_mcp_startup_context(monkeypatch, tmp_path, caplog) -> None:
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("USER", "codex-user")
    monkeypatch.setattr(mcp_server_mod.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(mcp_server_mod.sys, "argv", ["memory-mcp"])

    captured_run_payloads: list[object] = []
    monkeypatch.setattr(
        mcp_server_mod,
        "_run",
        lambda db_path, user_id: ("run", db_path, user_id),
    )
    monkeypatch.setattr(
        mcp_server_mod.asyncio,
        "run",
        lambda payload: captured_run_payloads.append(payload),
    )

    caplog.set_level("INFO", logger="headroom.memory.mcp")

    mcp_server_mod.main()

    assert captured_run_payloads == [
        ("run", str(project_dir / ".headroom" / "memory.db"), "codex-user")
    ]
    assert any(
        "Memory MCP startup: configured_db=" in record.message
        and "config_source=cwd-default" in record.message
        and "storage_scope=active-project" in record.message
        and "resolution=dynamic-cwd" in record.message
        for record in caplog.records
    )


def test_search_records_access_only_for_returned_memories() -> None:
    active = Memory(content="Active preference", user_id="alice")
    extra = Memory(content="Lower-ranked preference", user_id="alice")
    backend = SimpleNamespace(
        search_memories=AsyncMock(
            return_value=[
                SimpleNamespace(memory=active, score=0.9, related_entities=[]),
                SimpleNamespace(memory=extra, score=0.8, related_entities=[]),
            ]
        ),
        get_memory=AsyncMock(
            side_effect=lambda memory_id: {
                active.id: active,
                extra.id: extra,
            }[memory_id]
        ),
        record_access=AsyncMock(return_value=1),
    )

    result = asyncio.run(
        mcp_server_mod._handle_search(
            backend,
            {"query": "preference", "top_k": 1},
            "alice",
        )
    )

    backend.record_access.assert_awaited_once_with([active.id])
    assert "Active preference" in result[0].kwargs["text"]
    assert "Lower-ranked preference" not in result[0].kwargs["text"]


def test_search_does_not_record_superseded_memories() -> None:
    superseded = Memory(content="Old preference", user_id="alice")
    replacement = Memory(content="Current preference", user_id="alice")
    superseded.superseded_by = replacement.id
    backend = SimpleNamespace(
        search_memories=AsyncMock(
            return_value=[SimpleNamespace(memory=superseded, score=0.9, related_entities=[])]
        ),
        get_memory=AsyncMock(return_value=superseded),
        record_access=AsyncMock(),
    )

    result = asyncio.run(mcp_server_mod._handle_search(backend, {"query": "preference"}, "alice"))

    backend.record_access.assert_not_awaited()
    assert result[0].kwargs["text"] == "No memories found."


def test_search_fails_open_when_access_tracking_fails() -> None:
    memory = Memory(content="Useful preference", user_id="alice")
    backend = SimpleNamespace(
        search_memories=AsyncMock(
            return_value=[SimpleNamespace(memory=memory, score=0.9, related_entities=[])]
        ),
        get_memory=AsyncMock(return_value=memory),
        record_access=AsyncMock(side_effect=RuntimeError("write failed")),
    )

    result = asyncio.run(mcp_server_mod._handle_search(backend, {"query": "preference"}, "alice"))

    assert "Useful preference" in result[0].kwargs["text"]


def test_save_does_not_supersede_a_semantically_similar_memory() -> None:
    existing = Memory(content="Use compact output by default", user_id="alice")
    saved = Memory(content="Keep semantic compression disabled", user_id="alice")
    backend = SimpleNamespace(
        search_memories=AsyncMock(return_value=[SimpleNamespace(memory=existing, score=0.91)]),
        update_memory=AsyncMock(),
        save_memory=AsyncMock(return_value=saved),
    )

    result = asyncio.run(
        mcp_server_mod._handle_save(
            backend,
            {"facts": ["Keep semantic compression disabled"], "importance": 0.9},
            "alice",
        )
    )

    backend.search_memories.assert_not_awaited()
    backend.update_memory.assert_not_awaited()
    backend.save_memory.assert_awaited_once_with(
        content="Keep semantic compression disabled",
        user_id="alice",
        importance=0.9,
    )
    assert result[0].kwargs["text"].startswith("Saved 1 new, updated 0 existing")

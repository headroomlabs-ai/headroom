import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from headroom.memory.models import Memory
from tests._mcp_stub import import_module_with_mcp_stub

mcp_server_mod = import_module_with_mcp_stub("headroom.memory.mcp_server")


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

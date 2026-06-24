from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.proxy import memory_handler as memory_handler_module
from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler


@pytest.fixture
def handler(tmp_path: Path) -> MemoryHandler:
    return MemoryHandler(
        MemoryConfig(
            enabled=False,
            use_native_tool=True,
            native_memory_dir=str(tmp_path / "native"),
        ),
        agent_type="codex",
    )


class FakeBackend:
    def __init__(self) -> None:
        self.search_results: list[object] = []
        self.saved: list[dict[str, object]] = []
        self.updated: list[dict[str, object]] = []
        self.deleted: list[str] = []
        self.raise_on: str | None = None

    async def search_memories(self, **kwargs):  # noqa: ANN003
        if self.raise_on == "search":
            raise RuntimeError("search failed")
        return self.search_results

    async def save_memory(self, **kwargs):  # noqa: ANN003
        if self.raise_on == "save":
            raise RuntimeError("save failed")
        self.saved.append(kwargs)
        return SimpleNamespace(id=f"mem-{len(self.saved)}", content=kwargs["content"])

    async def update_memory(self, **kwargs):  # noqa: ANN003
        if self.raise_on == "update":
            raise RuntimeError("update failed")
        self.updated.append(kwargs)
        return SimpleNamespace(id=kwargs["memory_id"])

    async def delete_memory(self, memory_id: str):  # noqa: ANN201
        if self.raise_on == "delete":
            raise RuntimeError("delete failed")
        self.deleted.append(memory_id)
        return True


def make_result(
    memory_id: str,
    content: str,
    *,
    score: float = 0.9,
    metadata: dict[str, object] | None = None,
    related_entities: list[str] | None = None,
    created_at: str | None = None,
    importance: float = 0.5,
) -> object:
    return SimpleNamespace(
        memory=SimpleNamespace(
            id=memory_id,
            content=content,
            metadata=metadata or {},
            created_at=created_at,
            importance=importance,
        ),
        score=score,
        related_entities=related_entities or [],
    )


def test_resolve_native_path_blocks_traversal(handler: MemoryHandler) -> None:
    resolved = handler._resolve_native_path("/memories/topic.txt", "u1")
    assert resolved.name == "topic.txt"
    assert "u1" in str(resolved)

    with pytest.raises(ValueError, match="Path traversal detected"):
        handler._resolve_native_path("/memories/../escape.txt", "u1")


def test_native_view_lists_directory_and_reads_files(handler: MemoryHandler) -> None:
    root = handler._resolve_native_path("/memories", "u1")
    (root / "alpha.txt").write_text("line1\nline2\nline3", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "beta.txt").write_text("nested", encoding="utf-8")
    (root / ".hidden").write_text("skip", encoding="utf-8")
    (root / "node_modules").mkdir()

    listing = handler._native_view({"path": "/memories"}, "u1")
    assert "/memories/alpha.txt" in listing
    assert "/memories/nested/beta.txt" in listing
    assert ".hidden" not in listing
    assert "/memories/node_modules" not in listing

    file_view = handler._native_view({"path": "/memories/alpha.txt", "view_range": [2, 3]}, "u1")
    assert "2\tline2" in file_view
    assert "3\tline3" in file_view


def test_native_view_handles_missing_paths_and_latin1(handler: MemoryHandler) -> None:
    missing = handler._native_view({"path": "/memories/missing.txt"}, "u1")
    assert "does not exist" in missing

    latin_path = handler._resolve_native_path("/memories/latin.txt", "u1")
    latin_path.write_bytes("caf\xe9".encode("latin-1"))
    viewed = handler._native_view({"path": "/memories/latin.txt"}, "u1")
    assert "cafe" not in viewed
    assert "café" in viewed


def test_native_create_insert_delete_and_rename(handler: MemoryHandler) -> None:
    assert handler._native_create(
        {"path": "/memories/note.txt", "file_text": "a\nb"}, "u1"
    ).startswith("File created successfully")
    assert handler._native_create(
        {"path": "/memories/note.txt", "file_text": "dup"}, "u1"
    ).startswith("Error: File /memories/note.txt already exists")

    inserted = handler._native_insert(
        {"path": "/memories/note.txt", "insert_line": 1, "insert_text": "middle"},
        "u1",
    )
    assert inserted == "The file /memories/note.txt has been edited."
    assert "middle" in handler._resolve_native_path("/memories/note.txt", "u1").read_text(
        encoding="utf-8"
    )

    renamed = handler._native_rename(
        {"old_path": "/memories/note.txt", "new_path": "/memories/archive/renamed.txt"},
        "u1",
    )
    assert renamed == "Successfully renamed /memories/note.txt to /memories/archive/renamed.txt"

    deleted = handler._native_delete_file({"path": "/memories/archive"}, "u1")
    assert deleted == "Successfully deleted /memories/archive"


def test_native_insert_validates_range_and_path(handler: MemoryHandler) -> None:
    assert (
        handler._native_insert({"insert_line": 0, "insert_text": "x"}, "u1")
        == "Error: path is required"
    )
    assert "does not exist" in handler._native_insert(
        {"path": "/memories/missing.txt", "insert_line": 0, "insert_text": "x"},
        "u1",
    )

    note = handler._resolve_native_path("/memories/note.txt", "u1")
    note.write_text("a\nb", encoding="utf-8")
    invalid = handler._native_insert(
        {"path": "/memories/note.txt", "insert_line": 4, "insert_text": "x"},
        "u1",
    )
    assert "Invalid `insert_line` parameter: 4" in invalid


def test_native_str_replace_covers_missing_multiple_and_success(handler: MemoryHandler) -> None:
    note = handler._resolve_native_path("/memories/note.txt", "u1")
    note.write_text("hello\nhello\nworld", encoding="utf-8")

    assert (
        handler._native_str_replace({"old_str": "hello", "new_str": "bye"}, "u1")
        == "Error: path is required"
    )
    assert (
        handler._native_str_replace({"path": "/memories/note.txt", "new_str": "bye"}, "u1")
        == "Error: old_str is required"
    )

    multiple = handler._native_str_replace(
        {"path": "/memories/note.txt", "old_str": "hello", "new_str": "bye"},
        "u1",
    )
    assert "Multiple occurrences of old_str `hello` in lines: 1, 2" in multiple

    note.write_text("hello\nworld", encoding="utf-8")
    missing = handler._native_str_replace(
        {"path": "/memories/note.txt", "old_str": "nope", "new_str": "bye"},
        "u1",
    )
    assert "did not appear verbatim" in missing

    success = handler._native_str_replace(
        {"path": "/memories/note.txt", "old_str": "hello", "new_str": "bye"},
        "u1",
    )
    assert "The memory file has been edited." in success
    assert "bye" in note.read_text(encoding="utf-8")


def test_native_delete_and_rename_validate_inputs(handler: MemoryHandler) -> None:
    assert handler._native_delete_file({}, "u1") == "Error: path is required"
    assert "does not exist" in handler._native_delete_file({"path": "/memories/missing.txt"}, "u1")

    assert (
        handler._native_rename({"new_path": "/memories/new.txt"}, "u1")
        == "Error: old_path is required"
    )
    assert (
        handler._native_rename({"old_path": "/memories/old.txt"}, "u1")
        == "Error: new_path is required"
    )
    assert "does not exist" in handler._native_rename(
        {"old_path": "/memories/old.txt", "new_path": "/memories/new.txt"},
        "u1",
    )

    old = handler._resolve_native_path("/memories/old.txt", "u1")
    new = handler._resolve_native_path("/memories/new.txt", "u1")
    old.write_text("x", encoding="utf-8")
    new.write_text("y", encoding="utf-8")
    assert (
        handler._native_rename(
            {"old_path": "/memories/old.txt", "new_path": "/memories/new.txt"},
            "u1",
        )
        == "Error: The destination /memories/new.txt already exists"
    )


@pytest.mark.asyncio
async def test_execute_native_memory_tool_dispatches_and_wraps_errors(
    handler: MemoryHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler._backend = object()
    called: list[tuple[str, dict[str, object], str]] = []

    async def fake_ensure_initialized() -> None:
        return None

    async def fake_view(input_data, user_id):  # noqa: ANN001
        called.append(("view", input_data, user_id))
        return "viewed"

    async def fake_create(input_data, user_id):  # noqa: ANN001
        called.append(("create", input_data, user_id))
        return "created"

    monkeypatch.setattr(handler, "_ensure_initialized", fake_ensure_initialized)
    monkeypatch.setattr(handler, "_native_view_semantic", fake_view)
    monkeypatch.setattr(handler, "_native_create_semantic", fake_create)

    assert await handler._execute_native_memory_tool({"command": "view"}, "u1") == "viewed"
    assert await handler._execute_native_memory_tool({"command": "create"}, "u1") == "created"
    assert (
        await handler._execute_native_memory_tool({"command": "bad"}, "u1")
        == "Error: Unknown command 'bad'"
    )

    async def boom(input_data, user_id):  # noqa: ANN001
        raise RuntimeError("oops")

    monkeypatch.setattr(handler, "_native_view_semantic", boom)
    assert await handler._execute_native_memory_tool({"command": "view"}, "u1") == "Error: oops"
    assert [entry[0] for entry in called] == ["view", "create"]


@pytest.mark.asyncio
async def test_semantic_search_recent_all_and_overview(handler: MemoryHandler) -> None:
    backend = FakeBackend()
    handler._backend = backend
    backend.search_results = [
        make_result(
            "m1",
            "Alice likes pizza and pasta",
            score=0.91,
            related_entities=["Alice", "pizza"],
            created_at="2026-04-22",
        ),
        make_result("m2", "Bob prefers ramen", score=0.83),
    ]

    search_text = await handler._semantic_search("pizza", "u1")
    assert "Found 2 memories matching 'pizza'" in search_text
    assert "[91% match] Alice likes pizza and pasta" in search_text
    assert "Related: Alice, pizza" in search_text

    recent_text = await handler._get_recent_memories("u1", limit=2)
    assert "Recent memories:" in recent_text
    assert "(2026-04-22)" in recent_text

    all_text = await handler._list_all_memories("u1", limit=2)
    assert "Showing up to 2 memories:" in all_text
    assert "Showing first 2" in all_text

    overview = await handler._get_memory_overview("u1")
    assert "Memory System (2 memories stored)" in overview
    assert "view /memories/search/<your query>" in overview


@pytest.mark.asyncio
async def test_semantic_helpers_handle_empty_backend_and_errors(handler: MemoryHandler) -> None:
    assert await handler._semantic_search("x", "u1") == "Error: Memory backend not initialized"
    assert await handler._get_recent_memories("u1") == "Error: Memory backend not initialized"
    assert await handler._list_all_memories("u1") == "Error: Memory backend not initialized"
    assert await handler._get_memory_overview("u1") == "Error: Memory backend not initialized"

    backend = FakeBackend()
    handler._backend = backend
    assert "No memories found matching 'x'" in await handler._semantic_search("x", "u1")
    assert "No memories stored yet." in await handler._list_all_memories("u1")
    assert "No memories stored yet." in await handler._get_recent_memories("u1")

    backend.raise_on = "search"
    assert "Error searching memories: search failed" == await handler._semantic_search("x", "u1")
    assert "Error getting recent memories: search failed" == await handler._get_recent_memories(
        "u1"
    )
    assert "Error listing memories: search failed" == await handler._list_all_memories("u1")
    overview = await handler._get_memory_overview("u1")
    assert "📁 Memory System" in overview
    assert "To SEARCH memories" in overview


@pytest.mark.asyncio
async def test_native_view_semantic_routes_paths(
    handler: MemoryHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, object]] = []

    async def fake_search(query, user_id, top_k=5):  # noqa: ANN001
        seen.append(("search", query))
        return "search-result"

    async def fake_recent(user_id, limit=10):  # noqa: ANN001
        seen.append(("recent", limit))
        return "recent-result"

    async def fake_all(user_id, limit=20):  # noqa: ANN001
        seen.append(("all", limit))
        return "all-result"

    async def fake_overview(user_id):  # noqa: ANN001
        seen.append(("overview", user_id))
        return "overview-result"

    monkeypatch.setattr(handler, "_semantic_search", fake_search)
    monkeypatch.setattr(handler, "_get_recent_memories", fake_recent)
    monkeypatch.setattr(handler, "_list_all_memories", fake_all)
    monkeypatch.setattr(handler, "_get_memory_overview", fake_overview)

    assert (
        await handler._native_view_semantic({"path": "/memories/search/pizza"}, "u1")
        == "search-result"
    )
    assert (
        await handler._native_view_semantic({"path": "/memories/recent"}, "u1") == "recent-result"
    )
    assert await handler._native_view_semantic({"path": "/memories/all"}, "u1") == "all-result"
    assert await handler._native_view_semantic({"path": "/memories"}, "u1") == "overview-result"
    assert (
        await handler._native_view_semantic({"path": "/memories/work/projects"}, "u1")
        == "search-result"
    )
    assert (await handler._native_view_semantic({"path": "/memories/search/"}, "u1")).startswith(
        "Error: Please provide a search query"
    )
    assert seen == [
        ("search", "pizza"),
        ("recent", 10),
        ("all", 20),
        ("overview", "u1"),
        ("search", "work projects"),
    ]


@pytest.mark.asyncio
async def test_native_semantic_create_append_delete_and_rename(handler: MemoryHandler) -> None:
    backend = FakeBackend()
    handler._backend = backend

    assert await handler._native_create_semantic({}, "u1") == "Error: path is required"
    assert (
        await handler._native_create_semantic({"path": "/memories/topic.txt"}, "u1")
        == "Error: file_text is required (the memory content)"
    )

    created = await handler._native_create_semantic(
        {"path": "/memories/topic.txt", "file_text": "prefers pizza"},
        "u1",
    )
    assert created == "File created successfully at: /memories/topic.txt"
    assert backend.saved[-1]["metadata"] == {
        "virtual_path": "/memories/topic.txt",
        "topic": "topic",
    }

    assert await handler._native_append_semantic({}, "u1") == "Error: path is required"
    assert (
        await handler._native_append_semantic({"path": "/memories/topic.txt"}, "u1")
        == "Error: insert_text is required"
    )
    appended = await handler._native_append_semantic(
        {"path": "/memories/topic.txt", "insert_text": "and pasta"},
        "u1",
    )
    assert appended == "The file /memories/topic.txt has been edited."
    assert backend.saved[-1]["metadata"]["appended"] is True

    backend.search_results = [
        make_result(
            "m1", "prefers pizza", metadata={"virtual_path": "/memories/topic.txt"}, score=0.6
        ),
        make_result("m2", "prefers pasta", metadata={}, score=0.91),
    ]
    deleted = await handler._native_delete_semantic({"path": "/memories/topic.txt"}, "u1")
    assert deleted == "Successfully deleted /memories/topic.txt"
    assert backend.deleted == ["m1", "m2"]

    backend.search_results = [
        make_result(
            "m3", "old content", metadata={"virtual_path": "/memories/old.txt"}, importance=0.7
        )
    ]
    renamed = await handler._native_rename_semantic(
        {"old_path": "/memories/old.txt", "new_path": "/memories/new/topic.txt"},
        "u1",
    )
    assert renamed == "Successfully renamed /memories/old.txt to /memories/new/topic.txt"
    assert backend.deleted[-1] == "m3"
    assert backend.saved[-1]["metadata"] == {
        "virtual_path": "/memories/new/topic.txt",
        "topic": "new_topic",
    }


@pytest.mark.asyncio
async def test_native_semantic_update_delete_rename_and_backend_errors(
    handler: MemoryHandler,
) -> None:
    backend = FakeBackend()
    handler._backend = backend

    assert await handler._native_update_semantic({}, "u1") == "Error: path is required"
    assert (
        await handler._native_update_semantic({"path": "/memories/t.txt"}, "u1")
        == "Error: old_str is required"
    )

    backend.search_results = [
        make_result("m1", "hello hello world", metadata={"virtual_path": "/memories/t.txt"})
    ]
    multi = await handler._native_update_semantic(
        {"path": "/memories/t.txt", "old_str": "hello", "new_str": "bye"},
        "u1",
    )
    assert "Multiple occurrences of old_str `hello`" in multi

    backend.search_results = [
        make_result("m1", "hello world", metadata={"virtual_path": "/memories/t.txt"})
    ]
    edited = await handler._native_update_semantic(
        {"path": "/memories/t.txt", "old_str": "hello", "new_str": "bye"},
        "u1",
    )
    assert "The memory file has been edited." in edited
    assert backend.updated[-1]["new_content"] == "bye world"

    class NoUpdateBackend:
        def __init__(self) -> None:
            self.search_results: list[object] = []
            self.saved: list[dict[str, object]] = []
            self.deleted: list[str] = []

        async def search_memories(self, **kwargs):  # noqa: ANN003
            return self.search_results

        async def delete_memory(self, memory_id: str):  # noqa: ANN201
            self.deleted.append(memory_id)
            return True

        async def save_memory(self, **kwargs):  # noqa: ANN003
            self.saved.append(kwargs)
            return SimpleNamespace(id=f"mem-{len(self.saved)}")

    no_update_backend = NoUpdateBackend()
    no_update_backend.search_results = [
        make_result("m2", "alpha beta", metadata={"virtual_path": "/memories/t.txt"})
    ]
    handler._backend = no_update_backend
    fallback = await handler._native_update_semantic(
        {"path": "/memories/t.txt", "old_str": "alpha", "new_str": "omega"},
        "u1",
    )
    assert "The memory file has been edited." in fallback
    assert no_update_backend.deleted[-1] == "m2"
    assert no_update_backend.saved[-1]["content"] == "omega beta"

    backend = FakeBackend()
    handler._backend = backend
    assert await handler._native_delete_semantic({}, "u1") == "Error: path is required"
    assert await handler._native_rename_semantic({}, "u1") == "Error: old_path is required"
    assert (
        await handler._native_rename_semantic({"old_path": "/memories/a.txt"}, "u1")
        == "Error: new_path is required"
    )
    assert (
        await handler._native_delete_semantic({"path": "/memories/x.txt"}, "u1")
        == "Error: The path /memories/x.txt does not exist"
    )
    assert (
        await handler._native_rename_semantic(
            {"old_path": "/memories/x.txt", "new_path": "/memories/y.txt"},
            "u1",
        )
        == "Error: The path /memories/x.txt does not exist"
    )

    backend.search_results = [
        make_result("m9", "content", metadata={"virtual_path": "/memories/other.txt"}, score=0.1)
    ]
    assert (
        await handler._native_delete_semantic({"path": "/memories/x.txt"}, "u1")
        == "Error: The path /memories/x.txt does not exist"
    )
    assert (
        await handler._native_rename_semantic(
            {"old_path": "/memories/x.txt", "new_path": "/memories/y.txt"},
            "u1",
        )
        == "Error: The path /memories/x.txt does not exist"
    )

    backend.raise_on = "search"
    assert (
        await handler._native_create_semantic(
            {"path": "/memories/topic.txt", "file_text": "content"},
            "u1",
        )
        == "File created successfully at: /memories/topic.txt"
    )
    backend.raise_on = "save"
    assert (
        await handler._native_create_semantic(
            {"path": "/memories/topic.txt", "file_text": "content"}, "u1"
        )
    ).startswith("Error: ")
    assert (
        await handler._native_append_semantic(
            {"path": "/memories/topic.txt", "insert_text": "content"}, "u1"
        )
    ).startswith("Error: ")
    backend.raise_on = "search"
    assert (
        await handler._native_update_semantic(
            {"path": "/memories/t.txt", "old_str": "a", "new_str": "b"}, "u1"
        )
    ).startswith("Error: ")
    assert (await handler._native_delete_semantic({"path": "/memories/x.txt"}, "u1")).startswith(
        "Error: "
    )
    assert (
        await handler._native_rename_semantic(
            {"old_path": "/memories/x.txt", "new_path": "/memories/y.txt"},
            "u1",
        )
    ).startswith("Error: ")


@pytest.mark.asyncio
async def test_execute_search_update_delete_and_handler_status(
    handler: MemoryHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend()
    handler._backend = backend

    assert json.loads(await handler._execute_search({}, "u1")) == {
        "status": "error",
        "error": "query is required",
    }

    backend.search_results = [
        make_result("m1", "pizza", score=0.9123, related_entities=["food", "italy"])
    ]
    search_payload = json.loads(
        await handler._execute_search(
            {"query": "pizza", "top_k": 3, "include_related": False, "entities": ["food"]},
            "u1",
        )
    )
    assert search_payload["status"] == "found"
    assert search_payload["count"] == 1
    assert search_payload["memories"][0] == {
        "id": "m1",
        "content": "pizza",
        "score": 0.912,
        "entities": ["food", "italy"],
    }

    assert json.loads(await handler._execute_update({}, "u1")) == {
        "status": "error",
        "error": "memory_id is required",
    }
    assert json.loads(await handler._execute_update({"memory_id": "m1"}, "u1")) == {
        "status": "error",
        "error": "new_content is required",
    }

    backend.search_results = [make_result("m1", "old content")]
    update_payload = json.loads(
        await handler._execute_update(
            {"memory_id": "m1", "new_content": "new content", "reason": "cleanup"},
            "u1",
            provider="openai",
        )
    )
    assert update_payload == {"status": "updated", "memory_id": "m1"}
    assert backend.updated[-1]["new_content"] == "new content"

    class NoUpdateBackend:
        def __init__(self) -> None:
            self.deleted: list[str] = []
            self.saved: list[dict[str, object]] = []

        async def delete_memory(self, memory_id: str):  # noqa: ANN201
            self.deleted.append(memory_id)
            return True

        async def save_memory(self, **kwargs):  # noqa: ANN003
            self.saved.append(kwargs)
            return SimpleNamespace(id="m2")

    no_update_backend = NoUpdateBackend()
    handler._backend = no_update_backend
    update_fallback = json.loads(
        await handler._execute_update({"memory_id": "m1", "new_content": "replacement"}, "u1")
    )
    assert update_fallback == {
        "status": "updated",
        "memory_id": "m2",
        "note": "Replaced via delete+save",
    }
    assert no_update_backend.deleted == ["m1"]

    handler._backend = backend
    assert json.loads(await handler._execute_delete({}, "u1")) == {
        "status": "error",
        "error": "memory_id is required",
    }
    delete_payload = json.loads(await handler._execute_delete({"memory_id": "m1"}, "u1"))
    assert delete_payload == {"status": "deleted", "memory_id": "m1"}

    assert handler.health_status() == {
        "enabled": False,
        "backend": "local",
        "initialized": False,
        "native_tool": True,
        "bridge_enabled": False,
    }

    seen = {"count": 0}

    async def fake_ensure_initialized() -> None:
        seen["count"] += 1

    monkeypatch.setattr(handler, "_ensure_initialized", fake_ensure_initialized)
    await handler.ensure_initialized()
    assert seen["count"] == 1


@pytest.mark.asyncio
async def test_warmup_embedder_and_close(handler: MemoryHandler) -> None:
    assert await handler.warmup_embedder() is False

    class FakeEmbedder:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def embed(self, value: str) -> None:
            self.calls.append(value)

    embedder = FakeEmbedder()
    handler._initialized = True
    handler._backend = SimpleNamespace(_hierarchical_memory=SimpleNamespace(_embedder=embedder))
    assert await handler.warmup_embedder() is True
    assert embedder.calls == ["warmup"]

    handler._backend = SimpleNamespace(
        _hierarchical_memory=SimpleNamespace(_embedder=SimpleNamespace())
    )
    assert await handler.warmup_embedder() is False

    handler._backend = SimpleNamespace(close=lambda: None)
    await handler.close()
    assert handler.backend is None
    assert handler.initialized is False


@pytest.mark.asyncio
async def test_execute_memory_tool_save_and_background_dedup(
    handler: MemoryHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend()
    handler._backend = backend

    assert json.loads(await handler._execute_memory_tool("unknown", {}, "u1")) == {
        "error": "Unknown tool: unknown"
    }
    assert json.loads(await handler._execute_memory_tool("memory_save", {}, "u1")) == {
        "status": "error",
        "error": "content is required",
    }

    created_tasks: list[object] = []

    def fake_create_task(coro):  # noqa: ANN001
        created_tasks.append(coro)
        coro.close()
        return SimpleNamespace()

    monkeypatch.setattr("headroom.proxy.memory_handler.asyncio.create_task", fake_create_task)
    backend.search_results = [
        make_result(
            "other",
            "Very similar memory content " * 5,
            score=0.8,
            metadata={"source_agent": "claude"},
        ),
        make_result("mem-1", "self result", score=0.99),
    ]

    saved = json.loads(
        await handler._execute_memory_tool(
            "memory_save",
            {
                "content": "Useful fact",
                "importance": 0.7,
                "facts": ["fact"],
                "entities": ["entity"],
                "extracted_entities": ["entity"],
                "relationships": ["rel"],
                "extracted_relationships": ["rel"],
            },
            "u1",
            provider="openai",
        )
    )
    assert saved["status"] == "saved"
    assert saved["memory_id"] == "mem-1"
    assert "Similar memory exists" in saved["note"]
    assert "saved by claude" in saved["note"]
    assert backend.saved[-1]["metadata"]["source_provider"] == "openai"
    assert len(created_tasks) == 1

    backend.raise_on = "save"
    errored = json.loads(await handler._execute_memory_tool("memory_save", {"content": "x"}, "u1"))
    assert errored == {"status": "error", "error": "save failed"}


@pytest.mark.asyncio
async def test_execute_save_handles_search_failure_and_background_dedup_filters(
    handler: MemoryHandler,
) -> None:
    backend = FakeBackend()
    handler._backend = backend

    backend.raise_on = "search"
    saved = json.loads(await handler._execute_save({"content": "Useful fact"}, "u1"))
    assert saved == {"status": "saved", "memory_id": "mem-1", "content": "Useful fact"}

    backend.raise_on = None
    similar = [
        make_result("mem-1", "same", score=0.99),
        make_result("old-1", "duplicate", score=0.95, metadata={}),
        make_result("old-2", "already handled", score=0.99, metadata={"superseded_by": "new"}),
        make_result("old-3", "too low", score=0.5, metadata={}),
    ]
    await handler._background_dedup("mem-1", similar, "u1")
    assert backend.deleted == ["old-1"]

    backend.raise_on = "delete"
    await handler._background_dedup("mem-1", [make_result("old-4", "duplicate", score=0.95)], "u1")


def test_inject_tools_extract_query_and_has_tool_calls(
    handler: MemoryHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        handler,
        "_get_memory_tools",
        lambda: [
            {
                "function": {
                    "name": "memory_save",
                    "description": "save memory",
                    "parameters": {"type": "object"},
                }
            }
        ],
    )

    anthropic_tools, injected = handler.inject_tools([], "anthropic")
    assert injected is True
    assert anthropic_tools == [{"type": "memory_20250818", "name": "memory"}]

    custom_handler = MemoryHandler(
        MemoryConfig(enabled=False, inject_tools=True), agent_type="codex"
    )
    monkeypatch.setattr(
        custom_handler,
        "_get_memory_tools",
        lambda: [
            {
                "function": {
                    "name": "memory_save",
                    "description": "save memory",
                    "parameters": {"type": "object"},
                }
            }
        ],
    )
    anthropic_custom, injected_custom = custom_handler.inject_tools([], "anthropic")
    assert injected_custom is True
    assert anthropic_custom == [
        {"name": "memory_save", "description": "save memory", "input_schema": {"type": "object"}}
    ]
    openai_custom, _ = custom_handler.inject_tools([], "openai")
    assert openai_custom == [
        {
            "function": {
                "name": "memory_save",
                "description": "save memory",
                "parameters": {"type": "object"},
            }
        }
    ]
    existing, was_injected = custom_handler.inject_tools(
        [{"function": {"name": "memory_save"}}],
        "openai",
    )
    assert was_injected is False
    assert existing == [{"function": {"name": "memory_save"}}]

    assert handler._extract_user_query([{"role": "assistant", "content": "skip"}]) == ""
    # Pre-PR-this method truncated at 500 chars; that was a real bug
    # (none of Letta / Mem0 / Cognee / Supermemory truncate the
    # retrieval query). Now returns the full message — embedder
    # handles its own context window. See ``MemoryQuery``.
    assert handler._extract_user_query([{"role": "user", "content": "x" * 600}]) == "x" * 600
    assert (
        handler._extract_user_query(
            [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image"}]}]
        )
        == "hello"
    )

    anthropic_response = {
        "content": [{"type": "tool_use", "name": "memory_save", "id": "1", "input": {}}]
    }
    openai_response = {
        "choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "memory_save"}}]}}]
    }
    responses_api = {"output": [{"type": "function_call", "call_id": "2", "name": "memory"}]}
    assert handler.has_memory_tool_calls(anthropic_response, "anthropic") is True
    assert handler.has_memory_tool_calls(openai_response, "openai") is True
    assert handler.has_memory_tool_calls(responses_api, "openai") is True
    assert handler.has_memory_tool_calls({"content": []}, "anthropic") is False


@pytest.mark.asyncio
async def test_memory_handler_misc_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    handler = MemoryHandler(
        MemoryConfig(
            enabled=False,
            use_native_tool=True,
            inject_tools=True,
            native_memory_dir=str(tmp_path / "native"),
        ),
        agent_type="codex",
    )
    first_lock = handler._get_init_lock()
    assert handler._get_init_lock() is first_lock
    assert handler.get_beta_headers() == {"anthropic-beta": "context-management-2025-06-27"}

    disabled_headers = MemoryHandler(
        MemoryConfig(enabled=False, use_native_tool=False), agent_type="codex"
    )
    assert disabled_headers.get_beta_headers() == {}

    tools, injected = handler._inject_native_tool([])
    assert injected is True
    assert tools == [{"type": "memory_20250818", "name": "memory"}]
    same_tools, same_injected = handler._inject_native_tool([{"name": "memory"}])
    assert same_injected is False
    assert same_tools == [{"name": "memory"}]

    calls = {"count": 0}
    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.memory.tools",
        SimpleNamespace(
            get_memory_tools_optimized=lambda: (
                calls.__setitem__("count", calls["count"] + 1) or [{"name": "tool"}]
            )
        ),
    )
    cache_handler = MemoryHandler(MemoryConfig(enabled=False), agent_type="codex")
    assert cache_handler._get_memory_tools() == [{"name": "tool"}]
    assert cache_handler._get_memory_tools() == [{"name": "tool"}]
    assert calls["count"] == 1

    assert cache_handler._extract_tool_calls(
        {"content": [{"type": "tool_use", "id": "1"}]}, "anthropic"
    ) == [{"type": "tool_use", "id": "1"}]
    assert cache_handler._extract_tool_calls(
        {"choices": [{"message": {"tool_calls": [{"id": "2"}]}}]},
        "openai",
    ) == [{"id": "2"}]
    assert cache_handler._extract_tool_calls(
        {"output": [{"type": "function_call", "call_id": "3"}]},
        "openai",
    ) == [{"type": "function_call", "call_id": "3"}]
    assert cache_handler._extract_tool_calls({}, "other") == []

    closed: list[str] = []

    class Closable:
        async def close(self) -> None:
            closed.append("closed")

    await cache_handler._close_backend_instance(Closable(), reason="test")
    assert closed == ["closed"]
    await cache_handler._close_backend_instance(SimpleNamespace(), reason="test")

    class BrokenCloser:
        def close(self) -> None:
            raise RuntimeError("boom")

    await cache_handler._close_backend_instance(BrokenCloser(), reason="test")


@pytest.mark.asyncio
async def test_search_and_format_context_and_handle_memory_tool_calls(
    handler: MemoryHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend()
    handler._backend = backend
    handler._initialized = True
    backend.search_results = [
        make_result("m1", "Alice likes pizza", score=0.8, related_entities=["Alice", "pizza"]),
        make_result("m2", "below threshold", score=0.2),
    ]

    inject_none = await handler.search_and_format_context(
        "u1",
        [{"role": "assistant", "content": "skip"}],
    )
    assert inject_none is None

    context = await handler.search_and_format_context(
        "u1",
        [{"role": "user", "content": "What food does Alice like?"}],
    )
    assert "## Relevant Memories for This User" in context
    # Format now includes memory ID in brackets so the model can address
    # rows directly via memory_update / memory_delete without round-
    # tripping through memory_search.
    assert "1. [m1] Alice likes pizza" in context
    assert "(Related: Alice, pizza)" in context

    backend.raise_on = "search"
    assert (
        await handler.search_and_format_context("u1", [{"role": "user", "content": "Question"}])
        is None
    )
    backend.raise_on = None

    async def fake_ensure_initialized() -> None:
        return None

    async def fake_execute_memory_tool(
        tool_name, input_data, user_id, provider="anthropic", *, request_context=None
    ):  # noqa: ANN001
        return f"ran:{tool_name}:{user_id}:{provider}:{input_data}"

    async def fake_execute_native(input_data, user_id):  # noqa: ANN001
        return f"native:{user_id}:{input_data}"

    monkeypatch.setattr(handler, "_ensure_initialized", fake_ensure_initialized)
    monkeypatch.setattr(handler, "_execute_memory_tool", fake_execute_memory_tool)
    monkeypatch.setattr(handler, "_execute_native_memory_tool", fake_execute_native)

    anthropic_results = await handler.handle_memory_tool_calls(
        {
            "content": [
                {"type": "tool_use", "name": "memory_save", "id": "a1", "input": {"content": "x"}},
                {"type": "tool_use", "name": "memory", "id": "a2", "input": {"command": "view"}},
                {"type": "tool_use", "name": "other", "id": "a3", "input": {}},
            ]
        },
        "u1",
        "anthropic",
    )
    assert anthropic_results == [
        {
            "type": "tool_result",
            "tool_use_id": "a1",
            "content": "ran:memory_save:u1:anthropic:{'content': 'x'}",
        },
        {"type": "tool_result", "tool_use_id": "a2", "content": "native:u1:{'command': 'view'}"},
    ]

    openai_results = await handler.handle_memory_tool_calls(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "o1",
                                "function": {
                                    "name": "memory_search",
                                    "arguments": '{"query":"pizza"}',
                                },
                            },
                            {"id": "o2", "function": {"name": "other", "arguments": "{}"}},
                        ]
                    }
                }
            ]
        },
        "u1",
        "openai",
    )
    assert openai_results == [
        {
            "role": "tool",
            "tool_call_id": "o1",
            "content": "ran:memory_search:u1:openai:{'query': 'pizza'}",
        }
    ]

    handler._backend = None
    skipped = await handler.handle_memory_tool_calls(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "o3",
                                "function": {"name": "memory_delete", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        },
        "u1",
        "openai",
    )
    assert skipped == []


@pytest.mark.asyncio
async def test_ensure_initialized_timeout_and_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handler = MemoryHandler(
        MemoryConfig(
            enabled=True, use_native_tool=True, native_memory_dir=str(tmp_path / "native")
        ),
        agent_type="codex",
    )
    closed: list[str] = []

    class ClosableBackend:
        async def close(self) -> None:
            closed.append("closed")

    async def fake_init_backend_locked() -> None:
        handler._backend = ClosableBackend()

    monkeypatch.setattr(handler, "_init_backend_locked", fake_init_backend_locked)

    async def fake_wait_for_timeout(coro, timeout):  # noqa: ANN001
        await coro
        raise asyncio.TimeoutError

    monkeypatch.setattr(memory_handler_module.asyncio, "wait_for", fake_wait_for_timeout)
    await handler._ensure_initialized()
    assert handler.backend is None
    assert handler.initialized is False
    assert closed == ["closed"]

    async def fake_wait_for_cancel(coro, timeout):  # noqa: ANN001
        await coro
        raise asyncio.CancelledError

    monkeypatch.setattr(memory_handler_module.asyncio, "wait_for", fake_wait_for_cancel)
    with pytest.raises(asyncio.CancelledError):
        await handler._ensure_initialized()
    assert handler.backend is None
    assert handler.initialized is False
    assert closed == ["closed", "closed"]


@pytest.mark.asyncio
async def test_init_backend_locked_local_and_bridge_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handler = MemoryHandler(
        MemoryConfig(
            enabled=True,
            backend="local",
            db_path=str(tmp_path / "memory.db"),
            bridge_enabled=True,
            bridge_auto_import=True,
            bridge_md_paths=[str(tmp_path / "notes.md")],
            bridge_md_format="auto",
            bridge_export_path=str(tmp_path / "export"),
        ),
        agent_type="codex",
    )
    seen: dict[str, object] = {}

    class FakeLocalBackendConfig:
        def __init__(self, **kwargs):  # noqa: ANN003
            seen["config"] = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeLocalBackend:
        def __init__(self, config) -> None:  # noqa: ANN001
            seen["backend_config"] = config

        async def _ensure_initialized(self) -> None:
            seen["backend_initialized"] = True

    async def fake_init_and_import_bridge() -> None:
        seen["bridge_called"] = True

    monkeypatch.setitem(
        sys.modules,
        "headroom.memory.backends.local",
        SimpleNamespace(
            LocalBackend=FakeLocalBackend,
            LocalBackendConfig=FakeLocalBackendConfig,
        ),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    monkeypatch.setattr(handler, "_init_and_import_bridge", fake_init_and_import_bridge)

    await handler._init_backend_locked()

    assert handler.initialized is True
    assert seen["backend_initialized"] is True
    assert seen["bridge_called"] is True
    assert seen["config"] == {
        "db_path": str(tmp_path / "memory.db"),
        "embedder_backend": "onnx",
        "embedder_model": "all-MiniLM-L6-v2",
        "vector_dimension": 384,
    }


@pytest.mark.asyncio
async def test_init_and_import_bridge_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handler = MemoryHandler(
        MemoryConfig(
            enabled=True,
            bridge_enabled=True,
            bridge_md_paths=[str(tmp_path / "notes.md")],
            bridge_md_format="auto",
            bridge_export_path=str(tmp_path / "export"),
        ),
        agent_type="codex",
    )
    handler._backend = object()
    seen: dict[str, object] = {}

    class FakeMarkdownFormat(str):
        pass

    class FakeBridgeConfig:
        def __init__(self, **kwargs):  # noqa: ANN003
            seen["bridge_config"] = kwargs

    class FakeMemoryBridge:
        def __init__(self, config, backend) -> None:  # noqa: ANN001
            seen["bridge_backend"] = backend
            self.config = config

        async def import_from_markdown(self):
            return SimpleNamespace(sections_imported=2, sections_skipped_duplicate=1)

    monkeypatch.setitem(
        sys.modules,
        "headroom.memory.bridge",
        SimpleNamespace(MemoryBridge=FakeMemoryBridge),
    )
    monkeypatch.setitem(
        sys.modules,
        "headroom.memory.bridge_config",
        SimpleNamespace(
            BridgeConfig=FakeBridgeConfig,
            MarkdownFormat=FakeMarkdownFormat,
        ),
    )

    await handler._init_and_import_bridge()
    assert isinstance(handler._bridge, FakeMemoryBridge)
    assert seen["bridge_backend"] is handler._backend
    assert seen["bridge_config"] == {
        "md_paths": [tmp_path / "notes.md"],
        "md_format": "auto",
        "auto_import_on_startup": True,
        "export_path": tmp_path / "export",
    }

    class BrokenMemoryBridge(FakeMemoryBridge):
        async def import_from_markdown(self):
            raise RuntimeError("bridge failed")

    handler._bridge = None
    monkeypatch.setitem(
        sys.modules,
        "headroom.memory.bridge",
        SimpleNamespace(MemoryBridge=BrokenMemoryBridge),
    )
    await handler._init_and_import_bridge()
    assert isinstance(handler._bridge, BrokenMemoryBridge)


def test_memory_handler_init_defaults_and_tool_injection_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native_dir = tmp_path / "workspace-memory"
    monkeypatch.setattr(
        "headroom.paths.native_memory_dir",
        lambda: native_dir,
    )
    handler = MemoryHandler(MemoryConfig(enabled=False, use_native_tool=True), agent_type="codex")
    assert handler._native_memory_dir == native_dir
    assert native_dir.exists()

    disabled_injection = MemoryHandler(
        MemoryConfig(enabled=False, inject_tools=False, use_native_tool=False),
        agent_type="codex",
    )
    assert disabled_injection.inject_tools(None, "openai") == ([], False)

    same_type_tools, same_type_injected = handler._inject_native_tool(
        [{"type": "memory_20250818", "name": "other"}]
    )
    assert same_type_injected is False
    assert same_type_tools == [{"type": "memory_20250818", "name": "other"}]


@pytest.mark.asyncio
async def test_ensure_initialized_fast_paths_and_qdrant_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    disabled = MemoryHandler(MemoryConfig(enabled=False), agent_type="codex")
    await disabled._ensure_initialized()
    assert disabled.initialized is False

    initialized = MemoryHandler(MemoryConfig(enabled=True), agent_type="codex")
    initialized._initialized = True
    await initialized._ensure_initialized()
    assert initialized.initialized is True

    qdrant_handler = MemoryHandler(
        MemoryConfig(enabled=True, backend="qdrant-neo4j"),
        agent_type="codex",
    )
    seen: dict[str, object] = {}

    class FakeMem0Config:
        def __init__(self, **kwargs):  # noqa: ANN003
            seen["config"] = kwargs

    class FakeAdapter:
        def __init__(self, config) -> None:  # noqa: ANN001
            seen["adapter_config"] = config

        async def ensure_initialized(self) -> None:
            seen["initialized"] = True

    monkeypatch.setitem(
        sys.modules,
        "headroom.memory.backends.direct_mem0",
        SimpleNamespace(DirectMem0Adapter=FakeAdapter, Mem0Config=FakeMem0Config),
    )
    await qdrant_handler._init_backend_locked()
    assert qdrant_handler.initialized is True
    assert seen["initialized"] is True
    assert seen["config"] == {
        "qdrant_url": None,
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "qdrant_api_key": None,
        "neo4j_uri": "neo4j://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "enable_graph": True,
    }

    monkeypatch.setitem(sys.modules, "headroom.memory.backends.direct_mem0", None)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "headroom.memory.backends.direct_mem0":
            raise ImportError("missing mem0")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    broken_qdrant = MemoryHandler(
        MemoryConfig(enabled=True, backend="qdrant-neo4j"),
        agent_type="codex",
    )
    with pytest.raises(ImportError, match="missing mem0"):
        await broken_qdrant._init_backend_locked()
    monkeypatch.setattr(builtins, "__import__", real_import)

    unknown = MemoryHandler(MemoryConfig(enabled=True, backend="local"), agent_type="codex")
    unknown.config.backend = "mystery"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown memory backend"):
        await unknown._init_backend_locked()


@pytest.mark.asyncio
async def test_init_and_import_bridge_early_return_and_context_formatting_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handler = MemoryHandler(
        MemoryConfig(
            enabled=True, bridge_enabled=True, bridge_md_paths=[str(tmp_path / "notes.md")]
        ),
        agent_type="codex",
    )
    handler._bridge = object()
    await handler._init_and_import_bridge()
    assert handler._bridge is not None

    handler.config.inject_context = False
    assert (
        await handler.search_and_format_context("u1", [{"role": "user", "content": "hello"}])
        is None
    )

    handler.config.inject_context = True
    handler._backend = FakeBackend()
    handler._initialized = True
    handler._backend.search_results = [make_result("m1", "too low", score=0.1)]
    assert (
        await handler.search_and_format_context(
            "u1", [{"role": "user", "content": [{"type": "image"}]}]
        )
        is None
    )
    assert (
        await handler.search_and_format_context("u1", [{"role": "user", "content": "hello"}])
        is None
    )


@pytest.mark.asyncio
async def test_extract_tool_calls_and_handle_tool_calls_parse_edges(
    handler: MemoryHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert handler._extract_tool_calls({"content": "bad"}, "anthropic") == []
    assert handler._extract_tool_calls({"choices": []}, "openai") == []
    assert handler._extract_tool_calls({"output": "bad"}, "openai") == []

    backend = FakeBackend()
    handler._backend = backend

    async def fake_ensure_initialized() -> None:
        return None

    async def fake_execute(
        tool_name, input_data, user_id, provider="anthropic", *, request_context=None
    ):  # noqa: ANN001
        return f"ok:{tool_name}:{input_data}"

    monkeypatch.setattr(handler, "_ensure_initialized", fake_ensure_initialized)
    monkeypatch.setattr(handler, "_execute_memory_tool", fake_execute)

    results = await handler.handle_memory_tool_calls(
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "fc1",
                    "name": "memory_search",
                    "arguments": "{bad",
                },
                {"type": "function_call", "call_id": "fc2", "name": "other", "arguments": "{}"},
            ]
        },
        "u1",
        "openai",
    )
    assert results == [{"role": "tool", "tool_call_id": "fc1", "content": "ok:memory_search:{}"}]


# ── memory_list (browse without semantic query) ──────────────────────


@pytest.mark.asyncio
async def test_execute_list_returns_recent_memories_with_ids(handler: MemoryHandler) -> None:
    """memory_list returns memories in reverse-chronological order with
    IDs, content, and timestamps. Distinct from memory_search — no
    semantic query required; model can browse to discover IDs."""

    class ListBackend:
        async def search_memories(self, **kwargs):  # noqa: ANN003
            return []

        async def list_memories(self, *, user_id, limit):  # noqa: ANN001, ANN201
            assert user_id == "u1"
            assert limit == 5
            return [
                make_result("mem_002", "newest fact", created_at="2026-05-19T12:00:00+00:00"),
                make_result("mem_001", "older fact", created_at="2026-05-18T10:00:00+00:00"),
            ]

    handler._backend = ListBackend()  # type: ignore[assignment]
    handler._initialized = True
    out = await handler._execute_list({"limit": 5}, "u1")
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["count"] == 2
    assert payload["memories"][0]["id"] == "mem_002"
    assert payload["memories"][0]["content"] == "newest fact"
    assert payload["memories"][0]["created_at"] == "2026-05-19T12:00:00+00:00"
    assert payload["memories"][1]["id"] == "mem_001"


@pytest.mark.asyncio
async def test_execute_list_falls_back_to_search_when_list_unavailable(
    handler: MemoryHandler,
) -> None:
    """Backends without list_memories fall back to an empty-query
    search — most backends treat that as "return recent." Locks the
    fallback path so a future backend without list_memories still works."""

    class SearchOnlyBackend:
        async def search_memories(self, *, query, user_id, top_k, **kwargs):  # noqa: ANN001, ANN003
            assert query == ""  # fallback uses empty query
            assert top_k == 3
            return [make_result("mem_x", "anything", created_at=None)]

    handler._backend = SearchOnlyBackend()  # type: ignore[assignment]
    handler._initialized = True
    out = await handler._execute_list({"limit": 3}, "u1")
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["count"] == 1
    assert payload["memories"][0]["id"] == "mem_x"


@pytest.mark.asyncio
async def test_execute_list_caps_limit_to_1_100(handler: MemoryHandler) -> None:
    """Defensive: input limit is clamped to [1, 100]. Protects the
    backend from a runaway value the model might invent."""

    received_limits: list[int] = []

    class LimitWatcher:
        async def search_memories(self, **kwargs):  # noqa: ANN003
            return []

        async def list_memories(self, *, user_id, limit):  # noqa: ANN001, ANN201
            received_limits.append(limit)
            return []

    handler._backend = LimitWatcher()  # type: ignore[assignment]
    handler._initialized = True

    await handler._execute_list({"limit": 99999}, "u1")
    await handler._execute_list({"limit": 0}, "u1")
    await handler._execute_list({"limit": "bogus"}, "u1")  # type: ignore[dict-item]
    assert received_limits == [100, 1, 10]  # clamped to 100, 1, default 10


@pytest.mark.asyncio
async def test_memory_list_dispatched_via_execute_memory_tool(handler: MemoryHandler) -> None:
    """End-to-end: memory_list in MEMORY_TOOL_NAMES means the
    handler.handle_memory_tool_calls dispatcher routes it correctly."""

    class StubBackend:
        async def search_memories(self, **kwargs):  # noqa: ANN003
            return []

        async def list_memories(self, *, user_id, limit):  # noqa: ANN001, ANN201
            return [make_result("m1", "fact", created_at=None)]

    handler._backend = StubBackend()  # type: ignore[assignment]
    handler._initialized = True
    out = await handler._execute_memory_tool("memory_list", {"limit": 5}, "u1", "anthropic")
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["memories"][0]["id"] == "m1"

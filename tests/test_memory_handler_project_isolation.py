"""End-to-end isolation test for the per-project memory router (GH #462).

Verifies that two sessions running in different working directories
never see each other's memories — neither at search-time, nor in the
injected ``## Relevant Memories`` block.

The test stubs out the real backend so we don't have to load embedders
or open SQLite files. The interesting invariant is that the router
hands a *different* backend instance to each cwd, and that the handler
calls ``search_memories`` on the backend it received (not on the legacy
``self._backend``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from headroom.memory import storage_router as sr_mod
from headroom.proxy.memory_handler import (
    MemoryConfig,
    MemoryHandler,
    MemoryMode,
)


class _FakeBackend:
    """Minimal stand-in for ``LocalBackend`` used by the router cache.

    Each instance tracks its own write log so we can prove that
    Project A's saves and Project B's saves landed on different
    backends.
    """

    instances: list[_FakeBackend] = []

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.initialize_calls = 0
        self.saved_contents: list[str] = []
        self.search_results: list[Any] = []
        # Tag the backend with its db_path so tests can assert on it.
        self.db_path = getattr(cfg, "db_path", "<unknown>")
        _FakeBackend.instances.append(self)

    async def _ensure_initialized(self) -> None:
        self.initialize_calls += 1

    async def search_memories(self, **kwargs: Any) -> list[Any]:
        return list(self.search_results)

    async def save_memory(self, **kwargs: Any) -> Any:
        content = kwargs["content"]
        self.saved_contents.append(content)
        return SimpleNamespace(
            id=f"mem-{self.db_path}-{len(self.saved_contents)}",
            content=content,
            metadata={},
        )

    async def delete_memory(self, memory_id: str) -> bool:
        return True


@pytest.fixture(autouse=True)
def patch_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap LocalBackend out at every import site the handler/router uses."""

    _FakeBackend.instances.clear()
    monkeypatch.setattr(sr_mod, "LocalBackend", _FakeBackend)

    # The handler's _init_backend_locked imports LocalBackend locally;
    # patch the same target there too. The route below is the canonical
    # import path used by the handler.
    import headroom.memory.backends.local as _local_mod

    monkeypatch.setattr(_local_mod, "LocalBackend", _FakeBackend)


@pytest.fixture
def handler(tmp_path: Path) -> MemoryHandler:
    cfg = MemoryConfig(
        enabled=True,
        backend="local",
        db_path=str(tmp_path / "memory.db"),
        inject_context=True,
        mode=MemoryMode.AUTO_TAIL,
    )
    h = MemoryHandler(cfg, agent_type="test")
    return h


def _ctx_for_cwd(cwd: str, user_id: str = "alice") -> Any:
    return sr_mod.RequestContext(
        headers={"x-headroom-cwd": cwd},
        system_prompt="",
        base_user_id=user_id,
    )


def test_two_cwds_route_to_two_backends(handler: MemoryHandler) -> None:
    """Saves under different cwds must land on different backends."""

    async def run() -> None:
        await handler._ensure_initialized()
        ctx_a = _ctx_for_cwd("/Users/me/code/project-a")
        ctx_b = _ctx_for_cwd("/Users/me/code/project-b")

        await handler._execute_save(
            {"content": "Note about Project A's redis config"},
            "alice",
            "anthropic",
            request_context=ctx_a,
        )
        await handler._execute_save(
            {"content": "Note about Project B's auth setup"},
            "alice",
            "anthropic",
            request_context=ctx_b,
        )

        # The router must have created at least 3 backends:
        # 1 legacy (during init), 1 for project-a, 1 for project-b.
        backends_with_content = [b for b in _FakeBackend.instances if b.saved_contents]
        assert len(backends_with_content) == 2

        contents_per_backend = {tuple(b.saved_contents) for b in backends_with_content}
        assert ("Note about Project A's redis config",) in contents_per_backend
        assert ("Note about Project B's auth setup",) in contents_per_backend

    asyncio.run(run())


def test_search_returns_only_current_workspace_memories(handler: MemoryHandler) -> None:
    """Project A's search must not see Project B's memories."""

    async def run() -> None:
        await handler._ensure_initialized()
        ctx_a = _ctx_for_cwd("/Users/me/code/project-a")
        ctx_b = _ctx_for_cwd("/Users/me/code/project-b")

        # Save under A and B.
        await handler._execute_save(
            {"content": "A memory"}, "alice", "anthropic", request_context=ctx_a
        )
        await handler._execute_save(
            {"content": "B memory"}, "alice", "anthropic", request_context=ctx_b
        )

        # Seed each fake backend's search return so we can detect bleed.
        backend_a, _, _ = handler._resolve_for_request("alice", ctx_a)
        backend_b, _, _ = handler._resolve_for_request("alice", ctx_b)

        backend_a.search_results = [  # type: ignore[attr-defined]
            SimpleNamespace(
                memory=SimpleNamespace(id="a1", content="A memory", metadata={}),
                score=0.9,
                related_entities=[],
            )
        ]
        backend_b.search_results = [  # type: ignore[attr-defined]
            SimpleNamespace(
                memory=SimpleNamespace(id="b1", content="B memory", metadata={}),
                score=0.9,
                related_entities=[],
            )
        ]

        msgs_a = [{"role": "user", "content": "what's the redis config?"}]
        msgs_b = [{"role": "user", "content": "what's the auth setup?"}]

        ctx_block_a = await handler.search_and_format_context(
            "alice", msgs_a, request_context=ctx_a
        )
        ctx_block_b = await handler.search_and_format_context(
            "alice", msgs_b, request_context=ctx_b
        )

        assert ctx_block_a is not None
        assert ctx_block_b is not None

        # The block from Project A must contain A's memory and *not* B's.
        assert "A memory" in ctx_block_a
        assert "B memory" not in ctx_block_a
        # Symmetric assertion for B.
        assert "B memory" in ctx_block_b
        assert "A memory" not in ctx_block_b

        # Provenance header (Fix C) must include the workspace name.
        assert "workspace: project-a" in ctx_block_a
        assert "workspace: project-b" in ctx_block_b

    asyncio.run(run())


def test_legacy_callers_without_ctx_hit_legacy_backend(handler: MemoryHandler) -> None:
    """Callers that don't pass a RequestContext keep the pre-fix shape.

    Verifies the backward-compatibility seam: legacy tests / mocks that
    call ``search_and_format_context(user, messages)`` get the same
    behaviour they had before — the legacy single-DB backend, no scope
    header. This is the path tests + qdrant deployments take.
    """

    async def run() -> None:
        await handler._ensure_initialized()
        legacy_backend = handler._backend
        legacy_backend.search_results = [  # type: ignore[attr-defined]
            SimpleNamespace(
                memory=SimpleNamespace(id="g1", content="Global memory", metadata={}),
                score=0.9,
                related_entities=[],
            )
        ]

        block = await handler.search_and_format_context(
            "alice", [{"role": "user", "content": "anything"}]
        )

        assert block is not None
        assert "Global memory" in block
        # No provenance suffix in legacy header.
        assert block.startswith("## Relevant Memories for This User")

    asyncio.run(run())


def test_user_mode_partitions_by_user_id(tmp_path: Path) -> None:
    """``--memory-storage=user`` opens one DB per base_user_id."""

    cfg = MemoryConfig(
        enabled=True,
        backend="local",
        db_path=str(tmp_path / "memory.db"),
        storage_mode=sr_mod.MemoryStorageMode.USER,
    )
    h = MemoryHandler(cfg, agent_type="test")

    async def run() -> None:
        await h._ensure_initialized()
        ctx_alice = sr_mod.RequestContext(headers={}, system_prompt="", base_user_id="alice")
        ctx_bob = sr_mod.RequestContext(headers={}, system_prompt="", base_user_id="bob")

        backend_alice, scope_a, _ = h._resolve_for_request("alice", ctx_alice)
        backend_bob, scope_b, _ = h._resolve_for_request("bob", ctx_bob)

        assert scope_a.mode is sr_mod.MemoryStorageMode.USER
        assert scope_b.mode is sr_mod.MemoryStorageMode.USER
        assert backend_alice is not backend_bob

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Unresolved-project fail-closed (incident 2026-05-26).
#
# When `mode=PROJECT` and `unresolved_project_fallback="empty"` (the new
# default), an inbound request with no project-resolution signal
# (x-headroom-project-id / x-headroom-cwd / system-prompt cwd:) must
# return None from search_and_format_context — NOT silently pool the
# request's memory into the GLOBAL bucket. The old GLOBAL fallback was
# what surfaced a memory from a prior unrelated TAM-550 session into
# a live PR-review thread, where the agent misread it as a new command.
# ---------------------------------------------------------------------------


def test_unresolved_project_returns_no_context(tmp_path: Path) -> None:
    """No project signals + PROJECT mode + empty fallback → no memory injection."""
    cfg = MemoryConfig(
        enabled=True,
        backend="local",
        db_path=str(tmp_path / "memory.db"),
        inject_context=True,
        mode=MemoryMode.AUTO_TAIL,
        storage_mode=sr_mod.MemoryStorageMode.PROJECT,  # PROJECT mode triggers resolution.
        # unresolved_project_fallback="empty" — the new default applied
        # by MemoryHandler when building the BackendRouterConfig.
    )
    handler = MemoryHandler(cfg, agent_type="test")

    async def run() -> None:
        # Request with NO project-resolution signal: no header, no cwd,
        # no parseable system-prompt cwd: line.
        ctx_unresolved = sr_mod.RequestContext(
            headers={},  # No x-headroom-* headers.
            system_prompt="You are helpful.",  # No env block.
            base_user_id="alice",
        )

        # Seed a backend so search WOULD return something — to prove the
        # gate is at the scope-resolution layer, not just an empty store.
        for backend in _FakeBackend.instances:
            backend.search_results = [
                SimpleNamespace(
                    memory=SimpleNamespace(
                        id="should-not-leak", content="Stale prior content", metadata={}
                    ),
                    score=0.99,
                    related_entities=[],
                )
            ]

        msgs = [{"role": "user", "content": "Just a friendly hello"}]
        context = await handler.search_and_format_context("alice", msgs, ctx_unresolved)

        # Fail-closed: no memory injected even though backends have data.
        assert context is None, (
            "Unresolved project in PROJECT mode must skip injection — "
            "incident on 2026-05-26 (TAM-550) was caused by the GLOBAL "
            "fallback pooling prior-session content into a fresh thread."
        )
        assert _FakeBackend.instances == []

    asyncio.run(run())


def test_all_unresolved_tools_skip_without_backend_or_file_side_effects(
    tmp_path: Path,
) -> None:
    """Every custom/native operation returns the same fail-closed result."""
    handler = MemoryHandler(
        MemoryConfig(
            enabled=True,
            backend="local",
            db_path=str(tmp_path / "global-memory.db"),
            storage_mode=sr_mod.MemoryStorageMode.PROJECT,
            use_native_tool=True,
            native_memory_dir=str(tmp_path / "native"),
        ),
        agent_type="test",
    )
    unresolved = sr_mod.RequestContext(
        headers={},
        system_prompt="You are helpful.",
        base_user_id="alice",
    )
    custom_inputs = {
        "memory_save": {"content": "canary"},
        "memory_search": {"query": "canary"},
        "memory_update": {"memory_id": "canary", "new_content": "changed"},
        "memory_delete": {"memory_id": "canary"},
        "memory_list": {},
    }
    native_inputs = {
        "view": {"path": "/memories/canary.txt"},
        "create": {"path": "/memories/canary.txt", "file_text": "canary"},
        "str_replace": {
            "path": "/memories/canary.txt",
            "old_str": "canary",
            "new_str": "changed",
        },
        "insert": {
            "path": "/memories/canary.txt",
            "insert_line": 0,
            "insert_text": "changed",
        },
        "delete": {"path": "/memories/canary.txt"},
        "rename": {
            "old_path": "/memories/canary.txt",
            "new_path": "/memories/renamed.txt",
        },
    }
    content = [
        {
            "type": "tool_use",
            "id": name,
            "name": name,
            "input": input_data,
        }
        for name, input_data in custom_inputs.items()
    ]
    content += [
        {
            "type": "tool_use",
            "id": f"native-{command}",
            "name": "memory",
            "input": {"command": command, **input_data},
        }
        for command, input_data in native_inputs.items()
    ]
    native_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    results = asyncio.run(
        handler.handle_memory_tool_calls(
            {"content": content},
            "alice",
            "anthropic",
            request_context=unresolved,
        )
    )

    assert len(results) == len(content)
    assert all(
        result["content"] == '{"status": "skipped", "reason": "project_unresolved"}'
        for result in results
    )
    assert _FakeBackend.instances == []
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == native_before


def test_resolved_native_create_uses_project_backend(tmp_path: Path) -> None:
    """Native translation writes only to the resolved project backend."""
    handler = MemoryHandler(
        MemoryConfig(
            enabled=True,
            backend="local",
            db_path=str(tmp_path / "global-memory.db"),
            storage_mode=sr_mod.MemoryStorageMode.PROJECT,
            use_native_tool=True,
            native_memory_dir=str(tmp_path / "native"),
        ),
        agent_type="test",
    )

    results = asyncio.run(
        handler.handle_memory_tool_calls(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "native-create",
                        "name": "memory",
                        "input": {
                            "command": "create",
                            "path": "/memories/project.txt",
                            "file_text": "PROJECT_NATIVE_CANARY",
                        },
                    }
                ]
            },
            "alice",
            "anthropic",
            request_context=_ctx_for_cwd("/tmp/project-native"),
        )
    )

    assert results[0]["content"] == "File created successfully at: /memories/project.txt"
    writers = [backend for backend in _FakeBackend.instances if backend.saved_contents]
    assert len(writers) == 1
    assert writers[0].saved_contents == ["PROJECT_NATIVE_CANARY"]
    assert "projects" in Path(writers[0].db_path).parts


def test_unresolved_native_create_has_no_backend_side_effect(tmp_path: Path) -> None:
    """Native memory create must fail closed before touching the global backend."""
    handler = MemoryHandler(
        MemoryConfig(
            enabled=True,
            backend="local",
            db_path=str(tmp_path / "global-memory.db"),
            storage_mode=sr_mod.MemoryStorageMode.PROJECT,
            use_native_tool=True,
            native_memory_dir=str(tmp_path / "native"),
        ),
        agent_type="test",
    )
    unresolved = sr_mod.RequestContext(
        headers={},
        system_prompt="You are helpful.",
        base_user_id="alice",
    )

    async def run() -> None:
        results = await handler.handle_memory_tool_calls(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "native-create",
                        "name": "memory",
                        "input": {
                            "command": "create",
                            "path": "/memories/canary.txt",
                            "file_text": "UNRESOLVED_NATIVE_CANARY",
                        },
                    }
                ]
            },
            "alice",
            "anthropic",
            request_context=unresolved,
        )
        direct_result = await handler._execute_native_memory_tool(
            {
                "command": "create",
                "path": "/memories/direct-canary.txt",
                "file_text": "UNRESOLVED_DIRECT_NATIVE_CANARY",
            },
            "alice",
            request_context=unresolved,
        )

        assert '"reason": "project_unresolved"' in results[0]["content"]
        assert direct_result == '{"status": "skipped", "reason": "project_unresolved"}'
        assert not any(backend.initialize_calls for backend in _FakeBackend.instances)
        assert all(
            "UNRESOLVED_NATIVE_CANARY" not in backend.saved_contents
            for backend in _FakeBackend.instances
        )

    asyncio.run(run())

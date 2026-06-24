"""Unit tests for the per-project memory storage router (GH #462)."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.memory.backends.local import LocalBackendConfig
from headroom.memory.storage_router import (
    BackendRouter,
    BackendRouterConfig,
    MemoryStorageMode,
    ProjectResolver,
    RequestContext,
    extract_system_prompt,
)

# ---------------------------------------------------------------------------
# Resolver tier-order tests
# ---------------------------------------------------------------------------


def _ctx(
    *,
    headers: dict[str, str] | None = None,
    system_prompt: str = "",
    base_user_id: str = "alice",
    project_root_override: str | None = None,
) -> RequestContext:
    return RequestContext(
        headers=headers or {},
        system_prompt=system_prompt,
        base_user_id=base_user_id,
        project_root_override=project_root_override,
    )


def test_resolver_tier1_explicit_project_id_wins() -> None:
    r = ProjectResolver()
    # An explicit project id beats everything else.
    out = r.resolve(
        _ctx(
            headers={"x-headroom-project-id": "billing-svc"},
            system_prompt="Primary working directory: /Users/foo/code/other\n",
            project_root_override="/also/ignored",
        )
    )
    assert out is not None
    key, display = out
    assert key == "billing-svc"
    assert display == "billing-svc"


def test_resolver_tier2_explicit_cwd_header() -> None:
    r = ProjectResolver()
    out = r.resolve(_ctx(headers={"x-headroom-cwd": "/Users/foo/code/project-b"}))
    assert out is not None
    key, display = out
    assert display == "project-b"
    assert key.startswith("project-b-")
    assert len(key.split("-")[-1]) == 16  # sha256 prefix length


def test_resolver_tier3_cli_override() -> None:
    r = ProjectResolver()
    out = r.resolve(_ctx(project_root_override="/Users/foo/code/project-c"))
    assert out is not None
    _, display = out
    assert display == "project-c"


def test_resolver_tier4_env_block_primary_working_directory() -> None:
    r = ProjectResolver()
    prompt = (
        "You have been invoked in the following environment:\n"
        " - Primary working directory: /Users/foo/code/headroom\n"
        " - Is a git repo: yes\n"
    )
    out = r.resolve(_ctx(system_prompt=prompt))
    assert out is not None
    _, display = out
    assert display == "headroom"


def test_resolver_tier4_env_block_older_working_directory_format() -> None:
    r = ProjectResolver()
    prompt = "Working directory: /Users/foo/code/legacy-project\n"
    out = r.resolve(_ctx(system_prompt=prompt))
    assert out is not None
    _, display = out
    assert display == "legacy-project"


def test_resolver_tier4_env_block_cwd_format() -> None:
    r = ProjectResolver()
    prompt = "  cwd: /Users/foo/code/cwd-style\n"
    out = r.resolve(_ctx(system_prompt=prompt))
    assert out is not None
    _, display = out
    assert display == "cwd-style"


def test_resolver_returns_none_when_nothing_resolves() -> None:
    r = ProjectResolver()
    out = r.resolve(_ctx(system_prompt="A generic system prompt with no env block."))
    assert out is None


def test_resolver_same_cwd_yields_stable_key_across_calls() -> None:
    r = ProjectResolver()
    k1, _ = r.resolve(_ctx(headers={"x-headroom-cwd": "/Users/foo/code/x"}))  # type: ignore[misc]
    k2, _ = r.resolve(_ctx(headers={"x-headroom-cwd": "/Users/foo/code/x"}))  # type: ignore[misc]
    assert k1 == k2


def test_resolver_distinct_cwds_yield_distinct_keys() -> None:
    r = ProjectResolver()
    k1, _ = r.resolve(_ctx(headers={"x-headroom-cwd": "/Users/foo/code/a"}))  # type: ignore[misc]
    k2, _ = r.resolve(_ctx(headers={"x-headroom-cwd": "/Users/foo/code/b"}))  # type: ignore[misc]
    assert k1 != k2


def test_resolver_sanitises_unsafe_basename_chars() -> None:
    r = ProjectResolver()
    out = r.resolve(_ctx(headers={"x-headroom-project-id": "../etc/passwd; rm -rf /"}))
    assert out is not None
    key, _ = out
    # Path-separators and shell-metas must be neutralised.
    assert "/" not in key
    assert ";" not in key
    assert " " not in key


def test_extract_system_prompt_anthropic_string() -> None:
    assert extract_system_prompt({"system": "hello"}) == "hello"


def test_extract_system_prompt_anthropic_blocks() -> None:
    body = {"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert extract_system_prompt(body) == "a\nb"


def test_extract_system_prompt_openai_messages() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ]
    }
    assert extract_system_prompt(body) == "you are helpful"


def test_extract_system_prompt_missing_returns_empty() -> None:
    assert extract_system_prompt({"messages": []}) == ""


# ---------------------------------------------------------------------------
# BackendRouter path-layout tests (no real backend I/O — we stub the class).
# ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self, cfg: LocalBackendConfig) -> None:
        self.cfg = cfg

    async def _ensure_initialized(self) -> None:
        return


def _make_router(
    tmp_path: Path,
    mode: MemoryStorageMode,
    monkeypatch: pytest.MonkeyPatch,
) -> BackendRouter:
    # Patch out the real LocalBackend constructor so the router test
    # doesn't try to load embedders or open SQLite files.
    monkeypatch.setattr(
        "headroom.memory.storage_router.LocalBackend",
        _FakeBackend,
    )
    cfg = BackendRouterConfig(
        mode=mode,
        root_dir=tmp_path / "memories",
        global_db_path=tmp_path / "memory.db",
        max_open_backends=4,
        backend_config_template=LocalBackendConfig(db_path=str(tmp_path / "memory.db")),
    )
    return BackendRouter(cfg)


def test_router_project_mode_two_cwds_two_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = _make_router(tmp_path, MemoryStorageMode.PROJECT, monkeypatch)

    ctx_a = _ctx(headers={"x-headroom-cwd": "/code/a"})
    ctx_b = _ctx(headers={"x-headroom-cwd": "/code/b"})

    _, scope_a = router.backend_for(ctx_a)
    _, scope_b = router.backend_for(ctx_b)

    assert scope_a.mode is MemoryStorageMode.PROJECT
    assert scope_b.mode is MemoryStorageMode.PROJECT
    assert scope_a.db_path != scope_b.db_path
    assert scope_a.display_name == "a"
    assert scope_b.display_name == "b"


def test_router_project_mode_unresolved_fails_closed_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default `unresolved_project_fallback='empty'` → fail-closed signal, NOT GLOBAL pool.

    Updated 2026-05-26 from the prior GLOBAL-fallback assertion. The
    silent GLOBAL pooling was the root cause of the TAM-550
    "implémente X" cross-thread instruction misread (a memory from a
    prior unrelated session ended up in the live user turn and got
    treated as a new command). The new default is fail-closed: the
    router still returns a ResolvedScope (so callers don't need to
    handle None), but signals "no project" via
    ``mode=PROJECT & project_key=None``. The memory handler reads
    that sentinel and skips injection entirely.
    """
    router = _make_router(tmp_path, MemoryStorageMode.PROJECT, monkeypatch)

    _, scope = router.backend_for(_ctx(system_prompt="no env block"))
    # Fail-closed signal: PROJECT mode preserved, project_key is None.
    assert scope.mode is MemoryStorageMode.PROJECT
    assert scope.project_key is None
    assert scope.display_name == "unresolved (no memory)"


def test_router_project_mode_unresolved_global_fallback_when_opted_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy GLOBAL pooling is reachable via opt-in config."""
    monkeypatch.setattr(
        "headroom.memory.storage_router.LocalBackend",
        _FakeBackend,
    )
    cfg = BackendRouterConfig(
        mode=MemoryStorageMode.PROJECT,
        root_dir=tmp_path / "memories",
        global_db_path=tmp_path / "memory.db",
        max_open_backends=4,
        backend_config_template=LocalBackendConfig(db_path=str(tmp_path / "memory.db")),
        unresolved_project_fallback="global",
    )
    router = BackendRouter(cfg)

    _, scope = router.backend_for(_ctx(system_prompt="no env block"))
    assert scope.mode is MemoryStorageMode.GLOBAL
    assert scope.db_path == tmp_path / "memory.db"
    assert scope.display_name == "global (unresolved)"


def test_router_invalid_unresolved_fallback_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown values of `unresolved_project_fallback` fail loud, not silently."""
    monkeypatch.setattr(
        "headroom.memory.storage_router.LocalBackend",
        _FakeBackend,
    )
    cfg = BackendRouterConfig(
        mode=MemoryStorageMode.PROJECT,
        root_dir=tmp_path / "memories",
        global_db_path=tmp_path / "memory.db",
        max_open_backends=4,
        backend_config_template=LocalBackendConfig(db_path=str(tmp_path / "memory.db")),
        unresolved_project_fallback="nonsense_value",
    )
    router = BackendRouter(cfg)

    with pytest.raises(ValueError, match="not a recognised value"):
        router.backend_for(_ctx(system_prompt="no env block"))


def test_router_user_mode_partitions_by_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = _make_router(tmp_path, MemoryStorageMode.USER, monkeypatch)

    _, scope_a = router.backend_for(_ctx(base_user_id="alice"))
    _, scope_b = router.backend_for(_ctx(base_user_id="bob"))

    assert scope_a.mode is MemoryStorageMode.USER
    assert scope_b.mode is MemoryStorageMode.USER
    assert scope_a.db_path != scope_b.db_path
    assert scope_a.display_name == "alice"
    assert scope_b.display_name == "bob"


def test_router_global_mode_reuses_legacy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = _make_router(tmp_path, MemoryStorageMode.GLOBAL, monkeypatch)

    _, scope = router.backend_for(_ctx(headers={"x-headroom-cwd": "/code/anything"}))
    assert scope.mode is MemoryStorageMode.GLOBAL
    # GLOBAL mode hits the legacy DB regardless of cwd signals.
    assert scope.db_path == tmp_path / "memory.db"


def test_router_backend_cache_returns_same_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = _make_router(tmp_path, MemoryStorageMode.PROJECT, monkeypatch)

    ctx = _ctx(headers={"x-headroom-cwd": "/code/sticky"})
    b1, _ = router.backend_for(ctx)
    b2, _ = router.backend_for(ctx)
    assert b1 is b2


def test_router_lru_eviction_drops_oldest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # max_open_backends=4 in _make_router. Opening 5 different projects
    # should evict the first.
    router = _make_router(tmp_path, MemoryStorageMode.PROJECT, monkeypatch)
    for i in range(5):
        router.backend_for(_ctx(headers={"x-headroom-cwd": f"/code/p{i}"}))
    assert len(router.open_backends()) == 4

"""Tests for the shared cold-start warmup registry (Unit 1).

Covers:
- WarmupRegistry slot state transitions.
- Preload iterates both Anthropic + OpenAI pipelines and dedupes
  shared transforms by ``id(transform)``.
- Embedder warm-up encode runs once during startup (happy path).
- optimize=False leaves slots null.
- Memory backend init failure yields registry status=error, startup
  still completes, health reports degraded memory.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.warmup import WarmupRegistry, WarmupSlot

# -------------------------------------------------------------------
# WarmupSlot / WarmupRegistry unit tests
# -------------------------------------------------------------------


def test_warmup_slot_defaults_to_null():
    slot = WarmupSlot()
    assert slot.status == "null"
    assert slot.handle is None
    assert slot.error is None
    assert slot.to_dict() == {"status": "null"}


def test_warmup_slot_transitions():
    slot = WarmupSlot()
    slot.mark_loading()
    assert slot.status == "loading"

    slot.mark_loaded(handle="h", model="x")
    assert slot.status == "loaded"
    assert slot.handle == "h"
    assert slot.info == {"model": "x"}
    assert slot.to_dict() == {"status": "loaded", "info": {"model": "x"}}

    slot.mark_error("boom")
    assert slot.status == "error"
    assert slot.handle is None
    assert slot.error == "boom"
    assert slot.to_dict()["status"] == "error"
    assert slot.to_dict()["error"] == "boom"

    slot.mark_null()
    assert slot.status == "null"
    assert slot.handle is None
    assert slot.error is None


def test_warmup_registry_merges_enabled_status_into_loaded():
    reg = WarmupRegistry()
    reg.merge_transform_status(
        {
            "kompress": "enabled",
            "magika": "enabled",
            "code_aware": "enabled",
            "tree_sitter": "loaded (3 languages)",
            "smart_crusher": "ready",
        }
    )
    out = reg.to_dict()
    assert out["kompress"]["status"] == "loaded"
    assert out["magika"]["status"] == "loaded"
    assert out["code_aware"]["status"] == "loaded"
    assert out["tree_sitter"]["status"] == "loaded"
    assert out["smart_crusher"]["status"] == "loaded"


def test_warmup_registry_preserves_null_for_unavailable():
    reg = WarmupRegistry()
    reg.merge_transform_status({"kompress": "unavailable", "magika": "not installed"})
    assert reg.kompress.status == "null"
    assert reg.magika.status == "null"
    assert reg.kompress.info.get("source_status") == "unavailable"


def test_warmup_registry_to_dict_has_expected_keys():
    reg = WarmupRegistry()
    out = reg.to_dict()
    assert set(out.keys()) == {
        "kompress",
        "magika",
        "code_aware",
        "tree_sitter",
        "smart_crusher",
        "memory_backend",
        "memory_embedder",
    }


# -------------------------------------------------------------------
# Startup orchestration tests — use HeadroomProxy + stubbed transforms
# -------------------------------------------------------------------


@pytest.fixture
def _stub_pipelines(monkeypatch):
    """Build a minimal proxy whose pipelines share a single spy transform.

    The transform's ``eager_load_compressors`` increments a hit counter;
    assertion that dedup prevents double-loading relies on that counter.
    """
    pytest.importorskip("httpx")
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    class SpyTransform:
        def __init__(self) -> None:
            self.hits = 0

        def eager_load_compressors(self) -> dict[str, str]:
            self.hits += 1
            return {
                "kompress": "enabled",
                "magika": "enabled",
                "code_aware": "enabled",
                "smart_crusher": "ready",
            }

    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        code_aware_enabled=False,
    )
    proxy = HeadroomProxy(config)

    spy = SpyTransform()
    # Replace both pipeline transform lists with the SAME instance so the
    # dedupe-by-id() logic is what we actually assert against.
    proxy.anthropic_pipeline.transforms = [spy]
    proxy.openai_pipeline.transforms = [spy]
    return proxy, spy


@pytest.mark.asyncio
async def test_startup_runs_shared_transform_once(_stub_pipelines):
    proxy, spy = _stub_pipelines
    await proxy.startup()
    try:
        assert spy.hits == 1, "shared transform must be eager-loaded exactly once"
        assert proxy.warmup.kompress.status == "loaded"
        assert proxy.warmup.magika.status == "loaded"
        assert proxy.warmup.code_aware.status == "loaded"
        assert proxy.warmup.smart_crusher.status == "loaded"
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_startup_optimize_false_leaves_slots_null():
    pytest.importorskip("httpx")
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    proxy = HeadroomProxy(config)

    called = {"n": 0}

    class SpyTransform:
        def eager_load_compressors(self) -> dict[str, str]:
            called["n"] += 1
            return {"kompress": "enabled"}

    proxy.anthropic_pipeline.transforms = [SpyTransform()]
    proxy.openai_pipeline.transforms = [SpyTransform()]

    await proxy.startup()
    try:
        assert called["n"] == 0, "preload must not run when optimize=False"
        assert proxy.warmup.kompress.status == "null"
        assert proxy.warmup.memory_backend.status == "null"
        assert proxy.warmup.memory_embedder.status == "null"
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_startup_memory_embedder_warmup_encodes_once(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    proxy = HeadroomProxy(config)

    # Swap in a hand-rolled MemoryHandler whose backend exposes a mock
    # embedder. We don't want real ONNX here — just a spy.
    handler = MemoryHandler(
        MemoryConfig(enabled=True, backend="local", db_path=str(tmp_path / "mem.db"))
    )
    handler._initialized = True

    embed = AsyncMock(return_value=[0.0])

    class FakeHM:
        def __init__(self):
            self._embedder = type("_E", (), {"embed": embed})()

    class FakeBackend:
        def __init__(self):
            self._hierarchical_memory = FakeHM()

        async def close(self):
            pass

    handler._backend = FakeBackend()
    handler.ensure_initialized = AsyncMock()
    proxy.memory_handler = handler

    await proxy.startup()
    try:
        assert embed.await_count == 1
        assert embed.await_args[0][0] == "warmup"
        assert proxy.warmup.memory_backend.status == "loaded"
        assert proxy.warmup.memory_embedder.status == "loaded"
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_startup_memory_backend_error_surfaced_and_health_degraded(tmp_path):
    pytest.importorskip("httpx")
    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    proxy = HeadroomProxy(config)

    handler = MemoryHandler(
        MemoryConfig(enabled=True, backend="local", db_path=str(tmp_path / "mem.db"))
    )

    async def _boom() -> None:
        raise RuntimeError("synthetic backend init failure")

    handler.ensure_initialized = _boom  # type: ignore[assignment]
    proxy.memory_handler = handler

    # Startup must NOT raise — the memory slot must report error and the
    # rest of the startup pipeline keeps going (quota registry etc.).
    await proxy.startup()
    try:
        assert proxy.warmup.memory_backend.status == "error"
        assert "synthetic" in (proxy.warmup.memory_backend.error or "")
        # Memory embedder stays null because the backend never initialized.
        assert proxy.warmup.memory_embedder.status == "null"
        health = handler.health_status()
        assert health["initialized"] is False
    finally:
        await proxy.shutdown()

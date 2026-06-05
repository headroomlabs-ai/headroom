"""Tests for memory-architecture-v2 quick wins.

Covers:
1. HEADROOM_SKIP_MEMORY_WARMUP=1 — lazy embedder init (Option D):
   - When the env var is set, warmup_embedder() must NOT be called at startup.
   - The warmup slot must report "null", not "loaded".
2. /debug/rss endpoint (Option F):
   - Returns 200 with the expected JSON fields.
   - Loopback-only: non-loopback clients get 404.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app

# ---------------------------------------------------------------------------
# Option D — Lazy memory embedder warmup via HEADROOM_SKIP_MEMORY_WARMUP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_memory_warmup_env_prevents_eager_load(tmp_path, monkeypatch):
    """When HEADROOM_SKIP_MEMORY_WARMUP=1 the embedder must not be warmed.

    warmup_embedder is replaced with an AsyncMock spy; after startup the spy
    must have been called zero times and the warmup slot must be "null".
    """
    monkeypatch.setenv("HEADROOM_SKIP_MEMORY_WARMUP", "1")

    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    proxy = HeadroomProxy(config)

    # Install a pre-initialised handler whose warmup_embedder we can spy on.
    handler = MemoryHandler(
        MemoryConfig(enabled=True, backend="local", db_path=str(tmp_path / "mem.db"))
    )
    handler._initialized = True

    warmup_spy = AsyncMock(return_value=True)
    handler.warmup_embedder = warmup_spy  # type: ignore[assignment]
    handler.ensure_initialized = AsyncMock()  # bypass real init
    proxy.memory_handler = handler

    await proxy.startup()
    try:
        assert warmup_spy.await_count == 0, (
            f"warmup_embedder must NOT be called when HEADROOM_SKIP_MEMORY_WARMUP=1, "
            f"got {warmup_spy.await_count} calls"
        )
        assert proxy.warmup.memory_embedder.status == "null", (
            f"memory_embedder slot must be 'null' when warmup is skipped; "
            f"got {proxy.warmup.memory_embedder.status!r}"
        )
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_default_behavior_still_warms_embedder(tmp_path, monkeypatch):
    """Without HEADROOM_SKIP_MEMORY_WARMUP, warmup_embedder is called once."""
    # Ensure the env var is absent.
    monkeypatch.delenv("HEADROOM_SKIP_MEMORY_WARMUP", raising=False)

    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

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
    handler._initialized = True

    # Fake backend with an embedder spy
    embed = AsyncMock(return_value=[0.0])

    class _FakeHM:
        _embedder = type("_E", (), {"embed": embed})()

    class _FakeBackend:
        _hierarchical_memory = _FakeHM()

        async def close(self):
            pass

    handler._backend = _FakeBackend()
    handler.ensure_initialized = AsyncMock()
    proxy.memory_handler = handler

    await proxy.startup()
    try:
        assert embed.await_count == 1, (
            "warmup_embedder must fire one embed() call on startup by default"
        )
        assert proxy.warmup.memory_embedder.status == "loaded", (
            f"memory_embedder slot must be 'loaded' after eager warmup; "
            f"got {proxy.warmup.memory_embedder.status!r}"
        )
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_skip_memory_warmup_truthy_values(tmp_path, monkeypatch):
    """HEADROOM_SKIP_MEMORY_WARMUP accepts 'true' and 'yes' as truthy values."""
    for value in ("true", "yes", "1"):
        monkeypatch.setenv("HEADROOM_SKIP_MEMORY_WARMUP", value)

        from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

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
        handler._initialized = True
        warmup_spy = AsyncMock(return_value=True)
        handler.warmup_embedder = warmup_spy  # type: ignore[assignment]
        handler.ensure_initialized = AsyncMock()
        proxy.memory_handler = handler

        await proxy.startup()
        try:
            assert warmup_spy.await_count == 0, (
                f"HEADROOM_SKIP_MEMORY_WARMUP={value!r} must skip warmup"
            )
        finally:
            await proxy.shutdown()


# ---------------------------------------------------------------------------
# Option F — /debug/rss endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def loopback_client():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app, client=("127.0.0.1", 12345)) as tc:
        yield tc


@pytest.fixture
def external_client():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app, client=("10.0.0.1", 54321)) as tc:
        yield tc


def test_debug_rss_returns_200_with_expected_fields(loopback_client):
    """Happy path: loopback client gets a valid JSON response."""
    response = loopback_client.get("/debug/rss")
    assert response.status_code == 200, f"expected 200, got {response.status_code}"
    data = response.json()

    required_fields = {
        "pid",
        "peak_rss_mb",
        "python_version",
        "gc_stats",
        "top_types",
        "ml_models",
        "hnsw_elements",
        "compression_cache_sessions",
        "toin_patterns",
        "request_log_count",
        "memory_embedder_warmed",
    }
    missing = required_fields - set(data.keys())
    assert not missing, f"/debug/rss response missing fields: {missing}"


def test_debug_rss_pid_is_positive_int(loopback_client):
    data = loopback_client.get("/debug/rss").json()
    assert isinstance(data["pid"], int)
    assert data["pid"] > 0


def test_debug_rss_peak_rss_mb_is_positive_float(loopback_client):
    data = loopback_client.get("/debug/rss").json()
    assert isinstance(data["peak_rss_mb"], (int, float))
    assert data["peak_rss_mb"] > 0, "peak RSS must be > 0 for a live process"


def test_debug_rss_gc_stats_shape(loopback_client):
    data = loopback_client.get("/debug/rss").json()
    assert isinstance(data["gc_stats"], list)
    assert len(data["gc_stats"]) == 3, "CPython always has 3 GC generations"
    for entry in data["gc_stats"]:
        assert "generation" in entry
        assert "collections" in entry
        assert "collected" in entry


def test_debug_rss_top_types_shape(loopback_client):
    data = loopback_client.get("/debug/rss").json()
    assert isinstance(data["top_types"], list)
    assert len(data["top_types"]) <= 10
    for entry in data["top_types"]:
        assert "type" in entry
        assert "count" in entry
        assert isinstance(entry["count"], int)
        assert entry["count"] > 0


def test_debug_rss_is_loopback_only(external_client):
    """Non-loopback clients must receive 404 (invisible to scanners)."""
    response = external_client.get("/debug/rss")
    assert response.status_code == 404
    # Must be 404, not 403 — endpoints should be invisible.
    assert response.status_code != 403


def test_debug_rss_memory_embedder_warmed_is_bool(loopback_client):
    data = loopback_client.get("/debug/rss").json()
    assert isinstance(data["memory_embedder_warmed"], bool)

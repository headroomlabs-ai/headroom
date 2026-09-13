"""Regression tests for lazy memory warmup and RSS diagnostics."""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from headroom.proxy.server import (
    HeadroomProxy,
    ProxyConfig,
    _gc_snapshot,
    _peak_rss_mb,
    create_app,
)


def _proxy_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
async def test_skip_memory_warmup_truthy_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("HEADROOM_SKIP_MEMORY_WARMUP", value)
    proxy = HeadroomProxy(_proxy_config())
    handler = SimpleNamespace(
        ensure_initialized=AsyncMock(),
        warmup_embedder=AsyncMock(return_value=True),
        health_status=lambda: {"initialized": True, "backend": "local"},
    )
    proxy.memory_handler = handler

    await proxy.startup()

    handler.warmup_embedder.assert_not_awaited()
    assert proxy.warmup.memory_embedder.status == "null"


@pytest.mark.asyncio
async def test_memory_warmup_remains_eager_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_SKIP_MEMORY_WARMUP", raising=False)
    proxy = HeadroomProxy(_proxy_config())
    handler = SimpleNamespace(
        ensure_initialized=AsyncMock(),
        warmup_embedder=AsyncMock(return_value=True),
        health_status=lambda: {"initialized": True, "backend": "local"},
    )
    proxy.memory_handler = handler

    await proxy.startup()

    handler.warmup_embedder.assert_awaited_once_with()
    assert proxy.warmup.memory_embedder.status == "loaded"


def test_peak_rss_is_optional_without_resource(monkeypatch) -> None:
    real_import = builtins.__import__

    def import_without_resource(name, *args, **kwargs):
        if name == "resource":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_resource)
    assert _peak_rss_mb() is None


def test_gc_snapshot_does_not_force_collection(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.proxy.server.gc.collect", lambda: pytest.fail("gc.collect called")
    )
    stats, top_types = _gc_snapshot()
    assert stats
    assert all({"generation", "collections", "collected"} <= row.keys() for row in stats)
    assert len(top_types) <= 10


def test_debug_rss_schema_and_loopback_guard() -> None:
    app = create_app(_proxy_config())
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.get("/debug/rss", headers={"host": "127.0.0.1"})

    assert response.status_code == 200
    data = response.json()
    assert {
        "pid",
        "rss_mb",
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
    } <= data.keys()
    assert data["pid"] > 0
    assert isinstance(data["memory_embedder_warmed"], bool)

    with TestClient(app, client=("10.0.0.1", 12345)) as client:
        denied = client.get("/debug/rss", headers={"host": "127.0.0.1"})
    assert denied.status_code == 404

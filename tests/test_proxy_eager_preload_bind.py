"""Startup must bind its port even when eager preload hangs (#790).

``HeadroomProxy.startup()`` runs inside the ASGI lifespan, which completes
*before* uvicorn binds the socket. The eager compressor/parser preload used to
run synchronously there, so a hang or an uncatchable native stall during a model
load (observed on Windows) left the proxy "never opening its port". The preload
now runs off the event loop under ``asyncio.wait_for`` with
``EAGER_PRELOAD_TIMEOUT_SECONDS``; on timeout startup logs and continues so the
bind still happens and transforms fall back to lazy loading.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("fastapi")

import headroom.proxy.server as server_mod
from headroom.proxy.server import ProxyConfig, create_app


def _make_proxy(*, optimize: bool):
    config = ProxyConfig(
        optimize=optimize,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        subscription_tracking_enabled=False,
    )
    return create_app(config).state.proxy


class _FastTransform:
    def __init__(self, status):
        self._status = status

    def eager_load_compressors(self):
        return self._status


class _RaisingTransform:
    def eager_load_compressors(self):
        raise RuntimeError("boom")


class _NonDictTransform:
    def eager_load_compressors(self):
        return "not-a-dict"


class _HangingTransform:
    """Simulates a model load that hangs forever (released via the event)."""

    def __init__(self, release: threading.Event):
        self._release = release

    def eager_load_compressors(self):
        # Safety cap so a misbehaving test can never wedge the suite.
        self._release.wait(timeout=30)
        return {"hang": "done"}


class _FakePipeline:
    def __init__(self, transforms):
        self.transforms = transforms


def test_eager_preload_dedupes_and_swallows_failures():
    proxy = _make_proxy(optimize=False)
    shared = _FastTransform({"shared": "enabled"})
    proxy.anthropic_pipeline = _FakePipeline([shared, _FastTransform({"kompress": "enabled"})])
    # ``shared`` appears in both pipelines and must load exactly once; the
    # raising and non-dict transforms must be skipped without aborting.
    proxy.openai_pipeline = _FakePipeline([shared, _RaisingTransform(), _NonDictTransform()])

    eager_status, statuses = proxy._eager_preload_transforms()

    assert eager_status == {"shared": "enabled", "kompress": "enabled"}
    assert statuses == [{"shared": "enabled"}, {"kompress": "enabled"}]


async def test_startup_binds_despite_hung_preload(monkeypatch):
    monkeypatch.setattr(server_mod, "EAGER_PRELOAD_TIMEOUT_SECONDS", 0.3)
    proxy = _make_proxy(optimize=True)
    release = threading.Event()
    proxy.anthropic_pipeline = _FakePipeline([_HangingTransform(release)])
    proxy.openai_pipeline = _FakePipeline([])

    try:
        start = time.monotonic()
        await proxy.startup()  # must NOT wait on the hung load
        elapsed = time.monotonic() - start
        # Returns shortly after the 0.3s preload timeout, far below the 30s hang.
        assert elapsed < 10
    finally:
        release.set()
        await proxy.shutdown()


async def test_startup_merges_warmup_for_normal_transforms(monkeypatch):
    proxy = _make_proxy(optimize=True)
    captured: list[dict] = []
    monkeypatch.setattr(proxy.warmup, "merge_transform_status", captured.append)
    proxy.anthropic_pipeline = _FakePipeline([_FastTransform({"kompress": "enabled"})])
    proxy.openai_pipeline = _FakePipeline([])

    try:
        await proxy.startup()
        assert {"kompress": "enabled"} in captured
        assert proxy._kompress_status == "enabled"
    finally:
        await proxy.shutdown()

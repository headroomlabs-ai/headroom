"""Startup eager-preload must defer Kompress native loading before binding.

Regression for the production crash where ``eager_load_compressors`` entered
the cached Kompress native stack on the blocking startup/lifespan path.
"""

from __future__ import annotations

import pytest

from headroom import onnx_runtime
from headroom.transforms import kompress_compressor as kc
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.kompress_compressor import KompressModelNotCached


def test_local_first_no_network_when_disallowed(monkeypatch):
    """allow_network=False must never fall back to a network download."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls: list[bool] = []

    def fake_download(repo_id, filename, **kwargs):
        local_only = kwargs.get("local_files_only", False)
        calls.append(local_only)
        if local_only:
            raise LocalEntryNotFoundError("not cached")
        return "/cache/networked"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    with pytest.raises(LocalEntryNotFoundError):
        onnx_runtime.hf_hub_download_local_first("org/model", "f.onnx", allow_network=False)

    # Only the local-only lookup ran; the network branch was never taken.
    assert calls == [True]


def test_local_first_falls_back_to_network_by_default(monkeypatch):
    """allow_network=True (default) keeps the historic cold-start behavior."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls: list[bool] = []

    def fake_download(repo_id, filename, **kwargs):
        local_only = kwargs.get("local_files_only", False)
        calls.append(local_only)
        if local_only:
            raise LocalEntryNotFoundError("not cached")
        return "/cache/networked"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    path = onnx_runtime.hf_hub_download_local_first("org/model", "f.onnx")
    assert path == "/cache/networked"
    assert calls == [True, False]  # local-only miss, then network download


def test_load_kompress_onnx_cache_miss_raises_not_cached(monkeypatch):
    """A cache-only ONNX load surfaces KompressModelNotCached, not a network call."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setattr(kc, "_kompress_cache", {})

    def fake_local_first(repo_id, filename, *, allow_network=True):
        assert allow_network is False  # eager preload must request cache-only
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(kc, "hf_hub_download_local_first", fake_local_first)

    with pytest.raises(KompressModelNotCached):
        kc._load_kompress_onnx("org/model", allow_download=False)


def test_load_kompress_auto_does_not_pytorch_download_on_cache_miss(monkeypatch):
    """Auto mode must propagate the cache miss, not fall back to a PyTorch fetch."""
    monkeypatch.setattr(kc, "_kompress_cache", {})
    monkeypatch.setattr(kc, "_selected_backend", lambda: "auto")
    monkeypatch.setattr(kc, "_is_onnx_available", lambda: True)
    monkeypatch.setattr(kc, "_is_pytorch_available", lambda: True)

    def onnx_not_cached(model_id, *, use_coreml=False, allow_download=True):
        raise KompressModelNotCached(model_id)

    def pytorch_should_not_run(*args, **kwargs):
        raise AssertionError("PyTorch fallback must not download on a cache-only miss")

    monkeypatch.setattr(kc, "_load_kompress_onnx", onnx_not_cached)
    monkeypatch.setattr(kc, "_load_kompress_pytorch", pytorch_should_not_run)

    with pytest.raises(KompressModelNotCached):
        kc._load_kompress("org/model", allow_download=False)


class _StubCompressor:
    def __init__(self, *, cached: bool):
        self._cached = cached
        self.preload_calls: list[bool] = []

    def preload(self, *, allow_download: bool = True) -> str:
        self.preload_calls.append(allow_download)
        if self._cached:
            return "onnx"
        raise KompressModelNotCached("org/model")


class _FatalPreloadCompressor(_StubCompressor):
    def preload(self, *, allow_download: bool = True) -> str:
        self.preload_calls.append(allow_download)
        raise SystemExit("native Kompress preload")


def _router_kompress_only() -> ContentRouter:
    return ContentRouter(
        ContentRouterConfig(
            enable_kompress=True,
            enable_code_aware=False,
            enable_smart_crusher=False,
        )
    )


@pytest.mark.parametrize("cache_state", ["cached", "uncached"])
def test_eager_load_defers_kompress_regardless_of_cache_state(monkeypatch, cache_state):
    router = _router_kompress_only()
    stub = _StubCompressor(cached=cache_state == "cached")
    monkeypatch.setattr(router, "_get_kompress", lambda: stub)
    # Artifact prefetch is files-only, but it still reaches the network — keep it
    # out of this assertion so the test stays about the native-preload boundary.
    monkeypatch.setattr(router, "_prefetch_kompress_artifacts_async", lambda _cfg: False)

    status = router.eager_load_compressors()

    assert status["kompress"] == "deferred"
    assert stub.preload_calls == []


def test_eager_load_keeps_disabled_kompress_disabled(monkeypatch):
    router = ContentRouter(
        ContentRouterConfig(
            enable_kompress=False,
            enable_code_aware=False,
            enable_smart_crusher=False,
        )
    )
    stub = _StubCompressor(cached=True)
    monkeypatch.setattr(router, "_get_kompress", lambda: stub)

    status = router.eager_load_compressors()

    assert "kompress" not in status
    assert stub.preload_calls == []


def test_eager_load_reports_unavailable_kompress(monkeypatch):
    router = _router_kompress_only()
    monkeypatch.setattr(router, "_get_kompress", lambda: None)

    status = router.eager_load_compressors()

    assert status["kompress"] == "unavailable"


def test_non_kompress_warmups_continue_when_kompress_is_deferred(monkeypatch):
    router = _router_kompress_only()
    stub = _StubCompressor(cached=True)
    monkeypatch.setattr(router, "_get_kompress", lambda: stub)
    monkeypatch.setattr(router, "_prefetch_kompress_artifacts_async", lambda _cfg: False)
    monkeypatch.setattr("headroom.compression.detector._magika_available", lambda: True)
    monkeypatch.setattr("headroom.compression.detector._get_magika", lambda: object())

    status = router.eager_load_compressors()

    assert status["kompress"] == "deferred"
    assert status["magika"] == "enabled"
    assert stub.preload_calls == []


# ── Startup artifact prefetch (files only, never native init) ──────────────────
# The cold-start cost #2001 left behind: the ~4-minute model download began on the
# FIRST REQUEST, so every request in that window went silently uncompressed.
# Prefetching FILES at startup is safe because it is plain huggingface_hub HTTP —
# it never constructs an InferenceSession, which is the boundary that segfaults in
# libarrow/jemalloc on RHEL/CentOS 7-family hosts (#1908).


def test_eager_load_starts_artifact_prefetch_without_native_preload(monkeypatch):
    router = _router_kompress_only()
    stub = _StubCompressor(cached=False)
    monkeypatch.setattr(router, "_get_kompress", lambda: stub)
    prefetch_calls: list[object] = []
    monkeypatch.setattr(
        router,
        "_prefetch_kompress_artifacts_async",
        lambda cfg: (prefetch_calls.append(cfg), True)[1],
    )

    status = router.eager_load_compressors()

    assert status["kompress_artifacts"] == "prefetching"
    # The #2001 invariant still holds: no native preload on the startup path.
    assert status["kompress"] == "deferred"
    assert stub.preload_calls == []
    assert len(prefetch_calls) == 1


def test_prefetch_never_constructs_a_session_or_imports_transformers(monkeypatch):
    """The safety property that makes startup prefetch legal at all."""
    monkeypatch.setattr(kc, "_kompress_cache", {})
    requested: list[str] = []

    def fake_local_first(repo_id, filename, *, allow_network=True):
        requested.append(filename)
        return f"/cache/{filename}"

    monkeypatch.setattr(kc, "hf_hub_download_local_first", fake_local_first)

    def explode(*args, **kwargs):
        raise AssertionError("prefetch must not build the model")

    monkeypatch.setattr(kc, "_load_kompress", explode)
    monkeypatch.setattr(kc, "_load_kompress_onnx", explode)

    assert kc.prefetch_kompress_artifacts("org/model") is True
    # Stops at the first candidate that resolves — the loader tries the same order.
    assert requested == [kc._onnx_filename_candidates()[0]]


def test_prefetch_reports_false_when_no_artifact_resolves(monkeypatch):
    monkeypatch.setattr(kc, "_kompress_cache", {})

    def always_missing(repo_id, filename, *, allow_network=True):
        raise OSError("not found")

    monkeypatch.setattr(kc, "hf_hub_download_local_first", always_missing)

    assert kc.prefetch_kompress_artifacts("org/model") is False


def test_background_prefetch_is_noop_when_model_already_cached(monkeypatch):
    monkeypatch.setattr(kc, "_kompress_cache", {"org/model": object()})
    monkeypatch.setattr(kc, "is_kompress_available", lambda: True)

    assert kc.ensure_background_prefetch("org/model") is False


@pytest.mark.asyncio
async def test_proxy_startup_does_not_enter_cached_kompress_native_loader(monkeypatch):
    pytest.importorskip("httpx")
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            code_aware_enabled=False,
        )
    )
    # The background warm thread (GH #2730) is allowed to preload; this test
    # covers the *awaited* startup path only, so disable it for determinism.
    monkeypatch.setenv("HEADROOM_KOMPRESS_BACKGROUND_WARM", "0")
    router = _router_kompress_only()
    stub = _FatalPreloadCompressor(cached=True)
    monkeypatch.setattr(router, "_get_kompress", lambda: stub)
    proxy.anthropic_pipeline.transforms = [router]
    proxy.openai_pipeline.transforms = [router]

    await proxy.startup()
    try:
        assert stub.preload_calls == []
        assert proxy.warmup.kompress.info["source_status"] == "deferred"
        assert proxy._kompress_warm_thread is None
    finally:
        await proxy.shutdown()


def _kompress_proxy(monkeypatch, stub):
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            code_aware_enabled=False,
        )
    )
    router = _router_kompress_only()
    monkeypatch.setattr(router, "_get_kompress", lambda: stub)
    proxy.anthropic_pipeline.transforms = [router]
    proxy.openai_pipeline.transforms = [router]
    return proxy


@pytest.mark.asyncio
async def test_startup_background_warm_promotes_deferred_kompress(monkeypatch):
    """A cached model must reach the warmup slot without any request (GH #2730)."""
    pytest.importorskip("httpx")

    stub = _StubCompressor(cached=True)
    proxy = _kompress_proxy(monkeypatch, stub)

    await proxy.startup()
    try:
        thread = proxy._kompress_warm_thread
        assert thread is not None
        thread.join(timeout=10)
        assert not thread.is_alive()
        # Cache-only: the warm path must never hit the network.
        assert stub.preload_calls == [False]
        assert proxy.warmup.kompress.status == "loaded"
        assert proxy.warmup.kompress.info["backend"] == "onnx"
    finally:
        await proxy.shutdown()


@pytest.mark.asyncio
async def test_startup_background_warm_leaves_cold_cache_deferred(monkeypatch):
    """An uncached model stays deferred and says why."""
    pytest.importorskip("httpx")

    stub = _StubCompressor(cached=False)
    proxy = _kompress_proxy(monkeypatch, stub)

    await proxy.startup()
    try:
        thread = proxy._kompress_warm_thread
        assert thread is not None
        thread.join(timeout=10)
        assert stub.preload_calls == [False]
        assert proxy.warmup.kompress.status == "null"
        assert proxy.warmup.kompress.info["detail"] == "model not cached"
    finally:
        await proxy.shutdown()


class _LeakyPreloadCompressor(_StubCompressor):
    """Fails with an exception carrying a path and a token, as loaders do."""

    def preload(self, *, allow_download: bool = True) -> str:
        self.preload_calls.append(allow_download)
        raise RuntimeError("/opt/secret-path/model.onnx token=SENTINEL-LEAK-9f3a")


@pytest.mark.asyncio
async def test_startup_background_warm_failure_detail_is_bounded(monkeypatch):
    """The warm failure reason must not reach the auth-exempt health payload."""
    pytest.importorskip("httpx")

    stub = _LeakyPreloadCompressor(cached=True)
    proxy = _kompress_proxy(monkeypatch, stub)

    await proxy.startup()
    try:
        thread = proxy._kompress_warm_thread
        assert thread is not None
        thread.join(timeout=10)
        assert stub.preload_calls == [False]
        assert proxy.warmup.kompress.status == "null"
        assert proxy.warmup.kompress.info["detail"] == "warm failed"
    finally:
        await proxy.shutdown()

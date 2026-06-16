"""Startup eager-preload must be cache-only so a cold cache cannot block or
crash the proxy before it binds its port.

Regression for the production crash where ``eager_load_compressors`` ran a
network ``hf_hub_download`` of the Kompress ONNX model on the blocking
startup/lifespan path. On a cold cache that download could hang (300s bind
timeout) or hit a native ``SIGABRT`` in the download/ML stack, killing the
interpreter before it ever listened on its port.
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


def _router_kompress_only() -> ContentRouter:
    return ContentRouter(
        ContentRouterConfig(
            enable_kompress=True,
            enable_code_aware=False,
            enable_smart_crusher=False,
        )
    )


def test_eager_load_defers_when_model_not_cached(monkeypatch):
    router = _router_kompress_only()
    stub = _StubCompressor(cached=False)
    monkeypatch.setattr(router, "_get_kompress", lambda: stub)

    status = router.eager_load_compressors()

    assert status["kompress"] == "deferred"
    assert stub.preload_calls == [False]  # cache-only preload at startup


def test_eager_load_enabled_when_model_cached(monkeypatch):
    router = _router_kompress_only()
    stub = _StubCompressor(cached=True)
    monkeypatch.setattr(router, "_get_kompress", lambda: stub)

    status = router.eager_load_compressors()

    assert status["kompress"] == "enabled"
    assert status["kompress_backend"] == "onnx"
    assert stub.preload_calls == [False]

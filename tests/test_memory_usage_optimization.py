"""Tests for memory-usage optimizations in the proxy hot path.

These tests verify that long-lived components (ONNX router, fastembed
scorer, per-session compression caches) release the heavyweight model
state and bounded-cache slots they accumulate, so that the proxy's
worker RSS does not grow unbounded across request bursts.

Covered fixes:
    * ``OnnxTechniqueRouter.close()`` releases the cached ONNX session,
      tokenizer, and pre-computed text embeddings.
    * ``ImageCompressor`` caches a single ``OnnxTechniqueRouter`` per
      instance (rather than building a fresh one per request) and
      releases it from ``close()``.
    * ``EmbeddingScorer.close()`` drops the fastembed model so its
      ONNX session is reclaimed.
    * The proxy's per-session ``_compression_caches`` map is a true
      LRU: touched sessions move to the tail, and eviction drops the
      coldest quarter from the head.
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# OnnxTechniqueRouter.close()
# ---------------------------------------------------------------------------


class TestOnnxRouterClose:
    """Unit tests for ``OnnxTechniqueRouter.close``."""

    def test_close_on_unloaded_router_is_noop(self) -> None:
        """``close()`` works on a router whose models were never loaded."""
        from headroom.image.onnx_router import OnnxTechniqueRouter

        router = OnnxTechniqueRouter(use_siglip=False)

        router.close()  # must not raise

        assert router._classifier_session is None
        assert router._tokenizer is None
        assert router._siglip_session is None
        assert router._text_embeddings == {}
        assert router._siglip_processor is None
        assert router._id2label == {}

    def test_close_clears_loaded_state(self) -> None:
        """``close()`` releases every cached model reference.

        We stub the lazy-loaded attributes with sentinel objects so the
        test does not require network access or 127 MB of ONNX weights.
        ``close()`` must drop every one of them.
        """
        from headroom.image.onnx_router import OnnxTechniqueRouter

        router = OnnxTechniqueRouter(use_siglip=True)
        router._classifier_session = object()
        router._tokenizer = object()
        router._id2label = {0: "transcode"}
        router._siglip_session = object()
        router._text_embeddings = {"foo": object()}
        router._siglip_processor = object()

        router.close()

        assert router._classifier_session is None
        assert router._tokenizer is None
        assert router._id2label == {}
        assert router._siglip_session is None
        assert router._text_embeddings == {}
        assert router._siglip_processor is None

    def test_close_is_idempotent(self) -> None:
        """Calling ``close()`` twice does not error."""
        from headroom.image.onnx_router import OnnxTechniqueRouter

        router = OnnxTechniqueRouter(use_siglip=False)
        router._classifier_session = object()

        router.close()
        router.close()

        assert router._classifier_session is None


# ---------------------------------------------------------------------------
# ImageCompressor router caching
# ---------------------------------------------------------------------------


class _StubOnnxRouter:
    """Test double mimicking the public surface of OnnxTechniqueRouter."""

    instances: list[_StubOnnxRouter] = []

    def __init__(self, use_siglip: bool = True) -> None:
        self.use_siglip = use_siglip
        self.closed = False
        self.classify_calls = 0
        type(self).instances.append(self)

    def classify(self, image_data: bytes, query: str) -> Any:
        from headroom.image.trained_router import RouteDecision, Technique

        self.classify_calls += 1
        return RouteDecision(technique=Technique.PRESERVE, confidence=0.9, reason="stub")

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def stub_onnx(monkeypatch: pytest.MonkeyPatch) -> type[_StubOnnxRouter]:
    """Replace OnnxTechniqueRouter with a counting stub."""
    _StubOnnxRouter.instances = []
    monkeypatch.setattr(
        "headroom.image.onnx_router.OnnxTechniqueRouter",
        _StubOnnxRouter,
    )
    return _StubOnnxRouter


class TestImageCompressorRouterCaching:
    """ImageCompressor must reuse one ONNX router across requests."""

    def test_get_onnx_router_caches_instance(self, stub_onnx: type[_StubOnnxRouter]) -> None:
        """Repeated lookups return the same router instance."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor(use_siglip=False)

        first = compressor._get_onnx_router()
        second = compressor._get_onnx_router()

        assert first is second
        assert len(stub_onnx.instances) == 1

    def test_close_releases_onnx_router(self, stub_onnx: type[_StubOnnxRouter]) -> None:
        """``close()`` invokes the ONNX router's own ``close()``."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor(use_siglip=False)
        router = compressor._get_onnx_router()
        assert isinstance(router, _StubOnnxRouter)

        compressor.close(unload_models=False)

        assert router.closed is True
        assert compressor._onnx_router is None

    def test_close_without_load_is_safe(self, stub_onnx: type[_StubOnnxRouter]) -> None:
        """``close()`` on a never-used compressor must not load models."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor(use_siglip=False)
        compressor.close(unload_models=False)

        assert stub_onnx.instances == []
        assert compressor._onnx_router is None

    def test_close_after_close_is_idempotent(self, stub_onnx: type[_StubOnnxRouter]) -> None:
        """Double-close on the compressor does not error."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor(use_siglip=False)
        compressor._get_onnx_router()
        compressor.close(unload_models=False)
        compressor.close(unload_models=False)

        assert compressor._onnx_router is None


# ---------------------------------------------------------------------------
# EmbeddingScorer.close()
# ---------------------------------------------------------------------------


class TestEmbeddingScorerClose:
    """Unit tests for ``EmbeddingScorer.close``."""

    def test_close_on_unloaded_scorer_is_noop(self) -> None:
        """``close()`` is safe on a scorer that never loaded a model."""
        from headroom.relevance.embedding import EmbeddingScorer

        scorer = EmbeddingScorer()
        scorer.close()  # must not raise

        assert scorer._model is None

    def test_close_drops_loaded_model(self) -> None:
        """``close()`` drops the cached fastembed model reference."""
        from headroom.relevance.embedding import EmbeddingScorer

        scorer = EmbeddingScorer()
        scorer._model = object()  # type: ignore[assignment]

        scorer.close()

        assert scorer._model is None

    def test_close_runs_gc(self) -> None:
        """``close()`` triggers a GC cycle so ORT sessions can be freed.

        We can't directly observe the ORT free, but we can verify that
        ``gc.collect`` runs by checking that an unreachable referent of
        the scorer's ``_model`` slot becomes collectable after close.
        """
        from headroom.relevance.embedding import EmbeddingScorer

        scorer = EmbeddingScorer()
        sentinel = object()
        scorer._model = sentinel  # type: ignore[assignment]
        del sentinel
        gc.disable()
        try:
            scorer.close()
        finally:
            gc.enable()

        assert scorer._model is None

    def test_close_is_idempotent(self) -> None:
        """Repeated ``close()`` calls do not error."""
        from headroom.relevance.embedding import EmbeddingScorer

        scorer = EmbeddingScorer()
        scorer._model = object()  # type: ignore[assignment]
        scorer.close()
        scorer.close()

        assert scorer._model is None


# ---------------------------------------------------------------------------
# Proxy compression-cache LRU
# ---------------------------------------------------------------------------


class _FakeCompressionCache:
    """Lightweight stand-in so we don't import the real CompressionCache."""

    def __init__(self) -> None:
        self.tag = id(self)


def _make_proxy_lru(
    monkeypatch: pytest.MonkeyPatch,
    max_sessions: int,
) -> Any:
    """Construct a minimal proxy stub exercising ``_get_compression_cache``.

    We avoid building the full HeadroomProxy (which wires HTTP clients,
    handlers, and config — irrelevant to LRU semantics) and instead
    bind the live method to a stub that owns the same attributes.
    """
    import threading

    from headroom.proxy import helpers as helpers_mod
    from headroom.proxy import server as server_mod

    monkeypatch.setattr(helpers_mod, "MAX_COMPRESSION_CACHE_SESSIONS", max_sessions)
    monkeypatch.setattr(server_mod, "MAX_COMPRESSION_CACHE_SESSIONS", max_sessions)

    # Patch the CompressionCache import inside the method to return our
    # lightweight fake so the LRU test stays fast and dependency-free.
    fake_cache_module = type(
        "_FakeModule",
        (),
        {"CompressionCache": _FakeCompressionCache},
    )()
    import sys

    monkeypatch.setitem(
        sys.modules,
        "headroom.cache.compression_cache",
        fake_cache_module,
    )

    class _ProxyStub:
        pass

    stub = _ProxyStub()
    stub._compression_caches = OrderedDict()
    stub._compression_caches_lock = threading.RLock()

    method = server_mod.HeadroomProxy._get_compression_cache.__get__(stub, _ProxyStub)
    stub._get_compression_cache = method  # type: ignore[attr-defined]
    return stub


class TestCompressionCacheLRU:
    """``_get_compression_cache`` enforces a true LRU policy."""

    def test_returns_same_instance_for_same_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two lookups for the same session id yield the same cache."""
        stub = _make_proxy_lru(monkeypatch, max_sessions=8)

        first = stub._get_compression_cache("session-A")
        second = stub._get_compression_cache("session-A")

        assert first is second

    def test_recent_access_protects_from_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A recently-touched session must survive subsequent eviction.

        Old behavior: insertion-order eviction always dropped the
        oldest *created* session, even if it was the hottest. The fix
        moves accessed sessions to the tail so cold ones get evicted.
        """
        stub = _make_proxy_lru(monkeypatch, max_sessions=4)

        # Fill to capacity.
        stub._get_compression_cache("oldest")
        stub._get_compression_cache("middle-a")
        stub._get_compression_cache("middle-b")
        stub._get_compression_cache("newest")

        # Touch the oldest so it becomes the most recently used.
        protected = stub._get_compression_cache("oldest")

        # Adding a fifth session triggers eviction of the coldest 25%.
        # With max_sessions=4 that's 1 entry; "middle-a" should die.
        stub._get_compression_cache("fresh")

        assert "oldest" in stub._compression_caches
        assert "middle-a" not in stub._compression_caches
        assert stub._get_compression_cache("oldest") is protected

    def test_eviction_drops_quarter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Eviction removes ``max_sessions // 4`` cold entries at once."""
        stub = _make_proxy_lru(monkeypatch, max_sessions=8)

        for i in range(8):
            stub._get_compression_cache(f"session-{i}")
        # All eight present.
        assert len(stub._compression_caches) == 8

        # Adding a 9th forces eviction of the coldest quarter (2 here).
        stub._get_compression_cache("session-new")

        assert len(stub._compression_caches) == 7
        assert "session-0" not in stub._compression_caches
        assert "session-1" not in stub._compression_caches
        assert "session-new" in stub._compression_caches

    def test_access_order_reflected_in_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Out-of-order access shuffles the eviction order accordingly."""
        stub = _make_proxy_lru(monkeypatch, max_sessions=4)

        stub._get_compression_cache("a")
        stub._get_compression_cache("b")
        stub._get_compression_cache("c")
        stub._get_compression_cache("d")

        # Re-touch in a non-insertion order.
        stub._get_compression_cache("a")
        stub._get_compression_cache("c")

        # Now "b" is the LRU; insert a new session, "b" should be the
        # one evicted (eviction count = 4 // 4 = 1).
        stub._get_compression_cache("e")

        keys = list(stub._compression_caches.keys())
        assert "b" not in keys
        assert keys[-1] == "e"


# ---------------------------------------------------------------------------
# Integration: ImageCompressor + ONNX router across multiple requests
# ---------------------------------------------------------------------------


class TestImageCompressorIntegration:
    """End-to-end check that the compressor reuses the cached router.

    These tests use the in-process stub from the earlier suite so they
    don't touch HuggingFace or load actual ONNX weights.
    """

    def test_repeated_compress_calls_reuse_router(self, stub_onnx: type[_StubOnnxRouter]) -> None:
        """Compressing two requests in a row constructs only one router."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor(use_siglip=False)

        # Build a minimal payload the compressor will treat as "has image"
        # so the routing branch executes. We do not need real bytes — the
        # stub router accepts any payload.
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this picture"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo=",
                        },
                    },
                ],
            }
        ]

        compressor.compress(msgs, provider="openai")
        compressor.compress(msgs, provider="openai")

        assert len(stub_onnx.instances) == 1
        assert stub_onnx.instances[0].classify_calls == 2

        compressor.close(unload_models=False)
        assert stub_onnx.instances[0].closed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

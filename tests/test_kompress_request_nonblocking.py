"""The proxy request path must never block on a cold Kompress model download.

Counterpart to ``test_kompress_preload_deferral.py`` (which covers the startup
path). A first deep-path request used to resolve the 274MB ONNX artifact via an
inline ``hf_hub_download`` on the request thread, where it raced the proxy's
``HEADROOM_COMPRESSION_TIMEOUT_SECONDS`` budget (GH #946 / #1146): the fetch was
cancelled mid-transfer, nothing cached, and every request re-hung and failed
open. The request path now resolves the model cache-only and pulls it down once
in a background daemon thread instead.
"""

from __future__ import annotations

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.kompress_compressor import KompressCompressor


def test_compress_cache_only_passes_through_without_network(monkeypatch):
    """compress(allow_download=False) on a cold cache must not hit the network."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setattr(kc, "_kompress_cache", {})
    monkeypatch.setattr(kc, "_selected_backend", lambda: "onnx")

    def fake_local_first(repo_id, filename, *, allow_network=True):
        assert allow_network is False, "request path must resolve the model cache-only"
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(kc, "hf_hub_download_local_first", fake_local_first)

    text = " ".join(["token"] * 50)  # >= 10 words: not the short-content passthrough
    result = KompressCompressor().compress(text, allow_download=False)

    assert result.compressed == text
    assert result.compression_ratio == 1.0


def test_ensure_background_download_runs_one_thread_per_model(monkeypatch):
    """At most one download thread per model; retried after it dies; skipped once cached."""
    monkeypatch.setattr(kc, "_kompress_cache", {})
    monkeypatch.setattr(kc, "_download_threads", {})

    created: list[object] = []

    class FakeThread:
        def __init__(self, *, target, args, name, daemon):
            self.target, self.args, self.name, self.daemon = target, args, name, daemon
            self._alive = True
            created.append(self)

        def start(self):  # do not actually run — simulate a live download
            pass

        def is_alive(self):
            return self._alive

    monkeypatch.setattr(kc.threading, "Thread", FakeThread)

    kc.ensure_background_download("org/model", "cpu")
    kc.ensure_background_download("org/model", "cpu")  # thread alive -> no second start
    assert len(created) == 1
    assert created[0].daemon is True

    created[0]._alive = False  # simulate the download finishing/failing
    kc.ensure_background_download("org/model", "cpu")  # dead -> retry
    assert len(created) == 2

    kc._kompress_cache["org/model"] = ("model", "tokenizer", "onnx")
    kc.ensure_background_download("org/model", "cpu")  # cached -> no-op
    assert len(created) == 2


def _kompress_router() -> ContentRouter:
    return ContentRouter(
        ContentRouterConfig(
            enable_kompress=True,
            enable_code_aware=False,
            enable_smart_crusher=False,
        )
    )


def test_router_skips_deep_path_and_fetches_in_background_when_not_ready(monkeypatch):
    router = _kompress_router()
    calls = {"ensure": 0, "compress": 0}

    class NotReadyKompress:
        def is_ready(self) -> bool:
            return False

        def ensure_background_load(self) -> None:
            calls["ensure"] += 1

        def compress(self, *args, **kwargs):
            calls["compress"] += 1
            raise AssertionError("must not run the deep path before the model is cached")

    monkeypatch.setattr(router, "_get_kompress", lambda: NotReadyKompress())

    text = " ".join(["content"] * 40)
    out, tokens = router._try_ml_compressor(text, context="")

    assert out == text  # passthrough, unchanged
    assert calls["ensure"] == 1  # background fetch kicked off
    assert calls["compress"] == 0  # deep path skipped, no inline download


def test_router_compresses_cache_only_when_ready(monkeypatch):
    router = _kompress_router()
    seen: dict[str, object] = {}

    class ReadyResult:
        compressed = "kept words"
        compressed_tokens = 2

    class ReadyKompress:
        def is_ready(self) -> bool:
            return True

        def ensure_background_load(self) -> None:
            raise AssertionError("must not fetch when the model is already cached")

        def compress(
            self, content, *, context="", question=None, target_ratio=None, allow_download=True
        ):
            seen["allow_download"] = allow_download
            return ReadyResult()

    monkeypatch.setattr(router, "_get_kompress", lambda: ReadyKompress())

    text = " ".join(["content"] * 40)
    out, tokens = router._try_ml_compressor(text, context="")

    assert seen["allow_download"] is False  # request path stays cache-only even when ready
    assert out == "kept words"

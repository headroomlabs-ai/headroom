"""A Kompress model that is still loading is expected, not a connectivity problem.

The router used to log "Kompress model not ready ... Check HuggingFace
connectivity" at WARNING on the first request after every proxy start, because
the post-startup warm-up takes ~20 s and requests arrive meanwhile. It now says
"still loading" at INFO while a loader thread runs, and keeps the WARNING for
when nothing is loading the model (e.g. a failed download in its retry backoff).
"""

from __future__ import annotations

import logging
import threading

import pytest

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

_LOGGER = "headroom.transforms.content_router"
_CONTENT = "some plain text that the ML model would compress. " * 4


class _NotReadyKompress:
    def __init__(self, *, loading: bool) -> None:
        self.loading = loading
        self.background_loads = 0

    def is_ready(self) -> bool:
        return False

    def ensure_background_load(self) -> None:
        self.background_loads += 1

    def is_loading(self) -> bool:
        return self.loading


def _router(monkeypatch: pytest.MonkeyPatch, compressor: object) -> ContentRouter:
    router = ContentRouter(
        ContentRouterConfig(
            relevance_split=False,
            lossless_then_lossy=False,
            ccr_inject_marker=False,
        )
    )
    monkeypatch.setattr(router, "_get_kompress", lambda: compressor)
    return router


def _logged(caplog: pytest.LogCaptureFixture, level: int, text: str) -> int:
    return sum(
        1
        for r in caplog.records
        if r.name == _LOGGER and r.levelno == level and text in r.getMessage()
    )


def test_still_loading_is_info_not_a_connectivity_warning(monkeypatch, caplog):
    compressor = _NotReadyKompress(loading=True)
    router = _router(monkeypatch, compressor)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        for _ in range(3):
            assert router._try_ml_compressor(_CONTENT, "ctx")[0] == _CONTENT

    assert _logged(caplog, logging.INFO, "still loading in the background") == 1
    assert _logged(caplog, logging.WARNING, "Check HuggingFace connectivity") == 0
    assert compressor.background_loads == 3


def test_no_loader_running_still_warns_once(monkeypatch, caplog):
    router = _router(monkeypatch, _NotReadyKompress(loading=False))

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        for _ in range(3):
            router._try_ml_compressor(_CONTENT, "ctx")

    assert _logged(caplog, logging.WARNING, "Check HuggingFace connectivity") == 1
    assert _logged(caplog, logging.INFO, "still loading in the background") == 0


def test_warns_once_the_load_ends_without_a_model(monkeypatch, caplog):
    """A warm-up that fails leaves no loader running: then it is degraded."""
    compressor = _NotReadyKompress(loading=True)
    router = _router(monkeypatch, compressor)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        router._try_ml_compressor(_CONTENT, "ctx")
        compressor.loading = False
        router._try_ml_compressor(_CONTENT, "ctx")

    assert _logged(caplog, logging.INFO, "still loading in the background") == 1
    assert _logged(caplog, logging.WARNING, "Check HuggingFace connectivity") == 1


def test_compressor_without_is_loading_keeps_the_warning(monkeypatch, caplog):
    class _Legacy:
        def is_ready(self) -> bool:
            return False

        def ensure_background_load(self) -> None:
            pass

    router = _router(monkeypatch, _Legacy())

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        router._try_ml_compressor(_CONTENT, "ctx")

    assert _logged(caplog, logging.WARNING, "Check HuggingFace connectivity") == 1


def test_registered_warm_up_thread_counts_as_loading():
    """The real signal: the warm-up thread is the model's registered loader."""
    model_id = "test/not-ready-log-model"
    release = threading.Event()
    try:
        thread = kc.start_kompress_warmup_thread(model_id, lambda: release.wait(5))
        assert kc.is_kompress_load_in_progress(model_id)
        release.set()
        thread.join(timeout=5)
        assert not kc.is_kompress_load_in_progress(model_id)
    finally:
        release.set()
        with kc._download_threads_lock:
            kc._download_threads.pop(model_id, None)

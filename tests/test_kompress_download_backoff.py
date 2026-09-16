"""An unreachable HuggingFace must not turn every request into a download thread.

The request path calls ensure_background_download() on every Kompress miss. A
finished-or-failed thread is replaced on the next call, which is what lets a
transient blip recover — but with no floor, a permanently unreachable Hub means
one new thread per request forever, each importing transformers and holding the
GIL against the event loop.
"""

from __future__ import annotations

import threading

import pytest

from headroom.transforms import kompress_compressor as kc


@pytest.fixture(autouse=True)
def _clean_registry():
    with kc._download_threads_lock:
        kc._download_threads.clear()
        kc._download_failures.clear()
    yield
    with kc._download_threads_lock:
        kc._download_threads.clear()
        kc._download_failures.clear()


def _spawned(monkeypatch, *, fails: bool) -> list[str]:
    """Run ensure_background_download with the real load stubbed out."""
    started: list[str] = []

    def fake_load(model_id, device, allow_download=True):
        started.append(model_id)
        if fails:
            raise OSError("hub unreachable")
        return object(), object(), "onnx"

    monkeypatch.setattr(kc, "_load_kompress", fake_load)
    return started


def _drain():
    for t in list(kc._download_threads.values()):
        t.join(timeout=10)


def test_repeated_failure_stops_spawning_threads(monkeypatch):
    started = _spawned(monkeypatch, fails=True)
    for _ in range(25):
        kc.ensure_background_download("some/model")
        _drain()
    assert len(started) < 25, f"no backoff: spawned {len(started)} downloads for 25 calls"
    assert len(started) >= 1, "never even tried once"


def test_backoff_window_elapsing_allows_another_attempt(monkeypatch):
    started = _spawned(monkeypatch, fails=True)
    kc.ensure_background_download("some/model")
    _drain()
    assert len(started) == 1

    kc.ensure_background_download("some/model")
    _drain()
    assert len(started) == 1, "retried inside the backoff window"

    # Rewind the clock past the window instead of sleeping through it.
    with kc._download_threads_lock:
        failures, _ = kc._download_failures["some/model"]
        kc._download_failures["some/model"] = (failures, 0.0)
    kc.ensure_background_download("some/model")
    _drain()
    assert len(started) == 2, "backoff never expires"


def test_success_clears_the_backoff(monkeypatch):
    _spawned(monkeypatch, fails=False)
    kc.ensure_background_download("some/model")
    _drain()
    assert "some/model" not in kc._download_failures


def test_window_grows_with_consecutive_failures():
    kc._download_failures["m"] = (1, 0.0)
    assert kc._DOWNLOAD_RETRY_BASE_SECONDS == 5.0
    # Same last-attempt time, more failures -> still blocked at a later clock.
    import time as _t

    now = _t.monotonic()
    kc._download_failures["m"] = (1, now)
    with kc._download_threads_lock:
        first = kc._download_retry_blocked("m")
    kc._download_failures["m"] = (6, now)
    with kc._download_threads_lock:
        later = kc._download_retry_blocked("m")
    assert first and later


def test_a_live_thread_is_never_duplicated(monkeypatch):
    gate = threading.Event()
    started: list[str] = []

    def slow_load(model_id, device, allow_download=True):
        started.append(model_id)
        gate.wait(timeout=10)
        return object(), object(), "onnx"

    monkeypatch.setattr(kc, "_load_kompress", slow_load)
    for _ in range(10):
        kc.ensure_background_download("some/model")
    try:
        assert len(started) == 1
    finally:
        gate.set()
        _drain()

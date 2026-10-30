"""Kompress loads on a background thread after startup, not inside a request.

Startup cannot build the model (native init before the port binds segfaults on
RHEL/CentOS 7-family hosts, #1908), so the first request that needed it paid the
whole load: a gateway benchmark turn stalled 21 s. The warm-up moves that cost
off the request path, and stays off on the affected glibc family.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from headroom.proxy.server import HeadroomProxy
from headroom.transforms import kompress_compressor as kc

_MODEL_ID = "test/warmup-model"


class _Kompress:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_id=_MODEL_ID)
        self.preloaded = threading.Event()

    def preload(self, *, allow_download: bool = True) -> str:
        self.preloaded.set()
        return "onnx"


def _proxy_with(compressor: object | None) -> SimpleNamespace:
    router = SimpleNamespace(_get_kompress=lambda: compressor)
    pipeline = SimpleNamespace(transforms=[router])
    return SimpleNamespace(anthropic_pipeline=pipeline, openai_pipeline=pipeline)


@pytest.fixture(autouse=True)
def _no_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_KOMPRESS_WARMUP_DELAY_SECONDS", "0")
    monkeypatch.delenv("HEADROOM_KOMPRESS_WARMUP", raising=False)


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    with kc._download_threads_lock:
        thread = kc._download_threads.pop(_MODEL_ID, None)
        kc._download_failures.pop(_MODEL_ID, None)
    if thread is not None:
        thread.join(timeout=5)


def test_warms_on_a_background_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    kompress = _Kompress()
    assert HeadroomProxy._start_kompress_background_warmup(_proxy_with(kompress)) is True
    assert kompress.preloaded.wait(5)


def test_skipped_on_the_glibc_family_that_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.libc_ver", lambda: ("glibc", "2.17"))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    kompress = _Kompress()
    assert HeadroomProxy._start_kompress_background_warmup(_proxy_with(kompress)) is False
    assert not kompress.preloaded.wait(0.2)


def test_unreadable_interpreter_does_not_fail_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Microsoft Store Python's python.exe can't be opened, so libc_ver() raises."""

    def _raise() -> tuple[str, str]:
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr("platform.libc_ver", _raise)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    kompress = _Kompress()
    assert HeadroomProxy._start_kompress_background_warmup(_proxy_with(kompress)) is True
    assert kompress.preloaded.wait(5)


def test_env_switch_wins_both_ways(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.libc_ver", lambda: ("glibc", "2.17"))
    monkeypatch.setenv("HEADROOM_KOMPRESS_WARMUP", "1")
    forced = _Kompress()
    assert HeadroomProxy._start_kompress_background_warmup(_proxy_with(forced)) is True
    assert forced.preloaded.wait(5)

    monkeypatch.setattr("platform.libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setenv("HEADROOM_KOMPRESS_WARMUP", "0")
    off = _Kompress()
    assert HeadroomProxy._start_kompress_background_warmup(_proxy_with(off)) is False


def test_no_kompress_installed_means_no_thread() -> None:
    assert HeadroomProxy._start_kompress_background_warmup(_proxy_with(None)) is False


def test_test_processes_do_not_warm_unless_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    assert HeadroomProxy._start_kompress_background_warmup(_proxy_with(_Kompress())) is False


def test_request_during_warm_up_does_not_start_a_second_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The warm-up is the model's registered loader until it finishes."""
    monkeypatch.setattr("platform.libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(kc, "_kompress_cache", {})
    loads: list[str] = []
    monkeypatch.setattr(
        kc,
        "_load_kompress",
        lambda model_id, device, allow_download=True: (
            loads.append(model_id) or (None, None, "onnx")
        ),
    )
    release = threading.Event()

    class _SlowKompress(_Kompress):
        def preload(self, *, allow_download: bool = True) -> str:
            self.preloaded.set()
            release.wait(5)
            raise OSError("warm-up failed")

    kompress = _SlowKompress()
    assert HeadroomProxy._start_kompress_background_warmup(_proxy_with(kompress)) is True
    assert kompress.preloaded.wait(5)

    kc.ensure_background_download(_MODEL_ID)
    assert loads == []

    # Once the warm-up is over (here, failed), a request still starts its own load.
    release.set()
    kc._download_threads[_MODEL_ID].join(timeout=5)
    kc.ensure_background_download(_MODEL_ID)
    kc._download_threads[_MODEL_ID].join(timeout=5)
    assert loads == [_MODEL_ID]

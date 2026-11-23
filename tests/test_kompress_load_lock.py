"""The Kompress model load must not block callers for as long as the load takes.

``_kompress_lock`` is held across the whole load: ONNX/PyTorch session build,
``from transformers import AutoTokenizer`` (which pulls torch in through
``dlopen``) and the tokenizer load. On a degraded filesystem that import alone
was measured at 156.9 s wall for 2.8 s CPU (#4967), and every other Kompress
caller queues on the lock for the duration -- on the proxy's request path that
is a frozen request with no CPU burned and no asyncio timeout able to fire.

These tests pin the bound: a caller waits at most
``HEADROOM_KOMPRESS_LOAD_LOCK_TIMEOUT_SECONDS`` (default 10 s), then gets
:class:`KompressModelBusy` -- a :class:`KompressModelNotCached` subclass, so
every existing passthrough/fallback handler already covers it -- and a wait that
*does* succeed returns the loader's cached entry instead.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from headroom.transforms import kompress_compressor as kc

ENV = kc.KOMPRESS_LOAD_LOCK_TIMEOUT_ENV


@pytest.fixture(autouse=True)
def _clean_lock(monkeypatch):
    """Guarantee the module lock is free, so a leaked test cannot poison others."""
    monkeypatch.delenv(ENV, raising=False)
    if kc._kompress_lock.locked():  # pragma: no cover - defensive
        kc._kompress_lock.release()
    yield
    while kc._kompress_lock.locked():  # pragma: no cover - defensive
        kc._kompress_lock.release()


def test_budget_parsing(monkeypatch):
    assert kc._load_lock_budget_seconds() == kc._DEFAULT_LOAD_LOCK_TIMEOUT_SECONDS

    monkeypatch.setenv(ENV, "2.5")
    assert kc._load_lock_budget_seconds() == 2.5

    monkeypatch.setenv(ENV, "0")
    assert kc._load_lock_budget_seconds() == 0.0

    for raw in ("nan", "inf", "-inf", "1e999", "-1", "junk"):
        monkeypatch.setenv(ENV, raw)
        assert kc._load_lock_budget_seconds() == kc._DEFAULT_LOAD_LOCK_TIMEOUT_SECONDS, raw


def test_busy_is_a_not_cached_error():
    """Callers already fall back to the non-ML path on ``KompressModelNotCached``."""
    assert issubclass(kc.KompressModelBusy, kc.KompressModelNotCached)
    assert isinstance(kc.KompressModelBusy("m"), kc.KompressModelNotCached)


def test_bounded_wait_then_busy(monkeypatch, caplog):
    """A held lock must time out, not hang, and must say so in the log."""
    monkeypatch.setenv(ENV, "0.2")
    holder = threading.Event()

    def hold() -> None:
        kc._kompress_lock.acquire()
        holder.set()
        time.sleep(1.0)
        kc._kompress_lock.release()

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert holder.wait(5.0)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=kc.__name__):
        with pytest.raises(kc.KompressModelBusy):
            with kc._kompress_load_lock("kompress-test-model"):
                pytest.fail("acquired a lock another thread holds")
    elapsed = time.monotonic() - started
    thread.join(5.0)

    assert elapsed < 0.9, f"waited {elapsed:.2f}s, budget was 0.2s"
    assert "event=kompress_load_lock_timeout" in caplog.text


def test_zero_budget_fails_immediately():
    monkeypatch_env = pytest.MonkeyPatch()
    monkeypatch_env.setenv(ENV, "0")
    kc._kompress_lock.acquire()
    try:
        started = time.monotonic()
        with pytest.raises(kc.KompressModelBusy):
            with kc._kompress_load_lock("m"):
                pytest.fail("acquired a held lock with no budget")
        assert time.monotonic() - started < 0.5
    finally:
        kc._kompress_lock.release()
        monkeypatch_env.undo()


def test_waiting_caller_picks_up_the_loaded_entry(monkeypatch):
    """A wait that succeeds must return the loader's model, not a passthrough."""
    monkeypatch.setenv(ENV, "5")
    sentinel = (object(), object(), "onnx")
    model_id = "kompress-wait-wins"

    def slow_loader() -> None:
        kc._kompress_lock.acquire()
        try:
            time.sleep(0.2)
            kc._kompress_cache[model_id] = sentinel
        finally:
            kc._kompress_lock.release()

    thread = threading.Thread(target=slow_loader, daemon=True)
    thread.start()
    try:
        with kc._kompress_load_lock(model_id):
            # The loader finished during our bounded wait, so its entry is here.
            assert kc._kompress_cache[model_id] is sentinel
    finally:
        thread.join(5.0)
        kc._kompress_cache.pop(model_id, None)


def test_load_returns_cached_model_without_waiting(monkeypatch):
    """The cache fast path must not touch the lock budget at all."""
    monkeypatch.setenv(ENV, "0")
    model_id = "kompress-cached-fast-path"
    sentinel = (object(), object(), "onnx")
    kc._kompress_cache[model_id] = sentinel
    try:
        assert kc._load_kompress(model_id) is sentinel
    finally:
        kc._kompress_cache.pop(model_id, None)

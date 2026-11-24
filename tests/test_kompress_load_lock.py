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

import importlib.machinery
import logging
import sys
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


def _hold_lock_for(seconds: float) -> threading.Thread:
    holder = threading.Event()

    def hold() -> None:
        kc._kompress_lock.acquire()
        holder.set()
        time.sleep(seconds)
        kc._kompress_lock.release()

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert holder.wait(5.0)
    return thread


def test_auto_dispatch_spends_one_budget_not_two(monkeypatch, caplog):
    """``auto`` must not retry the second backend on a busy lock.

    With ONNX and PyTorch both available the dispatcher tries ONNX, and a
    ``KompressModelBusy`` from it used to be caught by the ``NotCached`` handler
    and fall through to PyTorch -- which queues on the *same* lock for a second
    full budget. The caller then reported busy twice as late as it was told.
    """
    monkeypatch.setenv(ENV, "0.25")
    monkeypatch.setattr(kc, "_selected_backend", lambda: "auto")
    monkeypatch.setattr(kc, "_is_onnx_available", lambda: True)
    monkeypatch.setattr(kc, "_is_pytorch_available", lambda: True)
    # The import is not what this test is about, and no backend may do real work.
    monkeypatch.setattr(kc, "_import_auto_tokenizer", lambda model_id, deadline: object())
    monkeypatch.setattr(kc, "_import_ml_stack", lambda model_id, deadline: (object(), object()))
    attempts: list[object] = []

    def pytorch_attempt(*args: object, **kwargs: object) -> None:
        attempts.append(kwargs.get("deadline"))
        raise kc.KompressModelBusy("kompress-auto-one-budget")

    monkeypatch.setattr(kc, "_load_kompress_pytorch", pytorch_attempt)
    thread = _hold_lock_for(1.5)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=kc.__name__):
        with pytest.raises(kc.KompressModelBusy):
            kc._load_kompress("kompress-auto-one-budget")
    elapsed = time.monotonic() - started
    thread.join(5.0)

    assert attempts == [], "the dispatcher retried the second backend after a busy lock"
    waits = [r for r in caplog.records if "event=kompress_load_lock_timeout" in r.getMessage()]
    assert len(waits) == 1, f"waited on the lock {len(waits)} times"
    assert elapsed < 0.4, f"waited {elapsed:.2f}s for a 0.25s budget"


class _BlockingTransformersImport:
    """Hold ``import transformers`` open, the way a cold page-in does."""

    def __init__(self, stall_seconds: float = 3.0) -> None:
        self.stall_seconds = stall_seconds
        self.entered = threading.Event()
        self.release = threading.Event()

    def find_spec(self, fullname, path=None, target=None):  # noqa: D102 - meta_path hook
        if fullname != "transformers":
            return None
        self.entered.set()
        self.release.wait(self.stall_seconds)
        return None

    def __enter__(self) -> _BlockingTransformersImport:
        sys.modules.pop("transformers", None)
        sys.meta_path.insert(0, self)
        self.thread = threading.Thread(target=self._import, daemon=True)
        self.thread.start()
        assert self.entered.wait(5.0), "the blocking finder never ran"
        return self

    def __exit__(self, *exc: object) -> None:
        self.release.set()
        self.thread.join(5.0)
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        sys.modules.pop("transformers", None)

    def _import(self) -> None:
        try:
            import transformers  # noqa: F401 - deliberately blocked
        except Exception:  # noqa: BLE001 - the finder refuses the import
            pass


def test_zero_budget_caller_does_not_wait_for_a_cold_transformers_import(monkeypatch, caplog):
    """A busy caller must reach its timeout *before* the import, not after it.

    importlib serialises a module's first import behind its own per-module lock,
    so a caller that starts by importing ``transformers`` inherits whoever is
    mid-import -- the 156.9 s/2.8 s stall from #4967 -- and only then reaches the
    lock it can time out on. With no budget that wait must collapse to nothing.
    """
    monkeypatch.setenv(ENV, "0")
    monkeypatch.setattr(kc, "_selected_backend", lambda: "auto")
    monkeypatch.setattr(kc, "_is_onnx_available", lambda: True)
    monkeypatch.setattr(kc, "_is_pytorch_available", lambda: True)
    monkeypatch.setattr(
        kc, "_load_kompress_pytorch", lambda *a, **k: pytest.fail("reached a second backend")
    )
    thread = _hold_lock_for(2.0)

    with caplog.at_level(logging.WARNING, logger=kc.__name__):
        with _BlockingTransformersImport():
            started = time.monotonic()
            with pytest.raises(kc.KompressModelBusy):
                kc._load_kompress("kompress-cold-import")
            elapsed = time.monotonic() - started
    thread.join(5.0)

    assert elapsed < 1.0, f"waited {elapsed:.2f}s inside a concurrent import"
    assert any("stage=import" in r.getMessage() for r in caplog.records), caplog.text


class _PausingTransformersInit:
    """Publish ``transformers`` and then stall inside ``exec_module``.

    importlib puts the module in ``sys.modules`` *before* running its body, so
    "already imported" and "still initialising" look identical from the outside.
    That window is where a caller that trusts ``sys.modules`` waits for a
    minute-scale ``dlopen`` (#4967).
    """

    def __init__(self, stall_seconds: float = 5.0) -> None:
        self.stall_seconds = stall_seconds
        self.entered = threading.Event()
        self.release = threading.Event()

    def find_spec(self, fullname, path=None, target=None):  # noqa: D102 - meta_path hook
        if fullname != "transformers":
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec):  # noqa: D102 - meta_path hook
        return None

    def exec_module(self, module):  # noqa: D102 - meta_path hook
        self.entered.set()
        self.release.wait(self.stall_seconds)
        module.AutoTokenizer = object

    def __enter__(self) -> _PausingTransformersInit:
        sys.modules.pop("transformers", None)
        sys.meta_path.insert(0, self)
        self.thread = threading.Thread(target=self._import, daemon=True)
        self.thread.start()
        assert self.entered.wait(5.0), "transformers never started initialising"
        return self

    def __exit__(self, *exc: object) -> None:
        self.release.set()
        self.thread.join(5.0)
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        sys.modules.pop("transformers", None)

    def _import(self) -> None:
        try:
            import transformers  # noqa: F401 - stalled inside exec_module
        except Exception:  # noqa: BLE001 - the finder may refuse the import
            pass


@pytest.mark.parametrize("backend", ["onnx", "pytorch"])
def test_deadline_holds_while_transformers_is_still_initialising(monkeypatch, caplog, backend):
    """Busy must be reported *during* the import, for both backends.

    Holding the model lock is what makes the two backends observable as one
    question: if the import is skipped or trusted via ``sys.modules``, the call
    reaches the lock and still returns busy -- after waiting for the paused
    initialisation to finish. The stage in the log is what tells them apart.
    """
    monkeypatch.setenv(ENV, "0")
    monkeypatch.setenv("HEADROOM_KOMPRESS_BACKEND", backend)
    thread = _hold_lock_for(3.0)

    with caplog.at_level(logging.WARNING, logger=kc.__name__):
        with _PausingTransformersInit():
            started = time.monotonic()
            with pytest.raises(kc.KompressModelBusy):
                kc._load_kompress(f"kompress-initialising-{backend}")
            elapsed = time.monotonic() - started
    thread.join(5.0)

    assert elapsed < 1.0, f"{backend} waited {elapsed:.2f}s for an initialising import"
    assert any("stage=import" in r.getMessage() for r in caplog.records), caplog.text

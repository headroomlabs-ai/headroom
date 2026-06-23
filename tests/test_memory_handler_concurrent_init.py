"""Concurrency + timeout tests for MemoryHandler._ensure_initialized (Unit 1).

Covers:
- 10 concurrent first-callers trigger exactly one backend init
  (singleflight via asyncio.Lock + double-check).
- Initialization timeout is surfaced via log + leaves _initialized=False so
  a subsequent call can retry (fail-open contract).
- End-to-end real-backend sanity test (no monkeypatching of internals)
  that exercises LocalBackend so the system-wide check is satisfied.
"""

from __future__ import annotations

import asyncio
import functools
import os
from typing import Any
from unittest.mock import patch

import pytest

from headroom.proxy.memory_handler import (
    STARTUP_INIT_TIMEOUT_SECONDS,
    MemoryConfig,
    MemoryHandler,
)
from tests._skip_helpers import external_model_skip_reason


def skip_offline_model_failures(func):
    """Skip real-backend smoke tests when the local embedder cannot start offline."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            reason = external_model_skip_reason(exc)
            if reason is not None:
                pytest.skip(reason)
            raise

    return wrapper


# -------------------------------------------------------------------
# Singleflight under concurrent callers
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_ensure_initialized_runs_init_once(tmp_path, monkeypatch):
    hits = {"n": 0}
    start_event = asyncio.Event()
    release_event = asyncio.Event()

    class FakeLocalBackend:
        def __init__(self, config):  # noqa: D401
            self.config = config

        async def _ensure_initialized(self) -> None:
            hits["n"] += 1
            start_event.set()
            # Simulate a slow cold-start so concurrent callers pile up on
            # the lock. Without singleflight, hits would exceed 1.
            await release_event.wait()

        async def close(self) -> None:
            pass

    import headroom.memory.backends.local as local_mod

    monkeypatch.setattr(local_mod, "LocalBackend", FakeLocalBackend)

    handler = MemoryHandler(
        MemoryConfig(enabled=True, backend="local", db_path=str(tmp_path / "mem.db"))
    )

    async def caller() -> None:
        await handler._ensure_initialized()

    # Fire 10 concurrent first-callers.
    tasks = [asyncio.create_task(caller()) for _ in range(10)]
    # Let one task enter the critical section, then release it.
    await start_event.wait()
    release_event.set()
    await asyncio.gather(*tasks)

    assert hits["n"] == 1, f"expected 1 backend init under concurrency, got {hits['n']}"
    assert handler._initialized is True


@pytest.mark.asyncio
async def test_ensure_initialized_noop_when_disabled(tmp_path):
    handler = MemoryHandler(
        MemoryConfig(enabled=False, backend="local", db_path=str(tmp_path / "mem.db"))
    )
    await handler._ensure_initialized()
    assert handler._initialized is False
    assert handler._backend is None


# -------------------------------------------------------------------
# Timeout fail-open
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_initialized_timeout_leaves_handler_unready(tmp_path, monkeypatch):
    class HangingBackend:
        def __init__(self, config):
            self.config = config

        async def _ensure_initialized(self) -> None:
            # Hang forever — timeout must cancel this.
            await asyncio.Event().wait()

        async def close(self) -> None:
            pass

    import headroom.memory.backends.local as local_mod

    monkeypatch.setattr(local_mod, "LocalBackend", HangingBackend)

    handler = MemoryHandler(
        MemoryConfig(enabled=True, backend="local", db_path=str(tmp_path / "mem.db"))
    )

    # Attach a handler directly to the module logger — caplog has trouble
    # when third-party conftest monkeys with propagation settings.
    import logging as _logging

    mem_logger = _logging.getLogger("headroom.proxy.memory_handler")
    captured: list[_logging.LogRecord] = []

    class _ListHandler(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            captured.append(record)

    handler_log = _ListHandler(level=_logging.ERROR)
    mem_logger.addHandler(handler_log)
    prev_level = mem_logger.level
    mem_logger.setLevel(_logging.DEBUG)
    try:
        # Shrink the module-level timeout to keep the test fast.
        with patch("headroom.proxy.memory_handler.STARTUP_INIT_TIMEOUT_SECONDS", 0.1):
            await handler._ensure_initialized()
    finally:
        mem_logger.removeHandler(handler_log)
        mem_logger.setLevel(prev_level)

    # Fail-open: no exception, _initialized=False, error logged.
    assert handler._initialized is False
    found_timeout_log = any("timed out" in rec.getMessage().lower() for rec in captured)
    assert found_timeout_log, (
        f"expected 'timed out' log record; got: {[(r.levelname, r.getMessage()) for r in captured]}"
    )

    # Confirm the default constant is unchanged (sanity).
    assert STARTUP_INIT_TIMEOUT_SECONDS == 30.0


@pytest.mark.asyncio
async def test_ensure_initialized_timeout_nulls_partially_initialized_backend(
    tmp_path, monkeypatch
):
    """If wait_for fires while _init_backend_locked has already set
    ``self._backend`` but before ``self._initialized = True``, the timeout
    handler must null ``_backend``. Otherwise callers doing
    ``if self.memory_handler._backend:`` see a truthy-but-broken backend.
    """

    close_hits = {"n": 0}

    class SlowBackend:
        def __init__(self, config):
            self.config = config

        async def _ensure_initialized(self) -> None:
            # Hang long enough to blow the 0.01s timeout below.
            await asyncio.sleep(5.0)

        async def close(self) -> None:
            close_hits["n"] += 1

    import headroom.memory.backends.local as local_mod

    monkeypatch.setattr(local_mod, "LocalBackend", SlowBackend)

    handler = MemoryHandler(
        MemoryConfig(enabled=True, backend="local", db_path=str(tmp_path / "mem.db"))
    )

    with patch("headroom.proxy.memory_handler.STARTUP_INIT_TIMEOUT_SECONDS", 0.01):
        await handler._ensure_initialized()

    # Both must be consistent after timeout.
    assert handler._initialized is False
    assert handler._backend is None
    assert close_hits["n"] == 1


@pytest.mark.asyncio
async def test_ensure_initialized_cancellation_propagates_and_resets_state(tmp_path, monkeypatch):
    """External cancellation of an in-flight ``_ensure_initialized`` must
    propagate (CancelledError is BaseException — not a swallowable error)
    and leave the handler in a clean state."""

    close_hits = {"n": 0}

    class HangingBackend:
        def __init__(self, config):
            self.config = config

        async def _ensure_initialized(self) -> None:
            await asyncio.Event().wait()

        async def close(self) -> None:
            close_hits["n"] += 1

    import headroom.memory.backends.local as local_mod

    monkeypatch.setattr(local_mod, "LocalBackend", HangingBackend)

    handler = MemoryHandler(
        MemoryConfig(enabled=True, backend="local", db_path=str(tmp_path / "mem.db"))
    )

    task = asyncio.create_task(handler._ensure_initialized())
    # Give the task a tick to enter _init_backend_locked and assign _backend.
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert handler._initialized is False
    assert handler._backend is None
    assert close_hits["n"] == 1


# -------------------------------------------------------------------
# Real backend init (no monkeypatching) — integration smoke test
# -------------------------------------------------------------------


@pytest.mark.asyncio
@skip_offline_model_failures
async def test_real_localbackend_initializes_via_public_entrypoint(tmp_path):
    """End-to-end sanity check: the public ``ensure_initialized`` path works
    against a real LocalBackend. This catches regressions where the new
    asyncio.Lock layering (Unit 1) breaks the actual init chain."""
    pytest.importorskip("sqlite3")  # bundled with Python but be explicit
    # Skip on environments without an embedder backend available.
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        try:
            import sentence_transformers  # noqa: F401
        except Exception:  # pragma: no cover - env-dependent
            pytest.skip("No embedder backend available in this environment")

    handler = MemoryHandler(
        MemoryConfig(
            enabled=True,
            backend="local",
            db_path=str(tmp_path / "mem.db"),
        )
    )

    # Concurrent callers should still produce one initialized backend.
    await asyncio.gather(*[handler.ensure_initialized() for _ in range(5)])
    assert handler._initialized is True
    assert handler._backend is not None

    # warmup_embedder is best-effort; on a real backend it should succeed.
    warmed = await handler.warmup_embedder()
    if not warmed and os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        pytest.skip("Skipped because required Hugging Face model files are unavailable offline")
    assert warmed is True
    await handler.close()


@pytest.mark.asyncio
async def test_warmup_embedder_returns_false_without_backend():
    handler = MemoryHandler(MemoryConfig(enabled=True, backend="local"))
    # _initialized=False, _backend=None → no-op, no crash.
    result = await handler.warmup_embedder()
    assert result is False


@pytest.mark.asyncio
async def test_warmup_embedder_swallows_exceptions():
    handler = MemoryHandler(MemoryConfig(enabled=True, backend="local"))
    handler._initialized = True

    class HM:
        class _E:
            async def embed(self, _text: Any) -> Any:
                raise RuntimeError("synthetic embedder failure")

        _embedder = _E()

    class Backend:
        _hierarchical_memory = HM()

    handler._backend = Backend()
    # Must not raise; returns False.
    result = await handler.warmup_embedder()
    assert result is False

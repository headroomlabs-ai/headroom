"""macOS libmalloc tuning: pre-main re-exec gating + periodic allocator trim (#2820)."""

from __future__ import annotations

import asyncio

import pytest

import headroom.cli.proxy as proxy_cli
from headroom.proxy import malloc_trim


class _ExecCalled(Exception):
    """Sentinel so a fake execv can stop execution the way real execv would."""


def _fake_execv(recorder: dict):
    def _execv(path, argv):  # noqa: ANN001
        recorder["path"] = path
        recorder["argv"] = list(argv)
        raise _ExecCalled

    return _execv


@pytest.fixture(autouse=True)
def _clean_malloc_env(monkeypatch):
    for var in (
        "HEADROOM_MALLOC_TUNING",
        "_HEADROOM_MALLOC_TUNED",
        "MallocAggressiveMadvise",
        "MallocLargeCache",
    ):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# _reexec_with_malloc_tuning
# --------------------------------------------------------------------------- #
def test_reexec_noop_off_darwin(monkeypatch):
    monkeypatch.setattr(proxy_cli.sys, "platform", "linux")
    rec: dict = {}
    monkeypatch.setattr(proxy_cli.os, "execv", _fake_execv(rec))
    proxy_cli._reexec_with_malloc_tuning()  # must not raise / exec
    assert rec == {}


def test_reexec_respects_opt_out(monkeypatch):
    monkeypatch.setattr(proxy_cli.sys, "platform", "darwin")
    monkeypatch.setenv("HEADROOM_MALLOC_TUNING", "0")
    rec: dict = {}
    monkeypatch.setattr(proxy_cli.os, "execv", _fake_execv(rec))
    proxy_cli._reexec_with_malloc_tuning()
    assert rec == {}


def test_reexec_guard_prevents_loop(monkeypatch):
    monkeypatch.setattr(proxy_cli.sys, "platform", "darwin")
    monkeypatch.setenv("_HEADROOM_MALLOC_TUNED", "1")
    rec: dict = {}
    monkeypatch.setattr(proxy_cli.os, "execv", _fake_execv(rec))
    proxy_cli._reexec_with_malloc_tuning()
    assert rec == {}


def test_reexec_skips_when_operator_already_set_vars(monkeypatch):
    monkeypatch.setattr(proxy_cli.sys, "platform", "darwin")
    # A real CLI launch, like the sibling exec test below: the tuning path is
    # only reachable when this process is the Headroom CLI entrypoint, and
    # under pytest argv[0] is pytest's own.
    monkeypatch.setattr(proxy_cli.sys, "argv", ["headroom", "proxy"])
    monkeypatch.setenv("MallocAggressiveMadvise", "1")
    monkeypatch.setenv("MallocLargeCache", "0")
    rec: dict = {}
    monkeypatch.setattr(proxy_cli.os, "execv", _fake_execv(rec))
    proxy_cli._reexec_with_malloc_tuning()
    # No re-exec (vars present), but the guard is still stamped.
    assert rec == {}
    assert proxy_cli.os.environ.get("_HEADROOM_MALLOC_TUNED") == "1"


def test_reexec_sets_vars_and_execs_once(monkeypatch):
    monkeypatch.setattr(proxy_cli.sys, "platform", "darwin")
    monkeypatch.setattr(proxy_cli.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(proxy_cli.sys, "argv", ["headroom", "proxy", "--port", "8787"])
    rec: dict = {}
    monkeypatch.setattr(proxy_cli.os, "execv", _fake_execv(rec))

    with pytest.raises(_ExecCalled):
        proxy_cli._reexec_with_malloc_tuning()

    # The tuning knobs and the loop guard are exported to the replacement process.
    assert proxy_cli.os.environ["MallocAggressiveMadvise"] == "1"
    assert proxy_cli.os.environ["MallocLargeCache"] == "0"
    assert proxy_cli.os.environ["_HEADROOM_MALLOC_TUNED"] == "1"
    # Re-exec normalizes to `python -m headroom.cli <args>`, preserving the PID.
    assert rec["path"] == "/usr/bin/python3"
    assert rec["argv"] == ["/usr/bin/python3", "-m", "headroom.cli", "proxy", "--port", "8787"]


# --------------------------------------------------------------------------- #
# malloc_trim.trim / trim_periodically
# --------------------------------------------------------------------------- #
def test_trim_calls_platform_fn(monkeypatch):
    def fake_fn(ptr, size):  # noqa: ANN001  (mac signature)
        return 4096

    monkeypatch.setattr(malloc_trim, "_resolve", lambda: ("darwin", fake_fn))
    assert malloc_trim.trim() == 4096


def test_trim_never_runs_python_gc(monkeypatch):
    # The periodic trim must NOT trigger a full cyclic collection: gc.collect()
    # holds the GIL for a whole-heap traversal, which would stall the event loop
    # even though the C purge itself is dispatched off-thread. Only the
    # GIL-releasing allocator C call may run.
    import gc

    ran: list[str] = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: ran.append("gc") or 0)
    monkeypatch.setattr(malloc_trim, "_resolve", lambda: ("glibc", lambda _size: 0))
    malloc_trim.trim()
    assert ran == []


def test_trim_is_noop_on_unsupported_platform(monkeypatch):
    monkeypatch.setattr(malloc_trim, "_resolve", lambda: ("unsupported", None))
    assert malloc_trim.trim() == 0


def test_trim_periodically_trims_each_interval(monkeypatch):
    monkeypatch.setattr(malloc_trim, "_resolve", lambda: ("glibc", object()))
    trims: list[int] = []
    monkeypatch.setattr(malloc_trim, "trim", lambda: trims.append(1) or 0)

    async def fake_sleep(_seconds):
        if len(trims) >= 2:  # let two ticks run, then break the loop
            raise asyncio.CancelledError

    monkeypatch.setattr(malloc_trim.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(malloc_trim.trim_periodically(interval_seconds=1))
    assert len(trims) == 2


def test_trim_periodically_is_disabled_on_unsupported_platform(monkeypatch):
    # No supported trim call: the task must return at once, never scheduling a
    # wakeup (so it is a true no-op on Windows/musl, not a 60s spinner).
    monkeypatch.setattr(malloc_trim, "_resolve", lambda: ("unsupported", None))
    trims: list[int] = []
    monkeypatch.setattr(malloc_trim, "trim", lambda: trims.append(1) or 0)

    async def _no_sleep(_seconds):
        raise AssertionError("unsupported platform must not schedule a trim wakeup")

    monkeypatch.setattr(malloc_trim.asyncio, "sleep", _no_sleep)
    asyncio.run(malloc_trim.trim_periodically(interval_seconds=60))  # returns, no raise
    assert trims == []


@pytest.mark.parametrize("bad_interval", [0, -5])
def test_trim_periodically_rejects_non_positive_interval(monkeypatch, bad_interval):
    # A non-positive interval would make asyncio.sleep return immediately and
    # spin a continuous collect/trim loop; it must fall back to the default.
    monkeypatch.setattr(malloc_trim, "_resolve", lambda: ("glibc", object()))
    monkeypatch.setattr(malloc_trim, "trim", lambda: 0)
    slept: list[float] = []

    async def capture_sleep(seconds):
        slept.append(seconds)
        raise asyncio.CancelledError  # stop after the first sleep

    monkeypatch.setattr(malloc_trim.asyncio, "sleep", capture_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(malloc_trim.trim_periodically(interval_seconds=bad_interval))
    assert slept == [malloc_trim._DEFAULT_TRIM_INTERVAL_SECONDS]


def test_trim_runs_off_the_event_loop_thread(monkeypatch):
    # The blocking trim must run in a worker thread (via asyncio.to_thread), not
    # on the event loop, so a slow trim cannot stall other async work.
    import threading

    monkeypatch.setattr(malloc_trim, "_resolve", lambda: ("glibc", object()))
    seen: dict[str, int] = {}

    def record():
        seen["thread"] = threading.get_ident()
        return 0

    monkeypatch.setattr(malloc_trim, "trim", record)
    calls = {"n": 0}

    async def sleeper(_seconds):
        calls["n"] += 1
        if calls["n"] >= 2:  # first sleep returns; after the trim, stop
            raise asyncio.CancelledError

    monkeypatch.setattr(malloc_trim.asyncio, "sleep", sleeper)

    async def _run() -> int:
        loop_thread = threading.get_ident()
        with pytest.raises(asyncio.CancelledError):
            await malloc_trim.trim_periodically(interval_seconds=60)
        return loop_thread

    loop_thread = asyncio.run(_run())
    assert "thread" in seen  # trim actually ran
    assert seen["thread"] != loop_thread  # ran off the event-loop thread


@pytest.mark.asyncio
async def test_slow_trim_does_not_stop_unrelated_async_work(monkeypatch):
    # The periodic trim is dispatched off the event-loop thread via
    # asyncio.to_thread and runs no Python gc.collect(), so even a slow purge
    # must not freeze the loop. It is modeled here with a worker-thread park
    # which, like the real GIL-releasing allocator C call, does not hold the
    # GIL while it waits: unrelated coroutines keep making progress meanwhile.
    import threading

    monkeypatch.setattr(malloc_trim, "_resolve", lambda: ("glibc", object()))
    started = threading.Event()
    release = threading.Event()

    def slow_trim() -> int:
        started.set()
        release.wait(5.0)  # hold the worker thread until the test lets go
        return 0

    monkeypatch.setattr(malloc_trim, "trim", slow_trim)

    # Fire the trim's interval immediately (the interval is >= 1s) while leaving
    # the counter's sub-second sleeps to behave normally.
    real_sleep = asyncio.sleep

    async def smart_sleep(seconds):
        if seconds >= 1:
            return
        await real_sleep(seconds)

    monkeypatch.setattr(malloc_trim.asyncio, "sleep", smart_sleep)

    ticks = 0

    async def counter() -> None:
        nonlocal ticks
        while True:
            await real_sleep(0.005)
            ticks += 1

    counter_task = asyncio.create_task(counter())
    trim_task = asyncio.create_task(malloc_trim.trim_periodically(interval_seconds=60))
    try:
        # Wait for the trim to actually start blocking a worker thread.
        for _ in range(400):
            if started.is_set():
                break
            await real_sleep(0.005)
        assert started.is_set(), "trim never started"

        # The trim is now parked off-loop. The event loop must keep ticking.
        ticks_before = ticks
        await real_sleep(0.2)
        ticks_during_trim = ticks - ticks_before
    finally:
        release.set()
        counter_task.cancel()
        trim_task.cancel()

    # On-loop blocking would freeze the counter (~0 ticks); off-thread it keeps
    # ticking (~40 in 0.2s). Generous floor for scheduler jitter.
    assert ticks_during_trim >= 10


# --------------------------------------------------------------------------- #
# ProxyConfig wiring
# --------------------------------------------------------------------------- #
def test_proxy_config_malloc_trim_default_is_darwin_scoped(monkeypatch):
    # Default-on only on macOS (the platform with the documented RSS ratchet);
    # elsewhere it is opt-in, so glibc deployments do not silently take on a
    # once-a-minute allocator purge.
    from headroom.proxy import models

    monkeypatch.setattr(models.sys, "platform", "darwin")
    assert models.ProxyConfig().periodic_malloc_trim_enabled is True

    monkeypatch.setattr(models.sys, "platform", "linux")
    assert models.ProxyConfig().periodic_malloc_trim_enabled is False

    # The interval knob is platform-independent.
    assert models.ProxyConfig().malloc_trim_interval_seconds == 60

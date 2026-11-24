"""Isolated tests for the agy cross-process savings inbox.

These exercise :mod:`headroom.proxy.agy_savings_inbox` in a temp HOME so no
shared state is touched. The proxy funnel is replaced by a fake object whose
async ``record_request`` just records the kwargs it was called with.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.proxy import agy_savings_inbox


class FakeMetrics:
    """Stand-in for PrometheusMetrics: async record_request captures kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def record_request(self, **kwargs) -> None:
        self.calls.append(kwargs)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point the workspace (and thus the inbox) at a throwaway dir."""

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(home / ".headroom"))
    return home


_FUNNEL_KWARGS = {
    "provider": "anthropic",
    "model": "claude-sonnet",
    "input_tokens": 1200,
    "output_tokens": 340,
    "tokens_saved": 800,
    "latency_ms": 42.5,
    "cached": False,
    "overhead_ms": 3.0,
    "ttfb_ms": 10.0,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "cache_write_5m_tokens": 0,
    "cache_write_1h_tokens": 0,
    "uncached_input_tokens": 1200,
    "attempted_input_tokens": 2000,
    "project": "myproj",
    "client": "agy",
}


def _evt_files() -> list[Path]:
    return sorted(agy_savings_inbox.inbox_dir().glob("evt-*.json"))


def test_emit_event_writes_one_roundtrippable_file(isolated_home):
    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)

    files = _evt_files()
    assert len(files) == 1

    envelope = json.loads(files[0].read_text())
    assert envelope["v"] == agy_savings_inbox.SCHEMA_VERSION
    assert isinstance(envelope["event_id"], str) and envelope["event_id"]

    kwargs = envelope["kwargs"]
    assert kwargs == _FUNNEL_KWARGS
    # Sanity: the funnel-only fields are present...
    assert kwargs["output_tokens"] == 340
    assert kwargs["latency_ms"] == 42.5
    # ...and SavingsTracker-only fields never leak in.
    assert "total_input_tokens" not in kwargs
    assert "total_input_cost_usd" not in kwargs
    assert "timestamp" not in kwargs


@pytest.mark.asyncio
async def test_drain_replays_each_event_once_then_deletes(isolated_home):
    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    fake = FakeMetrics()

    recorded = await agy_savings_inbox.drain_inbox(fake)

    assert recorded == 1
    assert fake.calls == [_FUNNEL_KWARGS]
    assert _evt_files() == []


@pytest.mark.asyncio
async def test_redrain_dedups_and_survives_crash_window(isolated_home):
    # Normal completed drain: event recorded, file gone, id in .processed.
    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    fake = FakeMetrics()
    assert await agy_savings_inbox.drain_inbox(fake) == 1
    assert len(fake.calls) == 1

    # Re-draining does NOT re-record the same event id.
    assert await agy_savings_inbox.drain_inbox(fake) == 0
    assert len(fake.calls) == 1

    # Crash window: an event whose id is already in .processed but whose evt
    # file still exists (recorded, crashed before unlink) must be unlinked
    # WITHOUT being recorded again.
    inbox = agy_savings_inbox.inbox_dir()
    seen_id = "crash-1"
    (inbox / ".processed").write_text(seen_id + "\n")
    (inbox / f"evt-{seen_id}.json").write_text(
        json.dumps({"v": 1, "event_id": seen_id, "kwargs": _FUNNEL_KWARGS})
    )

    assert await agy_savings_inbox.drain_inbox(fake) == 0
    assert len(fake.calls) == 1
    assert not (inbox / f"evt-{seen_id}.json").exists()


@pytest.mark.asyncio
async def test_two_events_from_two_pids_recorded_once_each(isolated_home):
    inbox = agy_savings_inbox.inbox_dir()
    for pid in (111, 222):
        eid = f"{pid}-0-abc"
        (inbox / f"evt-{eid}.json").write_text(
            json.dumps({"v": 1, "event_id": eid, "kwargs": _FUNNEL_KWARGS})
        )

    fake = FakeMetrics()
    recorded = await agy_savings_inbox.drain_inbox(fake)

    assert recorded == 2
    assert len(fake.calls) == 2
    assert _evt_files() == []


@pytest.mark.asyncio
async def test_malformed_event_skipped_not_fatal(isolated_home):
    inbox = agy_savings_inbox.inbox_dir()
    # Bad JSON file.
    bad = inbox / "evt-bad.json"
    bad.write_text("{ this is not json")
    # A good event alongside it.
    good_id = "999-0-def"
    (inbox / f"evt-{good_id}.json").write_text(
        json.dumps({"v": 1, "event_id": good_id, "kwargs": _FUNNEL_KWARGS})
    )

    fake = FakeMetrics()
    recorded = await agy_savings_inbox.drain_inbox(fake)

    # The malformed file is dropped; the good one still recorded.
    assert recorded == 1
    assert len(fake.calls) == 1
    assert not bad.exists()
    assert _evt_files() == []


@pytest.mark.asyncio
async def test_empty_inbox_returns_zero(isolated_home):
    fake = FakeMetrics()
    assert await agy_savings_inbox.drain_inbox(fake) == 0
    assert fake.calls == []
    # Non-AGY proxy must not create savings.d on disk when unused.
    assert not (isolated_home / ".headroom" / "savings.d").exists()


@pytest.mark.asyncio
async def test_stateless_mode_skips_emit_and_drain(isolated_home, monkeypatch):
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    inbox_path = isolated_home / ".headroom" / "savings.d"
    assert not inbox_path.exists()

    # Pre-existing event file must not be unlinked or written to .processed in stateless mode.
    inbox_path.mkdir(parents=True)
    evt_file = inbox_path / "evt-stateless-1.json"
    evt_file.write_text(
        json.dumps({"v": 1, "event_id": "stateless-1", "kwargs": _FUNNEL_KWARGS}),
        encoding="utf-8",
        newline="\n",
    )
    fake = FakeMetrics()
    assert await agy_savings_inbox.drain_inbox(fake) == 0
    assert fake.calls == []
    assert evt_file.exists()
    assert not (inbox_path / ".processed").exists()


@pytest.mark.windows_newline
@pytest.mark.asyncio
async def test_processed_file_uses_lf_newlines(isolated_home):
    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    fake = FakeMetrics()
    assert await agy_savings_inbox.drain_inbox(fake) == 1
    raw = (isolated_home / ".headroom" / "savings.d" / ".processed").read_bytes()
    assert b"\n" in raw
    assert b"\r\n" not in raw


def test_agy_emit_enabled_reflects_env(monkeypatch):
    monkeypatch.delenv(agy_savings_inbox.AGY_INBOX_EMIT_ENV, raising=False)
    assert agy_savings_inbox.agy_emit_enabled() is False

    monkeypatch.setenv(agy_savings_inbox.AGY_INBOX_EMIT_ENV, "1")
    assert agy_savings_inbox.agy_emit_enabled() is True

    monkeypatch.setenv(agy_savings_inbox.AGY_INBOX_EMIT_ENV, "0")
    assert agy_savings_inbox.agy_emit_enabled() is False


def test_two_processes_draining_same_inbox_record_event_once(isolated_home, tmp_path):
    """Two worker processes draining the same inbox must replay one event only once."""
    import os
    import subprocess
    import sys

    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    assert len(_evt_files()) == 1

    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()

    worker_script = """
import asyncio
import sys
import time
from pathlib import Path
from headroom.proxy import agy_savings_inbox

worker_id = sys.argv[1]
other_id = sys.argv[2]
sync_dir = Path(sys.argv[3])

class SyncMetrics:
    async def record_request(self, **kwargs):
        (sync_dir / f"{worker_id}.entered").write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if (sync_dir / f"{other_id}.entered").exists():
                break
            await asyncio.sleep(0.01)

count = asyncio.run(agy_savings_inbox.drain_inbox(SyncMetrics()))
print(count)
"""
    env = {
        **os.environ,
        "HOME": str(isolated_home),
        "HEADROOM_WORKSPACE_DIR": str(isolated_home / ".headroom"),
        "HEADROOM_OFFLINE": "1",
    }

    p1 = subprocess.Popen(
        [sys.executable, "-c", worker_script, "w1", "w2", str(sync_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    p2 = subprocess.Popen(
        [sys.executable, "-c", worker_script, "w2", "w1", str(sync_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    out1, err1 = p1.communicate(timeout=15)
    out2, err2 = p2.communicate(timeout=15)

    assert p1.returncode == 0, err1
    assert p2.returncode == 0, err2
    c1 = int(out1.strip())
    c2 = int(out2.strip())
    assert c1 + c2 == 1, f"expected 1 combined record across two workers, got {c1} + {c2}"
    assert _evt_files() == []


@pytest.mark.asyncio
async def test_cancelled_contended_drain_leaves_no_pending_lock_and_allows_other_worker(
    isolated_home, monkeypatch
):
    """Cancelling a drain waiting on a contended lock must leave no in-flight lock op."""
    import asyncio
    import os
    import sys

    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    inbox = agy_savings_inbox.inbox_dir()
    lock_path = inbox / ".drain.lock"

    holder_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    in_flight = 0
    errors_after_cancel: list[BaseException] = []
    cancelled = False

    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(holder_fd, 0, os.SEEK_SET)
            msvcrt.locking(holder_fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            real_flock = fcntl.flock

            def _spy_flock(fd: int, op: int) -> None:
                nonlocal in_flight
                in_flight += 1
                try:
                    real_flock(fd, op)
                except OSError as exc:
                    if cancelled:
                        errors_after_cancel.append(exc)
                    raise
                finally:
                    in_flight -= 1

            monkeypatch.setattr(fcntl, "flock", _spy_flock)

        contended_metrics = FakeMetrics()
        drain_task = asyncio.create_task(agy_savings_inbox.drain_inbox(contended_metrics))
        await asyncio.sleep(0.1)
        assert not drain_task.done()

        drain_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain_task
        cancelled = True

        assert in_flight == 0, "blocking flock call is still in flight after cancellation"
        assert contended_metrics.calls == []
    finally:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(holder_fd, 0, os.SEEK_SET)
            msvcrt.locking(holder_fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            # Use real_flock so the spy doesn't count the test harness's own unlock.
            real_flock(holder_fd, fcntl.LOCK_UN)
        os.close(holder_fd)

    assert errors_after_cancel == []

    # Another worker can now acquire the lock and drain the event cleanly.
    other_metrics = FakeMetrics()
    recorded = await agy_savings_inbox.drain_inbox(other_metrics)
    assert recorded == 1
    assert other_metrics.calls == [_FUNNEL_KWARGS]
    assert _evt_files() == []


def test_conftest_preserves_runner_headroom_offline_opt_in():
    """When pytest is launched with HEADROOM_OFFLINE=1, the opt-in reaches the test body."""
    import os
    import subprocess
    import sys

    if os.environ.get("_HEADROOM_OFFLINE_CHILD_PROBE") == "1":
        assert os.environ.get("HEADROOM_OFFLINE") == "1"
        return

    env = {
        **os.environ,
        "_HEADROOM_OFFLINE_CHILD_PROBE": "1",
        "HEADROOM_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{__file__}::test_conftest_preserves_runner_headroom_offline_opt_in",
            "-q",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.asyncio
async def test_abrupt_exit_mid_batch_does_not_replay_completed_event(isolated_home, tmp_path):
    """If a worker dies entering the 2nd event, restart must not replay the 1st event."""
    import os
    import subprocess
    import sys

    inbox = agy_savings_inbox.inbox_dir()
    first_kwargs = {**_FUNNEL_KWARGS, "project": "first"}
    second_kwargs = {**_FUNNEL_KWARGS, "project": "second"}
    (inbox / "evt-001-first.json").write_text(
        json.dumps({"v": 1, "event_id": "evt-first", "kwargs": first_kwargs}),
        encoding="utf-8",
        newline="\n",
    )
    (inbox / "evt-002-second.json").write_text(
        json.dumps({"v": 1, "event_id": "evt-second", "kwargs": second_kwargs}),
        encoding="utf-8",
        newline="\n",
    )

    sink_path = tmp_path / "durable_sink.log"

    crash_worker = """
import asyncio
import os
import sys
from pathlib import Path
from headroom.proxy import agy_savings_inbox

sink = Path(sys.argv[1])

class CrashingDurableMetrics:
    def __init__(self) -> None:
        self._seen = 0

    async def record_request(self, **kwargs) -> None:
        self._seen += 1
        if self._seen == 2:
            os._exit(99)
        with sink.open("a", encoding="utf-8", newline="\\n") as fh:
            fh.write(f"{kwargs['project']}\\n")
            fh.flush()

asyncio.run(agy_savings_inbox.drain_inbox(CrashingDurableMetrics()))
"""
    env = {
        **os.environ,
        "HOME": str(isolated_home),
        "HEADROOM_WORKSPACE_DIR": str(isolated_home / ".headroom"),
        "HEADROOM_OFFLINE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", crash_worker, str(sink_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 99, proc.stdout + proc.stderr
    assert sink_path.read_text(encoding="utf-8").splitlines() == ["first"]

    class DurableMetrics:
        async def record_request(self, **kwargs) -> None:
            with sink_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(f"{kwargs['project']}\n")
                fh.flush()

    recorded = await agy_savings_inbox.drain_inbox(DurableMetrics())
    assert recorded == 1
    assert sink_path.read_text(encoding="utf-8").splitlines() == ["first", "second"]
    assert _evt_files() == []


@pytest.mark.asyncio
async def test_drain_keeps_event_loop_responsive_while_checkpointing_per_event(
    isolated_home, monkeypatch
):
    """Offloaded per-event fsync lets an asyncio heartbeat run while preserving checkpoint order."""
    import asyncio
    import os
    import time

    inbox = agy_savings_inbox.inbox_dir()
    event_count = 100
    for i in range(event_count):
        eid = f"evt-{i:03d}"
        (inbox / f"evt-{i:03d}.json").write_text(
            json.dumps({"v": 1, "event_id": eid, "kwargs": {**_FUNNEL_KWARGS, "project": eid}}),
            encoding="utf-8",
            newline="\n",
        )

    real_fsync = os.fsync

    def _slow_fsync(fd: int) -> None:
        time.sleep(0.005)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _slow_fsync)

    processed_path = inbox / ".processed"
    seen: list[str] = []

    class OrderCheckingMetrics:
        async def record_request(self, **kwargs) -> None:
            if seen:
                prev_id = seen[-1]
                assert processed_path.exists()
                assert prev_id in processed_path.read_text(encoding="utf-8").splitlines()
                assert not (inbox / f"{prev_id}.json").exists()
            seen.append(kwargs["project"])

    heartbeat_ticks = 0
    stop_heartbeat = False

    async def _heartbeat() -> None:
        nonlocal heartbeat_ticks
        while not stop_heartbeat:
            await asyncio.sleep(0.01)
            if not stop_heartbeat:
                heartbeat_ticks += 1

    hb_task = asyncio.create_task(_heartbeat())
    try:
        recorded = await agy_savings_inbox.drain_inbox(OrderCheckingMetrics())
    finally:
        stop_heartbeat = True
        await hb_task

    assert recorded == event_count
    assert len(seen) == event_count
    assert heartbeat_ticks >= 5, (
        f"event loop was blocked during replay: heartbeat only ticked {heartbeat_ticks} time(s)"
    )
    assert _evt_files() == []


@pytest.mark.asyncio
async def test_cancel_during_offloaded_checkpoint_finishes_write_before_unlock(
    isolated_home, monkeypatch
):
    """Cancelling during an offloaded checkpoint finishes that write before releasing the lock."""
    import asyncio
    import os
    import threading
    import time

    inbox = agy_savings_inbox.inbox_dir()
    for name in ("first", "second"):
        (inbox / f"evt-{name}.json").write_text(
            json.dumps(
                {"v": 1, "event_id": f"id-{name}", "kwargs": {**_FUNNEL_KWARGS, "project": name}}
            ),
            encoding="utf-8",
            newline="\n",
        )

    in_fsync = threading.Event()
    fsync_finished = False
    real_fsync = os.fsync

    def _slow_first_fsync(fd: int) -> None:
        nonlocal fsync_finished
        if not in_fsync.is_set():
            in_fsync.set()
            time.sleep(0.1)
            real_fsync(fd)
            fsync_finished = True
            return
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _slow_first_fsync)

    first_metrics = FakeMetrics()
    drain_task = asyncio.create_task(agy_savings_inbox.drain_inbox(first_metrics))

    deadline = time.monotonic() + 5.0
    while not in_fsync.is_set():
        assert time.monotonic() < deadline, "drain did not reach checkpoint fsync"
        await asyncio.sleep(0.005)

    drain_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain_task

    assert fsync_finished is True, "drain returned and unlocked while checkpoint write was running"
    assert [c["project"] for c in first_metrics.calls] == ["first"]
    assert "id-first" in (inbox / ".processed").read_text(encoding="utf-8").splitlines()
    assert not (inbox / "evt-first.json").exists()
    assert (inbox / "evt-second.json").exists()

    second_metrics = FakeMetrics()
    recorded = await agy_savings_inbox.drain_inbox(second_metrics)
    assert recorded == 1
    assert [c["project"] for c in second_metrics.calls] == ["second"]
    assert _evt_files() == []


@pytest.mark.asyncio
async def test_repeated_cancellation_aborts_stalled_checkpoint_wait(isolated_home, monkeypatch):
    """A shutdown timeout exits while keeping .drain.lock held until the checkpoint finishes."""
    import asyncio
    import threading
    import time

    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)

    in_checkpoint = threading.Event()
    release_checkpoint = threading.Event()
    real_append = agy_savings_inbox._append_processed

    def _stalled_append(path: Path, event_id: str) -> None:
        # Block BEFORE the event ID becomes visible in .processed.
        in_checkpoint.set()
        release_checkpoint.wait(timeout=5.0)
        real_append(path, event_id)

    monkeypatch.setattr(agy_savings_inbox, "_append_processed", _stalled_append)

    first_metrics = FakeMetrics()
    drain_task = asyncio.create_task(agy_savings_inbox.drain_inbox(first_metrics))
    peer_task: asyncio.Task[int] | None = None
    try:
        deadline = time.monotonic() + 5.0
        while not in_checkpoint.is_set():
            assert time.monotonic() < deadline, "drain did not reach checkpoint"
            await asyncio.sleep(0.005)

        # First cancel (normal shutdown) then second cancel (shutdown timeout).
        drain_task.cancel()
        await asyncio.sleep(0.02)
        assert not drain_task.done()

        drain_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(drain_task, timeout=0.5)
        assert drain_task.done(), "drain did not exit on repeated cancellation while stalled"
        assert not release_checkpoint.is_set()

        # Peer drain while checkpoint is still blocked must wait on .drain.lock
        # and must not replay the already-recorded event.
        peer_metrics = FakeMetrics()
        peer_task = asyncio.create_task(agy_savings_inbox.drain_inbox(peer_metrics))
        await asyncio.sleep(0.1)
        assert not peer_task.done(), "peer entered replay before stalled checkpoint finished"
        assert peer_metrics.calls == []

        release_checkpoint.set()
        peer_recorded = await asyncio.wait_for(peer_task, timeout=5.0)
        assert peer_recorded == 0
        assert peer_metrics.calls == []
        assert len(first_metrics.calls) == 1
        assert _evt_files() == []
    finally:
        release_checkpoint.set()
        await asyncio.gather(drain_task, return_exceptions=True)
        if peer_task is not None:
            await asyncio.gather(peer_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_queued_helper_before_start_releases_drain_lock(isolated_home, monkeypatch):
    """If a queued disk helper is cancelled before starting, .drain.lock must be released."""
    import asyncio

    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)

    loop = asyncio.get_running_loop()
    real_run_in_executor = loop.run_in_executor
    queued_future: asyncio.Future | None = None
    queued_fn = None
    queued_ready = asyncio.Event()

    def _intercept_run_in_executor(executor, func, *args):
        nonlocal queued_future, queued_fn
        if queued_future is None:
            queued_future = loop.create_future()
            queued_fn = (func, args)
            queued_ready.set()
            return queued_future
        return real_run_in_executor(executor, func, *args)

    monkeypatch.setattr(loop, "run_in_executor", _intercept_run_in_executor)

    first_metrics = FakeMetrics()
    drain_task = asyncio.create_task(agy_savings_inbox.drain_inbox(first_metrics))
    await asyncio.wait_for(queued_ready.wait(), timeout=5.0)
    assert queued_future is not None and queued_fn is not None

    # Simulate shutdown cancelling the drain and event-loop teardown cancelling
    # the queued executor future before any worker thread starts it.
    drain_task.cancel()
    await asyncio.sleep(0)
    queued_future.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(drain_task, timeout=5.0)

    # Restore real executor and confirm a peer immediately acquires .drain.lock.
    monkeypatch.setattr(loop, "run_in_executor", real_run_in_executor)
    peer_metrics = FakeMetrics()
    recorded = await asyncio.wait_for(agy_savings_inbox.drain_inbox(peer_metrics), timeout=5.0)
    assert recorded == 1
    assert len(peer_metrics.calls) == 1
    assert _evt_files() == []

    # Even if the cancelled queued callable were somehow dequeued later, it is a no-op.
    func, args = queued_fn
    assert func(*args) is None

"""Cross-process savings inbox: agy -> shared proxy replay.

``agy`` runs as a **separate process** from the shared Headroom proxy, but its
per-request savings must show up on the shared dashboard, counted once, without
agy ever writing any shared durable state. This module is the bridge.

Mechanism (AT-LEAST-ONCE with best-effort dedup — *not* exactly-once):

* In the agy process, :func:`emit_event` drops one JSON file per request into a
  canonical inbox directory (``workspace_dir()/savings.d``). Each file carries
  the exact keyword arguments that :meth:`PrometheusMetrics.record_request`
  (the single dashboard funnel) expects, plus a unique ``event_id``.
* In the shared proxy, :func:`drain_inbox` reads those files and replays each
  event through its *own* ``record_request`` funnel — the one writer of shared
  durable state (savings ledger, SavingsTracker, OTEL). agy itself redirects all
  three of those to throwaway paths, so the proxy replay is the sole writer and
  savings are counted exactly once on the dashboard.

Everything on the agy side is best-effort: emit never raises into the request
path, and drain never raises out into the proxy's lifespan / stats handler.
"""

from __future__ import annotations

import asyncio
import errno
import itertools
import json
import logging
import os
import random
import sys
import tempfile
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from headroom.paths import persistence_allowed, workspace_dir

logger = logging.getLogger("headroom.proxy")

# Bump when the on-disk envelope shape changes incompatibly.
SCHEMA_VERSION = 1

# Env var (set only in the agy process) that turns on emit at the outcome hook.
AGY_INBOX_EMIT_ENV = "HEADROOM_AGY_INBOX_EMIT"

# Hard cap on pending event files; oldest are dropped (disclosed) past this.
MAX_INBOX = 5000

# Keep at most this many processed ids in the dedup file so it stays bounded.
MAX_PROCESSED_IDS = 20000

_INBOX_SUBDIR = "savings.d"
_PROCESSED_FILE = ".processed"
_DRAIN_LOCK_FILE = ".drain.lock"

# Monotonic per-process sequence so two events from the same pid never collide.
_seq = itertools.count()

# Serialize drains within this process so periodic replay and /stats never race.
_drain_lock = asyncio.Lock()

_WIN_LOCK_RETRY_ERRNOS = frozenset(
    code
    for name in ("EACCES", "EAGAIN", "EDEADLK", "EDEADLOCK")
    if (code := getattr(errno, name, None)) is not None
)
_WIN_LOCK_RETRY_WINERRORS = frozenset({32, 33})  # ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
_POSIX_LOCK_RETRY_ERRNOS = frozenset(
    code
    for name in ("EAGAIN", "EWOULDBLOCK", "EACCES", "EINTR")
    if (code := getattr(errno, name, None)) is not None
)


def _is_windows_lock_contention(exc: OSError) -> bool:
    """True when a Windows msvcrt.locking error indicates temporary lock contention."""
    return (
        exc.errno in _WIN_LOCK_RETRY_ERRNOS
        or getattr(exc, "winerror", None) in _WIN_LOCK_RETRY_WINERRORS
    )


def _unlock_and_close_fd(fd: int) -> None:
    """Release the OS lock held on ``fd`` and close it."""
    try:
        if sys.platform == "win32":
            import msvcrt

            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


class _DrainLockLease:
    """Owns the open OS lock descriptor until both drain and any in-flight worker finish.

    Normally the lock is released as soon as ``_cross_process_drain_lock`` exits.
    If repeated cancellation (such as a shutdown timeout) unwinds the drain task
    while a checkpoint thread is still blocked on disk I/O, ownership transfers to
    that thread so ``.drain.lock`` stays held until the checkpoint finishes and
    no peer worker can replay the same event twice.
    """

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd
        self._lock = threading.Lock()
        self._pending = 0
        self._released_by_owner = False

    def begin_offload(self) -> None:
        with self._lock:
            self._pending += 1

    def finish_offload(self) -> None:
        fd_to_close: int | None = None
        with self._lock:
            self._pending -= 1
            if self._released_by_owner and self._pending == 0 and self._fd is not None:
                fd_to_close = self._fd
                self._fd = None
        if fd_to_close is not None:
            _unlock_and_close_fd(fd_to_close)

    def release(self) -> None:
        with self._lock:
            self._released_by_owner = True
            if self._pending > 0:
                return
            fd = self._fd
            self._fd = None
        if fd is not None:
            _unlock_and_close_fd(fd)


@asynccontextmanager
async def _cross_process_drain_lock(directory: Path) -> AsyncIterator[_DrainLockLease]:
    """Serialize inbox claim and replay across worker processes.

    Acquiring the OS file lock guarantees that two workers (e.g. uvicorn
    ``--workers 2``) never read and replay the same event file concurrently,
    while OS-level lock release on process exit preserves crash recovery.
    """
    lock_path = directory / _DRAIN_LOCK_FILE
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    lease: _DrainLockLease | None = None
    try:
        try:
            os.chmod(lock_path, 0o600)
        except OSError:
            pass
        if sys.platform == "win32":
            import msvcrt

            # Windows byte-range locks can extend beyond EOF; do not write a
            # sentinel byte before acquiring the lock or two workers race.
            os.lseek(fd, 0, os.SEEK_SET)
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError as exc:
                    if not _is_windows_lock_contention(exc):
                        raise
                    await asyncio.sleep(0.05)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in _POSIX_LOCK_RETRY_ERRNOS:
                        raise
                    await asyncio.sleep(0.05)

        lease = _DrainLockLease(fd)
        yield lease
    finally:
        if lease is not None:
            lease.release()
        elif acquired:
            _unlock_and_close_fd(fd)
        else:
            os.close(fd)


def inbox_dir() -> Path:
    """Return the canonical inbox directory, creating it on demand."""

    path = workspace_dir() / _INBOX_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def agy_emit_enabled() -> bool:
    """True when the agy emit marker env var is set to ``"1"``."""

    return os.environ.get(AGY_INBOX_EMIT_ENV, "").strip() == "1"


def _new_event_id() -> str:
    """Return a process-unique, collision-resistant event id."""

    return f"{os.getpid()}-{next(_seq)}-{random.getrandbits(48):012x}"


def _json_safe(value: Any) -> Any:
    """Return ``value`` if it round-trips through JSON, else ``None``.

    Non-scalar funnel args (``pipeline_timing``, ``waste_signals``) are dicts of
    scalars and normally survive; anything that does not is dropped so a single
    weird value can never make the whole envelope unwritable.
    """

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return None


def _enforce_cap() -> None:
    """Drop the oldest event files if the inbox is at/over :data:`MAX_INBOX`."""

    try:
        files = sorted(
            inbox_dir().glob("evt-*.json"),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return
    excess = len(files) - MAX_INBOX
    if excess <= 0:
        return
    for stale in files[: excess + 1]:
        try:
            stale.unlink()
        except OSError:
            continue
    logger.warning(
        "agy savings inbox at cap (%d); dropped %d oldest event(s)",
        MAX_INBOX,
        excess + 1,
    )


def emit_event(**funnel_kwargs: Any) -> None:
    """Atomically write one inbox event carrying ``record_request`` kwargs.

    Best-effort: any failure is swallowed (logged at debug) so emit can never
    break the request that triggered it.
    """

    try:
        if not persistence_allowed("AGY savings inbox"):
            return
        directory = inbox_dir()
        _enforce_cap()

        safe_kwargs = {key: _json_safe(val) for key, val in funnel_kwargs.items()}
        event_id = _new_event_id()
        envelope = {
            "v": SCHEMA_VERSION,
            "event_id": event_id,
            "kwargs": safe_kwargs,
        }

        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".tmp-evt-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(envelope, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, directory / f"evt-{event_id}.json")
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except Exception as exc:  # noqa: BLE001 - best-effort, never raise into caller
        logger.debug("agy savings emit failed: %s", exc)


def _load_processed(path: Path) -> list[str]:
    """Return the processed-id list (order preserved), or ``[]`` if unreadable."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    ids: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            ids.append(line)
    return ids


def _append_processed(path: Path, event_id: str) -> None:
    """Append one completed event id (LF-terminated) and fsync in O(1) time.

    Appending per completed event before starting the next replay prevents an
    abrupt worker exit mid-batch from replaying already-committed events,
    without rewriting up to ``MAX_PROCESSED_IDS`` entries on every event.
    """
    try:
        with path.open("a+b") as fh:
            fh.seek(0, os.SEEK_END)
            needs_leading_lf = False
            if fh.tell() > 0:
                fh.seek(-1, os.SEEK_END)
                needs_leading_lf = fh.read(1) != b"\n"
                fh.seek(0, os.SEEK_END)
            payload = f"\n{event_id}\n" if needs_leading_lf else f"{event_id}\n"
            fh.write(payload.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        logger.debug("agy savings processed-file append failed: %s", exc)


def _checkpoint_event(processed_path: Path, event_id: str, event_file: Path) -> None:
    """Persist ``event_id`` to ``.processed`` and unlink ``event_file`` in order."""
    _append_processed(processed_path, event_id)
    _safe_unlink(event_file)


def _list_event_files(directory: Path) -> list[Path]:
    """Return sorted pending event files in ``directory``."""
    return sorted(directory.glob("evt-*.json"))


def _read_event_envelope(event_file: Path) -> dict[str, Any]:
    """Read and parse one JSON event file."""
    raw = event_file.read_text(encoding="utf-8")
    envelope = json.loads(raw)
    if not isinstance(envelope, dict):
        raise ValueError("event envelope must be a JSON object")
    return envelope


async def _run_in_thread_before_unlock(
    lease: _DrainLockLease,
    fn: Callable[..., Any],
    *args: Any,
    cancellable: bool = True,
) -> Any:
    """Run a bounded disk helper off the event loop under ``lease``.

    Offloading keeps the proxy event loop responsive during backlog replay. On a
    normal cancellation, wait for the single in-flight disk operation to finish
    before returning. On a second cancellation (such as a lifespan shutdown
    timeout), stop waiting immediately:
    - when ``cancellable=True`` (reads / pre-record ops), cancel ``fut`` so a
      still-queued helper releases ``.drain.lock`` immediately via ``_on_done``;
    - when ``cancellable=False`` (post-record checkpoint), leave a queued
      checkpoint scheduled so it still persists ``.processed``, unlinks the
      event, and releases ``.drain.lock`` once a worker runs it.
    """
    state_lock = threading.Lock()
    started = False
    cancelled_before_start = False
    finished = False

    def _finish_once() -> None:
        nonlocal finished
        with state_lock:
            if finished:
                return
            finished = True
        lease.finish_offload()

    def _run() -> Any:
        nonlocal started
        with state_lock:
            if cancelled_before_start:
                return None
            started = True
        try:
            return fn(*args)
        finally:
            _finish_once()

    def _on_done(done_fut: asyncio.Future[Any]) -> None:
        nonlocal cancelled_before_start
        if not done_fut.cancelled():
            _finish_once()
            return
        should_release_now = False
        with state_lock:
            if not started:
                cancelled_before_start = True
                should_release_now = True
        if should_release_now:
            _finish_once()

    lease.begin_offload()
    try:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.run_in_executor(None, _run)
    except BaseException:
        # Submission can raise after the pool already enqueued `_run` (e.g.
        # ThreadPoolExecutor failing to spawn another OS thread). Coordinate
        # with `_run` under state_lock so exactly one path owns the lease.
        claimed = False
        with state_lock:
            if not started:
                cancelled_before_start = True
                claimed = True
        if claimed:
            if not cancellable:
                try:
                    return fn(*args)
                finally:
                    _finish_once()
            _finish_once()
        raise
    fut.add_done_callback(_on_done)
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            # Second cancellation (e.g. shutdown timeout): cancel queued reads
            # immediately, but preserve queued post-record checkpoints so the
            # worker still commits `.processed` before releasing `.drain.lock`.
            if cancellable:
                fut.cancel()
            raise
        except Exception:  # noqa: BLE001
            pass
        raise


def _write_processed(path: Path, ids: list[str]) -> None:
    """Atomically persist the processed-id list, pruned to the newest N."""

    pruned = ids[-MAX_PROCESSED_IDS:]
    try:
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-proc-")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(pruned))
            if pruned:
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except OSError as exc:
        logger.debug("agy savings processed-file write failed: %s", exc)


async def drain_inbox(metrics: Any, *, max_events: int = 1000) -> int:
    """Replay pending inbox events through ``metrics.record_request``.

    Delivery is at-least-once with best-effort dedup: an event is only unlinked
    *after* it has been recorded (or found already-processed), so a crash
    between record and unlink re-delivers it next drain - the ``.processed`` set
    then suppresses the duplicate. Returns the number of events recorded.

    Never raises: the whole body is defended so a drain error can never crash
    the proxy lifespan loop or the stats handler.
    """
    if not persistence_allowed("AGY savings inbox"):
        return 0

    recorded = 0
    async with _drain_lock:
        try:
            directory = workspace_dir() / _INBOX_SUBDIR
            if not directory.is_dir():
                return 0
            try:
                if not any(directory.glob("evt-*.json")):
                    return 0
            except OSError:
                return 0

            async with _cross_process_drain_lock(directory) as lease:
                processed_path = directory / _PROCESSED_FILE
                processed_list = await _run_in_thread_before_unlock(
                    lease, _load_processed, processed_path
                )
                processed_set = set(processed_list)

                try:
                    files = await _run_in_thread_before_unlock(lease, _list_event_files, directory)
                except OSError:
                    return 0

                for event_file in files[:max_events]:
                    try:
                        try:
                            envelope = await _run_in_thread_before_unlock(
                                lease, _read_event_envelope, event_file
                            )
                        except (OSError, ValueError):
                            # Malformed / unreadable: skip and remove, never fatal.
                            logger.debug("agy savings: dropping malformed %s", event_file.name)
                            await _run_in_thread_before_unlock(lease, _safe_unlink, event_file)
                            continue

                        event_id = envelope.get("event_id")
                        if not isinstance(event_id, str) or not event_id:
                            await _run_in_thread_before_unlock(lease, _safe_unlink, event_file)
                            continue

                        if event_id in processed_set:
                            # Crash-window duplicate: already recorded, just remove.
                            await _run_in_thread_before_unlock(lease, _safe_unlink, event_file)
                            continue

                        kwargs = envelope.get("kwargs")
                        if not isinstance(kwargs, dict):
                            await _run_in_thread_before_unlock(lease, _safe_unlink, event_file)
                            continue

                        await metrics.record_request(**kwargs)
                        recorded += 1

                        processed_set.add(event_id)
                        processed_list.append(event_id)
                        await _run_in_thread_before_unlock(
                            lease,
                            _checkpoint_event,
                            processed_path,
                            event_id,
                            event_file,
                            cancellable=False,
                        )
                    except Exception as exc:  # noqa: BLE001 - one bad event never aborts drain
                        logger.debug("agy savings: error replaying %s: %s", event_file.name, exc)
                        continue

                if len(processed_list) > MAX_PROCESSED_IDS:
                    await _run_in_thread_before_unlock(
                        lease, _write_processed, processed_path, list(processed_list)
                    )
        except Exception as exc:  # noqa: BLE001 - drain never raises out
            logger.debug("agy savings drain failed: %s", exc)

    return recorded


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


__all__ = [
    "AGY_INBOX_EMIT_ENV",
    "SCHEMA_VERSION",
    "MAX_INBOX",
    "inbox_dir",
    "agy_emit_enabled",
    "emit_event",
    "drain_inbox",
]

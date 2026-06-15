"""Agy-session compression observability helpers.

Thread-safe, agy-scoped ONLY.  No imports from gemini.py, transport, or
compression_store — those are imported lazily at call time.

Public surface
--------------
FailOpenWarnHandler    logging.Handler that emits a one-time stderr notice on
                       the first Cloud-Code-Assist fail-open log record.
AgySesssionStats       Snapshot + delta + summary formatting; idempotent print.

Ref: headroom-30y.15
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

# The logger that gemini.py actually emits the fail-open warning on.
# CONFIRMED: gemini.py:25 is `logger = logging.getLogger("headroom.proxy")` and
# the fail-open warning at gemini.py:883 uses that logger. Python logging
# propagates child->parent (NOT parent->child), so a handler on the *child*
# "headroom.proxy.handlers.gemini" would NEVER receive these records — we must
# install on the actual emitting logger "headroom.proxy".
_GEMINI_LOGGER = "headroom.proxy"
# Substring that identifies the fail-open warning record (gemini.py:883). Used
# to filter out unrelated "headroom.proxy" warnings.
_FAIL_OPEN_SUBSTR = "Cloud Code Assist optimization failed"


class FailOpenWarnHandler(logging.Handler):
    """One-shot logging.Handler that prints a user-facing stderr notice on the
    FIRST fail-open compression warning emitted by the gemini handler, then
    counts all subsequent occurrences.

    Install on the ``"headroom.proxy"`` logger (the logger gemini.py actually
    emits the fail-open warning on) before launching agy.  Remove in finally to
    avoid leaking into the click process.

    Thread-safe: the one-shot flag and counter use a single lock.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self._lock: threading.Lock = threading.Lock()
        self._warned: bool = False
        self._count: int = 0

    # ------------------------------------------------------------------
    # logging.Handler interface
    # ------------------------------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        if _FAIL_OPEN_SUBSTR not in record.getMessage():
            return
        with self._lock:
            self._count += 1
            if self._warned:
                return
            self._warned = True
        # Print outside the lock to avoid holding it during I/O.
        print(
            "Headroom: compression failed for a request; forwarding it uncompressed"
            " (fail-open). Further occurrences are summarized at exit.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Accessors (called from the main thread after agy exits)
    # ------------------------------------------------------------------

    @property
    def fail_open_count(self) -> int:
        """Total number of fail-open log records observed (thread-safe)."""
        with self._lock:
            return self._count


def _get_compression_stats() -> dict[str, Any]:
    """Return get_compression_store().get_stats() — imported lazily so the
    compression stack is not pulled in unless actually called."""
    from headroom.cache.compression_store import get_compression_store

    return get_compression_store().get_stats()


class AgySessionStats:
    """Snapshot start/end compression-store stats, format a one-line summary.

    Usage::

        stats = AgySessionStats()           # call at session start (before agy)
        stats.snapshot_start()
        # ... agy runs ...
        stats.print_summary(handler)        # call in finally / SIGTERM handler

    The summary is idempotent: ``print_summary`` prints exactly once regardless
    of how many times it is called (safe for both the ``finally`` path and the
    SIGTERM handler running close together).
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._start: dict[str, Any] | None = None
        self._printed: bool = False

    def snapshot_start(self) -> None:
        """Capture the compression-store baseline before agy launches.

        Best-effort: if the store is unavailable the snapshot is omitted and
        ``print_summary`` will emit a reduced message.
        """
        try:
            snap = _get_compression_stats()
        except Exception:  # noqa: BLE001
            snap = None
        with self._lock:
            self._start = snap

    def print_summary(self, handler: FailOpenWarnHandler | None = None) -> None:
        """Print a one-line session compression summary to stderr.

        Idempotent: prints at most once per ``AgySessionStats`` instance.
        Safe to call from both the ``finally`` block and the SIGTERM handler.

        Args:
            handler: The ``FailOpenWarnHandler`` installed for this session, or
                     ``None`` if it was not installed (fail-open count omitted).
        """
        with self._lock:
            if self._printed:
                return
            self._printed = True
            start = self._start

        # Snapshot end outside the lock (I/O + potential lock in store).
        try:
            end = _get_compression_stats()
        except Exception:  # noqa: BLE001
            end = None

        fail_open = handler.fail_open_count if handler is not None else None
        summary = _format_summary(start, end, fail_open_count=fail_open)
        print(summary, file=sys.stderr)


def _format_summary(
    start: dict[str, Any] | None,
    end: dict[str, Any] | None,
    *,
    fail_open_count: int | None = None,
) -> str:
    """Format a session compression summary string.

    Pure function for testability — no I/O side-effects.

    Args:
        start:           ``get_stats()`` snapshot taken before the session.
        end:             ``get_stats()`` snapshot taken after the session.
        fail_open_count: Number of fail-open warnings observed, or ``None``
                         when the handler was not installed.

    Returns:
        A single-line string suitable for printing to stderr.
    """
    if start is None or end is None:
        fail_suffix = (
            f"  Fail-open requests: {fail_open_count}" if fail_open_count is not None else ""
        )
        return f"Headroom agy session summary: compression stats unavailable.{fail_suffix}"

    entries = max(0, end.get("entry_count", 0) - start.get("entry_count", 0))
    orig = max(0, end.get("total_original_tokens", 0) - start.get("total_original_tokens", 0))
    comp = max(0, end.get("total_compressed_tokens", 0) - start.get("total_compressed_tokens", 0))

    if orig > 0:
        ratio = comp / orig
        ratio_str = f"{ratio:.2f}x"
    else:
        ratio_str = "n/a (no compression)"

    parts = [
        f"Headroom agy session: {entries} entries compressed,",
        f"{orig:,} → {comp:,} tokens ({ratio_str} ratio)",
    ]
    if fail_open_count is not None:
        parts.append(f"| {fail_open_count} fail-open request(s)")

    return " ".join(parts)


def install_fail_open_handler() -> FailOpenWarnHandler:
    """Install a ``FailOpenWarnHandler`` on the gemini proxy logger.

    Returns the installed handler so the caller can:
    - read ``.fail_open_count`` after agy exits
    - pass it to ``remove_fail_open_handler`` in finally

    Safe to call multiple times (each call installs a fresh handler; the old
    one is not removed — call ``remove_fail_open_handler`` explicitly).
    """
    handler = FailOpenWarnHandler()
    # Target "headroom.proxy" — the logger gemini.py:25 actually uses to emit the
    # fail-open warning. emit() filters on _FAIL_OPEN_SUBSTR so unrelated
    # headroom.proxy warnings are ignored.
    logging.getLogger(_GEMINI_LOGGER).addHandler(handler)
    return handler


def remove_fail_open_handler(handler: FailOpenWarnHandler | None) -> None:
    """Remove a previously-installed ``FailOpenWarnHandler`` (best-effort).

    Called in the ``finally`` block of ``agy()`` to avoid leaking the handler
    into the click process or subsequent agent invocations.  Idempotent and
    exception-safe.  Accepts ``None`` to simplify callers that may not have
    installed the handler (e.g. if an exception was raised before install).
    """
    if handler is None:
        return
    try:
        logging.getLogger(_GEMINI_LOGGER).removeHandler(handler)
    except Exception:  # noqa: BLE001
        pass

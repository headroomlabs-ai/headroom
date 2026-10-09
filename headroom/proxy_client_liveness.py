"""Shared PID-liveness/identity primitives for wrap-client marker files.

Both wrap.py (tearing down a shared proxy) and the proxy itself (resolving
a registered workspace root) need to answer "is the process that wrote this
marker still alive?" -- kept here once, importable from both, instead of
duplicated. No CLI dependencies, so the proxy can import it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from headroom import fsutil
from headroom._subprocess import pid_alive

__all__ = ["identity_mismatch", "marker_pid_reused", "pid_alive", "proc_identity"]


def proc_identity(pid: int) -> tuple[str, float] | None:
    """Best-effort ``(source, start_time)`` identity for a PID.

    Used to defeat PID reuse: a marker is only trusted while the live PID is
    *the same process* that wrote it. Returns ``None`` when start time can't be
    determined (e.g. macOS without psutil), in which case callers fall back to
    existence-only liveness -- no regression, just no reuse protection there.

    The ``source`` tag ("psutil" vs "proc") guards against comparing values in
    different units; we only compare like-for-like.
    """
    try:
        import psutil  # type: ignore[import-untyped]  # optional dependency; portable when present

        return ("psutil", psutil.Process(pid).create_time())
    except Exception:
        pass
    # Linux fallback: field 22 of /proc/<pid>/stat is starttime in clock ticks
    # since boot -- a stable per-process value. `comm` (field 2) may contain
    # spaces/parens, so split after the final ')'.
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            fields = fh.read().rpartition(b")")[2].split()
        return ("proc", float(fields[19]))
    except (OSError, IndexError, ValueError):
        return None


def identity_mismatch(src: Any, recorded: Any, pid: int) -> bool:
    """True only if ``pid``'s current identity *provably* differs from the
    recorded ``(src, recorded)`` identity (i.e. the PID was recycled).

    Conservative by design: any uncertainty (unknown/legacy identity, unknown
    start time, mismatched source) returns ``False`` -- never claim a mismatch
    without proof, since callers use this to decide whether to trust or
    discard state tied to a live PID.
    """
    if not isinstance(src, str) or not isinstance(recorded, int | float):
        return False  # legacy / identity-less record -- can't tell
    ident = proc_identity(pid)
    if ident is None or ident[0] != src:
        return False  # can't compare like-for-like -- don't claim mismatch
    # Start times are stable per process; >1s apart means a different process.
    return abs(ident[1] - float(recorded)) > 1.0


def marker_pid_reused(marker: Path, pid: int) -> bool:
    """True only if the live ``pid`` is *provably* a different process than the
    one that wrote ``marker`` (i.e. the PID was recycled after a crash)."""
    import json

    try:
        rec = json.loads(fsutil.read_text(marker))
    except (OSError, ValueError):
        return False
    if not isinstance(rec, dict):
        return False
    return identity_mismatch(rec.get("start_src"), rec.get("start_time"), pid)

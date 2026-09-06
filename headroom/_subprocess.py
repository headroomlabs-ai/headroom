import os
import subprocess as _sp
import sys
from typing import Any


def _win32_pid_alive(pid: int) -> bool:
    """Non-destructive PID liveness probe for Windows via ``kernel32``."""
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    ERROR_ACCESS_DENIED = 5
    # ctypes GetLastError() is Any; wrap so mypy sees bool (matches pid_alive below).
    return bool(kernel32.GetLastError() == ERROR_ACCESS_DENIED)


def pid_alive(pid: int) -> bool:
    """Return True if *pid* names a live process (non-destructive on all platforms)."""
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore[import-untyped]  # optional dep

        return bool(psutil.pid_exists(pid))
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            return _win32_pid_alive(pid)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (ProcessLookupError, OSError, SystemError, OverflowError, ValueError):
        return False
    return True


def proc_identity(pid: int) -> tuple[str, float] | None:
    """Best-effort ``(source, start_time)`` identity for a PID.

    Used to defeat PID reuse: a marker is only trusted while the live PID is
    *the same process* that wrote it. Returns ``None`` when start time can't be
    determined (e.g. macOS without psutil), in which case callers fall back to
    existence-only liveness, no regression, just no reuse protection there.

    The ``source`` tag ("psutil" vs "proc") guards against comparing values in
    different units; we only compare like-for-like.
    """
    try:
        import psutil  # type: ignore[import-untyped]  # optional dependency; portable when present

        return ("psutil", psutil.Process(pid).create_time())
    except Exception:
        pass
    # Linux fallback: field 22 of /proc/<pid>/stat is starttime in clock ticks
    # since boot, a stable per-process value. `comm` (field 2) may contain
    # spaces/parens, so split after the final ')'.
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            fields = fh.read().rpartition(b")")[2].split()
        return ("proc", float(fields[19]))
    except (OSError, IndexError, ValueError):
        return None


def identity_mismatch(
    src: Any,
    recorded: Any,
    pid: int,
    *,
    identity_fn: Any = None,
) -> bool:
    """True only if ``pid``'s current identity *provably* differs from the
    recorded ``(src, recorded)`` identity (i.e. the PID was recycled).

    Conservative by design: any uncertainty (unknown/legacy identity, unknown
    start time, mismatched source) returns ``False``, never claim a mismatch
    without proof, since the caller uses this to decide whether to trust or
    discard state tied to a live PID. ``identity_fn`` defaults to
    :func:`proc_identity` and exists so callers with their own (mockable)
    identity source can share this comparison.
    """
    if identity_fn is None:
        identity_fn = proc_identity
    if not isinstance(src, str) or not isinstance(recorded, int | float):
        return False  # legacy / identity-less record, can't tell
    ident = identity_fn(pid)
    if ident is None or ident[0] != src:
        return False  # can't compare like-for-like, don't claim mismatch
    # Start times are stable per process; >1s apart means a different process.
    return bool(abs(ident[1] - float(recorded)) > 1.0)


def run(*args: Any, **kwargs: Any) -> _sp.CompletedProcess:
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "replace")
    return _sp.run(*args, **kwargs)


def Popen(*args: Any, **kwargs: Any) -> _sp.Popen:
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "replace")
    return _sp.Popen(*args, **kwargs)

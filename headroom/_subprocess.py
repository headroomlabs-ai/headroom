import os
import subprocess as _sp
from typing import Any


def pid_alive(pid: int) -> bool:
    """Return True if ``pid`` names a live process (Windows-safe).

    ``os.kill(pid, 0)`` is the usual Unix liveness probe, but on Windows it can
    fail against a detached/stale/invalid PID with ``WinError 87`` ("The
    parameter is incorrect"). CPython sometimes surfaces that as a
    ``SystemError`` rather than an ``OSError``; since ``SystemError`` is not an
    ``OSError`` subclass, a bare ``except OSError`` lets it escape and crash the
    caller — and in the detached-agent path it took down the supervised proxy
    (issue #1544). Prefer ``psutil.pid_exists`` and treat ``SystemError`` as
    "not alive".
    """
    if pid <= 0:
        return False  # non-positive PIDs are never valid liveness targets
    try:
        import psutil  # type: ignore[import-untyped]  # optional dep, already used elsewhere

        return bool(psutil.pid_exists(pid))
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True  # exists but owned by another user
    except (ProcessLookupError, OSError, SystemError):
        return False
    return True


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

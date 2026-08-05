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

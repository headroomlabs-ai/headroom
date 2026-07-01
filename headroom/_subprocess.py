import os
import subprocess as _sp
from typing import Any


def pid_alive(pid: int) -> bool:
    """Return True if *pid* names a live process (non-destructive)."""
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore[import-untyped]  # optional dep

        return bool(psutil.pid_exists(pid))
    except Exception:
        pass
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

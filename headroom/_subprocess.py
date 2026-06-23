import subprocess as _sp
from typing import Any


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

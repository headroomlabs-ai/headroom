"""PATH environment helpers for e2e test isolation."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _minimal_path_dirs() -> list[str]:
    """Directories always needed so Python / basic shell utilities work."""

    if os.name == "nt":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        return [
            rf"{system_root}\System32",
            system_root,
            rf"{system_root}\System32\Wbem",
            rf"{system_root}\System32\WindowsPowerShell\v1.0",
        ]
    # POSIX: keep enough for bash, python3, mkdir, chmod, etc.
    return ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]


@contextmanager
def with_clean_path(extra_dirs: list[Path] | None = None) -> Iterator[dict[str, str]]:
    """Set PATH to a minimal known-good value plus ``extra_dirs``.

    Yields the (already-mutated) environment dict so callers can pass it
    directly to ``subprocess.run(env=...)``. On exit, the previous PATH is
    restored.
    """

    extras = [str(Path(p)) for p in (extra_dirs or [])]
    new_path = os.pathsep.join(extras + _minimal_path_dirs())
    env = os.environ.copy()
    previous = env.get("PATH")
    env["PATH"] = new_path
    # Also mutate the real environment so ``shutil.which`` inside this process
    # sees the clean PATH. Restore on exit.
    real_previous = os.environ.get("PATH")
    os.environ["PATH"] = new_path
    try:
        yield env
    finally:
        if real_previous is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = real_previous
        if previous is None:
            env.pop("PATH", None)
        else:
            env["PATH"] = previous

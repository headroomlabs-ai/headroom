"""Shared filesystem helpers for the ``learn`` session plugins."""

from __future__ import annotations

from pathlib import Path


def path_exists(path: Path) -> bool:
    """Like ``Path.exists()`` but treats an unreadable path as absent.

    The ``learn`` plugins probe project paths reconstructed or recorded in
    session files written on other machines / by other users. ``Path.exists()``
    calls ``os.stat``, which raises ``PermissionError`` (not ``False``) when a
    parent directory isn't stat-able — e.g. a decoded candidate that collides
    with another user's home, or a recorded ``cwd`` behind a restricted mount.
    Unhandled, that crashes the whole ``learn`` command (issue #2443). Treat any
    ``OSError`` as "does not exist" so discovery skips the path instead.
    """
    try:
        return path.exists()
    except OSError:
        return False

"""Owner-only creation for the files Headroom writes at runtime.

Headroom's runtime log (``~/.headroom/logs/proxy-<port>.log``) and the optional
JSONL request log can both carry verbatim request and response content: wire
debug dumps, ``--log-messages`` bodies, and — when an operator opts in — CCR
payload previews. None of that should be created at the process umask, which on
a stock developer machine means ``0644``: world-readable.

Scope of the guarantee — read this before citing it in a threat model:

* **POSIX** (Linux, macOS, \\*BSD): enforced. Files are created ``0600`` and
  tightened with ``fchmod`` on the already-open descriptor, so a file left
  world-readable by an earlier run is fixed, and the mode cannot be applied to
  the wrong inode by a path that changed underneath us. ``O_NOFOLLOW`` means a
  symlink planted at the path fails the open outright rather than redirecting
  the write.
* **Windows**: *not* enforced, and deliberately not claimed. Who may read an
  NTFS file is decided by its ACL. ``os.chmod`` on Windows only toggles the
  read-only attribute and leaves the ACL untouched, so ``chmod(0o600)`` returns
  successfully while ``stat.S_IMODE`` still reports ``0666`` and the file stays
  readable per the inherited ACL. Python ships no ACL API, so Headroom does not
  pretend to set one: on Windows the log inherits the ACL of its parent
  directory and :data:`OWNER_ONLY_SUPPORTED` is ``False``. Callers that create
  sensitive files say so once in the log (see
  ``headroom.proxy.helpers._setup_file_logging``), and operators should treat
  the log directory itself as the access-control boundary — keep it under the
  user profile, and do not put it on a share.

``O_NOFOLLOW`` does not exist on Windows either, so the symlink protection
there is the explicit :func:`is_symlink` check callers make before opening, not
a kernel-enforced one.
"""

from __future__ import annotations

import os
import stat
from typing import IO, Any

#: Mode for every runtime file Headroom creates that may hold request content.
OWNER_ONLY_MODE = 0o600

#: ``True`` only where the mode bits above actually decide who can read the
#: file. See the module docstring for why Windows is excluded.
OWNER_ONLY_SUPPORTED = os.name == "posix"


def _open_flags() -> int:
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
    # Refuse to open through a symlink where the platform can enforce it, so a
    # planted link cannot redirect either the write or the chmod. Absent on
    # Windows, where getattr() leaves the flag out.
    flags |= getattr(os, "O_NOFOLLOW", 0)
    # Match the builtin open(): keep the OS layer byte-exact and let the text
    # wrapper do newline translation, instead of translating twice on Windows.
    flags |= getattr(os, "O_BINARY", 0)
    return flags


def restrict_fd_to_owner(fd: int) -> bool:
    """Make an open descriptor owner-only. Returns whether it took effect.

    Acts on the descriptor, not the path, so there is no window in which the
    mode could land on a different file, and no way for a symlink to move it.
    Returns ``False`` on platforms where the mode bits do not carry the
    guarantee rather than reporting a protection that was not established.
    """
    if not OWNER_ONLY_SUPPORTED:
        return False
    if stat.S_IMODE(os.fstat(fd).st_mode) != OWNER_ONLY_MODE:
        os.fchmod(fd, OWNER_ONLY_MODE)
    return True


def restrict_path_to_owner(path: str | os.PathLike[str]) -> bool:
    """Make an existing file owner-only. Returns whether it took effect.

    For files Headroom did not open itself — rotated log backups, and backups
    left behind by an older unhardened run. A path that is a symlink is left
    alone: the target is not ours to re-permission.
    """
    if not OWNER_ONLY_SUPPORTED:
        return False
    try:
        if os.path.islink(path) or not os.path.exists(path):
            return False
        if stat.S_IMODE(os.stat(path).st_mode) != OWNER_ONLY_MODE:
            os.chmod(path, OWNER_ONLY_MODE)
    except OSError:
        return False
    return True


def open_owner_only(
    path: str | os.PathLike[str],
    mode: str = "a",
    *,
    encoding: str | None = None,
    errors: str | None = None,
) -> IO[Any]:
    """Open *path* for append, creating it owner-only.

    Raises ``OSError`` — which every caller already treats as "logging is
    unavailable, carry on" — if the path cannot be opened, including when it is
    a symlink on a platform with ``O_NOFOLLOW``. Failing closed is deliberate:
    a redirected sensitive log is worse than no log.
    """
    fd = os.open(path, _open_flags(), OWNER_ONLY_MODE)
    try:
        restrict_fd_to_owner(fd)
        return open(fd, mode, encoding=encoding, errors=errors, closefd=True)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            # open() can have taken and closed the descriptor on its way out.
            pass
        raise

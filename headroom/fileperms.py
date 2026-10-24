"""Owner-only creation for the files Headroom writes at runtime.

Headroom's runtime log (``~/.headroom/logs/proxy-<port>.log``) and the optional
JSONL request log can both carry verbatim request and response content: wire
debug dumps, ``--log-messages`` bodies, and — when an operator opts in — CCR
payload previews. The CCR retrieval store holds verbatim tool output, and the
memory databases hold facts extracted from conversations together with the
user id they belong to. None of that should be created at the process umask,
which on a stock developer machine means ``0644``: world-readable.

This module is the one place that knows how to create such a file privately,
so every store uses the same rules and a reviewer can read the guarantee off
one file. Extensions that keep their own state (Shield, Foresight, Kiro, the
router) should call these helpers rather than ``sqlite3.connect`` /
``Path.write_text`` directly:

* :func:`open_owner_only` — open a log or text file for append or write.
* :func:`ensure_private_file` — make a path a private regular file *before*
  a library that opens by path (sqlite) touches it.
* :func:`connect_private_sqlite` — ``sqlite3.connect`` through the above.
* :func:`private_dir` — create a directory that will hold such files.

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

import functools
import os
import sqlite3
import stat
from pathlib import Path
from typing import IO, Any
from urllib.parse import parse_qs, unquote, urlsplit

#: Mode for every runtime file Headroom creates that may hold request content.
OWNER_ONLY_MODE = 0o600

#: Mode for directories that hold such files.
OWNER_ONLY_DIR_MODE = 0o700

#: ``True`` only where the mode bits above actually decide who can read the
#: file. See the module docstring for why Windows is excluded.
OWNER_ONLY_SUPPORTED = os.name == "posix"


def _open_flags(*, truncate: bool = False) -> int:
    flags = os.O_CREAT | os.O_WRONLY | (os.O_TRUNC if truncate else os.O_APPEND)
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
    newline: str | None = None,
) -> IO[Any]:
    """Open *path* for append (``"a"``) or write (``"w"``), creating it owner-only.

    Raises ``OSError`` — which every caller already treats as "logging is
    unavailable, carry on" — if the path cannot be opened, including when it is
    a symlink on a platform with ``O_NOFOLLOW``. Failing closed is deliberate:
    a redirected sensitive log is worse than no log.
    """
    if mode[:1] not in ("a", "w"):
        raise ValueError(f"open_owner_only: mode must start with 'a' or 'w', got {mode!r}")
    fd = os.open(path, _open_flags(truncate=mode.startswith("w")), OWNER_ONLY_MODE)
    try:
        restrict_fd_to_owner(fd)
        return open(fd, mode, encoding=encoding, errors=errors, newline=newline, closefd=True)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            # open() can have taken and closed the descriptor on its way out.
            pass
        raise


def ensure_private_file(path: str | os.PathLike[str], *, what: str = "file") -> None:
    """Make *path* an owner-only regular file before a by-path opener touches it.

    For libraries that open files by path and create them at the umask —
    sqlite above all. Creating the file here first, with an explicit
    ``0o600`` mode, means a *new* database is private from birth (sqlite
    treats an empty file as a fresh database, and creates its ``-wal`` /
    ``-shm`` sidecars with the same mode as the main file). A pre-existing
    file left wide by an earlier run is narrowed.

    Fails **closed**: a store of conversation or tool content must not be
    opened world-readable, so a failure to create or narrow the file privately
    raises :class:`PermissionError` rather than proceeding to open a wide
    file. The existing-file path is symlink-race resistant — ``O_EXCL`` on the
    create refuses to reuse a planted file or symlink; an existing file is
    re-opened with ``O_NOFOLLOW`` and narrowed through that descriptor
    (``fstat`` to confirm a regular file, ``fchmod`` to set the mode) rather
    than by re-resolving the path, which closes the check-then-chmod TOCTOU a
    ``chmod(path)`` would leave open. On Windows POSIX mode bits and
    ``O_NOFOLLOW`` do not apply, so there is nothing to narrow (see the module
    docstring for what is and is not claimed there).

    *what* names the store in the error message.
    """
    create_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        create_flags |= os.O_NOFOLLOW
    try:
        os.close(os.open(path, create_flags, OWNER_ONLY_MODE))
        return
    except FileExistsError:
        pass

    if not (hasattr(os, "O_NOFOLLOW") and hasattr(os, "fchmod")):
        return

    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise PermissionError(
            f"refusing to open {what} at {path}: not a regular file (symlink or open error: {exc})"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PermissionError(f"refusing to open {what} at {path}: not a regular file")
        # os.fchmod is POSIX-only (guarded by the hasattr check above); the
        # ignore keeps type-checking clean on Windows where it is absent.
        os.fchmod(fd, OWNER_ONLY_MODE)  # type: ignore[attr-defined]
    finally:
        os.close(fd)


@functools.cache
def _sqlite_always_uri() -> bool:
    """Whether this sqlite reads ``file:`` names as URIs even without ``uri=True``.

    True for builds compiled with ``SQLITE_USE_URI`` — Debian's, and so the
    official ``python`` Docker images.
    """
    conn = sqlite3.connect(":memory:")
    try:
        options = {row[0] for row in conn.execute("PRAGMA compile_options")}
    finally:
        conn.close()
    return bool(options & {"USE_URI", "USE_URI=1"})


def _sqlite_file_path(path: str | os.PathLike[str], *, uri: bool) -> str | None:
    """The file sqlite will create or open for *path*; ``None`` if there is none.

    Mirrors sqlite's own rules. Without ``uri=True`` the name is a literal
    path, even one that starts with ``file:``, unless this sqlite was built to
    always parse URIs (:func:`_sqlite_always_uri`). Parsed as a URI, ``file:`` names
    its percent-decoded path (relative to the working directory unless
    absolute), and is in-memory when that path is empty or ``:memory:`` or the
    query says ``mode=memory``. ``mode=ro`` / ``mode=rw`` never create the
    file, so a missing one is left for sqlite to report.
    """
    text = os.fspath(path)
    if text in ("", ":memory:"):
        return None
    if not (text.startswith("file:") and (uri or _sqlite_always_uri())):
        return text
    if not OWNER_ONLY_SUPPORTED:
        # No mode bits to set (module docstring), and Windows URIs carry a
        # drive letter sqlite strips its own way: pass them through.
        return None
    parts = urlsplit(text)
    # surrogateescape keeps an undecodable %XX byte as that raw byte, which is
    # the name sqlite opens; the default "replace" would secure a different one.
    file_path = unquote(parts.path, errors="surrogateescape")
    mode = parse_qs(parts.query).get("mode", [""])[-1]
    if file_path in ("", ":memory:") or mode == "memory":
        return None
    if parts.netloc not in ("", "localhost"):
        raise PermissionError(f"refusing SQLite URI with a remote authority: {text}")
    if mode in ("ro", "rw") and not os.path.lexists(file_path):
        return None
    return file_path


#: Files sqlite keeps next to a database and reopens by name.
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def connect_private_sqlite(
    path: str | os.PathLike[str],
    *,
    what: str = "database",
    **connect_kwargs: Any,
) -> sqlite3.Connection:
    """``sqlite3.connect`` that first makes the database file owner-only.

    Drop-in for ``sqlite3.connect(str(path), **kwargs)`` at every store that
    holds conversation-derived content. In-memory databases are passed
    straight through; a file-backed ``file:`` URI (``uri=True``) has its
    underlying file made private like a plain path. Existing ``-wal`` /
    ``-shm`` / ``-journal`` sidecars are narrowed too. The parent directory must
    already exist; create it with :func:`private_dir` if it is dedicated to
    this store.
    """
    file_path = _sqlite_file_path(path, uri=bool(connect_kwargs.get("uri")))
    if file_path is not None:
        ensure_private_file(file_path, what=what)
        # A crash can leave -wal/-shm/-journal sidecars behind with the mode
        # of an older, unhardened run, and sqlite reuses them as they are.
        # Narrow any that exist (refusing symlinks) before sqlite writes to
        # them. Missing ones are left alone: sqlite creates them with the
        # main file's mode, which is now 0600.
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            sidecar = file_path + suffix
            if os.path.lexists(sidecar):
                ensure_private_file(sidecar, what=f"{what} {suffix[1:]} file")
    # ``**connect_kwargs: Any`` makes mypy type the call as ``Any``; the
    # annotation pins it back to the real return type.
    conn: sqlite3.Connection = sqlite3.connect(os.fspath(path), **connect_kwargs)
    return conn


def private_dir(path: str | os.PathLike[str]) -> Path:
    """Create *path* (and missing parents) and make the leaf directory ``0o700``.

    For a directory that exists only to hold Headroom's sensitive files (the
    native memory directory); an existing directory left wider by an earlier
    run is narrowed. A symlink at the leaf is left alone: the target is not
    ours to re-permission. Returns the path.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    if OWNER_ONLY_SUPPORTED and not target.is_symlink():
        os.chmod(target, OWNER_ONLY_DIR_MODE)
    return target

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
* **Windows**: mode bits do not control read access there — who may read an
  NTFS file is decided by its ACL, and ``os.chmod`` only toggles the
  read-only attribute and leaves the ACL untouched, so ``chmod(0o600)``
  returns successfully while ``stat.S_IMODE`` still reports ``0666``. That is
  why :data:`OWNER_ONLY_SUPPORTED` — "mode bits enforce this" — is ``False``
  on Windows and stays that way regardless of the rest of this paragraph.
  :func:`restrict_fd_to_owner` and :func:`restrict_path_to_owner` nonetheless
  restrict the log, its rotated backups, and the JSONL request log there too,
  through :mod:`headroom._fileperms_windows`: a DACL with exactly one entry
  (the calling user, full control), built fresh rather than merged with the
  inherited entries that made the file readable in the first place, and
  applied with ``PROTECTED_DACL_SECURITY_INFORMATION`` so those inherited
  entries are dropped rather than left alongside it. Unlike the POSIX path
  this can fail per-call (a non-NTFS volume, a missing privilege), so callers
  that need to know whether it actually took effect call
  :func:`verify_owner_only` rather than trusting the static flag — see
  ``headroom.proxy.helpers._warn_once_if_not_protected``, which warns only
  when verification fails rather than unconditionally on Windows. Not
  extended (yet) to the sqlite-backed memory/CCR stores
  (:func:`ensure_private_file`/:func:`connect_private_sqlite`) or
  :func:`private_dir` — those remain unenforced on Windows, same as before.

``O_NOFOLLOW`` does not exist on Windows, so the CRT ``os.open`` underneath
Python's own ``open()`` has no way to refuse a symlink/junction there — it
silently follows one to whatever it points at. :func:`open_owner_only` does
not route through that on Windows for this reason: it opens via
``headroom._fileperms_windows.open_no_follow``, which uses
``FILE_FLAG_OPEN_REPARSE_POINT`` to open the reparse point itself regardless
of create-or-open disposition, and refuses it outright if the resulting
handle's own attributes show it is one — closing the same gap ``O_NOFOLLOW``
closes on POSIX, checked on the handle rather than the path so there is
nothing for a race to redirect. :func:`restrict_path_to_owner` uses the same
flag for the same reason when narrowing a file it did not open itself
(rotated backups): because that path only ever touches the reparse point's
own DACL, never a target's, it is safe even without the attribute check.
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
#: file. See the module docstring for why Windows is excluded -- it has its
#: own, separately-verified mechanism instead, not reflected in this flag.
OWNER_ONLY_SUPPORTED = os.name == "posix"

_win: Any = None
if os.name == "nt":
    from . import _fileperms_windows as _win


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
    On Windows this goes through :mod:`headroom._fileperms_windows` instead
    of mode bits (see the module docstring); returns ``False`` if that fails
    too, rather than reporting a protection that was not established.
    """
    if OWNER_ONLY_SUPPORTED:
        if stat.S_IMODE(os.fstat(fd).st_mode) != OWNER_ONLY_MODE:
            os.fchmod(fd, OWNER_ONLY_MODE)
        return True
    if _win is not None:
        return bool(_win.restrict_handle(fd))
    return False


def restrict_path_to_owner(path: str | os.PathLike[str]) -> bool:
    """Make an existing file owner-only. Returns whether it took effect.

    For files Headroom did not open itself — rotated log backups, and backups
    left behind by an older unhardened run. A path that is a symlink is left
    alone on every platform: the target is not ours to re-permission. On
    Windows the ACL path (see the module docstring) adds
    ``FILE_FLAG_OPEN_REPARSE_POINT`` underneath that check as a second guard.
    """
    try:
        if os.path.islink(path) or not os.path.exists(path):
            return False
        if OWNER_ONLY_SUPPORTED:
            if stat.S_IMODE(os.stat(path).st_mode) != OWNER_ONLY_MODE:
                os.chmod(path, OWNER_ONLY_MODE)
            return True
    except OSError:
        return False
    if _win is not None:
        return bool(_win.restrict_path(os.fspath(path)))
    return False


def verify_owner_only(path: str | os.PathLike[str]) -> bool:
    """Confirm *path* is restricted to its current owner right now.

    Unlike :data:`OWNER_ONLY_SUPPORTED` (a static per-platform flag) or the
    return value of :func:`restrict_fd_to_owner`/:func:`restrict_path_to_owner`
    (whether the API calls *reported* success), this re-reads the file's
    actual permissions. On Windows the ACL call can fail for reasons that
    have nothing to do with Headroom's code -- a non-NTFS volume, a missing
    privilege -- so a caller deciding whether to warn an operator should ask
    what happened to *this* file, not assume from the platform.
    """
    if OWNER_ONLY_SUPPORTED:
        try:
            return stat.S_IMODE(os.stat(path).st_mode) == OWNER_ONLY_MODE
        except OSError:
            return False
    if _win is not None:
        return bool(_win.verify_path(os.fspath(path)))
    return False


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
    a symlink or junction. Failing closed is deliberate: a redirected sensitive
    log is worse than no log.

    On POSIX, ``O_NOFOLLOW`` makes the ``os.open`` below refuse a symlink at
    the kernel level. Windows has no such flag -- plain ``os.open`` there
    silently follows a planted symlink/junction to whatever it points at, so
    a DACL applied afterward lands on the target, not the intended file; this
    goes through :func:`headroom._fileperms_windows.open_no_follow` instead,
    which opens the reparse point itself and verifies that before touching
    anything, closing the one path/open gap ``O_NOFOLLOW`` closes on POSIX.
    """
    if mode[:1] not in ("a", "w"):
        raise ValueError(f"open_owner_only: mode must start with 'a' or 'w', got {mode!r}")
    truncate = mode.startswith("w")
    if OWNER_ONLY_SUPPORTED:
        fd = os.open(path, _open_flags(truncate=truncate), OWNER_ONLY_MODE)
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
    if _win is not None:
        fd = _win.open_no_follow(os.fspath(path), truncate=truncate)
        try:
            return open(fd, mode, encoding=encoding, errors=errors, newline=newline, closefd=True)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
    return open(path, mode, encoding=encoding, errors=errors, newline=newline)


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

"""Encoding- and newline-safe text file I/O.

``Path.read_text()`` / ``Path.write_text()`` and the builtin ``open()`` default
to the *system locale* encoding and, in text mode, translate ``\\n`` to
``os.linesep`` on write. On non-UTF-8 Windows locales (e.g. GBK / cp936 on
zh-CN) this corrupts config files two ways:

1. Reading a UTF-8 file as GBK — or a GBK file as UTF-8 — raises
   ``UnicodeDecodeError``.
2. Writing re-translates ``\\n`` to ``\\r\\n``; content that already has
   ``\\r\\n`` becomes ``\\r\\r\\n``, which TOML parsers reject with
   "carriage return must be followed by newline".

These helpers always use UTF-8, fall back to the locale encoding when a file
predates the fix (tools may have written it in the locale encoding), and write
with ``newline=""`` so existing line endings pass through unchanged.

See issue #733.
"""

from __future__ import annotations

import locale
import os
import shutil
import tempfile
from pathlib import Path

# Sentinel so ``default=None`` can be a real return value if a caller wants it.
_RAISE = object()


def read_text(path: str | os.PathLike[str], *, default: object = _RAISE) -> str:
    """Read text, preferring UTF-8 and falling back to the locale encoding.

    Decoding order: UTF-8 (strict) → locale preferred encoding (strict) →
    UTF-8 with ``errors="replace"`` (never raises on content). If the file
    cannot be opened (missing/unreadable) and ``default`` is given, it is
    returned; otherwise the ``OSError`` propagates.

    Line endings are normalised to ``\\n`` (universal-newline semantics,
    matching the stdlib text-mode default) so callers that search or rewrite
    the text work on a single ending, and a later :func:`write_text` cannot
    re-double an existing ``\\r\\n``.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError:
        if default is not _RAISE:
            return default  # type: ignore[return-value]
        raise

    text: str | None = None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        loc = locale.getpreferredencoding(False)
        if loc and loc.lower().replace("-", "") != "utf8":
            try:
                text = raw.decode(loc)
            except (UnicodeDecodeError, LookupError):
                text = None
        if text is None:
            text = raw.decode("utf-8", errors="replace")

    return text.replace("\r\n", "\n").replace("\r", "\n")


def write_text(path: str | os.PathLike[str], content: str) -> None:
    """Write text as UTF-8 without translating line endings, atomically.

    ``newline=""`` disables the platform ``\\n`` → ``\\r\\n`` rewrite, so the
    bytes written match ``content`` exactly and existing ``\\r\\n`` endings are
    never doubled.

    The write goes to a temp file in the same directory, is fsynced, then moved
    into place with :func:`os.replace` (atomic on POSIX and Windows). Opening a
    config file ``"w"`` truncates it to zero bytes *before* the new content is
    written, so a crash, SIGKILL, OOM or ENOSPC mid-write left the user with a
    half-written ``~/.claude.json`` / ``settings.local.json``. Callers now see
    either the old file or the complete new one.

    A symlink is followed rather than replaced (dotfile managers symlink these
    configs), and an existing file's mode is preserved — ``mkstemp`` creates
    0600, which would otherwise silently tighten a 0644 config.

    Known tradeoff: this needs a writable *parent directory*, whereas an in-place
    truncate only needed a writable *file*, so a read-only directory holding a
    writable config now raises ``PermissionError``. That is deliberate — there is
    no in-place fallback, because falling back on ``OSError`` would turn the
    ENOSPC case into "file truncated, rewrite failed", the exact data loss this
    function exists to prevent. A ``PermissionError`` naming the directory is
    both rarer and actionable.
    """
    target = Path(path)
    if target.is_symlink():
        target = Path(os.path.realpath(target))
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if target.exists():
            shutil.copymode(target, tmp_path)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def append_text(path: str | os.PathLike[str], content: str) -> None:
    """Append text as UTF-8 without translating line endings (see ``write_text``)."""
    with Path(path).open("a", encoding="utf-8", newline="") as f:
        f.write(content)

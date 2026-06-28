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
    """Write text as UTF-8 without translating line endings.

    ``newline=""`` disables the platform ``\\n`` → ``\\r\\n`` rewrite, so the
    bytes written match ``content`` exactly and existing ``\\r\\n`` endings are
    never doubled.
    """
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        f.write(content)


def append_text(path: str | os.PathLike[str], content: str) -> None:
    """Append text as UTF-8 without translating line endings (see ``write_text``)."""
    with Path(path).open("a", encoding="utf-8", newline="") as f:
        f.write(content)

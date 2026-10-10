"""Payload-safe building blocks for log lines.

Log records are kept, rotated, attached to bug reports and shipped to log
collectors, so no level may carry request content, tool output or
credentials. Exception messages are the usual leak: a compressor, hook or
storage backend can quote the payload or a connection URL in its error. These
helpers describe a failure by what is safe to keep (exception types, code
locations, errno, ids, counts) and leave the message out.

Content is logged only when the operator opts in to it explicitly with
``HEADROOM_DEBUG_DUMP=full``, the same switch that gates full request dumps.
"""

from __future__ import annotations

import logging
import os
import threading
import traceback
from collections.abc import Hashable
from urllib.parse import urlsplit, urlunsplit

CONTENT_OPT_IN_ENV = "HEADROOM_DEBUG_DUMP"
_CONTENT_OPT_IN_VALUES = frozenset({"full", "all", "content"})
_MAX_FRAMES = 3
_MAX_CHAIN = 4
_ID_MAX_CHARS = 80
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")


def content_logging_enabled() -> bool:
    """True when the operator opted in to content in diagnostics (``HEADROOM_DEBUG_DUMP=full``)."""
    return os.environ.get(CONTENT_OPT_IN_ENV, "").strip().lower() in _CONTENT_OPT_IN_VALUES


def describe_exception(exc: BaseException) -> str:
    """Describe ``exc`` for a log line without its message.

    Gives each exception in the cause chain as its type plus the innermost code
    locations, e.g. ``ValueError at headroom/x.py:12 in parse; caused by
    KeyError at ...``. An ``OSError`` keeps its errno and strerror, which never
    carry payload. With ``HEADROOM_DEBUG_DUMP=full`` it returns the full
    traceback, messages included.
    """
    if content_logging_enabled():
        return "".join(traceback.format_exception(exc)).rstrip()
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(parts) < _MAX_CHAIN:
        seen.add(id(current))
        parts.append(_describe_one(current))
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return "; caused by ".join(parts)


def _describe_one(exc: BaseException) -> str:
    text = type(exc).__qualname__
    if isinstance(exc, OSError) and exc.errno is not None:
        # strerror is free text in some OSError subclasses (ssl.SSLError), so bound it.
        text += f" [Errno {exc.errno}] {str(exc.strerror)[:_ID_MAX_CHARS]}"
    frames = traceback.extract_tb(exc.__traceback__)[-_MAX_FRAMES:]
    if frames:
        where = " <- ".join(
            f"{_short_path(frame.filename)}:{frame.lineno} in {frame.name}"
            for frame in reversed(frames)
        )
        text += f" at {where}"
    return text


def _short_path(path: str) -> str:
    """``headroom/<module path>`` for Headroom's own files, the file name for anything else."""
    normalized = path.replace("\\", "/")
    if normalized.startswith(_PACKAGE_DIR + "/"):
        return "headroom/" + normalized[len(_PACKAGE_DIR) + 1 :]
    return normalized.rsplit("/", 1)[-1]


def redact_url(url: str) -> str:
    """Return ``url`` without userinfo, query or fragment, which can hold credentials.

    The path is kept, so do not pass URLs that carry a secret in the path
    (webhook URLs, ``/bot<token>/``). The host comes back lowercased.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return "<unparseable url>"
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    redacted = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    return f"{redacted}?<redacted>" if parts.query else redacted


def safe_id(value: object) -> str:
    """Render a client- or model-supplied id as one bounded log token.

    Uses ``repr`` so a newline in the id cannot forge a second log line, and
    cuts it to a fixed length so an oversized id cannot flood the log.
    """
    text = repr(value)
    if len(text) <= _ID_MAX_CHARS:
        return text
    return f"{text[:_ID_MAX_CHARS]}…(+{len(text) - _ID_MAX_CHARS} chars)"


class WarnOnce:
    """Warn once per key, for at most ``limit`` keys over the process lifetime.

    Past the limit it logs one overflow WARNING and then stays quiet, so a
    failure that hits many distinct keys (paths, rows, models) cannot flood the
    log on every pass. A key is used up only when WARNING is enabled, so a
    warning suppressed by the log level is still emitted later. ``forget``
    re-arms a key once its failure has cleared.
    """

    def __init__(self, limit: int, what: str) -> None:
        if limit < 1:
            raise ValueError(f"WarnOnce limit must be at least 1, got {limit}")
        self._limit = limit
        self._what = what
        self._keys: set[Hashable] = set()
        self._overflowed = False
        self._lock = threading.Lock()

    def first(self, key: Hashable, log: logging.Logger) -> bool:
        """Return True if the caller should log this key's WARNING now."""
        if not log.isEnabledFor(logging.WARNING):
            return False
        with self._lock:
            if key in self._keys:
                return False
            if len(self._keys) < self._limit:
                self._keys.add(key)
                return True
            report_overflow = not self._overflowed
            self._overflowed = True
        if report_overflow:
            log.warning(
                "More than %d distinct %s; further ones are logged at debug only",
                self._limit,
                self._what,
            )
        return False

    def forget(self, key: Hashable) -> None:
        """Re-arm ``key`` so its next failure warns again."""
        with self._lock:
            self._keys.discard(key)

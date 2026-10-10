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
import math
import os
import threading
import time
import traceback
from collections.abc import Hashable
from urllib.parse import urlsplit, urlunsplit

CONTENT_OPT_IN_ENV = "HEADROOM_DEBUG_DUMP"
_CONTENT_OPT_IN_VALUES = frozenset({"full", "all", "content"})
_MAX_FRAMES = 3
_MAX_CHAIN = 4
_ID_MAX_CHARS = 80
_MAX_ERRNO = 2**31 - 1
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")


def content_logging_enabled() -> bool:
    """True when the operator opted in to content in diagnostics (``HEADROOM_DEBUG_DUMP=full``)."""
    return os.environ.get(CONTENT_OPT_IN_ENV, "").strip().lower() in _CONTENT_OPT_IN_VALUES


def describe_exception(exc: BaseException) -> str:
    """Describe ``exc`` for a log line without its message.

    Gives each exception in the cause chain as its type plus the innermost code
    locations, e.g. ``ValueError at headroom/x.py:12 in parse; caused by
    KeyError at ...``. An ``OSError`` keeps its errno and the system's text for
    it, never the exception's own strerror. With ``HEADROOM_DEBUG_DUMP=full`` it returns the full
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
    if isinstance(exc, OSError) and isinstance(exc.errno, int):
        # The system's own text for the errno; exc.strerror is free text in some
        # subclasses (ssl.SSLError) and can quote a path or a peer's message.
        # An out-of-range errno is not formatted at all: converting a huge int
        # to text can itself raise, and a log helper must never raise.
        # Negative codes (socket.gaierror's EAI_*) have no os.strerror text.
        if 0 <= exc.errno <= _MAX_ERRNO:
            text += f" [Errno {exc.errno}] {_errno_text(exc.errno)}"
        elif -_MAX_ERRNO - 1 <= exc.errno < 0:
            text += f" [Errno {exc.errno}]"
        else:
            text += " [Errno out of range]"
    frames = traceback.extract_tb(exc.__traceback__)[-_MAX_FRAMES:]
    if frames:
        where = " <- ".join(
            f"{_short_path(frame.filename)}:{frame.lineno} in {frame.name}"
            for frame in reversed(frames)
        )
        text += f" at {where}"
    return text


def _errno_text(errno: int) -> str:
    try:
        return os.strerror(errno)
    except (ValueError, OverflowError):
        return "unknown error"


def _short_path(path: str) -> str:
    """``headroom/<module path>`` for Headroom's own files, the file name for anything else."""
    normalized = path.replace("\\", "/")
    if normalized.startswith(_PACKAGE_DIR + "/"):
        return "headroom/" + normalized[len(_PACKAGE_DIR) + 1 :]
    return normalized.rsplit("/", 1)[-1]


def redact_url(url: str) -> str:
    """Return ``url`` as scheme, host and port only, safe for any log line.

    Userinfo, query and fragment are always dropped, and so is the path, since
    a path segment can carry a credential (``/bot<token>/``, webhook URLs) and
    no rule tells such a segment from a route name. The path is kept only with
    the explicit content opt-in, ``HEADROOM_DEBUG_DUMP=full``. The host comes
    back lowercased.
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
    if content_logging_enabled():
        path = parts.path
    else:
        path = "/<path>" if parts.path not in ("", "/") else parts.path
    redacted = urlunsplit((parts.scheme, netloc, path, "", ""))
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
    """Warn once per key per window, for at most ``limit`` keys per window.

    The first warning opens a window of ``window_seconds`` (an hour by
    default). Inside it each key warns once; past ``limit`` keys it logs one
    overflow WARNING and stays quiet, so a failure that hits many distinct keys
    (paths, rows, models) cannot flood the log: at most ``limit + 1`` warnings
    per window. When the window ends the guard starts over, so a failure that
    keeps recurring, or a newly broken key, warns again once an hour. A key is
    used up only when WARNING is enabled, so a warning suppressed by the log
    level is still emitted later. ``forget`` re-arms a key once its failure has
    cleared; a re-armed key that fails again counts against the same window's
    budget, so forgetting can never push a window past ``limit + 1`` warnings.
    """

    def __init__(self, limit: int, what: str, *, window_seconds: float = 3600.0) -> None:
        if limit < 1:
            raise ValueError(f"WarnOnce limit must be at least 1, got {limit}")
        if window_seconds <= 0:
            raise ValueError(f"WarnOnce window_seconds must be positive, got {window_seconds}")
        self._limit = limit
        self._what = what
        self._window = window_seconds
        self._keys: set[Hashable] = set()
        self._issued = 0  # warnings issued this window; forget() does not refund them
        self._window_start: float | None = None
        self._overflowed = False
        self._lock = threading.Lock()

    def first(self, key: Hashable, log: logging.Logger) -> bool:
        """Return True if the caller should log this key's WARNING now."""
        if not log.isEnabledFor(logging.WARNING):
            return False
        with self._lock:
            now = time.monotonic()
            if self._window_start is not None and now - self._window_start >= self._window:
                self._keys.clear()
                self._issued = 0
                self._overflowed = False
                self._window_start = None
            if key in self._keys:
                return False
            if not self._overflowed and self._issued < self._limit:
                self._keys.add(key)
                self._issued += 1
                if self._window_start is None:
                    self._window_start = now
                return True
            report_overflow = not self._overflowed
            self._overflowed = True
            started = now if self._window_start is None else self._window_start
            remaining = self._window - (now - started)
        if report_overflow:
            log.warning(
                "More than %d distinct %s; further ones are not logged at WARNING "
                "for the next %d minutes",
                self._limit,
                self._what,
                max(1, math.ceil(remaining / 60)),
            )
        return False

    def forget(self, key: Hashable) -> None:
        """Re-arm ``key`` so its next failure warns again."""
        with self._lock:
            self._keys.discard(key)

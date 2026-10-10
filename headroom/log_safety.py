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

import ipaddress
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Hashable
from types import FrameType, ModuleType, TracebackType
from urllib.parse import urlsplit

CONTENT_OPT_IN_ENV = "HEADROOM_DEBUG_DUMP"
_CONTENT_OPT_IN_VALUES = frozenset({"full", "all", "content"})
_MAX_FRAMES = 3
_MAX_CHAIN = 4
_ID_MAX_CHARS = 80
_MAX_ERRNO = 2**31 - 1
# Schemes whose URLs have no host ("file:///path").
_LOCAL_SCHEMES = frozenset({"file", "sqlite", "unix"})
# DNS labels, with "_" allowed for container and service names (model_gateway).
_HOSTNAME = re.compile(
    r"[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?(?:\.[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?)*\.?"
)
_HOSTLESS_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:///")
_HEADROOM_MODULE = re.compile(r"headroom(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A class's __qualname__: identifiers joined by dots, with <locals> for nested ones.
_QUALIFIED_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.(?:<locals>|[A-Za-z_][A-Za-z0-9_]*))*")
_FIXED_CODE_NAMES = frozenset(
    {"<module>", "<lambda>", "<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>"}
)
# The base classes' own attribute descriptors. Reading through them means no
# subclass property (a __cause__ or errno that raises, a __qualname__ chosen by
# a metaclass) can run while a failure is being described.
_CAUSE = vars(BaseException)["__cause__"]
_CONTEXT = vars(BaseException)["__context__"]
_SUPPRESS_CONTEXT = vars(BaseException)["__suppress_context__"]
_TRACEBACK = vars(BaseException)["__traceback__"]
_ERRNO = vars(OSError)["errno"]
_QUALNAME = vars(type)["__qualname__"]
# Ints above this many bits have more than 4300 digits, which repr refuses.
_MAX_ID_INT_BITS = 14_000
# The module __dict__ descriptor itself, so a ModuleType subclass cannot override it.
_MODULE_DICT = vars(ModuleType)["__dict__"]
# Schemes a log may name; any other text before "://" could be a key someone
# pasted where a URL belongs, so it is replaced.
_KNOWN_SCHEMES = frozenset(
    {
        "http",
        "https",
        "ws",
        "wss",
        "grpc",
        "grpcs",
        "redis",
        "rediss",
        "postgres",
        "postgresql",
        "mysql",
        "mongodb",
        "mongodb+srv",
        "sqlite",
        "file",
        "unix",
        "s3",
        "gs",
    }
)


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
        current = _CAUSE.__get__(current) or (
            None if _SUPPRESS_CONTEXT.__get__(current) else _CONTEXT.__get__(current)
        )
    return "; caused by ".join(parts)


def _describe_one(exc: BaseException) -> str:
    # A class name is chosen by whoever defined it (type("Err\nforged", ...)).
    text = _class_name(type(exc)) or "<exception>"
    # An int subclass errno could format itself as anything, so only a plain int.
    errno = _ERRNO.__get__(exc) if issubclass(type(exc), OSError) else None
    if type(errno) is int:
        # The system's own text for the errno; exc.strerror is free text in some
        # subclasses (ssl.SSLError) and can quote a path or a peer's message.
        # An out-of-range errno is not formatted at all: converting a huge int
        # to text can itself raise, and a log helper must never raise.
        # Negative codes (socket.gaierror's EAI_*) have no os.strerror text.
        if 0 <= errno <= _MAX_ERRNO:
            text += f" [Errno {errno}] {_errno_text(errno)}"
        elif -_MAX_ERRNO - 1 <= errno < 0:
            text += f" [Errno {errno}]"
        else:
            text += " [Errno out of range]"
    frames = _innermost_headroom_frames(_TRACEBACK.__get__(exc))
    if frames:
        text += " at " + " <- ".join(reversed(frames))
    return text


def _errno_text(errno: int) -> str:
    try:
        return os.strerror(errno)
    except (ValueError, OverflowError):
        return "unknown error"


def _innermost_headroom_frames(tb: TracebackType | None) -> list[str]:
    """Locations of the innermost Headroom frames, at most ``_MAX_FRAMES``, in memory bounded by that."""
    frames: deque[str] = deque(maxlen=_MAX_FRAMES)
    while tb is not None:
        location = _headroom_frame_location(tb.tb_frame, tb.tb_lineno)
        if location is not None:
            frames.append(location)
        tb = tb.tb_next
    return list(frames)


def _headroom_frame_location(frame: FrameType, lineno: int) -> str | None:
    """``headroom/<module>.py:line in function`` for a frame of Headroom's own code, else None.

    Only Headroom's own frames are named; library and stdlib frames are skipped,
    since any text a frame carries (file, module or function name) can be set by
    code compiled at runtime and quote request data. A frame counts as Headroom's
    only when all of these hold, so none of its text is chosen at runtime:
    its globals are the namespace of the module ``sys.modules`` holds under that
    name; that name is a plain dotted ``headroom.*`` identifier; the code's file
    is the module's own file (compared by stem, so ``.pyc``-only and zip installs
    match); and the function name is a plain identifier. Headroom never execs code.
    """
    namespace = frame.f_globals
    # Read through dict's own methods: globals can be a dict subclass.
    name = dict.get(namespace, "__name__")
    if type(name) is not str or not _HEADROOM_MODULE.fullmatch(name):
        return None
    module = sys.modules.get(name)
    if not isinstance(module, ModuleType) or _MODULE_DICT.__get__(module) is not namespace:
        return None
    code = frame.f_code
    module_file = dict.get(namespace, "__file__")
    if type(module_file) is not str or _stem(module_file) != _stem(code.co_filename):
        return None
    function = code.co_name
    if not (_IDENTIFIER.fullmatch(function) or function in _FIXED_CODE_NAMES):
        return None
    package = dict.__contains__(namespace, "__path__")
    path = name.replace(".", "/") + ("/__init__.py" if package else ".py")
    return f"{path}:{lineno} in {function}"


def _stem(path: str) -> str:
    return os.path.splitext(path.replace("\\", "/").rsplit("/", 1)[-1])[0]


def redact_url(url: str) -> str:
    """Return ``url`` as scheme, host and port only, safe for any log line.

    Userinfo, query and fragment are always dropped, and so is the path, since
    a path segment can carry a credential (``/bot<token>/``, webhook URLs) and
    no rule tells such a segment from a route name. The path is kept only with
    the explicit content opt-in, ``HEADROOM_DEBUG_DUMP=full``. The host comes
    back lowercased, and only when it is a DNS name or IP address; a URL with
    no host is accepted only for ``file``, ``sqlite`` and ``unix``.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return "<unparseable url>"
    # Without "//" urlsplit reads "sk-key:secret" as scheme "sk-key", so a string
    # with no host is logged only for the hostless schemes ("file:///path").
    if not parts.netloc and not (
        parts.scheme in _LOCAL_SCHEMES and _HOSTLESS_URL.match(url.strip())
    ):
        return "<unparseable url>"
    if parts.netloc:
        checked = _checked_host(host)
        if checked is None:
            return "<unparseable url>"
        host = checked
    scheme = parts.scheme if parts.scheme in _KNOWN_SCHEMES else "<scheme>"
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    if content_logging_enabled():
        path = parts.path
    else:
        path = "/<path>" if parts.path not in ("", "/") else parts.path
    redacted = f"{scheme}://{netloc}{path}"
    return f"{redacted}?<redacted>" if parts.query else redacted


def _checked_host(host: str) -> str | None:
    """``host`` as a loggable name or address, or None when it may not be one.

    An IPv6 zone id (``fe80::1%eth0``) is free text, so it is dropped. A
    non-ASCII name is given in its ASCII (IDNA) form.
    """
    address = host.split("%", 1)[0]
    try:
        ipaddress.ip_address(address)
    except ValueError:
        pass
    else:
        return address
    try:
        ascii_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None
    return ascii_host if _HOSTNAME.fullmatch(ascii_host) else None


def safe_id(value: object) -> str:
    """Render a client- or model-supplied id as one bounded log token.

    Uses the built-in ``repr`` of ``str``, ``bytes`` and numbers, so a newline
    in the id cannot forge a second log line and no ``__repr__`` of the
    caller's object runs; any other type is shown by its name. Cuts the result
    to a fixed length so an oversized id cannot flood the log.
    """
    text = _builtin_repr(value)
    if len(text) <= _ID_MAX_CHARS:
        return text
    return f"{text[:_ID_MAX_CHARS]}…(+{len(text) - _ID_MAX_CHARS} chars)"


def _builtin_repr(value: object) -> str:
    # type() rather than isinstance: isinstance consults a __class__ the object
    # can fake, and then a built-in repr would run on the wrong type.
    kind = type(value)
    if value is None or kind is bool:
        return "None" if value is None else ("True" if value else "False")
    if issubclass(kind, str):
        return str.__repr__(value)  # type: ignore[arg-type]
    if issubclass(kind, bytes):
        return bytes.__repr__(value)  # type: ignore[arg-type]
    if issubclass(kind, int):
        return _int_repr(value)  # type: ignore[arg-type]
    if issubclass(kind, float):
        return float.__repr__(value)  # type: ignore[arg-type]
    return f"<{_class_name(kind) or 'object'}>"


def _int_repr(value: int) -> str:
    bits = int.bit_length(value)
    # Digits an int of this many bits can have, against the process's own limit
    # (sys.set_int_max_str_digits); past it, repr raises ValueError.
    limit = sys.get_int_max_str_digits()
    if bits > _MAX_ID_INT_BITS or (limit and bits * 0.30103 + 1 >= limit):
        return f"<int of {bits} bits>"
    return int.__repr__(value)


def _class_name(cls: type) -> str | None:
    """The class's qualified name when it is plain identifiers, else None.

    Whoever defines a class chooses its name (``type("sk-...\\r", ...)``), so only
    names shaped like source identifiers are logged.
    """
    name = _QUALNAME.__get__(cls)
    return name if type(name) is str and _QUALIFIED_NAME.fullmatch(name) else None


class WarnOnce:
    """Warn once per key per window, for at most ``limit`` keys per window.

    The first warning opens a window of ``window_seconds`` (an hour by
    default). Inside it each key warns once; past ``limit`` keys it logs one
    overflow WARNING and stays quiet, so a failure that hits many keys
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
        if not 0 < window_seconds < math.inf:  # also rejects nan
            raise ValueError(
                f"WarnOnce window_seconds must be positive and finite, got {window_seconds}"
            )
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
                "More than %d warnings about %s this window; further ones are not logged "
                "at WARNING "
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

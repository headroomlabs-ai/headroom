"""Redis connections for state that several Headroom replicas share.

The CCR store (``HEADROOM_CCR_BACKEND=redis``) and shared ``/v1/compress``
sessions keep their keys under ``HEADROOM_REDIS_KEY_PREFIX`` (default
``headroom``), so two deployments can share one Redis without sharing state.

Redis holds tool output and transcripts verbatim: it must be private to the
deployment, with authentication and TLS (``rediss://``). Plain ``redis://`` to a
host other than this machine is accepted with a warning.

Shared state is an optimisation of a request, never a condition for it: a Redis
error is a miss, logged when it starts and when Redis is back, and Redis is not
asked again for :data:`BACKOFF_SECONDS` after a connection failure, so an outage
costs requests nothing but the sharing.

Each :class:`Guard` registers itself in a module-level registry keyed by a short
label; :func:`guards` exposes a snapshot for ``/health`` and ``/metrics``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any, TypeVar
from urllib.parse import urlsplit

from headroom.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

PREFIX_ENV = "HEADROOM_REDIS_KEY_PREFIX"
#: Seconds Redis is skipped after a connection error.
BACKOFF_SECONDS = 5.0
#: Seconds a command may take. Commands carry small values (an entry, a few
#: messages), so a slow one is an outage, not a large transfer.
TIMEOUT_SECONDS = 1.0

T = TypeVar("T")

#: Registry of live guards, keyed by their short label (last wins). A
#: :class:`weakref.WeakValueDictionary` so a guard is dropped when its owning
#: store is closed or detached: registrations follow the store lifecycle
#: instead of outliving it.
_GUARDS: weakref.WeakValueDictionary[str, Guard] = weakref.WeakValueDictionary()


def guards() -> dict[str, Guard]:
    """Snapshot of every registered :class:`Guard`, keyed by label."""
    return dict(_GUARDS)


def unregister(guard: Guard) -> None:
    """Remove ``guard`` from the module-level registry if it is registered
    under its label. A no-op when another guard already holds that label or
    when ``guard`` was never registered."""
    label = getattr(guard, "_label", None)
    if label and _GUARDS.get(label) is guard:
        _GUARDS.pop(label, None)


def key_prefix() -> str:
    return os.environ.get(PREFIX_ENV, "").strip() or "headroom"


def connect(url: str, *, setting: str) -> Any:
    """A client for ``url``. Raises :class:`ConfigurationError`, naming
    ``setting``, when the URL is not Redis or the client is not installed.
    Does not connect: an unreachable Redis does not stop startup."""
    scheme = urlsplit(url).scheme
    if scheme not in ("redis", "rediss", "unix"):
        raise ConfigurationError(f"{setting} must be a redis://, rediss:// or unix:// URL")
    try:
        import redis
    except ImportError as exc:
        raise ConfigurationError(
            f"{setting} names Redis but the client is not installed: "
            "pip install 'headroom-ai[redis]'"
        ) from exc
    from headroom.proxy.loopback_guard import is_loopback_host

    host = urlsplit(url).hostname
    if scheme == "redis" and not is_loopback_host(host):
        logger.warning(
            "%s: plain redis:// to %s; Redis will hold tool output and transcripts, "
            "so use rediss:// with authentication on a private network",
            setting,
            host,
        )
    return redis.Redis.from_url(
        url, socket_timeout=TIMEOUT_SECONDS, socket_connect_timeout=TIMEOUT_SECONDS
    )


class Guard:
    """Runs Redis calls for one feature: an error is a miss, logged when that
    kind of call (``what``: read, write, ...) starts failing and when it works
    again; after a connection error Redis is skipped for :data:`BACKOFF_SECONDS`.
    Tracking each kind apart keeps a Redis that serves reads but refuses writes
    (``maxmemory`` with ``noeviction``) from logging on every request.

    ``label`` is a short stable key (``"ccr"``, ``"compress_session"``) used in
    the module-level registry so ``/health`` and ``/metrics`` can report
    availability without importing the ``redis`` package at module level.
    """

    def __init__(self, feature: str, label: str) -> None:
        import redis

        self._feature = feature
        self._label = label
        self._errors: tuple[type[BaseException], ...] = (redis.RedisError,)
        self._connection_errors = (redis.ConnectionError, redis.TimeoutError)
        self._lock = threading.Lock()
        self._skip_until = 0.0
        self._failing: set[str] = set()
        # ``connect`` never connects, so a configured-but-unchecked store is
        # "unknown" until ``call`` actually runs a command (success or error).
        self._observed = False
        _GUARDS[label] = self

    @property
    def state(self) -> str:
        """``"unknown"`` before any command ran, ``"available"`` when Redis
        answered its last call and no backoff is active, else
        ``"unavailable"``."""
        if not self._observed:
            return "unknown"
        if time.monotonic() >= self._skip_until and not self._failing:
            return "available"
        return "unavailable"

    @property
    def available(self) -> bool:
        """True when Redis answered its last call and no backoff is active.

        Kept as ``state == "available"``: an unobserved store reports
        ``False`` rather than advertising connectivity it never tested.
        """
        return self.state == "available"

    def close(self) -> None:
        """Unregister this guard from the module-level registry."""
        unregister(self)

    def call(self, what: str, fn: Callable[[], T], default: T) -> T:
        if time.monotonic() < self._skip_until:
            return default
        try:
            result = fn()
        except self._errors as exc:
            with self._lock:
                self._observed = True
                if isinstance(exc, self._connection_errors):
                    self._skip_until = time.monotonic() + BACKOFF_SECONDS
                if what not in self._failing:
                    self._failing.add(what)
                    logger.warning(
                        "%s: Redis %s failed (%s: %s); carrying on without shared state",
                        self._feature,
                        what,
                        type(exc).__name__,
                        _redact(str(exc)),
                    )
            return default
        if what in self._failing:
            with self._lock:
                if what in self._failing:
                    self._failing.discard(what)
                    logger.info("%s: Redis %s works again", self._feature, what)
        self._observed = True
        return result


def _redact(message: str) -> str:
    """``message`` with any URL credentials (``//user:password@``) masked."""
    return re.sub(r"//[^/@\s]*@", "//***@", message)

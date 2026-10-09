"""Logging for compression-hook failures on the proxy request path.

A hook is operator-supplied code (``ProxyConfig.hooks``). When it raises, the
handler carries on without its output, so the failure is otherwise invisible.
A broken hook usually fails on every request, so the first failure per hook
stage and exception type is a warning with its traceback; repeats go to debug.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("headroom.proxy")

_WARNED: set[tuple[str, str]] = set()
_WARNED_LOCK = threading.Lock()


def log_hook_failure(request_id: str, stage: str, err: BaseException) -> None:
    """Log one hook failure: a warning the first time, debug after that."""
    key = (stage, type(err).__name__)
    with _WARNED_LOCK:
        first = key not in _WARNED
        _WARNED.add(key)
    logger.log(
        logging.WARNING if first else logging.DEBUG,
        "[%s] %s hook failed; continuing without it: %s: %s%s",
        request_id,
        stage,
        type(err).__name__,
        err,
        "" if first else " (repeat; first occurrence logged at warning)",
        exc_info=first,
    )

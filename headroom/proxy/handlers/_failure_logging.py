"""Failure logging for the proxy request path.

Exceptions on this path can carry request content: a compression hook sees the
messages, and a catch-all handler catches whatever a compressor or router raised.
So warning and error lines name the failure (request id, provider, model, hook,
stage, exception type) without the exception text or traceback; those go to a
separate debug line, which an operator turns on explicitly. Upstream transport
errors keep their text: httpx writes it, and it is what diagnoses an outage.
"""

from __future__ import annotations

import logging
import threading

import httpx

logger = logging.getLogger("headroom.proxy")

# Hook warnings: one per (hook class, stage, exception type), so a hook failing
# on every request warns once. Bounded: cleared when full, after which a hook
# that is still failing warns again.
_MAX_WARNED = 256
_WARNED: set[tuple[str, str, str]] = set()
_WARNED_LOCK = threading.Lock()


def _hook_name(hooks: object) -> str:
    cls = type(hooks)
    return f"{cls.__module__}.{cls.__qualname__}"


def log_hook_failure(
    request_id: str,
    stage: str,
    hooks: object,
    err: BaseException,
    outcome: str = "continuing without it",
) -> None:
    """Log one compression-hook failure: a warning the first time per key, else debug.

    ``outcome`` says what the caller does next, e.g. that it skips compression.
    """
    hook = _hook_name(hooks)
    error_type = type(err).__name__
    key = (hook, stage, error_type)
    warn = False
    # Only consume the once-slot when the warning will actually be emitted.
    if logger.isEnabledFor(logging.WARNING):
        with _WARNED_LOCK:
            if key not in _WARNED:
                if len(_WARNED) >= _MAX_WARNED:
                    _WARNED.clear()
                _WARNED.add(key)
                warn = True
    if warn:
        logger.warning(
            "[%s] hook %s %s failed; %s: %s "
            "(repeats log at debug; enable debug logging for the error text)",
            request_id,
            hook,
            stage,
            outcome,
            error_type,
        )
    logger.debug("[%s] hook %s %s failed: %s", request_id, hook, stage, error_type, exc_info=err)


def log_request_failure(request_id: str, what: str, err: BaseException, **context: object) -> None:
    """Log a request-path failure at error, with its detail at debug.

    ``what`` names the operation (``"Request"``, ``"OpenAI request"``); ``context``
    adds ``key=value`` fields such as provider and model.
    """
    fields = "".join(f"{key}={value} " for key, value in context.items())
    detail = f": {err}" if isinstance(err, httpx.TransportError) else ""
    logger.error("[%s] %s failed: %s%s%s", request_id, what, fields, type(err).__name__, detail)
    logger.debug("[%s] %s failure detail", request_id, what, exc_info=err)

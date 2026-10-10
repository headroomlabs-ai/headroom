"""Failure logging for the proxy request path.

Exceptions on this path can carry request content: a compression hook sees the
messages, and a catch-all handler catches whatever a compressor or router raised.
So no line, at any level, carries an exception message or a traceback. A failure
is named by the request id, provider, model, hook, stage and
``describe_exception`` (exception types and code locations). Upstream transport
errors keep their text, which httpx writes and which diagnoses an outage, with
any URL in it redacted.
"""

from __future__ import annotations

import logging
import re

import httpx

from headroom.log_safety import WarnOnce, describe_exception, redact_url, safe_id

logger = logging.getLogger("headroom.proxy")

# One warning per (hook class, stage, exception type), so a hook failing on
# every request warns once and a different hook or stage still warns.
_HOOK_WARNINGS = WarnOnce(256, "failing compression hooks")

_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"<>]+")
_TRANSPORT_TEXT_MAX = 200


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
    first = _HOOK_WARNINGS.first((hook, stage, type(err).__name__), logger)
    logger.log(
        logging.WARNING if first else logging.DEBUG,
        "[%s] hook %s %s failed; %s: %s%s",
        request_id,
        hook,
        stage,
        outcome,
        describe_exception(err),
        " (repeats log at debug)" if first else "",
    )


def _transport_text(err: httpx.TransportError) -> str:
    text = _URL_RE.sub(lambda m: redact_url(m.group(0)), str(err))
    return text[:_TRANSPORT_TEXT_MAX]


def log_request_failure(request_id: str, what: str, err: BaseException, **context: object) -> None:
    """Log a request-path failure at error.

    ``what`` names the operation (``"Request"``, ``"OpenAI request"``); ``context``
    adds ``key=value`` fields such as provider and model.
    """
    fields = "".join(f"{key}={safe_id(value)} " for key, value in context.items())
    detail = f" ({_transport_text(err)})" if isinstance(err, httpx.TransportError) else ""
    logger.error(
        "[%s] %s failed: %s%s%s", request_id, what, fields, describe_exception(err), detail
    )

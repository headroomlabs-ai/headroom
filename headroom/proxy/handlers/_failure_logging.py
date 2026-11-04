"""Failure logging for the proxy request path.

Exceptions on this path can carry request content: a compression hook sees the
messages, and a catch-all handler catches whatever a compressor or router raised.
So no line, at any level, carries an exception message or a traceback. A failure
is named by the request id, provider, model, hook, stage and
``describe_exception`` (exception types, code locations, errno). That holds for
upstream transport errors too: an httpx ``ProxyError`` quotes the proxy's reason
phrase, which a peer controls. Messages and tracebacks appear only under the
operator's content opt-in, ``HEADROOM_DEBUG_DUMP=full``.
"""

from __future__ import annotations

import logging

from headroom.log_safety import WarnOnce, describe_class, describe_exception, safe_id

logger = logging.getLogger("headroom.proxy")

# One warning per (hook class, stage, exception type), so a hook failing on
# every request warns once and a different hook or stage still warns.
_HOOK_WARNINGS = WarnOnce(256, "failing compression hooks")


class _HookClassKey:
    """Use class identity without running plugin metaclass hashing/equality.

    Retaining the class for the bounded warning window also prevents id reuse.
    """

    def __init__(self, cls: type) -> None:
        self.cls = cls

    def __hash__(self) -> int:
        return id(self.cls)

    def __eq__(self, other: object) -> bool:
        return type(other) is _HookClassKey and self.cls is other.cls


def _hook_name(hooks: object) -> str:
    return describe_class(type(hooks))


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
    first = _HOOK_WARNINGS.first((_HookClassKey(type(hooks)), stage, type(err).__name__), logger)
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


def log_request_failure(request_id: str, what: str, err: BaseException, **context: object) -> None:
    """Log a request-path failure at error.

    ``what`` names the operation (``"Request"``, ``"OpenAI request"``); ``context``
    adds ``key=value`` fields such as provider and model.
    """
    fields = "".join(f"{key}={safe_id(value)} " for key, value in context.items())
    logger.error("[%s] %s failed: %s%s", request_id, what, fields, describe_exception(err))

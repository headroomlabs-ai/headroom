"""The ASGI wrapper for a buffered-CCR turn, shared by both provider handlers.

Server-side CCR retrieval needs the whole upstream reply in hand before it can
answer, so a ``stream: true`` turn is flipped to ``stream: false`` upstream and
resynthesized as SSE on the way out. That leaves a window — the entire
generation — where the proxy is holding a request open with nothing to say yet,
and two constraints pull in opposite directions across it:

* **Status fidelity.** Committing ``200 text/event-stream`` before the outcome
  is known destroys it. Any reply that then fails to become SSE reaches the
  client as a 200 with no ``message_start`` ("API returned an empty or
  malformed response (HTTP 200)"), and the real status goes with it, so
  client-side 429/5xx backoff never fires. This is what #2997 fixed by never
  committing early.

* **Liveness.** Sending nothing at all for the whole wait trips the client's
  *stream-idle* watchdog. That timer is separate from the total-request budget
  (``x-stainless-timeout``), and it is the one #2465 was about. #2479 fixed it
  with a keepalive preamble, which #2997 removed as collateral — putting #2465's
  condition back (#3079).

Neither property is worth trading for the other, so this keeps both:

1. For ``grace_seconds`` nothing is sent, and anything resolving inside that
   window is handed to the client untouched — real status, real headers. Fast
   failures (a 4xx, or a 429/529 that resolves once ``_retry_request`` has
   honored ``Retry-After``) land here.
2. Past the window the response is committed as SSE and a heartbeat starts, so
   a first byte always precedes any client idle watchdog.
3. A failure arriving *after* the commit can no longer carry an HTTP status, so
   it is translated into the provider's own typed SSE error instead of a generic
   one. A rate limit still reads as a rate limit, and client backoff still
   fires — the property that made early commits harmful in the first place.

Set ``grace_seconds`` to 0 or less to disable the heartbeat entirely and always
wait for full fidelity.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("headroom.proxy")

DEFAULT_BUFFERED_CCR_GRACE_SECONDS = 5.0
"""Seconds to hold out for full status fidelity before committing to SSE.

Comfortably under any client stream-idle watchdog, while still covering the
fast failures whose status matters most.
"""

_HEARTBEAT_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True)
class BufferedCCRErrorFormat:
    """The provider-specific error shapes this wrapper has to emit.

    Anthropic and OpenAI disagree on both the JSON envelope and the SSE framing,
    and the difference is pure formatting — the decision logic above is shared.
    """

    provider: str
    #: Build the pre-commit JSON body for a 502.
    json_body: Callable[[str], bytes]
    #: Build a post-commit SSE error event from an error type and message.
    sse_event: Callable[[str, str], bytes]
    #: Map an upstream HTTP status onto this provider's error type string.
    error_type_for_status: Callable[[int], str]
    #: The keepalive frame to emit while waiting, once committed.
    heartbeat: bytes


def _anthropic_error_type(status: int) -> str:
    # The wire types Anthropic documents; clients switch their retry behaviour
    # on these, so a 429 must not arrive labelled `api_error`.
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
        529: "overloaded_error",
    }.get(status, "api_error")


def _openai_error_type(status: int) -> str:
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "invalid_request_error",
        429: "rate_limit_error",
    }.get(status, "server_error")


ANTHROPIC_ERROR_FORMAT = BufferedCCRErrorFormat(
    provider="anthropic",
    json_body=lambda message: json.dumps(
        {"type": "error", "error": {"type": "api_error", "message": message}}
    ).encode(),
    sse_event=lambda error_type, message: (
        "event: error\ndata: "
        + json.dumps({"type": "error", "error": {"type": error_type, "message": message}})
        + "\n\n"
    ).encode(),
    error_type_for_status=_anthropic_error_type,
    # Anthropic's stream carries a real `ping` event type.
    heartbeat=b'event: ping\ndata: {"type":"ping"}\n\n',
)

OPENAI_ERROR_FORMAT = BufferedCCRErrorFormat(
    provider="openai",
    json_body=lambda message: json.dumps(
        {"error": {"message": message, "type": "server_error", "code": "proxy_error"}}
    ).encode(),
    sse_event=lambda error_type, message: (
        "data: " + json.dumps({"error": {"message": message, "type": error_type}}) + "\n\n"
    ).encode(),
    error_type_for_status=_openai_error_type,
    # OpenAI has no ping event; an SSE comment keeps the socket warm without
    # putting a frame the client would try to parse on the wire.
    heartbeat=b": ping\n\n",
)

_GENERIC_FAILURE_MESSAGE = "An error occurred while processing your request. Please try again."


def _upstream_message(body: bytes | None, fallback: str) -> str:
    """Prefer the upstream's own error text over a synthesized one.

    A committed stream cannot carry the upstream status, so the message is the
    only place its detail survives. Falls back whenever the body is not a
    recognizable error envelope.
    """

    if not body:
        return fallback
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return fallback
    if not isinstance(parsed, dict):
        return fallback
    error = parsed.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    message = parsed.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return fallback


async def _send_committed_failure(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    fmt: BufferedCCRErrorFormat,
    *,
    status: int,
    body: bytes | None,
) -> None:
    """Emit a typed SSE error for a failure that arrived after the commit."""

    error_type = fmt.error_type_for_status(status)
    message = _upstream_message(body, _GENERIC_FAILURE_MESSAGE)
    await send(
        {
            "type": "http.response.body",
            "body": fmt.sse_event(error_type, message),
            "more_body": False,
        }
    )


async def _forward_after_commit(
    result: Any,
    send: Callable[[dict[str, Any]], Awaitable[None]],
    fmt: BufferedCCRErrorFormat,
    *,
    request_id: str,
) -> None:
    """Relay a resolved result once SSE headers are already on the wire."""

    status = int(getattr(result, "status_code", 200) or 200)
    body_iterator = getattr(result, "body_iterator", None)

    if body_iterator is not None:
        # Streaming results are already SSE — both the success path and the
        # 502 CCR-failure paths, whose bodies are error events. Forwarding the
        # bytes keeps whatever detail they carry.
        async for chunk in body_iterator:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        return

    # A non-streaming result here is an upstream reply that never became SSE.
    # Its status cannot reach the client anymore, so preserve its meaning in
    # the error type instead of degrading to a generic failure (#3079).
    body = getattr(result, "body", None)
    if status != 200:
        logger.warning(
            f"[{request_id}] CCR: buffered upstream returned {status} after the response "
            "was committed as SSE; relaying it as a typed stream error"
        )
    await _send_committed_failure(send, fmt, status=status, body=body)


def buffered_ccr_asgi_call(
    *,
    operation: asyncio.Task,
    fmt: BufferedCCRErrorFormat,
    grace_seconds: float,
    record_failed: Callable[..., Awaitable[None]],
    request_id: str,
) -> Callable[[Any, Any, Any], Awaitable[None]]:
    """Build the ``__call__`` body for a buffered-CCR ASGI response.

    Returned rather than subclassed so both handlers can keep their existing
    ``Response`` shells and differ only in the error format they pass in.
    """

    async def __call__(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401
        loop = asyncio.get_running_loop()
        committed = False
        deadline = loop.time() + grace_seconds
        heartbeat_enabled = grace_seconds > 0
        try:
            while True:
                if heartbeat_enabled:
                    timeout = (
                        _HEARTBEAT_INTERVAL_SECONDS
                        if committed
                        else max(0.0, deadline - loop.time())
                    )
                else:
                    timeout = None
                done, _pending = await asyncio.wait({operation}, timeout=timeout)

                if done:
                    try:
                        result = operation.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        await record_failed(provider=fmt.provider)
                        logger.error(f"[{request_id}] Request failed: {type(exc).__name__}: {exc}")
                        if committed:
                            await _send_committed_failure(send, fmt, status=502, body=None)
                            return
                        await send(
                            {
                                "type": "http.response.start",
                                "status": 502,
                                "headers": [(b"content-type", b"application/json")],
                            }
                        )
                        await send(
                            {
                                "type": "http.response.body",
                                "body": fmt.json_body(_GENERIC_FAILURE_MESSAGE),
                                "more_body": False,
                            }
                        )
                        return

                    if not committed:
                        # Nothing is on the wire yet, so the result keeps its
                        # own status and headers. This is the fidelity #2997
                        # was protecting.
                        await result(scope, receive, send)
                        return

                    await _forward_after_commit(result, send, fmt, request_id=request_id)
                    return

                if not committed:
                    # The grace window expired without an outcome. Commit now so
                    # a first byte beats the client's stream-idle watchdog.
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 200,
                            "headers": [(b"content-type", b"text/event-stream")],
                        }
                    )
                    committed = True
                    logger.debug(
                        f"[{request_id}] CCR: buffered turn exceeded the {grace_seconds}s "
                        "grace window; committing SSE and starting the heartbeat"
                    )
                await send({"type": "http.response.body", "body": fmt.heartbeat, "more_body": True})
        except asyncio.CancelledError:
            raise
        finally:
            if not operation.done():
                operation.cancel()
            try:
                await operation
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    return __call__

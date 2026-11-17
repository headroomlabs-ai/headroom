"""Request-body ceiling enforced as a raw ASGI layer.

The proxy already bounds request bodies inside its handlers: every route that
reads JSON goes through ``helpers._read_request_body_bytes``, which refuses
anything over :data:`headroom.proxy.helpers.MAX_REQUEST_BODY_SIZE` while it is
still streaming in. That protects the handlers, and only the handlers. Anything
that reads the body *before* a handler runs, which in practice means an ASGI
middleware installed by a proxy extension, never sees that check. An extension
that buffers the body to inspect or rewrite it (a router that reads
``service_tier``, a redactor, a relay) therefore holds an unbounded body for
as long as the client cares to send one.

This module makes the ceiling a property of the ASGI stack instead of a
convention every extension has to know about. :class:`RequestBodyLimitMiddleware`
is registered by :func:`headroom.proxy.server.create_app` immediately inside
the inbound security gate and *outside* every extension, so the guarantee an
extension author gets is:

* an unauthenticated request never reaches extension code (the gate), and
* an authenticated request whose body exceeds the ceiling is refused with the
  same status and payload the handlers use, before any extension buffers it.

Enforcement is the same two-step the handlers use: a declared
``Content-Length`` above the ceiling is refused before the body is read at all,
and the body stream itself is metered so an absent or understated
``Content-Length`` (chunked transfer) is bounded too. Once the stream trips the
ceiling the client gets the refusal, the application's own attempt to respond
is discarded, and :class:`~headroom.proxy.helpers.RequestBodyTooLarge` is
raised into whatever was reading the stream so it stops.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from headroom.proxy import helpers as _helpers

logger = logging.getLogger("headroom.proxy")

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

# Paths whose clients speak the Anthropic error dialect. Everything else gets
# the OpenAI-style payload, which is what the handlers' own 413s use.
_ANTHROPIC_PATH_MARKERS = ("/v1/messages",)

# The AWS Bedrock InvokeModel passthrough speaks a third dialect: its handler
# (``handle_bedrock_invoke``) answers a too-large body with
# ``{"error": {"type": "request_too_large", "message": ...}}``. Its routes are
# ``/model/{model_id:path}/invoke`` and
# ``/model/{model_id:path}/invoke-with-response-stream``
# (``headroom/providers/proxy_routes.py``), and model ids may contain slashes
# (ARN-style inference profiles), so the match is a suffix check under the
# ``/model/`` prefix rather than a fixed-path table.
_BEDROCK_INVOKE_PREFIX = "/model/"
_BEDROCK_INVOKE_SUFFIXES = ("/invoke", "/invoke-with-response-stream")


def _is_bedrock_invoke_path(path: str) -> bool:
    return path.startswith(_BEDROCK_INVOKE_PREFIX) and path.endswith(_BEDROCK_INVOKE_SUFFIXES)


def _too_large_payload(path: str, limit: int) -> bytes:
    """Mirror the handlers' own 413 payloads so clients see one shape per dialect."""
    message = f"Request body too large. Maximum size is {limit // (1024 * 1024)}MB"
    if any(marker in path for marker in _ANTHROPIC_PATH_MARKERS):
        payload: dict[str, Any] = {
            "type": "error",
            "error": {"type": "request_too_large", "message": message},
        }
    elif _is_bedrock_invoke_path(path):
        payload = {
            "error": {
                "type": "request_too_large",
                "message": message,
            }
        }
    else:
        payload = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "request_too_large",
            }
        }
    return json.dumps(payload).encode("utf-8")


def _declared_content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers") or ():
        if name.lower() == b"content-length":
            try:
                return int(value.decode("latin-1").strip())
            except ValueError:
                return None
    return None


class RequestBodyLimitMiddleware:
    """Refuse HTTP request bodies above the proxy's ceiling before anything buffers them.

    ``max_bytes`` defaults to :data:`headroom.proxy.helpers.MAX_REQUEST_BODY_SIZE`,
    resolved per request so a test (or a future policy knob) that adjusts the
    module constant is honoured without rebuilding the app. Non-HTTP scopes
    (WebSocket, lifespan) pass straight through: WebSocket frames have their own
    limits in the relay handlers.
    """

    def __init__(self, app: Any, *, max_bytes: int | None = None) -> None:
        self.app = app
        self._max_bytes = max_bytes

    @property
    def max_bytes(self) -> int:
        if self._max_bytes is not None:
            return self._max_bytes
        return int(_helpers.MAX_REQUEST_BODY_SIZE)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        limit = self.max_bytes
        path = str(scope.get("path") or "")

        declared = _declared_content_length(scope)
        if declared is not None and declared > limit:
            logger.warning(
                "event=request_body_rejected path=%s reason=content_length declared=%d limit=%d",
                path,
                declared,
                limit,
            )
            await self._reject(send, path, limit)
            return

        # ``received`` meters the stream; ``responded`` records that *we* have
        # answered, after which anything the application tries to send is
        # discarded (it is reacting to a body we cut off); ``started`` records
        # that the application answered first, in which case it is too late
        # for us to substitute a refusal and we only stop feeding the body.
        state = {"received": 0, "responded": False, "started": False}

        async def metered_receive() -> MutableMapping[str, Any]:
            message = await receive()
            if message.get("type") == "http.request":
                state["received"] += len(message.get("body") or b"")
                if state["received"] > limit:
                    logger.warning(
                        "event=request_body_rejected path=%s reason=stream received>%d limit=%d",
                        path,
                        limit,
                        limit,
                    )
                    if not state["started"] and not state["responded"]:
                        state["responded"] = True
                        await self._reject(send, path, limit)
                    raise _helpers.RequestBodyTooLarge(
                        f"Request body exceeds {limit // (1024 * 1024)}MB"
                    )
            return message

        async def guarded_send(message: MutableMapping[str, Any]) -> None:
            if state["responded"]:
                return
            if message.get("type") == "http.response.start":
                state["started"] = True
            await send(message)

        try:
            await self.app(scope, metered_receive, guarded_send)
        except _helpers.RequestBodyTooLarge:
            # Raised by metered_receive above, or by a handler's own bounded
            # read that nothing caught. Either way the answer is the same 413;
            # if the application already started a response there is nothing
            # left to say on the wire.
            if state["responded"] or state["started"]:
                return
            state["responded"] = True
            await self._reject(send, path, limit)

    @staticmethod
    async def _reject(send: Send, path: str, limit: int) -> None:
        body = _too_large_payload(path, limit)
        await send(
            {
                "type": "http.response.start",
                "status": _helpers.get_body_too_large_status(),
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


__all__ = ["RequestBodyLimitMiddleware"]

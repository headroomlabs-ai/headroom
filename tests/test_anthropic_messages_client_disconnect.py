"""``handle_anthropic_messages`` must not crash when the client disconnects
mid-body-read.

Companion to the passthrough-path regression coverage in
``tests/test_proxy_handler_helpers.py`` (``test_handle_passthrough_client_disconnect``
/ ``test_handle_streaming_passthrough_client_disconnect``, added in #2033/#2067).
Those two guard the three named passthrough paths. They do not cover
``read_request_json_with_bytes`` / ``_read_request_body_bytes`` in
``headroom/proxy/helpers.py`` — the shared reader documented as used by the
Anthropic, OpenAI, and Bedrock handlers — so the main ``/v1/messages`` handler
had no guard at all: a real client disconnect (an interrupted turn, a
client-side restart) crashed the request as an unhandled 500 instead of a
quiet 204.

Reproduces the real-world trigger at the ASGI level: ``receive()`` returns
``{"type": "http.disconnect"}``, which is exactly what Starlette's own
``Request.body()`` checks for before raising ``ClientDisconnect`` — not a
mocked exception, the actual protocol-level signal.
"""

from __future__ import annotations

import asyncio

from fastapi import Request

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin


class _MinimalAnthropicHandler(AnthropicHandlerMixin):
    """The smallest handler that can reach the ClientDisconnect guard.

    The guard sits before any of the heavier machinery (compression,
    memory, CCR, upstream dispatch) executes, so none of that needs
    stubbing here — only ``_next_request_id``, which the real
    implementation backs with an ``asyncio.Lock`` + counter this
    minimal handler never initializes.
    """

    async def _next_request_id(self) -> str:
        return "req-test"


def _build_disconnecting_request() -> Request:
    """A real ``Request`` whose ASGI receive channel signals disconnect
    before any body bytes arrive — the actual protocol-level trigger,
    not a mocked ``.body()`` override."""

    async def receive():
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer sk-ant-api-test"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


def test_handle_anthropic_messages_client_disconnect_returns_204():
    """Before the fix this raised an unhandled starlette.requests.ClientDisconnect
    (observed live in production as a 500 traceback through
    handle_anthropic_messages -> read_request_json_with_bytes ->
    _read_request_body_bytes -> request.body()). After the fix it returns a
    quiet 204, matching the existing guard on the sibling batch-passthrough
    handler a few hundred lines below in the same file."""
    handler = _MinimalAnthropicHandler()
    response = asyncio.run(handler.handle_anthropic_messages(_build_disconnecting_request()))
    assert response.status_code == 204

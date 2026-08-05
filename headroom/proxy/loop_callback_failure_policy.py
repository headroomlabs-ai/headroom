"""Classification policy for event-loop callback failures."""

from __future__ import annotations

from typing import Any

KNOWN_WEBSOCKET_CALLBACK_MESSAGE = (
    "Exception in callback Connection.connection_lost(ConnectionResetError())"
)
KNOWN_WEBSOCKET_CALLBACK_EXCEPTION = "'ClientConnection' object has no attribute 'recv_messages'"


def is_known_websocket_callback_failure(context: dict[str, Any]) -> bool:
    """Return True iff this exact websockets callback failure shape is observed."""
    if context.get("message") != KNOWN_WEBSOCKET_CALLBACK_MESSAGE:
        return False
    exception = context.get("exception")
    return isinstance(exception, AttributeError) and str(exception) == (
        KNOWN_WEBSOCKET_CALLBACK_EXCEPTION
    )

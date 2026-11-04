"""Shared WebSocket handshake header policy."""

WS_HOP_BY_HOP_HEADERS = frozenset(
    {
        "host",
        "connection",
        "upgrade",
        "sec-websocket-key",
        "sec-websocket-version",
        "sec-websocket-extensions",
        "sec-websocket-accept",
        "sec-websocket-protocol",
        "content-length",
        "transfer-encoding",
    }
)

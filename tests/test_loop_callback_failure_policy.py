from __future__ import annotations

from headroom.proxy.loop_callback_failure_policy import (
    KNOWN_WEBSOCKET_CALLBACK_EXCEPTION,
    KNOWN_WEBSOCKET_CALLBACK_MESSAGE,
    is_known_websocket_callback_failure,
)


def test_known_websocket_callback_failure_matches_exact_shape() -> None:
    assert is_known_websocket_callback_failure(
        {
            "message": KNOWN_WEBSOCKET_CALLBACK_MESSAGE,
            "exception": AttributeError(KNOWN_WEBSOCKET_CALLBACK_EXCEPTION),
        }
    )


def test_known_websocket_callback_failure_rejects_other_message() -> None:
    assert not is_known_websocket_callback_failure(
        {
            "message": "Exception in callback something_else",
            "exception": AttributeError(KNOWN_WEBSOCKET_CALLBACK_EXCEPTION),
        }
    )


def test_known_websocket_callback_failure_rejects_other_exception() -> None:
    assert not is_known_websocket_callback_failure(
        {
            "message": KNOWN_WEBSOCKET_CALLBACK_MESSAGE,
            "exception": AttributeError("different attribute"),
        }
    )
    assert not is_known_websocket_callback_failure(
        {
            "message": KNOWN_WEBSOCKET_CALLBACK_MESSAGE,
            "exception": RuntimeError(KNOWN_WEBSOCKET_CALLBACK_EXCEPTION),
        }
    )

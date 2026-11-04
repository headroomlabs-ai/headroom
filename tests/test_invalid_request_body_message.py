"""Client-facing 400 text for unreadable request bodies (CodeQL py/stack-trace-exposure).

The handlers used to echo ``str(exc)`` back to the client. These pin that the
replacement still tells the client *what kind* of problem it was, and never
carries the exception's own text.
"""

import json

import pytest

from headroom.proxy.helpers import (
    RequestBodyNotObject,
    RequestBodyTooLarge,
    invalid_request_body_message,
)


def _unicode_error() -> ValueError:
    try:
        b"\xff".decode("utf-8")
    except UnicodeDecodeError as cause:
        try:
            raise ValueError(f"Request body is not valid UTF-8: {cause}") from cause
        except ValueError as exc:
            return exc
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RequestBodyTooLarge("Request body exceeds 100MB (Content-Length: 1)"), "too large"),
        (RequestBodyNotObject("Request body must be a JSON object, not list"), "JSON object"),
        (json.JSONDecodeError("Expecting value", "secret-doc", 0), "malformed JSON"),
        (_unicode_error(), "not valid UTF-8"),
        (ValueError("zstd frame header at offset 3"), None),
    ],
)
def test_message_names_the_category_without_exception_text(exc, expected):
    message = invalid_request_body_message(exc)

    assert message.startswith("Invalid request body")
    if expected is None:
        assert message == "Invalid request body"
    else:
        assert expected in message
    assert str(exc) not in message
    assert "secret-doc" not in message


def test_not_object_is_still_a_value_error():
    # Existing ``except (json.JSONDecodeError, ValueError)`` call sites must keep
    # catching it.
    assert issubclass(RequestBodyNotObject, ValueError)

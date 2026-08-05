"""Error classification ordering: specific categories must not be shadowed by
the generic RUNTIME_ERROR catch-all."""

from __future__ import annotations

from headroom.learn._shared import classify_error
from headroom.learn.models import ErrorCategory


def test_timeout_repr_is_not_shadowed_by_runtime_error() -> None:
    # "TimeoutError: ..." contains "Error:", which the generic RUNTIME_ERROR
    # pattern also matches; TIMEOUT must still win.
    assert classify_error("TimeoutError: timed out after 30s") == ErrorCategory.TIMEOUT
    assert classify_error("operation timed out") == ErrorCategory.TIMEOUT


def test_connection_repr_is_not_shadowed_by_runtime_error() -> None:
    assert (
        classify_error("ConnectionError: [Errno 111] Connection refused")
        == ErrorCategory.CONNECTION_ERROR
    )
    assert classify_error("ECONNREFUSED") == ErrorCategory.CONNECTION_ERROR


def test_generic_error_still_classifies_as_runtime() -> None:
    # A plain exception repr with no more-specific token stays RUNTIME_ERROR
    # (matches the opencode scanner's expectation).
    assert classify_error("Error: command failed with exit code 1") == ErrorCategory.RUNTIME_ERROR
    assert classify_error("Traceback (most recent call last):") == ErrorCategory.RUNTIME_ERROR


def test_non_error_text_is_unknown() -> None:
    assert classify_error("all good, tests passed") == ErrorCategory.UNKNOWN

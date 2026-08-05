from __future__ import annotations

import pytest

from headroom.proxy.request_limit_policy import (
    BODY_TOO_LARGE_STATUS_DEFAULT,
    SSE_EVENT_MAX_BYTES_DEFAULT,
    resolve_body_too_large_status,
    resolve_sse_event_max_bytes,
)


def test_resolve_sse_event_max_bytes_uses_default_for_missing_value() -> None:
    assert resolve_sse_event_max_bytes(None) == SSE_EVENT_MAX_BYTES_DEFAULT
    assert resolve_sse_event_max_bytes("") == SSE_EVENT_MAX_BYTES_DEFAULT


def test_resolve_sse_event_max_bytes_accepts_positive_integer() -> None:
    assert resolve_sse_event_max_bytes("2048") == 2048


@pytest.mark.parametrize("raw", ["0", "-1", "not-int"])
def test_resolve_sse_event_max_bytes_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        resolve_sse_event_max_bytes(raw)


def test_resolve_body_too_large_status_uses_default_for_missing_value() -> None:
    assert resolve_body_too_large_status(None) == BODY_TOO_LARGE_STATUS_DEFAULT
    assert resolve_body_too_large_status("") == BODY_TOO_LARGE_STATUS_DEFAULT


def test_resolve_body_too_large_status_accepts_4xx_or_5xx_status() -> None:
    assert resolve_body_too_large_status("413") == 413
    assert resolve_body_too_large_status("529") == 529


@pytest.mark.parametrize("raw", ["399", "600", "not-int"])
def test_resolve_body_too_large_status_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        resolve_body_too_large_status(raw)

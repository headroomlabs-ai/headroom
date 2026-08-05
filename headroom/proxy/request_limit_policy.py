"""Validation policy for proxy request and stream limits."""

from __future__ import annotations

SSE_EVENT_MAX_BYTES_ENV = "HEADROOM_SSE_BUFFER_MAX_BYTES"
SSE_EVENT_MAX_BYTES_DEFAULT = 1 * 1024 * 1024

BODY_TOO_LARGE_STATUS_ENV = "HEADROOM_PROXY_BODY_TOO_LARGE_STATUS"
BODY_TOO_LARGE_STATUS_DEFAULT = 413


def resolve_sse_event_max_bytes(raw: str | None) -> int:
    """Resolve the per-event SSE size cap from an optional env string."""
    if raw is None or raw == "":
        return SSE_EVENT_MAX_BYTES_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{SSE_EVENT_MAX_BYTES_ENV} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{SSE_EVENT_MAX_BYTES_ENV} must be positive, got {value}")
    return value


def resolve_body_too_large_status(raw: str | None) -> int:
    """Resolve the HTTP status code for body-too-large rejections."""
    if raw is None or raw == "":
        return BODY_TOO_LARGE_STATUS_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{BODY_TOO_LARGE_STATUS_ENV} must be an integer, got {raw!r}") from exc
    if not 400 <= value < 600:
        raise ValueError(f"{BODY_TOO_LARGE_STATUS_ENV} must be a 4xx/5xx status, got {value}")
    return value

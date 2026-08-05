"""Secret redaction policy for opt-in proxy wire debug capture."""

from __future__ import annotations

from typing import Any

WIRE_DEBUG_REDACTED = "[REDACTED]"
WIRE_DEBUG_SECRET_KEYS = (
    "authorization",
    "cookie",
    "set-cookie",
    "api-key",
    "x-api-key",
    "openai-api-key",
    "anthropic-api-key",
    "access_token",
    "refresh_token",
    "id_token",
    "bearer",
    "password",
    "secret",
    "token",
    "credential",
)


def should_redact_key(key: str) -> bool:
    """Return whether a wire-debug field name should be redacted."""
    normalized = key.lower().replace("-", "_")
    if normalized in {marker.replace("-", "_") for marker in WIRE_DEBUG_SECRET_KEYS}:
        return True
    return (
        normalized.endswith("_api_key")
        or normalized.endswith("_secret")
        or normalized.endswith("_password")
        or normalized.endswith("_access_token")
        or normalized.endswith("_refresh_token")
    )


def redact_for_wire_debug(value: Any) -> Any:
    """Redact obvious secrets while preserving request/response shape."""
    if isinstance(value, dict):
        return {
            key: (
                WIRE_DEBUG_REDACTED if should_redact_key(str(key)) else redact_for_wire_debug(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_for_wire_debug(item) for item in value]
    return value

"""Secret redaction policy for opt-in proxy wire debug capture."""

from __future__ import annotations

from typing import Any

WIRE_DEBUG_REDACTED = "[REDACTED]"
WIRE_DEBUG_SECRET_KEYS = (
    "authorization",
    # A proxy sees Proxy-Authorization on the hop it makes itself; it carries
    # the same kind of credential as Authorization.
    "proxy-authorization",
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
    "secret_key",
    "token",
    "credential",
    "credentials",
)

# Suffixes that make a prefixed key a secret. ``_token`` covers the auth,
# session, api, id and security token names providers use, and does not touch
# the usage counters (``max_tokens``, ``input_tokens``, ...), which are plural.
_SECRET_KEY_SUFFIXES = (
    "_api_key",
    "_secret",
    "_secret_key",
    "_password",
    "_token",
    "_credential",
    "_credentials",
)


def should_redact_key(key: str) -> bool:
    """Return whether a wire-debug field name should be redacted."""
    normalized = key.lower().replace("-", "_")
    if normalized in {marker.replace("-", "_") for marker in WIRE_DEBUG_SECRET_KEYS}:
        return True
    return normalized.endswith(_SECRET_KEY_SUFFIXES)


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

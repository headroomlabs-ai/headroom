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
    "apikey",
    "x-api-key",
    "openai-api-key",
    "anthropic-api-key",
    # Codex's signed device attestation, sent with every request.
    "x-oai-attestation",
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
    # Any ``*-key`` header carries a key: Ocp-Apim-Subscription-Key,
    # X-Functions-Key, x-goog-api-key.
    "_key",
)

# A name with any of these as a ``_``-separated word is a secret wherever the
# word sits: X-Internal-Ops-Key, X-Auth-Token-Id, X-Amz-Signature, session ids.
# Word match, not substring, so the usage counters (``max_tokens``,
# ``input_tokens``) and ``author`` stay readable.
_SECRET_KEY_WORDS = frozenset(
    {
        "auth",
        "key",
        "apikey",
        "secret",
        "token",
        "password",
        "passwd",
        "credential",
        "credentials",
        "session",
        "signature",
        "sig",
        "cookie",
    }
)

# Substrings that mark a secret even inside a run-together name
# (``xauthtoken``, ``clientsecret``, ``x-apikey``).
_SECRET_KEY_SUBSTRINGS = (
    "authorization",
    "apikey",
    "authtoken",
    "accesstoken",
    "secret",
    "password",
    "passwd",
    "credential",
    "signature",
)

# Identity headers set by a fronting platform: Easy Auth's encoded principal
# and the mTLS client certificate. Headroom's own x-headroom-* identity headers
# stay readable.
_SECRET_EXACT_KEYS = frozenset({"x_ms_client_principal", "x_forwarded_client_cert"})

# Usage counters (``token_count``, ``max_tokens``) are numbers, not secrets,
# ``key_id`` names a key without being one, and ``auth_mode`` /
# ``is_chatgpt_auth`` are the routing enum and flag the Codex captures record.
_COUNTER_SUFFIXES = ("_count", "_tokens")
_READABLE_KEYS = frozenset({"key_id", "auth_mode", "is_chatgpt_auth"})


def should_redact_key(key: str) -> bool:
    """Return whether a wire-debug field name should be redacted."""
    normalized = key.lower().replace("-", "_")
    if normalized in {marker.replace("-", "_") for marker in WIRE_DEBUG_SECRET_KEYS}:
        return True
    if normalized in _SECRET_EXACT_KEYS or normalized.endswith(_SECRET_KEY_SUFFIXES):
        return True
    if normalized in _READABLE_KEYS or normalized.endswith(_COUNTER_SUFFIXES):
        return False
    if any(marker in normalized for marker in _SECRET_KEY_SUBSTRINGS):
        return True
    return any(word in _SECRET_KEY_WORDS for word in normalized.split("_"))


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

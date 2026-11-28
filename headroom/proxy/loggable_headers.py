"""Allow-list of headers that may reach a log line or a debug file.

Any header not named here is dropped, not redacted: a new or unknown header
never reaches a log, whatever its name. The list is a code constant on
purpose; it is not configurable. What is kept still goes through the secret
redaction policy as a second layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from headroom.proxy.wire_debug_redaction_policy import redact_for_wire_debug

LOGGABLE_HEADER_NAMES: frozenset[str] = frozenset(
    {
        "host",
        "user-agent",
        "accept",
        "accept-encoding",
        "content-type",
        "content-length",
        "anthropic-version",
        "anthropic-beta",
        "x-app",
        "x-claude-code-session-id",
        "x-request-id",
        "x-client-request-id",
        "x-forwarded-for",
        "x-envoy-external-address",
        "x-headroom-user",
        "x-headroom-principal",
        "x-headroom-app",
        # Anthropic/OpenAI SDK metadata, named one by one: no prefix match.
        "x-stainless-arch",
        "x-stainless-lang",
        "x-stainless-os",
        "x-stainless-package-version",
        "x-stainless-retry-count",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
        "x-stainless-timeout",
        # Upstream response headers the wire capture needs to explain retries
        # and rate limits.
        "retry-after",
        "retry-after-ms",
        "request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "anthropic-ratelimit-requests-limit",
        "anthropic-ratelimit-requests-remaining",
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-tokens-limit",
        "anthropic-ratelimit-tokens-remaining",
        "anthropic-ratelimit-tokens-reset",
    }
)

_AUTH_HEADER_NAMES = frozenset({"authorization", "x-api-key"})


def loggable_headers(headers: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only the allow-listed ``headers``, plus two non-secret summary fields.

    ``auth_present`` says whether ``authorization`` or ``x-api-key`` was sent;
    ``dropped_headers`` counts the headers left out, without naming them.
    """
    kept: dict[str, Any] = {}
    dropped = 0
    auth_present = False
    for name, value in (headers or {}).items():
        lower = str(name).lower()
        if lower in _AUTH_HEADER_NAMES:
            auth_present = True
        if lower in LOGGABLE_HEADER_NAMES:
            kept[str(name)] = value
        else:
            dropped += 1
    safe: dict[str, Any] = redact_for_wire_debug(kept)
    safe["auth_present"] = auth_present
    safe["dropped_headers"] = dropped
    return safe

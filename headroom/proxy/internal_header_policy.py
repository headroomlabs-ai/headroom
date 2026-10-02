"""Policy for stripping proxy-internal request headers before upstream calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

INTERNAL_HEADER_PREFIX = "x-headroom-"
STRIP_INTERNAL_HEADERS_ENV = "HEADROOM_STRIP_INTERNAL_HEADERS"
StripInternalHeadersMode = Literal["enabled", "disabled"]
STRIP_INTERNAL_HEADERS_DEFAULT: StripInternalHeadersMode = "enabled"

# The proxy's own credential header. Read by the security gate and the
# WebSocket auth middleware; it must never be copied into tags, logs, metrics
# labels or exports, which is what ``is_credential_header`` guards.
PROXY_TOKEN_HEADER = "x-headroom-proxy-token"

# Name components that mark an ``x-headroom-*`` header (or the tag key derived
# from it by stripping the prefix) as credential material rather than a label.
# Names are lower-cased, ``_`` is folded to ``-`` and the name is split on
# ``-``, so ``X-Headroom-License_Token`` -> ``license-token`` -> {license, token}.
# Whole components only: ``proxy-token`` is a credential, the numeric
# ``tool-search-deferred-tokens`` tag is not.
CREDENTIAL_NAME_COMPONENTS: frozenset[str] = frozenset(
    {
        "token",
        "secret",
        "password",
        "passwd",
        "credential",
        "credentials",
        "authorization",
        "apikey",
        "key",  # x-headroom-key (Headroom Cloud), x-headroom-api-key
        "bearer",
        "cookie",
        "signature",
    }
)


def _name_components(name: str) -> list[str]:
    return [part for part in name.strip().lower().replace("_", "-").split("-") if part]


def is_credential_tag_key(key: str) -> bool:
    """True if a tag key (header name minus the ``x-headroom-`` prefix) is a credential.

    Tag keys become request-log fields, dashboard facets and export labels,
    so this is the single rule every tag producer and consumer applies.
    """
    parts = _name_components(key)
    if parts[:2] == ["x", "headroom"]:
        parts = parts[2:]
    return any(part in CREDENTIAL_NAME_COMPONENTS for part in parts)


def _normalise_name(name: str) -> str:
    return "-".join(_name_components(name))


def is_credential_header(name: str) -> bool:
    """True if an inbound header carries a credential and must not be tagged."""
    normalised = _normalise_name(name)
    if normalised == PROXY_TOKEN_HEADER:
        return True
    if normalised in ("authorization", "proxy-authorization", "cookie", "api-key", "x-api-key"):
        return True
    return normalised.startswith(INTERNAL_HEADER_PREFIX) and is_credential_tag_key(normalised)


def resolve_strip_internal_headers_mode(raw: str | None) -> StripInternalHeadersMode:
    """Resolve the configured internal-header strip mode."""

    normalized = (raw or "").strip().lower()
    if not normalized:
        return STRIP_INTERNAL_HEADERS_DEFAULT
    if normalized in ("enabled", "disabled"):
        return cast(StripInternalHeadersMode, normalized)
    raise ValueError(
        f"Invalid {STRIP_INTERNAL_HEADERS_ENV}={normalized!r}; expected 'enabled' or 'disabled'"
    )


def strip_internal_headers(
    headers: Mapping[str, str],
    *,
    mode: StripInternalHeadersMode,
) -> dict[str, str]:
    """Return a copy of headers with internal x-headroom-* request headers removed."""

    if mode == "disabled":
        return dict(headers)
    return {
        key: value
        for key, value in headers.items()
        if not key.lower().startswith(INTERNAL_HEADER_PREFIX)
    }

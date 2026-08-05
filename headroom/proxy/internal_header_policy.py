"""Policy for stripping proxy-internal request headers before upstream calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

INTERNAL_HEADER_PREFIX = "x-headroom-"
STRIP_INTERNAL_HEADERS_ENV = "HEADROOM_STRIP_INTERNAL_HEADERS"
StripInternalHeadersMode = Literal["enabled", "disabled"]
STRIP_INTERNAL_HEADERS_DEFAULT: StripInternalHeadersMode = "enabled"


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

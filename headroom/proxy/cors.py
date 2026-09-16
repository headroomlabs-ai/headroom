"""CORS policy helpers for the local proxy."""

from __future__ import annotations

import os
from typing import Protocol

CORS_ORIGINS_ENV = "HEADROOM_CORS_ORIGINS"
DEFAULT_LOOPBACK_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"


class CorsProxyConfig(Protocol):
    """Minimal config protocol for CORS origin resolution."""

    port: int


def cors_origins_for_config(config: CorsProxyConfig) -> list[str]:
    """Resolve CORS origins for the proxy.

    A wildcard CORS policy lets arbitrary browser pages read local proxy
    content endpoints, so ``*`` is only honored when explicitly set through
    ``HEADROOM_CORS_ORIGINS``. Without an override, origins are handled by
    ``cors_origin_regex_for_config`` so loopback clients work on any proxy port.
    """

    configured = os.environ.get(CORS_ORIGINS_ENV, "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return []


def cors_origin_regex_for_config(config: CorsProxyConfig) -> str | None:
    """Resolve the default loopback regex when no explicit CORS list is set."""

    if os.environ.get(CORS_ORIGINS_ENV, "").strip():
        return None
    return DEFAULT_LOOPBACK_ORIGIN_REGEX

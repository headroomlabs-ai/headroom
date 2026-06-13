"""CORS policy helpers for the local proxy."""

from __future__ import annotations

import os
from typing import Protocol

CORS_ORIGINS_ENV = "HEADROOM_CORS_ORIGINS"


class CorsProxyConfig(Protocol):
    """Minimal config protocol for CORS origin resolution."""

    port: int


def cors_origins_for_config(config: CorsProxyConfig) -> list[str]:
    """Resolve CORS origins for the proxy.

    Default to the effective localhost port from config. A wildcard CORS policy
    lets arbitrary browser pages read local proxy content endpoints, so ``*`` is
    only honored when explicitly set through ``HEADROOM_CORS_ORIGINS``.
    """

    configured = os.environ.get(CORS_ORIGINS_ENV, "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [f"http://127.0.0.1:{config.port}", f"http://localhost:{config.port}"]

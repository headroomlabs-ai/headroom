"""Grok Build install-time helpers."""

from __future__ import annotations

from .runtime import build_proxy_targets


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Grok Build."""
    del backend
    target = build_proxy_targets(port)
    return {
        "GROK_MODEL_GROK_BUILD_BASE_URL": target.base_url,
    }
"""Grok install-time helpers."""

from __future__ import annotations

from .runtime import PROXY_ENV_KEY, proxy_base_url


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Grok CLI."""
    del backend
    return {PROXY_ENV_KEY: proxy_base_url(port)}

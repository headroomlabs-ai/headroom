"""Grok install-time helpers."""

from __future__ import annotations

from .runtime import GROK_PROXY_ENV, proxy_base_url


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Grok Build."""
    # Accepted for the shared install-provider interface; Grok only needs the proxy URL.
    _ = backend
    return {GROK_PROXY_ENV: proxy_base_url(port)}

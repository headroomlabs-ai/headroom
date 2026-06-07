"""Hermes install-time helpers."""

from __future__ import annotations

from .runtime import OPENAI_BASE_ENV, proxy_base_url


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build persistent install env for OpenAI clients using Hermes upstream."""
    _ = backend
    return {OPENAI_BASE_ENV: proxy_base_url(port)}
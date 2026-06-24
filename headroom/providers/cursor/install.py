"""Cursor install-time helpers."""

from __future__ import annotations

from .runtime import build_proxy_targets


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Cursor."""
    del backend
    targets = build_proxy_targets(port)
    return {
        "OPENAI_BASE_URL": targets.openai_base_url,
        "ANTHROPIC_BASE_URL": targets.anthropic_base_url,
    }

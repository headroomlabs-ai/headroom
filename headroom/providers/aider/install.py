"""Aider install-time helpers."""

from __future__ import annotations

from .runtime import build_launch_env


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Aider."""
    del backend
    env, _lines = build_launch_env(port=port, environ={})
    return {key: env[key] for key in ("OPENAI_API_BASE", "ANTHROPIC_BASE_URL")}

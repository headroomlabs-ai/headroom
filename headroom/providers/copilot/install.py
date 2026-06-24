"""Copilot install-time helpers."""

from __future__ import annotations

from .wrap import build_launch_env, resolve_provider_type


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Copilot."""
    provider_type = resolve_provider_type(backend, "auto", {"HEADROOM_BACKEND": backend})
    env, _lines = build_launch_env(
        port=port,
        provider_type=provider_type,
        wire_api=None,
        environ={},
    )
    return {
        key: env[key]
        for key in (
            "COPILOT_PROVIDER_TYPE",
            "COPILOT_PROVIDER_BASE_URL",
            "COPILOT_PROVIDER_WIRE_API",
        )
        if key in env
    }

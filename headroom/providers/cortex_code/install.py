"""Cortex Code install-time helpers."""

from __future__ import annotations

from headroom.providers._setup_text import project_attribution_lines

from .runtime import build_launch_env, proxy_base_url


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Cortex Code."""
    del backend
    return {"OPENAI_BASE_URL": proxy_base_url(port)}


def render_setup_lines(port: int, project: str | None = None) -> list[str]:
    """Render the Cortex Code setup instructions for the local proxy."""
    _, env_lines = build_launch_env(port=port, environ={}, project=project)
    lines = [
        "  Headroom proxy is running. Configure Cortex Code (CoCo):",
        "",
        "  Set the following environment variable before launching cortex:",
    ]
    lines += [f"    {line}" for line in env_lines]
    return lines + project_attribution_lines(project)

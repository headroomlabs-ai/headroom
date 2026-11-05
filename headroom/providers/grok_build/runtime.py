"""Runtime helpers for Grok Build integrations."""

from __future__ import annotations

from dataclasses import dataclass

from headroom.providers._setup_text import project_attribution_lines
from headroom.providers.codex import proxy_base_url
from headroom.providers.grok.runtime import DEFAULT_API_URL
from headroom.proxy.project_context import with_project_prefix


@dataclass(frozen=True)
class GrokBuildProxyTarget:
    """Resolved local proxy target shown in Grok Build setup instructions."""

    base_url: str


def build_proxy_targets(port: int, project: str | None = None) -> GrokBuildProxyTarget:
    """Build the local proxy URL shown to Grok Build users.

    ``project`` (the wrap launch directory) is encoded as a ``/p/<name>``
    base-URL prefix because Grok cannot send custom headers; the proxy
    strips it and attributes savings per project.
    """
    return GrokBuildProxyTarget(
        base_url=with_project_prefix(proxy_base_url(port), project),
    )


def render_setup_lines(port: int, project: str | None = None) -> list[str]:
    """Render Grok Build setup instructions for the local proxy."""
    target = build_proxy_targets(port, project)
    lines = [
        "  Headroom proxy is running. Configure Grok Build:",
        "",
        "  ~/.grok/config.toml has been updated with:",
        "    [model.grok-build]",
        f'    base_url = "{target.base_url}"',
        "",
        f"  Proxy upstream (OpenAI-compatible): {DEFAULT_API_URL}",
        "",
        "  Start Grok Build in this project directory:",
        "    grok",
        "",
        "  Or switch models in an existing session:",
        "    /model grok-build",
    ]
    return lines + project_attribution_lines(project)

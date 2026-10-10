"""Runtime helpers for Antigravity IDE integrations."""

from __future__ import annotations

from dataclasses import dataclass

from headroom.providers._setup_text import project_attribution_lines
from headroom.providers.claude import proxy_base_url as claude_proxy_base_url
from headroom.providers.codex import proxy_base_url as codex_proxy_base_url
from headroom.proxy.project_context import with_project_prefix


@dataclass(frozen=True)
class AntigravityProxyTargets:
    """Resolved local proxy targets shown in Antigravity setup instructions."""

    openai_base_url: str
    anthropic_base_url: str


def build_proxy_targets(port: int, project: str | None = None) -> AntigravityProxyTargets:
    """Build the local proxy URLs shown to Antigravity users.

    ``project`` (the wrap launch directory) is encoded as a ``/p/<name>``
    base-URL prefix because Antigravity's custom model-provider settings
    cannot send custom headers; the proxy strips it and attributes savings
    per project.
    """
    return AntigravityProxyTargets(
        openai_base_url=with_project_prefix(codex_proxy_base_url(port), project),
        anthropic_base_url=with_project_prefix(claude_proxy_base_url(port), project),
    )


def render_setup_lines(port: int, project: str | None = None) -> list[str]:
    """Render the Antigravity setup instructions for the local proxy."""
    targets = build_proxy_targets(port, project)
    lines = [
        "  Headroom proxy is running. Configure Antigravity:",
        "",
        "  Antigravity reads model endpoints from its model-provider settings,",
        "  not from environment variables. Add a custom OpenAI-compatible",
        "  model provider pointing at the proxy:",
        "",
        f"    Base URL:  {targets.openai_base_url}",
        "    API Key:   your-openai-api-key",
        "",
        "  Antigravity fetches the model list from GET /v1/models automatically,",
        "  so there is nothing else to register.",
    ]
    return lines + project_attribution_lines(project)

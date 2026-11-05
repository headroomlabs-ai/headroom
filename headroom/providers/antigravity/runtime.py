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
    base-URL prefix because clients pointing at these URLs cannot be
    counted on to send custom headers; the proxy strips it and attributes
    savings per project.
    """
    return AntigravityProxyTargets(
        openai_base_url=with_project_prefix(codex_proxy_base_url(port), project),
        anthropic_base_url=with_project_prefix(claude_proxy_base_url(port), project),
    )


def render_setup_lines(port: int, project: str | None = None) -> list[str]:
    """Render the Antigravity setup instructions for the local proxy."""
    targets = build_proxy_targets(port, project)
    # The Gemini-compatible base URL (scheme://host; Gemini clients append
    # /v1beta/...) — the one agy expects for GOOGLE_GEMINI_BASE_URL.
    gemini_base_url = with_project_prefix(f"http://127.0.0.1:{port}", project)
    lines = [
        "  Headroom proxy is running. Antigravity setup:",
        "",
        "  The Antigravity IDE does not support third-party model providers —",
        "  its agent models are Google-managed, so there is no settings panel",
        "  to point at the proxy. What the proxy does offer Antigravity users:",
        "",
        "  - MCP tools: `headroom mcp install --agent antigravity` registers",
        "    the Headroom MCP server in Antigravity's global mcp_config.json.",
        "  - Native traffic: the proxy accepts `google-antigravity` client",
        "    traffic on POST /v1internal:streamGenerateContent, routed",
        "    upstream like any other Cloud Code request.",
        "",
        "  The Antigravity CLI (agy), unlike the IDE, accepts a custom",
        "  endpoint via its GOOGLE_GEMINI_BASE_URL setting — point it at",
        "  the proxy's Gemini-compatible base URL:",
        "",
        f"    GOOGLE_GEMINI_BASE_URL={gemini_base_url}",
        "",
        "  For OpenAI-speaking clients, the proxy's OpenAI-compatible",
        "  base URL is:",
        "",
        f"    Base URL:  {targets.openai_base_url}",
        "    API Key:   your-openai-api-key",
    ]
    return lines + project_attribution_lines(project)

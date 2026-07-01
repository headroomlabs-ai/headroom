"""Runtime helpers for Grok Build integrations."""

from __future__ import annotations

from dataclasses import dataclass

from headroom.proxy.project_context import with_project_prefix


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL for OpenAI-compatible Grok traffic."""
    return f"http://127.0.0.1:{port}/v1"


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
        f"    [model.grok-build]",
        f'    base_url = "{target.base_url}"',
        "",
        "  Start Grok Build in this project directory:",
        "    grok",
        "",
        "  Or switch models in an existing session:",
        "    /model grok-build",
    ]
    if project:
        lines += [
            "",
            f"  Dashboard savings will be attributed to project '{project}'",
            "  (the directory this command was run from). Re-run from another",
            "  project directory to get that project's URL.",
        ]
    return lines
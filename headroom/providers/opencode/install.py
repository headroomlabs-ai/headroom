"""OpenCode install-time helpers."""

from __future__ import annotations

import json

from .runtime import CONFIG_SCHEMA_URL, build_provider_overrides, proxy_base_url


def render_provider_config(port: int, project: str | None = None) -> str:
    """Return the ``opencode.json`` snippet that routes OpenCode through Headroom."""
    config = {"$schema": CONFIG_SCHEMA_URL, "provider": build_provider_overrides(port, project)}
    return json.dumps(config, indent=2)


def render_setup_lines(port: int, project: str | None = None) -> list[str]:
    """Render the OpenCode manual-setup instructions for the local proxy.

    Used for ``--no-proxy`` / proxy-only displays and documentation.  OpenCode
    cannot be pointed at a base URL via environment variables, so the snippet
    below is merged into ``opencode.json`` automatically by
    ``headroom wrap opencode``; these lines describe the equivalent manual edit.
    """
    snippet = render_provider_config(port, project).splitlines()
    lines = [
        "  Headroom proxy is running. Configure OpenCode:",
        "",
        "  Add the following to opencode.json (project root) so the built-in",
        f"  anthropic + openai providers route through the proxy at {proxy_base_url(port)}:",
        "",
    ]
    lines += [f"    {line}" for line in snippet]
    if project:
        lines += [
            "",
            f"  Dashboard savings will be attributed to project '{project}'",
            "  (the directory this command was run from). Re-run from another",
            "  project directory to get that project's URL.",
        ]
    return lines

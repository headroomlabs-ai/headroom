"""Runtime helpers for Aider integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from headroom.providers.claude import proxy_base_url as claude_proxy_base_url
from headroom.providers.codex import proxy_base_url as codex_proxy_base_url
from headroom.proxy.project_context import with_project_prefix


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for Aider through the local proxy.

    ``project`` (the wrap launch directory) is encoded as a ``/p/<name>``
    base-URL prefix because aider cannot send custom headers; the proxy
    strips it and attributes savings per project.
    """
    env = dict(environ or os.environ)
    openai_base_url = with_project_prefix(codex_proxy_base_url(port), project)
    anthropic_base_url = with_project_prefix(claude_proxy_base_url(port), project)
    env["OPENAI_API_BASE"] = openai_base_url
    env["ANTHROPIC_BASE_URL"] = anthropic_base_url
    return env, [
        f"OPENAI_API_BASE={openai_base_url}",
        f"ANTHROPIC_BASE_URL={anthropic_base_url}",
    ]

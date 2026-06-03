"""Runtime helpers for Hermes (Nous Research) agent integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from headroom.providers.claude import proxy_base_url as claude_proxy_base_url
from headroom.providers.codex import proxy_base_url as codex_proxy_base_url


def build_launch_env(
    port: int, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for Hermes through the local proxy.

    Hermes (Nous Research) is an OpenAI-compatible agent CLI.  Both
    OPENAI_BASE_URL and ANTHROPIC_BASE_URL are set so that whichever
    provider Hermes is configured to use is transparently routed through
    the Headroom proxy.
    """
    env = dict(environ or os.environ)
    openai_base_url = codex_proxy_base_url(port)
    anthropic_base_url = claude_proxy_base_url(port)
    env["OPENAI_BASE_URL"] = openai_base_url
    env["ANTHROPIC_BASE_URL"] = anthropic_base_url
    return env, [
        f"OPENAI_BASE_URL={openai_base_url}",
        f"ANTHROPIC_BASE_URL={anthropic_base_url}",
    ]

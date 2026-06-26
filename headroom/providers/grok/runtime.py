"""Runtime helpers for Grok Build CLI integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from headroom.providers.codex import proxy_base_url as codex_proxy_base_url


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for Grok Build CLI through the local proxy."""
    env = dict(environ or os.environ)
    grok_proxy_url = codex_proxy_base_url(port)
    env["GROK_PROXY_URL"] = grok_proxy_url
    _ = project
    return env, [f"GROK_PROXY_URL={grok_proxy_url}"]

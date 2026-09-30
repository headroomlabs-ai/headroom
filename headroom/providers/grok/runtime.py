"""Runtime helpers for Grok CLI integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from headroom.proxy.project_context import with_project_prefix

DEFAULT_API_URL = "https://api.x.ai"
PROXY_ENV_KEY = "GROK_MODELS_BASE_URL"


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by Grok CLI integrations."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for Grok CLI through the local proxy.

    Grok routes inference traffic through ``GROK_MODELS_BASE_URL`` when set.
    The proxy forwards OpenAI-compatible chat requests upstream to xAI while
    Grok keeps its native settings and authentication routing.

    ``project`` (the wrap launch directory) is encoded as a ``/p/<name>``
    base-URL prefix because Grok cannot send custom attribution headers;
    the proxy strips it and attributes savings per project.
    """
    env = dict(environ or os.environ)
    base_url = with_project_prefix(proxy_base_url(port), project)
    env[PROXY_ENV_KEY] = base_url
    return env, [f"{PROXY_ENV_KEY}={base_url}"]

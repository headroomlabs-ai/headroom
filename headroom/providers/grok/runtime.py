"""Runtime helpers for Grok Build CLI integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_API_URL = "https://cli-chat-proxy.grok.com/v1"
_PROXY_ENV = "GROK_CLI_CHAT_PROXY_BASE_URL"


def proxy_base_url(port: int) -> str:
    """Return the local Headroom proxy base URL for Grok."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for Grok through the local Headroom proxy."""
    env = dict(environ or os.environ)
    base_url = proxy_base_url(port)
    env[_PROXY_ENV] = base_url
    return env, [f"{_PROXY_ENV}={base_url}"]

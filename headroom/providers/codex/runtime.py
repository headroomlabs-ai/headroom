"""Runtime helpers for Codex/OpenAI-facing integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_API_URL = "https://api.openai.com"


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenAI-compatible integrations."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for Codex through the local proxy."""
    env = dict(environ or os.environ)
    base_url = proxy_base_url(port)
    env["OPENAI_BASE_URL"] = base_url
    return env, [f"OPENAI_BASE_URL={base_url}"]

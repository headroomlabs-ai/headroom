"""Runtime helpers for Hermes llm-proxy integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_HERMES_API_URL = os.environ.get(
    "HEADROOM_HERMES_BASE_URL", "http://127.0.0.1:38765/v1"
).rstrip("/")
OPENAI_BASE_ENV = "OPENAI_BASE_URL"


def proxy_base_url(port: int) -> str:
    """Return the local Headroom proxy base URL for OpenAI clients."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Build environment for OpenAI clients through Headroom → Hermes."""
    env = dict(os.environ if environ is None else environ)
    base_url = proxy_base_url(port)
    env[OPENAI_BASE_ENV] = base_url
    return env, [f"{OPENAI_BASE_ENV}={base_url}"]
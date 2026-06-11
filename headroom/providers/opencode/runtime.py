"""Runtime helpers for OpenCode integrations.

OpenCode supports multiple LLM backends. The primary path for users who
rely on GitHub Copilot is the ``github-copilot/`` backend prefix that the
headroom proxy already understands (see :mod:`headroom.proxy.auth_mode`).

For users who configure Anthropic or OpenAI directly, the proxy is reached
via the standard ``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL`` env vars that
OpenCode passes straight through to the AI SDK.

The github-copilot provider in OpenCode reads its base URL from
``opencode.json`` (patched by :func:`headroom.providers.opencode.install.apply_provider_scope`),
not from an environment variable.  Do NOT set ``GITHUB_COPILOT_HOST`` here;
that env var controls GitHub Enterprise host resolution in
:mod:`headroom.copilot_auth` and would break credential discovery.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

# Headroom proxy speaks the Anthropic wire protocol on the root path and the
# OpenAI wire protocol under /v1, exactly what OpenCode expects.
DEFAULT_API_URL = "https://api.anthropic.com"


def proxy_base_url(port: int) -> str:
    """Return the Anthropic-protocol proxy URL for OpenCode."""
    return f"http://127.0.0.1:{port}"


def proxy_openai_url(port: int) -> str:
    """Return the OpenAI-protocol proxy URL for OpenCode."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    *,
    backend: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables to route OpenCode through the headroom proxy.

    OpenCode reads standard AI SDK env vars.  We set:

    * ``ANTHROPIC_BASE_URL``  — for ``anthropic/*`` models
    * ``OPENAI_BASE_URL``     — for ``openai/*`` models

    The ``github-copilot`` provider gets its base URL from ``opencode.json``
    (see :func:`~headroom.providers.opencode.install.apply_provider_scope`).
    """
    env = dict(environ if environ is not None else os.environ)

    anthropic_url = proxy_base_url(port)
    openai_url = proxy_openai_url(port)

    env["ANTHROPIC_BASE_URL"] = anthropic_url
    env["OPENAI_BASE_URL"] = openai_url

    display = [
        f"ANTHROPIC_BASE_URL={anthropic_url}",
        f"OPENAI_BASE_URL={openai_url}",
    ]
    return env, display

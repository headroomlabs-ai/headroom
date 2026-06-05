"""Runtime helpers for OpenCode integrations.

OpenCode supports multiple LLM backends. The primary path for users who
rely on GitHub Copilot is the ``github-copilot/`` backend prefix that the
headroom proxy already understands (see :mod:`headroom.proxy.auth_mode`).

For users who configure Anthropic or OpenAI directly, the proxy is reached
via the standard ``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL`` env vars that
OpenCode passes straight through to the AI SDK.
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


def proxy_copilot_url(port: int) -> str:
    """Return the GitHub Copilot backend proxy URL for OpenCode.

    The headroom proxy routes requests prefixed with ``github-copilot/``
    to ``api.githubcopilot.com``.  Setting ``HEADROOM_COPILOT_BASE_URL``
    tells headroom to use its own proxy as the upstream for Copilot calls.
    """
    return f"http://127.0.0.1:{port}/github-copilot"


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    *,
    backend: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables to route OpenCode through the headroom proxy.

    OpenCode reads standard AI SDK env vars.  We set all three so it works
    regardless of which provider the user has configured:

    * ``ANTHROPIC_BASE_URL``  — for ``anthropic/*`` models
    * ``OPENAI_BASE_URL``     — for ``openai/*`` models
    * ``GITHUB_COPILOT_HOST`` — headroom's copilot backend prefix for
                                 ``github-copilot/*`` models (e.g. the user's
                                 primary provider)
    """
    env = dict(environ if environ is not None else os.environ)

    anthropic_url = proxy_base_url(port)
    openai_url = proxy_openai_url(port)

    env["ANTHROPIC_BASE_URL"] = anthropic_url
    env["OPENAI_BASE_URL"] = openai_url
    # headroom.copilot_auth reads GITHUB_COPILOT_HOST to override the upstream
    env["GITHUB_COPILOT_HOST"] = f"127.0.0.1:{port}"

    display = [
        f"ANTHROPIC_BASE_URL={anthropic_url}",
        f"OPENAI_BASE_URL={openai_url}",
        f"GITHUB_COPILOT_HOST=127.0.0.1:{port}",
    ]
    return env, display

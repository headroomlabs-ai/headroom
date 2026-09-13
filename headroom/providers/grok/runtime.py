"""Runtime helpers for Grok CLI integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from headroom.proxy.project_context import with_project_prefix

DEFAULT_API_URL = "https://api.x.ai"
PROXY_ENV_KEY = "GROK_MODELS_BASE_URL"

# Official Grok CLI / Grok Build stamps this on inference requests (observed on
# grok-shell 0.2.x). Used for per-request xAI routing when the shared proxy's
# process-wide OPENAI target is still api.openai.com (Claude/Codex-started).
_XAI_TOKEN_AUTH_HEADER = "x-xai-token-auth"
_XAI_TOKEN_AUTH_VALUE = "xai-grok-cli"
# UA product tokens emitted by Grok CLI 0.2.x. Matched against whitespace-split
# tokens so unrelated clients ("litellm-grok/1.0") cannot collide.
_GROK_UA_PREFIXES = ("grok-pager/", "grok-shell/")


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup for plain mappings and Starlette Headers."""
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def is_grok_cli_request(headers: Mapping[str, str]) -> bool:
    """Return True when inbound headers identify the official Grok CLI.

    Grok cannot stamp ``x-headroom-base-url`` (no custom attribution headers),
    so shared-proxy routing must recognize the CLI from wire signals instead.
    Detection is intentionally narrow: only the official token-auth marker and
    known Grok UA product tokens (prefix match on whitespace-split tokens) —
    never model-id heuristics.
    """
    token_auth = _header_value(headers, _XAI_TOKEN_AUTH_HEADER)
    if token_auth is not None and token_auth.strip().lower() == _XAI_TOKEN_AUTH_VALUE:
        return True

    user_agent = _header_value(headers, "user-agent")
    if not user_agent:
        return False
    return any(token.startswith(_GROK_UA_PREFIXES) for token in user_agent.lower().split())


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

"""Runtime helpers for Grok CLI integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from headroom.proxy.project_context import with_project_prefix

DEFAULT_API_URL = "https://api.x.ai"
# Where the Grok CLI sends inference for a `grok login` session. The CLI's own
# default base for session auth is this host; api.x.ai only accepts `xai-`
# API keys and answers a session token 401.
SESSION_API_URL = "https://cli-chat-proxy.grok.com"
PROXY_ENV_KEY = "GROK_MODELS_BASE_URL"
_XAI_TOKEN_AUTH_HEADER = "x-xai-token-auth"
_XAI_TOKEN_AUTH_VALUE = "xai-grok-cli"
_GROK_UA_PREFIXES = ("grok-shell/", "grok/")


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


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def is_grok_cli_request(headers: Mapping[str, str]) -> bool:
    """Return True when the inbound headers identify the official Grok CLI."""
    token_auth = _header(headers, _XAI_TOKEN_AUTH_HEADER)
    if token_auth is not None and token_auth.strip().lower() == _XAI_TOKEN_AUTH_VALUE:
        return True
    user_agent = (_header(headers, "user-agent") or "").lower()
    return any(token.startswith(_GROK_UA_PREFIXES) for token in user_agent.split())


def session_upstream(headers: Mapping[str, str]) -> str | None:
    """Return the upstream a Grok CLI session-login request must go to.

    ``headroom wrap grok`` points the proxy at ``api.x.ai``, which is right for
    an ``xai-`` API key and wrong for a ``grok login`` session: the CLI then
    sends its auth.x.ai session token as the bearer, api.x.ai rejects it, and
    every message fails with "authentication required" even though login
    succeeded. Session tokens are only valid at ``cli-chat-proxy.grok.com``,
    the CLI's own default for that mode, so route them there.

    ``None`` for anything that is not a Grok CLI session request: other
    clients keep their configured target, and a Grok CLI carrying an ``xai-``
    key keeps ``api.x.ai``.
    """
    if not is_grok_cli_request(headers) or _bearer_is_api_key(headers):
        return None
    return SESSION_API_URL


def _bearer_is_api_key(headers: Mapping[str, str]) -> bool:
    value = (_header(headers, "authorization") or "").strip().lower()
    return value.startswith("bearer xai-")

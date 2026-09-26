"""Runtime helpers for Grok CLI integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlparse

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
#: User agents trusted to REDIRECT a credential, which is a stricter question
#: than "is this a Grok CLI". The bare ``grok/`` prefix is deliberately absent:
#: it is already claimed with a different meaning in
#: ``headroom.proxy.auth_policy`` (SUBSCRIPTION_UA_PREFIXES, CLIENT_UA_MAP), so
#: reusing it here would let any client that borrows that loose prefix steer
#: where its Authorization header is sent.
_SESSION_UA_PREFIXES = ("grok-shell/",)
#: Kill switch for an operator who never wants a request re-pointed at
#: grok.com, whatever it claims to be. Any value but "0" leaves routing on.
SESSION_ROUTING_ENV_KEY = "HEADROOM_GROK_SESSION_ROUTING"


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
    """Return True when the inbound headers identify the official Grok CLI.

    Attribution only. Every signal it reads is client-controlled, so it must
    never decide where a credential is SENT - see :func:`session_upstream`,
    which asks a deliberately narrower question.
    """
    token_auth = _header(headers, _XAI_TOKEN_AUTH_HEADER)
    if token_auth is not None and token_auth.strip().lower() == _XAI_TOKEN_AUTH_VALUE:
        return True
    user_agent = (_header(headers, "user-agent") or "").lower()
    return any(token.startswith(_GROK_UA_PREFIXES) for token in user_agent.split())


def session_upstream(headers: Mapping[str, str], configured_target: str | None) -> str | None:
    """Return the upstream a Grok CLI session-login request must go to.

    ``headroom wrap grok`` points the proxy at ``api.x.ai``, which is right for
    an ``xai-`` API key and wrong for a ``grok login`` session: the CLI then
    sends its auth.x.ai session token as the bearer, api.x.ai rejects it, and
    every message fails with "authentication required" even though login
    succeeded. Session tokens are only valid at ``cli-chat-proxy.grok.com``,
    the CLI's own default for that mode, so route them there.

    ``configured_target`` is the operator's OpenAI target. Only when it is
    already xAI is the reroute allowed: the credential was bound for xAI
    anyway and moves between two xAI hosts. Any other target (an internal
    gateway, api.openai.com) keeps every request, since the headers and a
    JWT-shaped bearer prove nothing about who issued the token.

    ``None`` for anything that is not a Grok CLI session request: other
    clients keep their configured target, and a Grok CLI carrying an ``xai-``
    key keeps ``api.x.ai``.
    """
    if os.environ.get(SESSION_ROUTING_ENV_KEY, "").strip() == "0":
        return None
    if not _is_xai_target(configured_target):
        return None
    if not _is_grok_session_client(headers) or _bearer_is_api_key(headers):
        return None
    # The credential itself has the final say. Headers that merely claim to be
    # the Grok CLI are trivially forged, and being wrong here does not degrade
    # a request - it forwards somebody's Authorization to a public third-party
    # host. A `grok login` session token is always a JWT; the gateway keys this
    # must never redirect (`sk-`, `sk-proj-`, `sk-litellm-`, `ghu_`) never are.
    if not _bearer_is_session_jwt(headers):
        return None
    return SESSION_API_URL


def _is_xai_target(url: str | None) -> bool:
    """True when ``url`` is the canonical xAI API origin (any path)."""
    parsed = urlparse((url or "").strip())
    return (
        parsed.scheme.lower() == "https"
        and parsed.hostname == "api.x.ai"
        and parsed.port in (None, 443)
    )


def _is_grok_session_client(headers: Mapping[str, str]) -> bool:
    """Grok CLI signals strong enough to justify re-pointing a credential."""
    token_auth = _header(headers, _XAI_TOKEN_AUTH_HEADER)
    if token_auth is not None and token_auth.strip().lower() == _XAI_TOKEN_AUTH_VALUE:
        return True
    user_agent = (_header(headers, "user-agent") or "").lower()
    return any(token.startswith(_SESSION_UA_PREFIXES) for token in user_agent.split())


def _bearer_is_api_key(headers: Mapping[str, str]) -> bool:
    value = (_header(headers, "authorization") or "").strip().lower()
    return value.startswith("bearer xai-")


def _bearer_is_session_jwt(headers: Mapping[str, str]) -> bool:
    """True when the bearer is JWT-shaped, as a `grok login` token always is.

    Not case-folded: base64url is case-sensitive and a real JWT header segment
    starts with the exact bytes ``eyJ``.
    """
    raw = (_header(headers, "authorization") or "").strip()
    if len(raw) < 7 or raw[:7].lower() != "bearer ":
        return False
    token = raw[7:].strip()
    segments = token.split(".")
    return len(segments) == 3 and token.startswith("eyJ") and all(segments)

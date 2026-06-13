"""Generic OAuth2 client-credentials upstream-auth extension for the Headroom proxy.

Enable: `--proxy-extension oauth2` (or HEADROOM_PROXY_EXTENSIONS=oauth2).
No-op unless HEADROOM_OAUTH2_TOKEN_URL is set. See README for env config.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .middleware import OAuth2Middleware
from .provider import OAuth2ClientCredentials, OAuth2Error

__all__ = ["install", "OAuth2ClientCredentials", "OAuth2Error", "OAuth2Middleware", "parse_headers"]
__version__ = "0.1.0"
log = logging.getLogger("headroom_oauth2")


def _split(s):
    return [x.strip() for x in (s or "").replace(",", " ").split() if x.strip()]


def _ctrl(s):
    return any(ord(c) < 32 or ord(c) == 127 for c in s)


def parse_headers(s: str | None) -> dict[str, str]:
    """Parse ``K=V,K2=V2`` into a dict. Drops pairs whose key/value contain control
    characters, or whose key contains a space or colon -- prevents HTTP header injection
    from a malformed env value."""
    out: dict[str, str] = {}
    for pair in (s or "").split(","):
        if "=" not in pair:
            continue
        k, v = (x.strip() for x in pair.split("=", 1))
        if not k:
            continue
        if _ctrl(k) or _ctrl(v) or " " in k or ":" in k:
            log.warning("headroom-oauth2: dropping malformed static header: %r", k)
            continue
        out[k] = v
    return out


def _int(env, key):
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{key}={raw!r} is not an integer") from None


def provider_from_env(env: dict | None = None) -> OAuth2ClientCredentials | None:
    """Build a provider from ``HEADROOM_OAUTH2_*`` env vars, or None if TOKEN_URL is unset.

    Raises ValueError on malformed config so callers can fail closed.
    """
    env = os.environ if env is None else env
    token_url = env.get("HEADROOM_OAUTH2_TOKEN_URL")
    if not token_url:
        return None
    allow_insecure = env.get("HEADROOM_OAUTH2_ALLOW_INSECURE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if allow_insecure:
        log.warning("headroom-oauth2: ALLOW_INSECURE set -- token endpoint TLS check disabled")
    resource = env.get("HEADROOM_OAUTH2_RESOURCE")  # RFC 8707 target service
    timeout = _int(env, "HEADROOM_OAUTH2_TIMEOUT")
    skew = _int(env, "HEADROOM_OAUTH2_SKEW")
    kwargs = {}
    if timeout is not None:
        kwargs["timeout_seconds"] = float(timeout)
    if skew is not None:
        kwargs["skew_seconds"] = skew
    return OAuth2ClientCredentials(
        token_url=token_url,
        client_id=env.get("HEADROOM_OAUTH2_CLIENT_ID", ""),
        client_secret=env.get("HEADROOM_OAUTH2_CLIENT_SECRET", ""),
        scopes=_split(env.get("HEADROOM_OAUTH2_SCOPES")),
        audience=env.get("HEADROOM_OAUTH2_AUDIENCE") or None,
        grant_type=env.get("HEADROOM_OAUTH2_GRANT_TYPE", "client_credentials"),
        auth_style=env.get("HEADROOM_OAUTH2_AUTH_STYLE", "post"),
        extra_params={"resource": resource} if resource else None,
        allow_insecure=allow_insecure,
        **kwargs,
    )


def install(app: Any, config: Any) -> None:
    """Headroom proxy-extension entry point: install(app, config) -> None."""
    try:
        provider = provider_from_env()
    except ValueError as e:
        raise RuntimeError(f"headroom-oauth2 misconfigured: {e}") from None  # fail-closed
    if provider is None:
        log.info("headroom-oauth2 loaded but HEADROOM_OAUTH2_TOKEN_URL unset; no-op")
        return
    static = parse_headers(os.environ.get("HEADROOM_OAUTH2_HEADERS"))
    if static:
        try:
            # litellm's import runs load_dotenv and can inject .env values into os.environ;
            # snapshot and restore so we never leak unrelated keys into the process env.
            _before = dict(os.environ)
            import litellm

            # drop keys litellm/load_dotenv added, restore any it changed (no empty-env window)
            for k in list(os.environ):
                if k not in _before:
                    del os.environ[k]
            os.environ.update(_before)
            litellm.headers = {**(getattr(litellm, "headers", None) or {}), **static}
            log.info("headroom-oauth2: static upstream headers: %s", list(static))
        except Exception as e:  # pragma: no cover
            log.warning("headroom-oauth2: could not set litellm.headers: %s", e)
    # The litellm backend auths bedrock/vertex/sagemaker from env and ignores a forwarded
    # bearer, so this extension is a no-op there -- warn loudly rather than silently do nothing.
    backend = str(getattr(config, "backend", "") or "").lower()
    if any(p in backend for p in ("bedrock", "vertex", "sagemaker")):
        log.warning(
            "headroom-oauth2: backend %r authenticates from env (bedrock/vertex/sagemaker) and "
            "ignores the injected bearer -- this extension will have NO effect. Use an "
            "OpenAI-compatible / passthrough backend (e.g. --backend litellm-openai).",
            backend or "<default>",
        )
    app.add_middleware(OAuth2Middleware, provider=provider)
    log.info(
        "headroom-oauth2: client-credentials auth installed (token_url=%s, style=%s)",
        provider.token_url,
        provider.auth_style,
    )

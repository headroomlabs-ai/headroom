"""Async HTTP client for Anthropic's OAuth usage API.

Endpoint: GET https://api.anthropic.com/api/oauth/usage
Required header: anthropic-beta: oauth-2025-04-20
Auth: Authorization: Bearer <oauth_access_token>

Token resolution order (highest → lowest priority):
  1. Explicit token passed to :meth:`fetch`
  2. ``CLAUDE_CODE_OAUTH_TOKEN`` env-var
  3. ``~/.claude/.credentials.json`` → ``claudeAiOauth.accessToken``
     (respects ``CLAUDE_CONFIG_DIR`` env-var override)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from headroom.subscription.models import SubscriptionSnapshot

logger = logging.getLogger(__name__)

_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
_BETA_HEADER = "oauth-2025-04-20"
_TOKEN_EXPIRY_BUFFER_S = 60


def _credentials_path() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    return Path(base) / ".credentials.json"


def _load_credentials_file() -> dict[str, Any] | None:
    """Load raw credentials dict from the Claude Code credentials file."""
    path = _credentials_path()
    try:
        with path.open() as fh:
            return json.load(fh)  # type: ignore[no-any-return]
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("Cannot read credentials file %s: %s", path, exc)
        return None


def read_cached_oauth_token() -> str | None:
    """Resolve a stored OAuth token for background polling (no request needed).

    Returns the raw access token string if found and not expired, else None.
    """
    # 1. Env var
    env_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if env_token:
        return env_token

    # 2. Credentials file
    creds = _load_credentials_file()
    if not creds:
        return None
    oauth = creds.get("claudeAiOauth") or {}
    token = oauth.get("accessToken") or ""
    if not token:
        return None

    # Check expiry (Anthropic stores timestamp in milliseconds)
    expires_at_ms = oauth.get("expiresAt")
    if expires_at_ms is not None:
        import time

        now_ms = time.time() * 1000
        if now_ms >= (expires_at_ms - _TOKEN_EXPIRY_BUFFER_S * 1000):
            logger.debug("Cached OAuth token expired; skipping background poll")
            return None

    return token


class SubscriptionClient:
    """Thin async wrapper around the Anthropic OAuth usage endpoint."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    async def fetch(self, token: str | None = None) -> SubscriptionSnapshot | None:
        """Fetch current subscription window data.

        :param token: OAuth access token.  When *None*, falls back to
            :func:`read_cached_oauth_token`.
        :returns: :class:`SubscriptionSnapshot` or *None* on auth failure /
            unsupported account.
        """
        resolved = (token or "").strip() or read_cached_oauth_token()
        if not resolved:
            logger.debug("No OAuth token available for subscription polling")
            return None

        headers = {
            "Authorization": f"Bearer {resolved}",
            "anthropic-beta": _BETA_HEADER,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(_USAGE_URL, headers=headers)

            if resp.status_code == 401:
                logger.debug("OAuth token rejected (401) by Anthropic usage API")
                return None
            if resp.status_code == 404:
                # API key accounts (non-subscription) return 404
                logger.debug("Subscription usage API returned 404; likely API-key account")
                return None
            if resp.status_code != 200:
                logger.warning("Anthropic usage API returned %s", resp.status_code)
                return None

            data: dict[str, Any] = resp.json()
            return SubscriptionSnapshot.from_api_response(data, token=resolved)

        except httpx.TimeoutException:
            logger.debug("Timeout fetching Anthropic subscription window")
            return None
        except Exception as exc:
            logger.warning("Error fetching subscription window: %s", exc)
            return None

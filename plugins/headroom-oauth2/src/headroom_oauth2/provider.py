"""Generic OAuth2 client-credentials token provider (RFC 6749 section 4.4).

Mints a bearer from a configurable token endpoint, caches it, and refreshes
single-flight before expiry. Standard library only; `urllib` uses the system
cert store (works behind corporate SSL inspection). No vendor specifics.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError

log = logging.getLogger("headroom_oauth2")


def _https_or_local(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https":
        return True
    # only numeric loopback -- "localhost" can be repointed via /etc/hosts or DNS rebinding
    return parts.scheme == "http" and (parts.hostname or "") in ("127.0.0.1", "::1")


class OAuth2Error(RuntimeError):
    """Raised when a token cannot be minted."""


class OAuth2ClientCredentials:
    """Mints and caches an OAuth2 client-credentials bearer (RFC 6749 section 4.4).

    Thread-safe: ``token()`` refreshes single-flight before expiry; ``cached()`` is a
    lock-free read for the request hot path.
    """

    def __init__(
        self,
        *,
        token_url: str,
        client_id: str,
        client_secret: str,
        scopes=None,
        audience: str | None = None,
        grant_type: str = "client_credentials",
        auth_style: str = "post",
        extra_params=None,
        skew_seconds: int = 60,
        timeout_seconds: float = 30.0,
        allow_insecure: bool = False,
    ):
        if not token_url:
            raise ValueError("token_url is required")
        if not client_id or not client_secret:
            raise ValueError("client_id and client_secret are required")
        if auth_style not in ("post", "basic"):
            raise ValueError("auth_style must be 'post' or 'basic'")
        if not allow_insecure and not _https_or_local(token_url):
            raise ValueError(
                "token_url must be https (loopback http allowed for tests; set "
                "allow_insecure=True / HEADROOM_OAUTH2_ALLOW_INSECURE=1 to override)"
            )
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = list(scopes or [])
        self.audience = audience
        self.grant_type = grant_type
        self.auth_style = auth_style
        self.extra_params = dict(extra_params or {})
        self.skew = max(0, int(skew_seconds))
        self.timeout = timeout_seconds
        self._lock = threading.Lock()
        self._token: str | None = None
        self._exp = 0.0
        self._eff_skew = self.skew

    def _valid(self) -> bool:
        return self._token is not None and time.monotonic() < self._exp - self._eff_skew

    def cached(self) -> str | None:
        """Return the cached token if still valid, else None. No minting -- hot-path read."""
        return self._token if self._valid() else None

    def token(self) -> str:
        """Return a valid bearer, minting/refreshing single-flight if needed."""
        if self._valid():
            return self._token  # type: ignore[return-value]
        with self._lock:  # single-flight: one mint per burst
            if self._valid():
                return self._token  # type: ignore[return-value]
            token, ttl = self._mint()
            # Publish _exp/_eff_skew BEFORE _token so a concurrent cached() reader never
            # sees a fresh token paired with a stale expiry.
            self._eff_skew = min(self.skew, max(0, ttl // 2))
            self._exp = time.monotonic() + ttl
            self._token = token
            return self._token

    def _mint(self):
        form = dict(self.extra_params)  # caller extras first; canonical fields below always win
        form["grant_type"] = self.grant_type
        if self.scopes:
            form["scope"] = " ".join(self.scopes)
        if self.audience:
            form["audience"] = self.audience
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        if self.auth_style == "basic":
            creds = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            headers["Authorization"] = "Basic " + creds
        else:
            form["client_id"] = self.client_id
            form["client_secret"] = self.client_secret
        req = urllib.request.Request(
            self.token_url,
            data=urllib.parse.urlencode(form).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.load(resp)
        except HTTPError as e:
            try:
                e.read()  # drain; do NOT surface the IdP body (may echo sensitive context)
            except Exception:
                pass
            raise OAuth2Error(f"token endpoint returned HTTP {e.code}") from None
        except (URLError, OSError) as e:
            raise OAuth2Error(f"token endpoint unreachable: {e}") from None
        except json.JSONDecodeError:
            raise OAuth2Error("token endpoint returned non-JSON") from None
        token = payload.get("access_token")
        if not token:
            raise OAuth2Error("token endpoint response had no access_token")
        raw = payload.get("expires_in")
        try:
            ttl = int(float(raw))  # tolerate "3600", "3600.0", 3600, or a JSON float
        except (TypeError, ValueError):
            ttl = 300
        ttl = max(1, ttl)  # 0/negative would cause a stale token or per-request minting
        log.info("oauth2: minted token (ttl=%ss, scopes=%s)", ttl, self.scopes or "-")
        return token, ttl

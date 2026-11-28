"""Reusable hostile-flow-safe OAuth transaction state machines."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlencode, urlsplit


class OAuthValidationError(ValueError):
    pass


class OAuthReplayRejected(OAuthValidationError):
    pass


class DeviceFlowError(RuntimeError):
    pass


class DeviceFlowCancelled(DeviceFlowError):
    pass


@dataclass(frozen=True, slots=True)
class OAuthCredential:
    account_ref: str
    authorization_code: str | None = None
    access_token: str | None = field(default=None, repr=False)


@dataclass(slots=True)
class BrowserAuthorizationTransaction:
    authorization_endpoint: str
    issuer: str
    client_id: str
    redirect_uri: str
    scopes: tuple[str, ...]
    allowed_account: str
    state: str
    code_verifier: str = field(repr=False)
    expires_at: float
    _consumed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        authorization_endpoint: str,
        issuer: str,
        client_id: str,
        redirect_uri: str,
        scopes: tuple[str, ...],
        allowed_account: str,
        lifetime_seconds: float,
        now: float | None = None,
    ) -> BrowserAuthorizationTransaction:
        if lifetime_seconds <= 0:
            raise ValueError("OAuth lifetime must be positive")
        cls._validate_loopback_redirect(redirect_uri)
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        return cls(
            authorization_endpoint=authorization_endpoint,
            issuer=issuer.rstrip("/"),
            client_id=client_id,
            redirect_uri=redirect_uri,
            scopes=scopes,
            allowed_account=allowed_account,
            state=state,
            code_verifier=verifier,
            expires_at=(time.monotonic() if now is None else now) + lifetime_seconds,
        )

    @property
    def authorization_url(self) -> str:
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(self.code_verifier.encode("ascii")).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "scope": " ".join(self.scopes),
                "state": self.state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        separator = "&" if "?" in self.authorization_endpoint else "?"
        return f"{self.authorization_endpoint}{separator}{query}"

    def consume(self, *, callback_uri: str, now: float | None = None) -> OAuthCredential:
        if self._consumed:
            raise OAuthReplayRejected("OAuth callback state was already consumed")
        current = time.monotonic() if now is None else now
        if current > self.expires_at:
            raise OAuthValidationError("OAuth callback expired")
        callback = urlsplit(callback_uri)
        if (
            callback_uri.split("?", 1)[0] != self.redirect_uri
            or callback.fragment
            or "#" in callback_uri
            or callback.username is not None
            or callback.password is not None
        ):
            raise OAuthValidationError("OAuth callback URI mismatch")
        values = parse_qs(callback.query, keep_blank_values=True)
        names = {part.partition("=")[0] for part in callback.query.split("&")}
        if names != {"state", "issuer", "account", "code"} or set(values) != names:
            raise OAuthValidationError("OAuth callback query mismatch")
        if values.get("state") != [self.state]:
            raise OAuthValidationError("OAuth state mismatch")
        if values.get("issuer") != [self.issuer]:
            raise OAuthValidationError("OAuth issuer mismatch")
        if values.get("account") != [self.allowed_account]:
            raise OAuthValidationError("OAuth account mismatch")
        codes = values.get("code")
        if codes is None or len(codes) != 1 or not codes[0]:
            raise OAuthValidationError("OAuth authorization code missing")
        self._consumed = True
        return OAuthCredential(account_ref=self.allowed_account, authorization_code=codes[0])

    @staticmethod
    def _validate_loopback_redirect(redirect_uri: str) -> None:
        parsed = urlsplit(redirect_uri)
        if (
            parsed.scheme != "http"
            or not redirect_uri.startswith("http://")
            or parsed.username is not None
            or parsed.password is not None
            or any(character in redirect_uri for character in "%\\#")
            or any(ord(character) <= 32 or ord(character) >= 127 for character in redirect_uri)
            or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or parsed.port is None
            or not parsed.path.startswith("/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("OAuth redirect must be an exact loopback HTTP URI")


Poller = Callable[[], Awaitable[Mapping[str, str]]]
Sleeper = Callable[[float], Awaitable[None]]


class DeviceAuthorizationTransaction:
    def __init__(
        self,
        *,
        poller: Poller,
        interval_seconds: float,
        deadline: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleeper = asyncio.sleep,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("device polling interval must be positive")
        self._poller = poller
        self._interval = interval_seconds
        self._deadline = deadline
        self._clock = clock
        self._sleep = sleep

    async def poll(self, cancel_event: asyncio.Event) -> OAuthCredential:
        interval = self._interval
        while True:
            if cancel_event.is_set():
                raise DeviceFlowCancelled("device authorization cancelled")
            if self._clock() >= self._deadline:
                raise DeviceFlowError("expired_token")
            reply = await self._poller()
            token = reply.get("access_token")
            account = reply.get("account_ref")
            if token and account:
                return OAuthCredential(account_ref=account, access_token=token)
            error = reply.get("error")
            if error == "slow_down":
                interval += 5.0
            elif error != "authorization_pending":
                raise DeviceFlowError(error or "invalid_device_response")
            await self._sleep(interval)

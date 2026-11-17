"""Startup policy for the listener bind: no unauthenticated non-loopback bind.

The proxy's inbound trust model has two legs. Loopback callers are trusted
(same boundary as the ``/admin`` and ``/debug`` routes); everyone else must
present ``HEADROOM_PROXY_TOKEN``. That model has a hole when the token is not
configured *and* the listener is bound to a non-loopback interface: every
``/v1/*`` data-plane route -- which relays to the upstream provider with the
operator's own credentials where ``*_extra_headers`` or a subscription token
is configured -- answers any peer that can reach the port.

Until 0.40 the proxy only logged a warning for that shape. A warning is not a
control: launchers that set ``HEADROOM_HOST=0.0.0.0`` (systemd units, the
enterprise bundle's own start script, any ``docker run -p 8787:8787``) start
successfully and serve an unauthenticated relay. This module makes the
decision explicit and refuses to start unless the operator either configures
a token or *acknowledges* the open bind with
:data:`OPEN_BIND_ACK_ENV` -- the documented shape for a container that binds
``0.0.0.0`` internally but is published on ``127.0.0.1`` by the runtime, which
the process itself cannot observe.

The policy is evaluated in one place (:func:`evaluate_bind_policy`) and
enforced from three: :func:`headroom.proxy.server.create_app` (covers every
programmatic embedding and the multi-worker factory), :func:`run_server`
(before uvicorn forks workers, so a refusal is one clean error and not a
crash loop) and the ``headroom proxy`` CLI (before the banner, so the
operator sees the reason and not a traceback).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

from headroom.exceptions import ConfigurationError
from headroom.proxy.loopback_guard import is_loopback_host

__all__ = [
    "DEFAULT_BIND_HOST",
    "OPEN_BIND_ACK_ENV",
    "PROXY_TOKEN_ENV",
    "BindPolicyDecision",
    "OpenBindRefused",
    "enforce_bind_policy",
    "evaluate_bind_policy",
]

logger = logging.getLogger(__name__)

#: Explicit operator acknowledgement that an unauthenticated non-loopback bind
#: is intended (for example a container published on a loopback port by the
#: runtime). Any of ``1``, ``true``, ``yes``, ``on`` (case-insensitive) enables it.
OPEN_BIND_ACK_ENV = "HEADROOM_ALLOW_UNAUTHENTICATED_BIND"

#: The inbound credential the policy is looking for. Named here so the error
#: text and the gate cannot drift on the variable name.
PROXY_TOKEN_ENV = "HEADROOM_PROXY_TOKEN"

#: uvicorn's default when no host is configured at all.
DEFAULT_BIND_HOST = "127.0.0.1"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class OpenBindRefused(ConfigurationError):
    """The proxy was asked to bind a non-loopback interface with no inbound token.

    Raised by :func:`enforce_bind_policy`. Callers that own a process (the CLI,
    :func:`run_server`) turn it into a clean non-zero exit; embedders see a
    :class:`~headroom.exceptions.ConfigurationError` they can handle.
    """

    def __init__(self, decision: BindPolicyDecision) -> None:
        self.decision = decision
        super().__init__(decision.message())


@dataclass(frozen=True)
class BindPolicyDecision:
    """Outcome of evaluating the listener bind against the inbound auth config."""

    host: str
    token_configured: bool
    loopback: bool
    acknowledged: bool

    @property
    def open_bind(self) -> bool:
        """True when non-loopback callers can reach ``/v1/*`` with no credential."""
        return not self.token_configured and not self.loopback

    @property
    def refused(self) -> bool:
        """True when the proxy must not start: open bind, no acknowledgement."""
        return self.open_bind and not self.acknowledged

    def message(self) -> str:
        """Operator-facing explanation. Names both remedies explicitly."""
        if not self.open_bind:
            return f"bind {self.host}: inbound auth OK"
        if self.acknowledged:
            return (
                f"bind {self.host}: non-loopback bind with NO {PROXY_TOKEN_ENV} — "
                f"/v1/* is UNAUTHENTICATED; acknowledged via {OPEN_BIND_ACK_ENV}=1"
            )
        return (
            f"Refusing to bind {self.host!r}: the proxy would serve its /v1/* data-plane "
            f"routes to every peer on the network without authentication, relaying "
            f"requests with the operator's upstream credentials. Either set "
            f"{PROXY_TOKEN_ENV} (e.g. `openssl rand -hex 32`) so non-loopback callers "
            f"must present a bearer token, bind a loopback address (--host 127.0.0.1), "
            f"or — only when the runtime already restricts who can reach the port, such "
            f"as a container published on 127.0.0.1 — set {OPEN_BIND_ACK_ENV}=1 to "
            f"acknowledge the open bind explicitly."
        )


def _env_flag(environ: Mapping[str, str], name: str) -> bool:
    value = environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in _TRUE_VALUES


def evaluate_bind_policy(
    host: str | None,
    proxy_token: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> BindPolicyDecision:
    """Decide whether ``host`` may be bound with the given inbound token.

    ``host`` ``None``/empty means "not configured", which uvicorn resolves to
    :data:`DEFAULT_BIND_HOST`; it is resolved the same way here so an absent
    host is never mistaken for a non-loopback one. ``proxy_token`` ``None``
    falls back to :data:`PROXY_TOKEN_ENV`, matching the security gate.
    ``environ`` is injectable for tests; production passes nothing.
    """
    env = os.environ if environ is None else environ
    resolved_host = (host or "").strip() or DEFAULT_BIND_HOST
    token = (proxy_token or env.get(PROXY_TOKEN_ENV) or "").strip()
    return BindPolicyDecision(
        host=resolved_host,
        token_configured=bool(token),
        loopback=is_loopback_host(resolved_host),
        acknowledged=_env_flag(env, OPEN_BIND_ACK_ENV),
    )


def enforce_bind_policy(
    host: str | None,
    proxy_token: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> BindPolicyDecision:
    """Evaluate the bind policy and refuse an unacknowledged open bind.

    Raises :class:`OpenBindRefused` when the bind must not proceed. When the
    open bind is acknowledged, emits the ``proxy_open_bind`` warning (kept from
    the pre-0.40 behaviour so log-based monitoring keeps working) and returns.
    """
    decision = evaluate_bind_policy(host, proxy_token, environ=environ)
    if decision.refused:
        logger.error("event=proxy_open_bind_refused host=%s", decision.host)
        raise OpenBindRefused(decision)
    if decision.open_bind:
        logger.warning(
            "event=proxy_open_bind host=%s — proxy is bound to a non-loopback "
            "interface with no %s set; the /v1/* data-plane routes are reachable "
            "WITHOUT authentication. This was acknowledged via %s=1; set %s to "
            "require a bearer token from non-loopback callers instead.",
            decision.host,
            PROXY_TOKEN_ENV,
            OPEN_BIND_ACK_ENV,
            PROXY_TOKEN_ENV,
        )
    return decision

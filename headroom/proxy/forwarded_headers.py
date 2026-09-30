"""Trusted-gateway gate for ``X-Forwarded-*`` headers — Phase F PR-F4.

The proxy must not blindly trust ``X-Forwarded-For``,
``X-Forwarded-Proto``, or ``X-Forwarded-Host`` from arbitrary clients —
a malicious upstream client can forge any of those values and spoof
their origin IP, scheme, or host. We trust them ONLY when the
connecting peer's IP is in a configured CIDR allow-list (i.e. behind
a known reverse proxy / API gateway).

Configuration
-------------

Single env var, comma-separated CIDR blocks::

    HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS=10.0.0.0/8,172.16.0.0/12,fd00::/8

Whitespace around the commas is tolerated. Empty / unset is the
**default** and the **most secure** setting — it means *no gateway is
trusted*, so every ``X-Forwarded-*`` header is ignored regardless of
peer.

Behaviour matrix
----------------

============================  =====================  ========
Allow-list state              Peer in list?          Result
============================  =====================  ========
unset / empty (default)       n/a                    headers IGNORED
configured                    yes                    headers HONORED
configured                    no                     headers IGNORED + ``forwarded_headers_rejected`` event
============================  =====================  ========

Public API
----------

* :func:`resolve_client_ip` — the IP to log / rate-limit / authorize on.
* :func:`trusted_forwarded_headers` — sanitized ``{proto, host, for}``
  dict; values are empty strings when the gate fails.

Both helpers cache their result on ``request.state`` so they run at
most once per request.

Constraints (per project memory)
--------------------------------

* configurable: env var only, no other config surface.
* no hardcodes: every CIDR comes from the env var.
* no regexes: parsing uses :mod:`ipaddress` from the stdlib.
* no silent fallbacks: a malformed CIDR raises ``ValueError`` at
  startup; every spoof rejection emits a structured log event.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from typing import TYPE_CHECKING, Any

from headroom.proxy.forwarded_policy import (
    ForwardedHeaderInputs,
    header_first,
    normalize_ip,
    parse_cidr_list,
    peer_is_trusted_gateway,
    resolve_forwarded_headers,
)

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)

__all__ = [
    "TRUSTED_DASHBOARD_CLIENT_CIDRS_ENV",
    "TRUSTED_GATEWAY_CIDRS_ENV",
    "load_trusted_dashboard_client_cidrs",
    "load_trusted_gateway_cidrs",
    "peer_is_trusted_gateway",
    "resolve_client_ip",
    "trusted_forwarded_headers",
]


#: Environment variable that holds the comma-separated CIDR allow-list.
TRUSTED_GATEWAY_CIDRS_ENV = "HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS"
TRUSTED_DASHBOARD_CLIENT_CIDRS_ENV = "HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS"


def _parse_cidr_list(
    raw: str,
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Parse a comma-separated CIDR list. Empty / whitespace → empty tuple.

    Whitespace around commas is tolerated. Empty individual entries
    (e.g. trailing comma) are skipped. Malformed entries raise
    :class:`ValueError` — we *deliberately* do not silently skip bad
    CIDRs, because a config typo that quietly empties the allow-list
    would silently downgrade the proxy from "strict" to "more strict",
    masking the operator's intent.
    """
    return parse_cidr_list(raw)


def load_trusted_gateway_cidrs(
    raw: str | None = None,
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Load and parse the trusted-gateway CIDR allow-list.

    ``raw`` is exposed for tests and direct callers; production code
    passes nothing and we read :data:`TRUSTED_GATEWAY_CIDRS_ENV` from
    the process environment. A malformed entry raises
    :class:`ValueError` — let it propagate so the failure is loud at
    startup instead of silently disabling the gate.
    """
    if raw is None:
        raw = os.environ.get(TRUSTED_GATEWAY_CIDRS_ENV, "")
    return _parse_cidr_list(raw)


def load_trusted_dashboard_client_cidrs(
    raw: str | None = None,
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Parse the Dashboard client CIDR allow-list from its environment variable."""
    if raw is None:
        raw = os.environ.get(TRUSTED_DASHBOARD_CLIENT_CIDRS_ENV, "")
    try:
        return _parse_cidr_list(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {TRUSTED_DASHBOARD_CLIENT_CIDRS_ENV} entry: {exc}") from exc


def _normalize_ip(
    host: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse ``host`` into an IPv4/IPv6 address, unmapping ``::ffff:*``.

    IPv4-mapped IPv6 addresses (``::ffff:10.0.0.1``) — emitted by Linux
    dual-stack sockets — are normalized to their underlying IPv4 form
    so a CIDR allow-list of ``10.0.0.0/8`` matches them naturally.
    Returns ``None`` on malformed input; callers treat that as "not a
    trusted gateway".
    """
    return normalize_ip(host)


def _peer_host(request: Any) -> str | None:
    """Pull ``request.client.host`` defensively (TestClient may omit)."""
    client = getattr(request, "client", None)
    if client is None:
        return None
    return getattr(client, "host", None)


def _header_first(value: str) -> str:
    """Return the leftmost element of a comma-separated header value.

    ``X-Forwarded-For: client, proxy1, proxy2`` → ``"client"``. Empty
    input returns ``""``. We intentionally do NOT walk the chain — the
    leftmost hop is the only one whose authenticity the immediate
    gateway can vouch for, and beyond that we have no trust signal.
    """
    return header_first(value)


def _read_header(request: Any, name: str) -> str:
    """Read a header case-insensitively, ``""`` on miss."""
    headers = getattr(request, "headers", None)
    if headers is None:
        return ""
    try:
        value = headers.get(name)
    except AttributeError:
        return ""
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("latin-1")
        except UnicodeDecodeError:  # pragma: no cover - defensive
            return ""
    return str(value)


def _emit_rejection_event(
    peer_host: str | None,
    fwd_for: str,
    fwd_proto: str,
    fwd_host: str,
) -> None:
    """One-line structured log for every spoof-rejection.

    Loud-by-design: an operator running a misconfigured network MUST
    see this so they can either widen their CIDR allow-list or fix the
    upstream proxy. The event name is stable for grep / Prometheus
    log-based alerts.
    """
    logger.warning(
        "forwarded_headers_rejected",
        extra={
            "event": "forwarded_headers_rejected",
            "peer_ip": peer_host or "",
            "forwarded_for": fwd_for,
            "forwarded_proto": fwd_proto,
            "forwarded_host": fwd_host,
        },
    )


def _resolve(request: Any) -> tuple[str, dict[str, str]]:
    """Compute (client_ip, sanitized_forwarded_dict) once.

    Cached on ``request.state.client_ip`` /
    ``request.state.forwarded`` so repeated calls within a single
    request are free.
    """
    state = getattr(request, "state", None)
    if state is not None:
        cached_ip = getattr(state, "client_ip", None)
        cached_fwd = getattr(state, "forwarded", None)
        if cached_ip is not None and cached_fwd is not None:
            return cached_ip, cached_fwd

    inputs = ForwardedHeaderInputs(
        peer_host=_peer_host(request) or "",
        forwarded_for=_read_header(request, "x-forwarded-for"),
        forwarded_proto=_read_header(request, "x-forwarded-proto"),
        forwarded_host=_read_header(request, "x-forwarded-host"),
    )
    resolution = resolve_forwarded_headers(inputs, load_trusted_gateway_cidrs())
    if resolution.rejected:
        _emit_rejection_event(
            inputs.peer_host or None,
            inputs.forwarded_for,
            inputs.forwarded_proto,
            inputs.forwarded_host,
        )
    client_ip = resolution.client_ip
    forwarded = resolution.forwarded

    if state is not None:
        try:
            state.client_ip = client_ip
            state.forwarded = forwarded
        except Exception:  # pragma: no cover - defensive
            # Some test fakes use a frozen ``state`` namespace; don't
            # crash — the helpers still return the right value, just
            # without caching.
            pass
    return client_ip, forwarded


def resolve_client_ip(request: Request) -> str:
    """Return the client IP to use for logging / auth / rate-limit.

    Always falls back to ``request.client.host`` when the gate fails
    or no usable forwarded value is present. Returns ``""`` only if
    even ``request.client`` is ``None`` (TestClient / UDS).
    """
    ip, _ = _resolve(request)
    return ip


def trusted_forwarded_headers(request: Request) -> dict[str, str]:
    """Return the sanitized ``X-Forwarded-*`` triple.

    Keys: ``"for"``, ``"proto"``, ``"host"``. Every value is the empty
    string when the gateway gate fails, so callers can use simple
    truthiness checks (``if fwd["proto"]: ...``).
    """
    _, fwd = _resolve(request)
    # Defensive copy: callers writing into the dict must not poison
    # the request-state cache.
    return dict(fwd)

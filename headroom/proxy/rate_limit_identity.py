"""Who a request is charged to, for the proxy's request/token rate limits.

One rule for every provider handler (Anthropic, OpenAI, Gemini). Before this
module each handler keyed its buckets differently, and every rule could be
walked around (VAPT finding 01-F3):

* OpenAI keyed on an HMAC of the full ``Authorization`` / ``api-key`` value, so
  a caller rotating that header got a fresh bucket per request; callers with no
  credential all shared one ``"default"`` bucket.
* Anthropic keyed on the first 16 characters of the key plus the peer address —
  mostly the common ``sk-ant-api03-`` prefix.
* Gemini keyed on the first 20 characters of ``x-goog-api-key`` with no peer.

The rule now:

1. Every identity is **owned by the peer** — ``resolve_client_ip``, which honours
   ``X-Forwarded-For`` only from ``HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS`` peers.
   IPv6 peers are grouped by ``/64`` so rotating addresses inside one
   allocation does not rotate buckets.
2. For a **trusted** request the provider credential is part of the bucket, so
   distinct principals behind one address (a team gateway, a NAT) do not share a
   limit — decision D2, one proxy is shared by several principals. A request is
   trusted when it presented the proxy token (the security gate records that on
   ``request.state``) or came directly from loopback. A trusted-gateway peer is
   trusted to report the caller's address, not to vouch that the caller
   authenticated, so forwarded requests without the token are untrusted. A
   direct request from a gateway address that forwards no client address is
   judged like any other direct request.
3. For an **untrusted** request (no token configured and a remote caller) the
   credential is ignored: the bucket is the peer. Rotating a header cannot mint
   a new bucket for someone the proxy cannot authenticate.
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets
from typing import Any

from headroom.proxy.forwarded_headers import load_trusted_gateway_cidrs, resolve_client_ip
from headroom.proxy.forwarded_policy import peer_is_trusted_gateway
from headroom.proxy.loopback_guard import is_loopback_host

# Set by the security gate when a caller presented a valid HEADROOM_PROXY_TOKEN.
PROXY_AUTHENTICATED_STATE_ATTR = "proxy_authenticated"

# Checked in order; the first present header is the caller's provider credential.
_CREDENTIAL_HEADERS = ("authorization", "x-api-key", "api-key", "x-goog-api-key")

# Process-local key: bucket names never contain recoverable credential material.
_IDENTITY_SECRET = secrets.token_bytes(32)


def _peer_group(ip: str) -> str:
    """Normalise a peer address; IPv6 is grouped by its /64."""
    if not ip:
        return "peer:unknown"
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return f"peer:{ip[:64]}"
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return f"peer:{addr.ipv4_mapped}"
        network = ipaddress.IPv6Network((int(addr) >> 64 << 64, 64))
        return f"peer:{network}"
    return f"peer:{addr}"


def _credential(headers: Any) -> str | None:
    for name in _CREDENTIAL_HEADERS:
        try:
            value = headers.get(name)
        except AttributeError:
            return None
        if value:
            return f"{name}:{value}"
    return None


def _relayed_by_trusted_gateway(request: Any) -> bool:
    """True when a trusted-gateway peer forwarded a client address for this request.

    Any ``X-Forwarded-For`` value counts, even one the forwarded-header policy
    cannot use, so a relayed request is never mistaken for a direct one.
    """
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    if not host or not peer_is_trusted_gateway(host, load_trusted_gateway_cidrs()):
        return False
    headers = getattr(request, "headers", None)
    return bool(headers is not None and headers.get("x-forwarded-for"))


def is_trusted_request(request: Any) -> bool:
    """True when the caller presented the proxy token, or connected directly from loopback.

    A request relayed by a trusted gateway (even one on loopback) is keyed by
    the forwarded client address unless it presented the proxy token: the
    gateway vouches for that address, not for the caller's credential. A direct
    request from a loopback gateway address, with no forwarded client address,
    is still a direct loopback caller.
    """
    state = getattr(request, "state", None)
    if state is not None and getattr(state, PROXY_AUTHENTICATED_STATE_ATTR, False):
        return True
    if _relayed_by_trusted_gateway(request):
        return False
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    return bool(host and is_loopback_host(host))


def rate_limit_identity(request: Any, headers: Any = None) -> str:
    """Return the bucket this request is charged to. See the module docstring."""
    headers = headers if headers is not None else getattr(request, "headers", {})
    peer = _peer_group(resolve_client_ip(request) or "")
    credential = _credential(headers) if is_trusted_request(request) else None
    if credential is None:
        return peer
    digest = hmac.digest(_IDENTITY_SECRET, credential.encode("utf-8", "replace"), "sha256")
    return f"{peer}|cred:{digest.hex()[:32]}"

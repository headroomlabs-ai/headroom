"""SSRF guard for client-supplied upstream base URLs (WEB-01).

Clients may redirect the proxy's upstream via the ``x-headroom-base-url`` header
(BYOK / custom OpenAI-compatible endpoints). Without validation this lets a
caller turn the proxy into a confused deputy — reaching cloud-metadata
(``169.254.169.254``) or internal RFC1918 hosts the caller cannot reach directly.

Policy:
  * Default: reject destinations that resolve to private, loopback, link-local,
    or otherwise non-public addresses. Public hosts (api.openai.com, api.x.ai,
    Azure, ...) are allowed so ordinary BYOK keeps working.
  * When ``HEADROOM_ALLOWED_BASE_URLS`` is set (comma-separated hosts or URLs),
    bare hosts permit every safe scheme/port for that host, while URLs permit
    only their exact normalized origin. Because that is an explicit operator
    choice, allowlisted destinations may point at internal/on-prem endpoints.

Answering "is this safe?" is only half the job: the answer has to survive to the
socket. Every accepted destination therefore has the addresses it was judged on
recorded against *the request that asked* (see :func:`guarded_pin`), and the
proxy's HTTP client dials one of *those* instead of resolving the name a second
time — ``headroom/proxy/upstream_pinning.py`` is the connect-time half.

The record is a :mod:`contextvars` scope, not a process-global cache, and that
distinction is the safety property rather than a tidiness one. A hostname-keyed
cache can only answer "is there a pin for this name right now?", so an absent
one is indistinguishable from a name that was never guarded — which means a
connection for a guarded destination whose pin is missing falls back to ordinary
DNS, the exact re-resolution this module exists to prevent. A scope answers the
question that actually matters, "was *this* request's destination guarded?", so
a guarded destination with no usable pin is a denial. It also stops one caller's
pin from being visible to another caller's connection.

This module intentionally depends only on the standard library so it is safe to
import from any handler without risking an import cycle.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from contextvars import ContextVar
from dataclasses import dataclass
from urllib.parse import urlparse

ALLOWED_BASE_URLS_ENV = "HEADROOM_ALLOWED_BASE_URLS"

# `socket.getaddrinfo` has no timeout parameter and runs on whatever thread
# calls it -- which, for the proxy, is the event loop. A caller-supplied host
# that resolves slowly therefore stalls every other in-flight request, so the
# lookup is bounded here and fails closed when it overruns. Callers already in
# async context should prefer `is_safe_upstream_url_async`, which keeps the
# wait off the loop entirely.
RESOLVE_TIMEOUT_ENV = "HEADROOM_UPSTREAM_RESOLVE_TIMEOUT_S"
_DEFAULT_RESOLVE_TIMEOUT_S = 3.0
_RESOLVER_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hr-upstream-dns")


def _resolve_timeout_seconds() -> float:
    raw = (os.environ.get(RESOLVE_TIMEOUT_ENV) or "").strip()
    if not raw:
        return _DEFAULT_RESOLVE_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_RESOLVE_TIMEOUT_S
    return value if value > 0 else _DEFAULT_RESOLVE_TIMEOUT_S


_SAFE_SCHEMES = {"http", "https", "ws", "wss"}

# Pinning the addresses a check was made against.
#
# `socket.getaddrinfo` below and the HTTP client's own lookup at connect time
# are two separate resolutions of the same name, and an attacker who controls
# its authoritative DNS answers them differently: a public address for the
# check, 169.254.169.254 (or an RFC1918 host) for the connection. That is DNS
# rebinding, and it makes a verdict based only on the first answer advisory --
# the socket never sees it. So the addresses that were actually judged are
# recorded for the connection to dial.
#
# The record lives in a contextvar, which scopes it to the request that made the
# check: a task inherits its creating context, so the pin an `async def` route
# writes is the pin its own upstream call reads, and no other caller's. The
# entry is also the *marker* that this destination went through the guard --
# see `guarded_pin`.
#
# The TTL bounds how stale a judged address may be by the time a socket opens.
# It is short on purpose and deliberately not configurable: a pin only has to
# bridge a check and the connection it authorises, while an operator who
# stretched it to hours would be authorising a connection on a DNS answer from
# hours ago. Running out is NOT a fallback to ordinary resolution -- the whole
# point is that the destination is known to be caller-supplied, so the
# connect-time layer denies rather than resolving the attacker's name again.
_PIN_TTL_SECONDS = 60.0
# Hostnames are caller-supplied, so a scope is bounded even though it dies with
# the request that owns it. Overflow rejects the destination rather than leaving
# it unpinned, since an unpinned entry is one the connect layer would have to
# resolve itself. A request guards one or two upstreams; this is not reachable
# by any legitimate route.
_PIN_MAX_ENTRIES = 64


@dataclass(frozen=True)
class UpstreamPin:
    """The addresses one guarded destination was judged on, and when they lapse.

    An instance existing at all is the statement "this request had this host
    validated by the guard". ``addresses`` is what the check accepted;
    ``expires_at`` is a :func:`time.monotonic` deadline past which the answer is
    too old to authorise a socket.
    """

    addresses: tuple[str, ...]
    expires_at: float

    def is_expired(self, now: float | None = None) -> bool:
        return (time.monotonic() if now is None else now) >= self.expires_at


_PIN_SCOPE: ContextVar[dict[str, UpstreamPin] | None] = ContextVar(
    "headroom_upstream_pin_scope", default=None
)


def begin_pin_scope() -> dict[str, UpstreamPin]:
    """Create (or return) this context's pin scope. Returns the live mapping.

    Call this on the thread/task that will *make* the request, before handing
    the check off anywhere else. A contextvar ``set`` inside a worker thread
    lands in that thread's copy of the context and is lost when it returns,
    so :func:`is_safe_upstream_url_async` establishes the mapping here and then
    mutates it from the worker -- the mapping object is shared, the binding is
    not.
    """
    scope = _PIN_SCOPE.get()
    if scope is None:
        scope = {}
        _PIN_SCOPE.set(scope)
    return scope


def _record_validated_addresses(
    host: str,
    addresses: tuple[str, ...],
    scope: dict[str, UpstreamPin] | None = None,
) -> bool:
    """Remember what ``host`` was judged on, for the connection that follows.

    Returns False when the scope is full, which the caller turns into a
    rejection: a destination we cannot pin is one the connection would resolve
    for itself.
    """
    if not host or not addresses:
        return False
    if scope is None:
        scope = begin_pin_scope()
    if host not in scope and len(scope) >= _PIN_MAX_ENTRIES:
        return False
    scope[host] = UpstreamPin(addresses, time.monotonic() + _PIN_TTL_SECONDS)
    return True


def guarded_pin(host: str) -> UpstreamPin | None:
    """Return this request's pin for ``host``, expired or not, else ``None``.

    ``None`` means "the current request never had this destination checked" --
    a configured provider upstream, a proxy endpoint, an allowlisted host --
    and such a destination resolves normally, exactly as it always has.

    A returned pin means the opposite, and it says so even when it has expired.
    That is the whole reason expiry does not delete the entry: the connect-time
    layer has to be able to tell "never guarded" from "guarded, but the judged
    addresses are too old to use", because only the first of those is safe to
    resolve by name.
    """
    scope = _PIN_SCOPE.get()
    if not scope:
        return None
    key = (host or "").strip().lower()
    if not key:
        return None
    return scope.get(key)


def validated_addresses(host: str) -> tuple[str, ...] | None:
    """Return the still-valid addresses this request's check accepted for ``host``.

    ``None`` covers both "never guarded here" and "guarded but lapsed", so it is
    not a safe basis for a connect decision -- use :func:`guarded_pin`, which
    keeps those apart. Retained for callers that only want the addresses.
    """
    pin = guarded_pin(host)
    if pin is None or pin.is_expired():
        return None
    return pin.addresses


def clear_validated_addresses() -> None:
    """Drop this context's pins. For tests, and between reused contexts."""
    _PIN_SCOPE.set(None)


def _allowlisted_destinations() -> tuple[set[str], set[tuple[str, str, int]]] | None:
    raw = os.environ.get(ALLOWED_BASE_URLS_ENV)
    if not raw or not raw.strip():
        return None
    hosts: set[str] = set()
    origins: set[tuple[str, str, int]] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "://" not in item:
            parsed = urlparse(f"//{item}")
            if parsed.hostname:
                hosts.add(parsed.hostname.lower())
            continue
        parsed = urlparse(item)
        if parsed.scheme.lower() not in _SAFE_SCHEMES or not parsed.hostname:
            continue
        try:
            port = parsed.port
        except ValueError:
            continue
        if port is None:
            port = 443 if parsed.scheme.lower() in {"https", "wss"} else 80
        origins.add((parsed.scheme.lower(), parsed.hostname.lower(), port))
    return hosts, origins


# RFC 6052 / RFC 8215: these IPv6 prefixes embed an IPv4 address in their low
# 32 bits, and `ipaddress` reports the well-known one as globally routable. On a
# NAT64 network `64:ff9b::7f00:1` reaches 127.0.0.1, so the embedded address is
# what has to be judged. 6to4, Teredo and IPv4-mapped forms are already caught
# by the `is_global` test below.
_NAT64_PREFIXES = (
    ipaddress.IPv6Network("64:ff9b::/96"),
    ipaddress.IPv6Network("64:ff9b:1::/48"),
)


def _nat64_embedded_ipv4(addr: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if not any(addr in prefix for prefix in _NAT64_PREFIXES):
        return None
    try:
        return ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
    except (ipaddress.AddressValueError, ValueError):  # pragma: no cover - defensive
        return None


def _is_internal_address(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable (e.g. scoped link-local) -> treat as unsafe
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    ):
        return True
    # Anything not globally routable. This is what catches RFC 6598 shared
    # address space (100.64.0.0/10) -- which `is_private` does not flag, and
    # which reaches ISP and cloud-internal infrastructure -- along with
    # benchmarking (198.18/15), TEST-NET, 240/4, 6to4 and Teredo tunnels that
    # embed an internal IPv4, and any future special-use range the stdlib
    # learns about.
    if not addr.is_global:
        return True
    if isinstance(addr, ipaddress.IPv6Address):
        embedded = _nat64_embedded_ipv4(addr)
        if embedded is not None and _is_internal_address(str(embedded)):
            return True
    return False


def is_safe_upstream_url(url: str, *, scope: dict[str, UpstreamPin] | None = None) -> bool:
    """Return True if ``url`` is a safe client-chosen upstream destination.

    In allowlist mode only allowlisted hosts pass. Otherwise the host is
    resolved and rejected if any resolved address is internal/metadata, which
    also catches DNS names that point at private space.

    A destination accepted on its addresses also has them pinned against the
    calling request, so the connection that follows cannot be re-pointed by a
    second DNS answer. Allowlist mode pins nothing: it admits a host by name
    without resolving it at all, precisely so split-horizon and on-prem
    endpoints -- whose addresses are the operator's business, and may
    legitimately move or round-robin -- keep working.

    ``scope`` is the pin mapping to write into; it defaults to the calling
    context's. :func:`is_safe_upstream_url_async` passes its caller's mapping
    explicitly because the check itself runs on a worker thread.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in _SAFE_SCHEMES:
        return False
    host = parsed.hostname
    if not host:
        return False

    allow = _allowlisted_destinations()
    if allow is not None:
        hosts, origins = allow
        if host.lower() in hosts:
            return True
        try:
            port = parsed.port
        except ValueError:
            return False
        if port is None:
            port = 443 if parsed.scheme.lower() in {"https", "wss"} else 80
        return (parsed.scheme.lower(), host.lower(), port) in origins

    try:
        infos = _RESOLVER_POOL.submit(
            socket.getaddrinfo, host, None, 0, 0, socket.IPPROTO_TCP
        ).result(timeout=_resolve_timeout_seconds())
    except (OSError, _FutureTimeout):
        # Resolution and connection are separate operations, so allowing a DNS
        # miss here would fail open if the name resolves on the later lookup.
        # A lookup that overruns the budget is treated the same way.
        # Operators can explicitly allowlist split-horizon/internal endpoints.
        return False
    addresses = tuple(dict.fromkeys(str(info[4][0]) for info in infos))
    if any(_is_internal_address(address) for address in addresses):
        return False
    # Every address in this answer passed, so pin them all: the connection may
    # dial any one of them and still be dialling something this check accepted.
    # Keeping the whole set (in resolver order) rather than a single winner
    # leaves the connection somewhere to go when the first address is an AAAA on
    # a host with no IPv6 route -- the fallback the OS resolver would otherwise
    # have done for us.
    #
    # A destination that cannot be pinned is rejected rather than allowed
    # through unpinned: an unpinned client-supplied host is one the connection
    # would have to resolve for itself, which is the hole this closes.
    return _record_validated_addresses(host.lower(), addresses, scope)


async def is_safe_upstream_url_async(url: str) -> bool:
    """Async form of :func:`is_safe_upstream_url` for event-loop callers.

    Same policy; the blocking resolution runs off the loop so a hostile or
    slow-resolving hostname cannot stall unrelated in-flight requests.

    The pin scope is opened *here*, on the caller's task, and handed to the
    worker to fill in. Letting the worker open it instead would bind the
    contextvar in the thread's throwaway copy of the context, so the pin would
    vanish before the request it authorises ever reached a socket -- and a
    guarded destination with no pin is a denied one.
    """
    scope = begin_pin_scope()
    return await asyncio.to_thread(is_safe_upstream_url, url, scope=scope)

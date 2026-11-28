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
import functools
import ipaddress
import logging
import os
import socket
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from contextvars import ContextVar
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

ALLOWED_BASE_URLS_ENV = "HEADROOM_ALLOWED_BASE_URLS"

# `socket.getaddrinfo` has no timeout parameter and runs on whatever thread
# calls it -- which, for the proxy, is the event loop. A caller-supplied host
# that resolves slowly therefore stalls every other in-flight request, so the
# lookup is bounded here and fails closed when it overruns. Callers already in
# async context should prefer `is_safe_upstream_url_async`, which keeps the
# wait off the loop entirely.
RESOLVE_TIMEOUT_ENV = "HEADROOM_UPSTREAM_RESOLVE_TIMEOUT_S"
_DEFAULT_RESOLVE_TIMEOUT_S = 3.0
_RESOLVER_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="hr-upstream-dns")

# An overrun lookup is abandoned, not cancelled: `getaddrinfo` keeps its worker
# until the system resolver gives up, so slow names could occupy the resolver
# and get every other client's upstream refused. Hence: one shared lookup per
# name; each charged to the caller's network (never a credential), with a few
# in flight per network and all of them below the worker count (a small extra
# reserve is for refreshes); and a memory of hosts judged public. A remembered
# host is answered from memory, pinned like a fresh answer, and refreshed in
# the background once its answer is a minute old. Only a refresh can drop it:
# one that comes back internal or NXDOMAIN does; one that is slow, fails or
# cannot start leaves the old public answer serving. A network may fill only a
# small share of the memory.
#
# Not covered: many networks together can fill the lookup cap, so a name not
# already remembered is refused while that lasts; evicting remembered hosts
# takes about `_REMEMBER_MAX_ENTRIES / _REMEMBER_MAX_PER_CLIENT` networks.
_MAX_LOOKUPS_PER_CLIENT = 4
_MAX_LOOKUPS_IN_FLIGHT = 12
_REFRESH_RESERVE = 2
_REFRESH_AFTER_S = 60.0
_REMEMBER_MAX_ENTRIES = 8192
_REMEMBER_MAX_PER_CLIENT = 32
# Answers that say the name has no address, as opposed to a resolver that is
# slow or failing (EAI_AGAIN, EAI_FAIL): these drop a remembered host.
_NO_ADDRESS_ERRORS = frozenset(
    code for code in (socket.EAI_NONAME, getattr(socket, "EAI_NODATA", None)) if code is not None
)

# Lookups made outside any request are charged here, apart from every client.
_INTERNAL_CLIENT = "internal"
_UPSTREAM_CLIENT: ContextVar[str] = ContextVar("headroom_upstream_client", default=_INTERNAL_CLIENT)


def upstream_client_key(client_ip: str) -> str:
    """The network a request's upstream lookups are charged to.

    IPv4 by address; IPv6 by /48, the usual end-site allocation, so rotating
    addresses inside one allocation does not mint new allowances.
    """
    try:
        addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return f"peer:{client_ip[:64]}"
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return f"peer:{addr.ipv4_mapped}"
        return f"peer:{ipaddress.IPv6Network((int(addr) >> 80 << 80, 48))}"
    return f"peer:{addr}"


def bind_upstream_client(client: str) -> None:
    """Charge this context's upstream lookups to ``client`` (see ``upstream_client_key``)."""
    _UPSTREAM_CLIENT.set(client)


@dataclass
class _Remembered:
    addresses: tuple[str, ...]
    judged_at: float
    owner: str


class _Resolver:
    """Shared lookups, their limits, and the remembered answers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight: dict[str, tuple[Future[list], str]] = {}
        self._per_client: dict[str, int] = {}
        self._memory: OrderedDict[str, _Remembered] = OrderedDict()
        self._remembered_by: dict[str, int] = {}

    def remembered(self, host: str) -> tuple[tuple[str, ...], bool] | None:
        """Return ``(addresses, due_for_refresh)`` for a remembered host, else ``None``."""
        with self._lock:
            entry = self._memory.get(host)
            if entry is None:
                return None
            self._memory.move_to_end(host)
            return entry.addresses, time.monotonic() - entry.judged_at >= _REFRESH_AFTER_S

    def start(self, host: str, client: str, *, refresh: bool) -> Future[list] | None:
        """Return the in-flight lookup for ``host``, starting one if allowed."""
        with self._lock:
            running = self._in_flight.get(host)
            if running is not None:
                return running[0]
            limit = _MAX_LOOKUPS_IN_FLIGHT + (_REFRESH_RESERVE if refresh else 0)
            if (
                len(self._in_flight) >= limit
                or self._per_client.get(client, 0) >= _MAX_LOOKUPS_PER_CLIENT
            ):
                return None
            future = _RESOLVER_POOL.submit(socket.getaddrinfo, host, None, 0, 0, socket.IPPROTO_TCP)
            self._in_flight[host] = (future, client)
            self._per_client[client] = self._per_client.get(client, 0) + 1
        future.add_done_callback(lambda done: self._finished(host, done))
        return future

    def _finished(self, host: str, done: Future[list]) -> None:
        with self._lock:
            running = self._in_flight.get(host)
            if running is None or running[0] is not done:
                return
            del self._in_flight[host]
            owner = running[1]
            _decrement(self._per_client, owner)
            if done.cancelled():
                return
            error = done.exception()
            if error is None:
                addresses = _addresses(done.result())
                if addresses and not any(_is_internal_address(a) for a in addresses):
                    self._remember(host, addresses, owner)
                    return
                self._forget(host)
            elif isinstance(error, socket.gaierror) and error.errno in _NO_ADDRESS_ERRORS:
                self._forget(host)

    def _remember(self, host: str, addresses: tuple[str, ...], owner: str) -> None:
        existing = self._memory.get(host)
        if existing is not None:
            existing.addresses, existing.judged_at = addresses, time.monotonic()
            self._memory.move_to_end(host)
            return
        if self._remembered_by.get(owner, 0) >= _REMEMBER_MAX_PER_CLIENT:
            self._forget(next(h for h, e in self._memory.items() if e.owner == owner))
        while len(self._memory) >= _REMEMBER_MAX_ENTRIES:
            self._forget(next(iter(self._memory)))
        self._memory[host] = _Remembered(addresses, time.monotonic(), owner)
        self._remembered_by[owner] = self._remembered_by.get(owner, 0) + 1

    def _forget(self, host: str) -> None:
        entry = self._memory.pop(host, None)
        if entry is not None:
            _decrement(self._remembered_by, entry.owner)


def _decrement(counts: dict[str, int], key: str) -> None:
    if counts[key] > 1:
        counts[key] -= 1
    else:
        del counts[key]


_RESOLVER = _Resolver()


def _addresses(infos: list) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(info[4][0]) for info in infos))


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


def _connect_name(host: str) -> str | None:
    """Return ``host`` as the connection will name it: lower-case IDNA A-label.

    httpx IDNA-encodes a non-ASCII request host (IDNA 2008) and httpcore hands
    that A-label to ``connect_tcp``, so the check has to resolve and pin the
    same name. ``socket.getaddrinfo`` would encode a Unicode name itself, with
    IDNA 2003, which spells some names differently (``faß.de`` vs
    ``xn--fa-hia.de``). ``None`` when the host cannot be encoded at all.
    """
    key = (host or "").strip().lower()
    if key.isascii():
        return key
    import httpx  # local: this module otherwise needs only the standard library

    try:
        return httpx.URL(scheme="https", host=key).raw_host.decode("ascii").lower()
    except (httpx.InvalidURL, UnicodeError):
        return None


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
    key = _connect_name(host)
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


def allowlisted_destinations(
    env: str = ALLOWED_BASE_URLS_ENV,
) -> tuple[frozenset[str], frozenset[tuple[str, str, int]]] | None:
    """Parse a comma-separated host/URL list from ``env``; ``None`` when unset.

    Hosts are kept as the name the connection dials (lower-case IDNA A-label),
    so every spelling of a host matches the same entry.
    """
    raw = os.environ.get(env)
    if not raw or not raw.strip():
        return None
    return _parse_destinations(env, raw)


@functools.lru_cache(maxsize=16)
def _parse_destinations(
    env: str, raw: str
) -> tuple[frozenset[str], frozenset[tuple[str, str, int]]]:
    hosts: set[str] = set()
    origins: set[tuple[str, str, int]] = set()
    # Entries can carry credentials (``https://user:pass@host``), so a rejected
    # one is reported by its position, never by its text.
    for position, item in enumerate(raw.split(","), start=1):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = urlparse(item if "://" in item else f"//{item}")
            port = parsed.port
        except ValueError:
            parsed = None
        host = _connect_name(parsed.hostname or "") if parsed is not None else None
        if parsed is None or not host:
            logger.warning("ignoring %s entry %d: not a valid host, port or URL", env, position)
            continue
        if "://" not in item:
            hosts.add(host)
            continue
        scheme = parsed.scheme.lower()
        if scheme not in _SAFE_SCHEMES:
            logger.warning("ignoring %s entry %d: unsupported scheme", env, position)
            continue
        if port is None:
            port = 443 if scheme in {"https", "wss"} else 80
        origins.add((scheme, host, port))
    return frozenset(hosts), frozenset(origins)


def destination_listed(
    allow: tuple[frozenset[str], frozenset[tuple[str, str, int]]],
    scheme: str,
    host: str,
    port: int | None,
) -> bool:
    """Whether a destination matches a parsed host/origin list.

    A bare host admits every port for that host; a URL admits only its exact
    origin, with the scheme's default port filled in when ``port`` is ``None``.
    """
    hosts, origins = allow
    host = _connect_name(host) or ""
    if host in hosts:
        return True
    scheme = scheme.lower()
    if port is None:
        port = 443 if scheme in {"https", "wss"} else 80
    return (scheme, host, port) in origins


# RFC 6052 well-known prefix. A DNS64 resolver answers a host that has only an
# A record with `64:ff9b::<its IPv4>`, and the NAT64 translator delivers that to
# the IPv4 address in the low 32 bits, so the embedded address is what has to
# be judged: `64:ff9b::7f00:1` reaches 127.0.0.1. It has to be judged *before*
# the category checks, because `ipaddress` calls the whole prefix reserved (it
# sits in ::/8) -- judging the IPv6 form rejected every IPv4-only upstream on a
# DNS64 network. The local-use prefix (RFC 8215, 64:ff9b:1::/48) is not
# unwrapped: where its IPv4 bits sit depends on the operator's prefix length,
# so it stays rejected as non-global. Teredo and IPv4-mapped forms are caught
# by the `is_global` test below.
_NAT64_WELL_KNOWN_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")
# 6to4 (RFC 3056) embeds an IPv4 address in bits 16-47: the tunnel endpoint the
# IPv6 traffic is encapsulated to. A 6to4 destination embedding 169.254.169.254
# or an RFC 1918 address therefore sends the proxy's traffic into internal IPv4
# space, where any 6to4 router at that address delivers it. Only Python releases that
# carry the 2024 `ipaddress` table fix call 2002::/16 non-global; older 3.11
# patch releases call it global, so the guard cannot lean on `is_global` here.
# The mechanism is deprecated (RFC 7526) and no API upstream is reached through
# it, so the whole prefix is refused on every Python.
_SIX_TO_FOUR_PREFIX = ipaddress.IPv6Network("2002::/16")


def _nat64_embedded_ipv4(addr: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if addr not in _NAT64_WELL_KNOWN_PREFIX:
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
    if isinstance(addr, ipaddress.IPv6Address):
        embedded = _nat64_embedded_ipv4(addr)
        if embedded is not None:
            return _is_internal_address(str(embedded))
        if addr in _SIX_TO_FOUR_PREFIX:
            return True
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
    # benchmarking (198.18/15), TEST-NET, 240/4, Teredo tunnels that embed an
    # internal IPv4, and any future special-use range the stdlib
    # learns about.
    if not addr.is_global:
        return True
    return False


def _check_before_lookup(url: str) -> bool | str:
    """Decide ``url`` without DNS where possible.

    Returns the verdict, or the name to resolve when only an answer can decide.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in _SAFE_SCHEMES:
        return False
    host = parsed.hostname
    if not host:
        return False

    allow = allowlisted_destinations()
    if allow is not None:
        try:
            port = parsed.port
        except ValueError:
            port = -1  # matches no origin; a listed bare host still passes, as before
        return destination_listed(allow, parsed.scheme, host, port)

    # Resolve and pin the name the connection will dial, not the spelling the
    # client sent, or a non-ASCII host would be checked under one name and
    # connected under another.
    connect_host = _connect_name(host)
    if not connect_host:
        return False
    return connect_host


def _judge_answer(connect_host: str, infos: list, scope: dict[str, UpstreamPin] | None) -> bool:
    addresses = _addresses(infos)
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
    return _record_validated_addresses(connect_host, addresses, scope)


PROXIED_BASE_URLS_ENV = "HEADROOM_PROXIED_BASE_URLS"


def proxied_destination_listed(url: str) -> bool:
    """Whether ``url``'s destination is listed in ``HEADROOM_PROXIED_BASE_URLS``.

    The one matcher for that list: the guard asks it whether a host may be
    answered from memory, and the outbound transport asks it whether a guarded
    request may leave through the proxy by name, so the two cannot disagree
    about a destination.
    """
    allow = allowlisted_destinations(PROXIED_BASE_URLS_ENV)
    if allow is None:
        return False
    parsed = urlparse((url or "").strip())
    try:
        port = parsed.port
    except ValueError:
        return False
    return destination_listed(allow, parsed.scheme, parsed.hostname or "", port)


def _from_memory(url: str, connect_host: str, scope: dict[str, UpstreamPin] | None) -> bool | None:
    """Answer a remembered host at once; refresh it in the background when due.

    ``None`` when the host is not remembered and has to be looked up. A
    destination listed in ``HEADROOM_PROXIED_BASE_URLS`` is always looked up:
    its connection leaves through the outbound proxy by name, so no pin binds
    it, and only an answer from this request keeps the check current.
    """
    if proxied_destination_listed(url):
        return None
    remembered = _RESOLVER.remembered(connect_host)
    if remembered is None:
        return None
    addresses, due_for_refresh = remembered
    if due_for_refresh:
        _RESOLVER.start(connect_host, _UPSTREAM_CLIENT.get(), refresh=True)
    return _record_validated_addresses(connect_host, addresses, scope)


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
    context's.
    """
    decided = _check_before_lookup(url)
    if isinstance(decided, bool):
        return decided
    remembered = _from_memory(url, decided, scope)
    if remembered is not None:
        return remembered
    lookup = _RESOLVER.start(decided, _UPSTREAM_CLIENT.get(), refresh=False)
    if lookup is None:
        return False
    try:
        infos = lookup.result(timeout=_resolve_timeout_seconds())
    except (OSError, _FutureTimeout):
        # Resolution and connection are separate operations, so allowing a
        # DNS miss here would fail open if the name resolves on the later
        # lookup. Operators can allowlist split-horizon/internal endpoints.
        return False
    return _judge_answer(decided, infos, scope)


def _consume_outcome(future: asyncio.Future[list]) -> None:
    if not future.cancelled():
        future.exception()


async def is_safe_upstream_url_async(url: str) -> bool:
    """Async form of :func:`is_safe_upstream_url` for event-loop callers.

    Same policy; the lookup is awaited rather than waited on, so a hostile or
    slow-resolving hostname neither stalls the event loop nor holds one of its
    worker threads for the length of the budget.

    The pin scope is opened *here*, on the caller's task: the mapping object
    is what the check fills in, so it has to belong to the request's context.
    """
    scope = begin_pin_scope()
    decided = _check_before_lookup(url)
    if isinstance(decided, bool):
        return decided
    remembered = _from_memory(url, decided, scope)
    if remembered is not None:
        return remembered
    lookup = _RESOLVER.start(decided, _UPSTREAM_CLIENT.get(), refresh=False)
    if lookup is None:
        return False
    try:
        # Shielded: the lookup may be shared with other callers, and timing out
        # or being cancelled here must not cancel it for them.
        # The wrapper is consumed when it settles, so a failure that lands
        # after this caller gave up is not reported as never retrieved.
        waited = asyncio.wrap_future(lookup)
        waited.add_done_callback(_consume_outcome)
        infos = await asyncio.wait_for(asyncio.shield(waited), _resolve_timeout_seconds())
    except (OSError, asyncio.TimeoutError):
        return False
    return _judge_answer(decided, infos, scope)

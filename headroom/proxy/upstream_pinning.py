"""Connect-time half of the SSRF guard: dial the address that was validated.

:mod:`headroom.proxy.upstream_guard` resolves a caller-supplied upstream and
judges the answer -- and then hands the *hostname* to httpx, which resolves it
again when it opens the socket. Two resolutions mean two chances to answer, and
an attacker who controls the name's authoritative DNS takes both: a public
address for the check, ``169.254.169.254`` for the connection. The verdict was
never wrong; it just never reached the socket (DNS rebinding / TOCTOU).

This module carries it there. It replaces the connection pool's network backend
with one that substitutes a validated address for the hostname at ``connect_tcp``
time, and *only* that: the request still carries the hostname in its URL, so

  * httpx still sends ``Host: <hostname>`` -- an upstream that routes by Host
    (every gateway, every multi-tenant provider) still routes correctly; and
  * httpcore still passes the hostname to ``start_tls`` as ``server_hostname``,
    so SNI and certificate hostname verification are untouched.

That is the whole reason the substitution happens down at the socket instead of
by rewriting the URL to an IP literal. A rewritten URL takes the Host header and
the certificate identity down with it -- the upstream misroutes, and the
handshake either fails or has to be weakened to an unverified one to work at
all. It would also collapse two hostnames sharing an address into one pooled
origin, letting a connection whose certificate was checked for one host serve
requests for the other.

What "guarded" means here
-------------------------
The guard records its verdict in a contextvar scope owned by the request that
asked (``upstream_guard.guarded_pin``), so this layer can tell three states
apart rather than two:

  * **no pin** -- the current request never had this destination checked. Every
    operator-configured provider upstream (Anthropic, OpenAI, Gemini, Bedrock,
    Copilot, LiteLLM, a Kong gateway), every allowlisted host, and the proxy
    endpoint itself land here, and they resolve exactly as they always have.
  * **a live pin** -- dial one of the addresses the check accepted.
  * **a pin whose addresses have lapsed** -- *deny*. This is the state a
    hostname-keyed cache could not express: there, an aged-out pin was simply an
    absent one, so a guarded request that waited out the TTL (pool saturation,
    connection limits, a slow upstream ahead of it in the queue) fell back to
    resolving the attacker's name a second time -- reopening the exact hole.
    Denying costs a caller-supplied upstream a request it can immediately retry,
    which re-checks and re-pins; the alternative costs the guarantee.

Proxies are refused, not silently unpinned
------------------------------------------
With ``HTTP_PROXY``/``HTTPS_PROXY``/``ALL_PROXY`` or a SOCKS proxy configured,
the pool dials the *proxy*, and the target hostname travels onward inside the
protocol -- in the request line for a forwarding HTTP proxy, in ``CONNECT`` for
a tunnelling one, in the SOCKS5 address for SOCKS. Every one of those addresses
the target *by name*, and the proxy resolves it, on the proxy's network, well
after the guard ran. Pinning the proxy's own address does nothing about that, so
a caller-supplied upstream routed through a proxy is refused outright with
:class:`UnpinnableUpstreamError` rather than forwarded on an unenforced verdict.

Only guarded destinations are refused. Operator-configured upstreams have no pin
and are unaffected, so proxied deployments keep working exactly as before unless
they also accept ``x-headroom-base-url`` -- in which case the refusal is the
honest answer, and ``HEADROOM_ALLOWED_BASE_URLS`` is the supported way to admit
specific internal endpoints through a proxy.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import httpcore
import httpx

from headroom.proxy.upstream_guard import guarded_pin


class UnpinnableUpstreamError(RuntimeError):
    """A guarded upstream could not be dialled at an address the guard accepted.

    Deliberately not an ``httpcore`` connect error: those are retried, fallen
    back from, and mapped into httpx's transport errors, and none of that is
    right for a refusal. httpx's exception mapping re-raises types it does not
    know, so this surfaces to the caller as itself.
    """


def _refusal(host: str, reason: str) -> UnpinnableUpstreamError:
    return UnpinnableUpstreamError(
        f"refusing to connect to the client-supplied upstream {host!r}: {reason}. "
        "The SSRF guard accepted this destination on addresses it resolved, and "
        "those addresses cannot be enforced on this connection, so the name "
        "would be resolved a second time and could answer differently (DNS "
        "rebinding). Retry the request, or allowlist the endpoint with "
        "HEADROOM_ALLOWED_BASE_URLS — see headroom/proxy/upstream_pinning.py."
    )


class PinnedAddressBackend(httpcore.AsyncNetworkBackend):
    """Network backend that dials a validated address in place of a pinned name.

    Everything else is delegated untouched, including the ``connect_tcp``
    keyword arguments -- ``local_address`` and ``socket_options`` are how
    operators bind egress to a chosen interface, so dropping them here would
    quietly change which source address the proxy connects from.
    """

    def __init__(self, inner: httpcore.AsyncNetworkBackend) -> None:
        self._inner = inner

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        # httpcore hands us the origin host as `str`; be tolerant anyway, since
        # a bytes host would silently miss every pin and fail open.
        name = host.decode("ascii", "ignore") if isinstance(host, bytes) else host
        pin = guarded_pin(name)

        async def dial(target: str) -> httpcore.AsyncNetworkStream:
            return await self._inner.connect_tcp(
                target,
                port,
                timeout=timeout,
                local_address=local_address,
                socket_options=socket_options,
            )

        if pin is None:
            # This request never had this destination checked -- an operator
            # upstream, an allowlisted host, the proxy itself. Resolve as before.
            return await dial(host)

        if pin.is_expired():
            # A guarded destination whose judged addresses are too old to
            # authorise a socket. Resolving the name again is what this module
            # exists to prevent, so the connection is refused instead.
            raise _refusal(name, "the addresses it was validated on have expired")

        # Try each validated address in turn, in the resolver's own order. A
        # later one is reached only when an earlier one could not be connected
        # to at all -- the answer is unreachable on this host's network, e.g. an
        # AAAA record where there is no IPv6 route. That is the fallback the OS
        # resolver would have made had we handed it the name, and it costs
        # nothing in safety because every address here passed the same check.
        # Only connect failures fall through: once a socket is open, whatever
        # happens on it belongs to the caller. The last address is dialled
        # outside the loop so its failure propagates as itself.
        addresses = pin.addresses
        for address in addresses[:-1]:
            try:
                return await dial(address)
            except (httpcore.ConnectError, httpcore.ConnectTimeout):
                continue
        return await dial(addresses[-1])

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return await self._inner.connect_unix_socket(
            path, timeout=timeout, socket_options=socket_options
        )

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class GuardedUpstreamRefusingTransport(httpx.AsyncBaseTransport):
    """Refuses guarded destinations on a transport that cannot pin them.

    Wraps proxy transports, and any transport whose connection pool this module
    does not recognise. The check runs here rather than in the network backend
    because this is the last layer that still knows the *target* -- below it a
    proxied connection only ever mentions the proxy's own host.

    ``handle_async_request`` runs on the task that issued the request, which is
    the task whose contextvar scope holds the guard's verdict.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport, reason: str) -> None:
        self._inner = inner
        self._reason = reason

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if guarded_pin(host) is not None:
            raise _refusal(host, self._reason)
        return await self._inner.handle_async_request(request)

    async def __aenter__(self) -> GuardedUpstreamRefusingTransport:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._inner.__aexit__(*exc)

    async def aclose(self) -> None:
        await self._inner.aclose()


_PROXY_POOLS: tuple[type, ...] = tuple(
    pool
    for pool in (
        getattr(httpcore, "AsyncHTTPProxy", None),
        getattr(httpcore, "AsyncSOCKSProxy", None),
    )
    if pool is not None
)


def _transport_slots(client: httpx.AsyncClient) -> Iterator[tuple[Any, Any]]:
    """Yield ``(key, transport)`` for every transport ``client`` may dial through.

    The key is ``None`` for the primary transport and a ``URLPattern`` for a
    mounted one. Mounted transports matter as much as the primary: httpx builds
    them for ``HTTPS_PROXY``/``NO_PROXY``-style environments, and a request
    routed to one of those would otherwise leave through an unhooked path.
    """
    yield None, getattr(client, "_transport", None)
    yield from getattr(client, "_mounts", {}).items()


def _install(transport: Any) -> tuple[Any, bool]:
    """Return ``(transport to use, whether it pins)`` for one transport slot."""
    if transport is None:  # a mount that defers to the primary transport
        return None, False
    if isinstance(transport, GuardedUpstreamRefusingTransport):  # already installed
        return transport, False
    pool = getattr(transport, "_pool", None)
    if pool is None:
        return (
            GuardedUpstreamRefusingTransport(
                transport, "this transport exposes no connection pool to pin through"
            ),
            False,
        )
    if _PROXY_POOLS and isinstance(pool, _PROXY_POOLS):
        # The proxy resolves the target itself, from its own network, after the
        # guard ran. Pinning the proxy's address would look like a fix and be
        # none, so guarded destinations are refused on this route instead.
        return (
            GuardedUpstreamRefusingTransport(
                transport,
                "it routes through a proxy, which resolves the target hostname "
                "itself and cannot be told which address the guard accepted",
            ),
            False,
        )
    backend = getattr(pool, "_network_backend", None)
    if backend is None:
        return (
            GuardedUpstreamRefusingTransport(
                transport, "its connection pool exposes no network backend to pin through"
            ),
            False,
        )
    if not isinstance(backend, PinnedAddressBackend):
        pool._network_backend = PinnedAddressBackend(backend)
    return transport, True


def install_upstream_pinning(client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Make ``client`` dial validated addresses for guarded hosts. Returns it.

    Transports that cannot honour a pin -- proxy transports above all -- are
    wrapped so a guarded destination routed through them is refused rather than
    forwarded on a verdict nothing enforces.

    Raises ``RuntimeError`` when no direct transport could be pinned at all,
    which means httpx/httpcore no longer expose the connection pool this hooks.
    That is fail-closed on purpose: the alternative is a proxy that starts
    happily and silently re-opens the rebinding hole, and the versions involved
    are locked in ``uv.lock``, so it can only fire on a deliberate dependency
    change -- exactly when someone should be looking.
    """
    pinned = 0
    mounts = getattr(client, "_mounts", None)
    for key, transport in _transport_slots(client):
        replacement, did_pin = _install(transport)
        pinned += did_pin
        if replacement is not transport:
            if key is None:
                client._transport = replacement
            elif mounts is not None:
                mounts[key] = replacement
    if not pinned:
        raise RuntimeError(
            "cannot pin validated upstream addresses: this httpx/httpcore "
            "exposes no direct connection pool to hook. Refusing to run "
            "unpinned — see headroom/proxy/upstream_pinning.py."
        )
    return client

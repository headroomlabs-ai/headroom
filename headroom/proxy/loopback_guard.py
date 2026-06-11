"""Loopback-only access guard for /debug/* endpoints.

Unit 5 of the Codex-proxy resilience plan. A FastAPI dependency that
raises :class:`fastapi.HTTPException` with status 404 — *not* 403 — for
any request whose client address is not the loopback interface. 404 is
deliberate: debug endpoints should be invisible to external scanners,
not merely forbidden.

The guard is a ``Depends(...)``-friendly function (rather than a
middleware) because:

* FastAPI's dependency injection makes the guard explicit on each
  route, so ``ruff``/reviewers can see which endpoints are guarded.
* ``TestClient`` lets us override a dependency with
  ``app.dependency_overrides``, which is the cleanest way to simulate
  a non-loopback client in tests.
* The set of debug endpoints is small and co-located; a middleware
  would be disproportionate.

DNS-rebinding defence
---------------------
A loopback-IP check alone is not enough to keep these endpoints local.
A malicious site can use DNS rebinding to make a victim's browser send
requests to ``127.0.0.1`` while the ``Host:`` header (and the JS
``fetch`` URL) still reads ``attacker.com``. From the proxy's point of
view ``request.client.host`` is ``127.0.0.1`` (the browser, which IS on
loopback) and the IP check passes. The proxy ships a wide-open CORS
policy (``allow_origins=['*']``), so attacker JS can then read the
response.

To close that gap the guard also requires the ``Host:`` header to name
loopback — ``127.0.0.1[:port]``, ``[::1][:port]``, or
``localhost[:port]``. Same-origin XHR from a real local tool always
sets one of those values; cross-origin rebinding does not. This is the
canonical Host-header allowlist mitigation called out in OWASP's
CSRF / DNS-rebinding guidance and the standard Starlette
``TrustedHostMiddleware`` pattern.
"""

from __future__ import annotations

import ipaddress

try:
    from fastapi import HTTPException, Request
except ImportError:  # pragma: no cover - fastapi is a hard dep in practice
    HTTPException = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]


__all__ = [
    "LOOPBACK_HOSTS",
    "is_loopback_host",
    "is_loopback_host_header",
    "require_loopback",
]


# Legacy canonical loopback literal set. Retained for backwards
# compatibility with callers/tests that still import it; the real check
# now goes through :func:`ipaddress.ip_address(...).is_loopback` so we
# also accept IPv6-mapped IPv4 (``::ffff:127.0.0.1``) and other valid
# loopback literals on dual-stack sockets.
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})


def is_loopback_host(host: str | None) -> bool:
    """Return True if ``host`` represents a loopback interface.

    ``None`` is treated as loopback — this covers ``TestClient`` /
    UDS-style requests where FastAPI does not populate
    ``request.client``.

    ``"localhost"`` is special-cased as a string since it is not a
    valid IP literal. The comparison is case-insensitive because
    hostnames are (RFC 4343), so a ``Host: LOCALHOST`` from a local
    tool is still accepted. Every other host is parsed with
    :func:`ipaddress.ip_address`; this accepts IPv6-mapped IPv4
    (``::ffff:127.0.0.1``) which Linux dual-stack sockets emit by
    default. Malformed input returns ``False``.
    """
    if host is None:
        return True
    if host.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def is_loopback_host_header(header_value: str | None) -> bool:
    """Return True if a ``Host:`` header names a loopback address.

    The header can include a port (``127.0.0.1:8787``,
    ``[::1]:8787``, ``localhost:8787``) and uses bracket notation for
    raw IPv6 literals per RFC 3986. This helper strips brackets and
    the trailing ``:port`` (if any) and delegates the address-vs-name
    decision to :func:`is_loopback_host`.

    Missing / empty headers return ``False`` rather than ``True`` —
    a real local browser or CLI always sets ``Host:``, so absence is
    suspicious. Server-internal callers that bypass HTTP entirely
    (``TestClient`` with a manual call) do not hit the guard.
    """
    if not header_value:
        return False
    candidate = header_value.strip()
    if not candidate:
        return False
    # Bracketed IPv6: [::1] or [::1]:8787 — strip the brackets and
    # everything after the matching ``]`` (which is the port suffix).
    if candidate.startswith("["):
        closing = candidate.find("]")
        if closing == -1:
            return False
        host_part = candidate[1:closing]
    elif candidate.count(":") == 1:
        # Single colon = host:port for IPv4 / hostname. A bare IPv6
        # literal without brackets has multiple colons and would be
        # ambiguous, so we don't strip in that case.
        host_part = candidate.rsplit(":", 1)[0]
    else:
        host_part = candidate
    return is_loopback_host(host_part)


def require_loopback(request: Request) -> None:  # type: ignore[valid-type]
    """FastAPI dependency: 404 any non-loopback caller.

    Usage::

        @app.get("/debug/tasks", dependencies=[Depends(require_loopback)])
        async def debug_tasks() -> list[dict]:
            ...

    Two gates have to pass:

    1. ``request.client.host`` must be a loopback IP. Stops anyone
       who actually reaches the listener from outside ``127.0.0.0/8``
       / ``::1``.
    2. The inbound ``Host:`` header must also name loopback. Stops
       DNS-rebinding attacks where a browser sends requests to the
       loopback IP but the page origin is ``attacker.com`` — the IP
       check alone passes, but the ``Host:`` header still reads
       ``attacker.com`` and we reject the request here.

    Returning 404 (not 403) keeps debug endpoints invisible to
    external scanners — indistinguishable from "no such route".
    """
    if HTTPException is None:  # pragma: no cover - defensive
        raise RuntimeError("FastAPI is required for the loopback guard")

    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    if not is_loopback_host(host):
        # No body: minimal FastAPI default, behaves like "no route".
        raise HTTPException(status_code=404)

    headers = getattr(request, "headers", None)
    if headers is None:
        # Manual ``Request`` stub with no ``headers`` attribute — used
        # by older unit tests that pre-date this gate. Treat the same
        # way as the IP-only path did and accept.
        return
    try:
        host_header = headers.get("host")
    except AttributeError:
        host_header = None
    if not is_loopback_host_header(host_header):
        raise HTTPException(status_code=404)

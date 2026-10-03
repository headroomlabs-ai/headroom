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
import os
import socket
import struct

try:
    from fastapi import HTTPException, Request
except ImportError:  # pragma: no cover - fastapi is a hard dep in practice
    HTTPException = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]


__all__ = [
    "LOOPBACK_HOSTS",
    "get_container_host_gateway",
    "is_container_environment",
    "is_container_host_gateway",
    "is_ip_literal_host_header",
    "is_loopback_host",
    "is_loopback_host_header",
    "require_loopback",
    "require_loopback_or_container_gateway",
    "require_same_origin",
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


def is_ip_literal_host_header(header_value: str | None) -> bool:
    """Return whether ``Host:`` contains an IPv4 or bracketed IPv6 literal.

    Dashboard clients may use a non-loopback server address, but retaining an
    IP-literal Host requirement prevents DNS-rebinding requests from using an
    attacker-controlled hostname. Ports are accepted in normal HTTP forms.
    """
    if not header_value:
        return False

    candidate = header_value.strip()
    if not candidate or "/" in candidate or "@" in candidate:
        return False

    if candidate.startswith("["):
        closing = candidate.find("]")
        if closing == -1 or candidate.count("[") != 1 or candidate.count("]") != 1:
            return False
        host_part = candidate[1:closing]
        suffix = candidate[closing + 1 :]
        if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
            return False
        try:
            return isinstance(ipaddress.ip_address(host_part), ipaddress.IPv6Address)
        except ValueError:
            return False

    if candidate.count(":") == 1:
        host_part, port = candidate.rsplit(":", 1)
        if not port.isdigit():
            return False
    else:
        host_part = candidate

    try:
        return isinstance(ipaddress.ip_address(host_part), ipaddress.IPv4Address)
    except ValueError:
        return False


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


def is_container_environment() -> bool:
    """Return True if running inside a container or containerized deployment."""
    if os.environ.get("HEADROOM_CONTAINER_HOST_GATEWAY"):
        return True
    if os.environ.get("HEADROOM_DEPLOYMENT_RUNTIME") in ("docker", "podman", "container"):
        return True
    if os.environ.get("HEADROOM_DEPLOYMENT_PRESET") == "persistent-docker":
        return True
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    try:
        with open("/proc/1/cgroup", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if any(marker in content for marker in ("docker", "containerd", "kubepods", "libpod")):
                return True
    except OSError:
        pass
    return False


def _read_linux_default_gateway() -> str | None:
    """Read the default IPv4 gateway address from /proc/net/route."""
    try:
        with open("/proc/net/route", encoding="ascii") as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) >= 3 and fields[1] == "00000000":
                    gw_hex = fields[2]
                    if len(gw_hex) == 8:
                        gw_int = int(gw_hex, 16)
                        if gw_int != 0:
                            return socket.inet_ntoa(struct.pack("<L", gw_int))
    except (OSError, ValueError):
        pass
    return None


def get_container_host_gateway() -> str | None:
    """Return the container host gateway IP if running in a container.

    Checks ``HEADROOM_CONTAINER_HOST_GATEWAY`` first, and falls back to
    resolving the default gateway from ``/proc/net/route`` when running
    inside a container environment. Returns None when not in a container or
    if no gateway can be determined.
    """
    env_gw = os.environ.get("HEADROOM_CONTAINER_HOST_GATEWAY")
    if env_gw:
        env_gw = env_gw.strip()
        if env_gw:
            return env_gw
    if not is_container_environment():
        return None
    return _read_linux_default_gateway()


def _normalize_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def is_container_host_gateway(host: str | None) -> bool:
    """Return True if ``host`` matches the container host gateway IP.

    Always returns False if not in a container environment or if ``host``
    is empty. Correctly handles IPv4-mapped IPv6 addresses (e.g.
    ``::ffff:172.17.0.1``).
    """
    if not host or not is_container_environment():
        return False
    gateway = get_container_host_gateway()
    if not gateway:
        return False
    parsed_host = _normalize_ip(host)
    parsed_gw = _normalize_ip(gateway)
    if parsed_host is not None and parsed_gw is not None:
        return parsed_host == parsed_gw
    return host.strip() == gateway.strip()


def require_loopback_or_container_gateway(request: Request) -> None:  # type: ignore[valid-type]
    """FastAPI dependency: allow loopback callers or container host gateway.

    Used by routes like ``/v1/compress`` and ``/v1/usage``. When running inside
    a container environment (Docker / Podman), incoming host traffic forwarded
    over the container bridge arrives with the host gateway IP (e.g. 172.17.0.1)
    rather than a loopback interface.

    Enforces two gates:
    1. Client address must be either loopback (:func:`is_loopback_host`) or
       the container host default gateway (:func:`is_container_host_gateway`).
       Arbitrary peer containers on the same bridge network are rejected.
    2. The inbound ``Host:`` header must name a loopback host
       (:func:`is_loopback_host_header`). This blocks DNS-rebinding attacks and
       rejects callers addressing the container's bridge IP or hostname directly.
    """
    if HTTPException is None:  # pragma: no cover - defensive
        raise RuntimeError("FastAPI is required for the loopback guard")

    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    if not (is_loopback_host(host) or is_container_host_gateway(host)):
        raise HTTPException(status_code=404)

    headers = getattr(request, "headers", None)
    if headers is None:
        return
    try:
        host_header = headers.get("host")
    except AttributeError:
        host_header = None
    if not is_loopback_host_header(host_header):
        raise HTTPException(status_code=404)


def require_same_origin(request: Request) -> None:  # type: ignore[valid-type]
    """FastAPI dependency: reject cross-origin browser requests on mutating routes.

    ``require_loopback``'s Host-header check stops DNS-rebinding, but not a
    plain CSRF where a remote page's JS targets a known
    ``http://127.0.0.1:<port>`` URL directly with a non-preflighted "simple"
    request (e.g. ``Content-Type: text/plain`` carrying a JSON body) -- the
    browser's ``Host:`` header still reads the real destination (loopback),
    but its ``Origin:`` header reflects the page's actual origin. CORS alone
    does not stop this: CORS only blocks the attacker's JS from *reading* the
    response, not the server from acting on the request.

    Reject when ``Origin`` is present and does not itself name a loopback
    host, or is the opaque literal ``"null"`` (sandboxed iframe / ``file://``
    page). Requests with no ``Origin`` header (CLI tools, curl, ``TestClient``,
    same-origin simple navigations) pass through unchanged -- a real browser
    always sets ``Origin`` on cross-origin fetch/XHR.
    """
    if HTTPException is None:  # pragma: no cover - defensive
        raise RuntimeError("FastAPI is required for the same-origin guard")

    headers = getattr(request, "headers", None)
    origin = headers.get("origin") if headers is not None else None
    if not origin:
        return
    if origin == "null":
        raise HTTPException(status_code=403, detail="cross-origin request rejected")
    host_part = origin.split("://", 1)[-1].split("/", 1)[0]
    if not is_loopback_host_header(host_part):
        raise HTTPException(status_code=403, detail="cross-origin request rejected")

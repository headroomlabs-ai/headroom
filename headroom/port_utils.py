"""Centralized port management for Headroom proxy processes.

Provides shared constants and utilities for port allocation, Headroom-specific
proxy detection, and multi-port lifecycle management. All wrap/unwrap/install
commands should use these utilities instead of ad-hoc port scanning.
"""

from __future__ import annotations

import socket

DEFAULT_PROXY_PORT = 8787

_MAX_PORT_SCAN = 50


def allocate_ports(
    base_port: int,
    count: int,
    *,
    max_scan: int = _MAX_PORT_SCAN,
) -> list[int]:
    """Return *count* free ports starting from *base_port*.

    Skips ports where a TCP connect succeeds (anything is listening).
    Raises ValueError if no free port is found within *max_scan* attempts.

    Raises ValueError if *base_port* is 0 (OS-assigned ports are not supported).
    """
    if base_port == 0:
        raise ValueError(
            "port 0 (OS-assigned) is not supported — specify a concrete base port"
        )
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []

    ports: list[int] = []
    candidate = base_port
    bound = base_port + count + max_scan

    while len(ports) < count and candidate < bound:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(("127.0.0.1", candidate))
        except (TimeoutError, ConnectionRefusedError, OSError):
            ports.append(candidate)
        candidate += 1

    if len(ports) < count:
        raise ValueError(
            f"could not find {count} free port(s) in range "
            f"{base_port}–{candidate - 1} — found {len(ports)}. "
            f"Try a different --port offset or stop the processes on those ports"
        )

    return ports


def is_headroom_proxy(port: int, *, timeout: float = 2.0) -> bool:
    """True when a Headroom proxy is listening on *port*.

    Performs a TCP connect followed by an HTTP GET to ``/healthz``.
    Returns False for non-Headroom services (e.g. nginx, local dev servers)
    and for ports with nothing listening.
    """
    import http.client

    try:
        conn = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=timeout
        )
        conn.request("GET", "/healthz")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status == 200
    except Exception:
        return False


def find_opencode_ports(
    base_port: int = DEFAULT_PROXY_PORT,
    max_scan: int = 20,
) -> list[int]:
    """Return ports where a Headroom proxy is running for OpenCode.

    Scans from *base_port* upward. Used by ``headroom unwrap opencode``
    to discover all proxy instances that need to be stopped.
    """
    ports: list[int] = []
    for port in range(base_port, base_port + max_scan):
        if is_headroom_proxy(port):
            ports.append(port)
    return ports


def format_unbindable_port_error(
    port: int,
    error: OSError,
    agent_type: str,
) -> str:
    """Build an actionable message for ports that fail before binding.

    This mirrors the existing ``_format_unbindable_port_error`` pattern
    in wrap.py so it can be called from there and from port_utils consumers.
    """
    from textwrap import dedent

    command = "headroom proxy"
    if agent_type != "unknown":
        command = f"headroom wrap {agent_type}"
    suggested_port = port + 1
    return dedent(f"""\
        Port {port} is unavailable on 127.0.0.1 before the proxy can start: {error}.
        On Windows this can happen when the port is in an excluded or reserved range.
        Try a different port, e.g. {command} --port {suggested_port}""")

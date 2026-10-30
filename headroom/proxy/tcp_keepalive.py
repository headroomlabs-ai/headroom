"""TCP keepalive on the proxy's upstream connections.

A request waiting on a model reads nothing for minutes, legitimately, so the
read timeout cannot tell a slow answer from a dead link. When the network goes
away without a reset (a train entering a tunnel, a Wi-Fi handover, a NAT
dropping its state, a laptop changing networks) the socket stays open and the
proxy waits out the whole read timeout: 600s on the buffered Anthropic path,
for each of ``retry_max_attempts`` attempts. The buffered path heartbeats the
client meanwhile, so the client's own stall detection never fires either, and
the user watches a spinner for 10-30 minutes where a direct connection would
have failed over in about three.

Keepalive asks the peer's kernel, not the model, whether the connection is
alive: after ``idle_seconds`` of silence it sends a probe every
``PROBE_INTERVAL_SECONDS`` and drops the connection after ``PROBE_COUNT``
unanswered ones. A live upstream answers probes while the model thinks, so slow
answers are untouched. A dead link fails after about ``idle_seconds + 60`` as a
transport error, which ``_retry_request`` already retries on a fresh
connection. Keepalive never fires while sent data is unacknowledged; a stalled
upload is ``write_timeout_seconds``' job (#3259, #3327).
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Iterable
from typing import Any

import httpcore
import httpx

logger = logging.getLogger(__name__)

PROBE_INTERVAL_SECONDS = 10
PROBE_COUNT = 6


def keepalive_socket_options(idle_seconds: int) -> list[tuple[int, int, int]]:
    """Keepalive options for this platform: on, idle time, probe interval, probe count."""
    # Linux and Windows call the idle time TCP_KEEPIDLE, macOS TCP_KEEPALIVE.
    idle = getattr(socket, "TCP_KEEPIDLE", None)
    if idle is None:
        idle = getattr(socket, "TCP_KEEPALIVE", None)
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name, value in (
        (idle, idle_seconds),
        (getattr(socket, "TCP_KEEPINTVL", None), PROBE_INTERVAL_SECONDS),
        (getattr(socket, "TCP_KEEPCNT", None), PROBE_COUNT),
    ):
        if name is not None:
            options.append((socket.IPPROTO_TCP, name, value))
    return options


class KeepaliveNetworkBackend(httpcore.AsyncNetworkBackend):
    """Turns keepalive on for every TCP connection the wrapped backend opens.

    Options are set after the connect rather than passed as httpcore
    ``socket_options``: httpcore maps a failing ``setsockopt`` to ConnectError,
    so a platform refusing one option would fail every request. Here a refused
    option is skipped and the connection goes ahead as it would have without it.
    """

    def __init__(
        self, inner: httpcore.AsyncNetworkBackend, options: list[tuple[int, int, int]]
    ) -> None:
        self._inner = inner
        self._options = options

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        stream = await self._inner.connect_tcp(
            host,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )
        sock = stream.get_extra_info("socket")
        if sock is not None:
            for option in self._options:
                try:
                    sock.setsockopt(*option)
                except OSError as exc:
                    logger.debug("upstream TCP keepalive option %r refused: %s", option, exc)
        return stream

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


def install_tcp_keepalive(client: httpx.AsyncClient, idle_seconds: int) -> httpx.AsyncClient:
    """Probe every upstream connection ``client`` opens; ``idle_seconds <= 0`` leaves it as is.

    Wraps the network backend of each transport's connection pool: the primary
    one and every proxy mount httpx built from HTTP(S)_PROXY or the system proxy
    settings, so direct dials, CONNECT tunnels and SOCKS proxies all get it.
    ``httpx.AsyncHTTPTransport(socket_options=...)`` cannot do this: passing a
    transport switches off environment proxies, and httpcore's proxy pools do
    not forward ``socket_options`` to the connections they open. Call before
    ``install_upstream_pinning``, which hides proxy pools behind a refusing
    wrapper.
    """
    if idle_seconds <= 0:
        return client
    options = keepalive_socket_options(idle_seconds)
    for transport in (client._transport, *client._mounts.values()):
        pool: Any = getattr(transport, "_pool", None)
        backend = getattr(pool, "_network_backend", None)
        if backend is not None and not isinstance(backend, KeepaliveNetworkBackend):
            pool._network_backend = KeepaliveNetworkBackend(backend, options)
    return client

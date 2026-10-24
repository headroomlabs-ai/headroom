"""Upstream connections must notice a dead link without waiting for the model.

The read timeout cannot tell a model thinking from a link that died without a
reset (a network change, a tunnel, a NAT dropping state), so the proxy held such
a request for the whole read timeout, 600s on the buffered Anthropic path, per
attempt, while heartbeating the client so its own stall detection never fired.
TCP keepalive asks the peer's kernel instead, so every socket the upstream
client opens, direct or through a proxy, has to carry it.
"""

from __future__ import annotations

import asyncio
import socket

import httpcore
import httpx
import pytest

from headroom.proxy.models import ProxyConfig
from headroom.proxy.tcp_keepalive import (
    PROBE_COUNT,
    PROBE_INTERVAL_SECONDS,
    KeepaliveNetworkBackend,
    install_tcp_keepalive,
    keepalive_socket_options,
)

_IDLE = getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None)


class _RecordingBackend(httpcore.AsyncNetworkBackend):
    """Hands out real connections and remembers their sockets."""

    def __init__(self) -> None:
        self._real = httpcore.AnyIOBackend()
        self.sockets: list[socket.socket] = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        stream = await self._real.connect_tcp(
            host,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )
        self.sockets.append(stream.get_extra_info("socket"))
        return stream

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise NotImplementedError

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


async def _http_server(seen_paths: list[str]) -> asyncio.Server:
    """Answers every request 200; works as an origin and as a forward proxy."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            head = await reader.readuntil(b"\r\n\r\n")
            seen_paths.append(head.split(b" ", 2)[1].decode())
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()

    async def guarded(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await handle(reader, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            writer.close()

    return await asyncio.start_server(guarded, "127.0.0.1", 0)


def _assert_keepalive(sock: socket.socket, idle_seconds: int) -> None:
    assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
    if _IDLE is not None:
        assert sock.getsockopt(socket.IPPROTO_TCP, _IDLE) == idle_seconds
    if hasattr(socket, "TCP_KEEPINTVL"):
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == PROBE_INTERVAL_SECONDS
    if hasattr(socket, "TCP_KEEPCNT"):
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == PROBE_COUNT


def _record_under_keepalive(client: httpx.AsyncClient) -> _RecordingBackend:
    """Put a recorder beneath every keepalive layer ``client`` dials through."""
    recorder = _RecordingBackend()
    for transport in (client._transport, *client._mounts.values()):
        backend = transport._pool._network_backend
        assert isinstance(backend, KeepaliveNetworkBackend)
        backend._inner = recorder
    return recorder


def test_options_turn_keepalive_on_with_the_idle_time_this_platform_names() -> None:
    options = keepalive_socket_options(42)

    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options
    if _IDLE is not None:
        assert (socket.IPPROTO_TCP, _IDLE, 42) in options


async def test_a_direct_upstream_connection_carries_keepalive() -> None:
    """The regression itself: a real socket to the upstream, probed when idle."""
    seen: list[str] = []
    server = await _http_server(seen)
    port = server.sockets[0].getsockname()[1]
    client = install_tcp_keepalive(httpx.AsyncClient(trust_env=False), 25)
    recorder = _record_under_keepalive(client)
    try:
        response = await client.get(f"http://127.0.0.1:{port}/v1/messages")
        assert response.status_code == 200
        assert len(recorder.sockets) == 1
        _assert_keepalive(recorder.sockets[0], 25)
    finally:
        await client.aclose()
        server.close()


async def test_a_proxied_upstream_connection_carries_keepalive() -> None:
    """httpcore's proxy pools drop ``socket_options``; the backend layer does not.

    The socket to the proxy is the one that dies with the user's network, so it
    is the one that has to be probed.
    """
    seen: list[str] = []
    proxy = await _http_server(seen)
    port = proxy.sockets[0].getsockname()[1]
    client = install_tcp_keepalive(
        httpx.AsyncClient(proxy=f"http://127.0.0.1:{port}", trust_env=False), 25
    )
    recorder = _record_under_keepalive(client)
    try:
        response = await client.get("http://upstream.example/v1/messages")
        assert response.status_code == 200
        assert seen == ["http://upstream.example/v1/messages"]  # it went via the proxy
        assert len(recorder.sockets) == 1
        _assert_keepalive(recorder.sockets[0], 25)
    finally:
        await client.aclose()
        proxy.close()


async def test_environment_proxy_mounts_are_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    """System and env proxies become mounts httpx builds itself."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
    client = install_tcp_keepalive(httpx.AsyncClient(), 30)
    try:
        transports = [client._transport, *client._mounts.values()]
        assert len(transports) > 1
        for transport in transports:
            assert isinstance(transport._pool._network_backend, KeepaliveNetworkBackend)
    finally:
        await client.aclose()


async def test_zero_leaves_the_client_untouched() -> None:
    client = httpx.AsyncClient(trust_env=False)
    backend = client._transport._pool._network_backend
    try:
        assert install_tcp_keepalive(client, 0) is client
        assert client._transport._pool._network_backend is backend
    finally:
        await client.aclose()


async def test_installing_twice_does_not_stack() -> None:
    client = install_tcp_keepalive(
        install_tcp_keepalive(httpx.AsyncClient(trust_env=False), 30), 30
    )
    try:
        backend = client._transport._pool._network_backend
        assert isinstance(backend, KeepaliveNetworkBackend)
        assert not isinstance(backend._inner, KeepaliveNetworkBackend)
    finally:
        await client.aclose()


async def test_a_refused_option_never_fails_the_connection() -> None:
    """Keepalive is best effort; a platform that rejects an option still connects."""
    seen: list[str] = []
    server = await _http_server(seen)
    port = server.sockets[0].getsockname()[1]
    bogus = (socket.IPPROTO_TCP, 0x7FFF, 1)
    backend = KeepaliveNetworkBackend(
        httpcore.AnyIOBackend(), [bogus, (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    )
    try:
        stream = await backend.connect_tcp("127.0.0.1", port)
        sock = stream.get_extra_info("socket")
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
        await stream.aclose()
    finally:
        server.close()


def test_keepalive_defaults_on_and_fails_over_well_inside_the_read_timeouts() -> None:
    config = ProxyConfig()
    detection = config.upstream_tcp_keepalive_seconds + PROBE_INTERVAL_SECONDS * PROBE_COUNT

    assert config.upstream_tcp_keepalive_seconds > 0
    assert detection < config.request_timeout_seconds
    assert detection < config.anthropic_buffered_request_timeout_seconds


def test_env_sets_and_disables_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from headroom.proxy.server import _proxy_config_from_env

    monkeypatch.setenv("HEADROOM_UPSTREAM_TCP_KEEPALIVE_SECONDS", "45")
    assert _proxy_config_from_env().upstream_tcp_keepalive_seconds == 45
    monkeypatch.setenv("HEADROOM_UPSTREAM_TCP_KEEPALIVE_SECONDS", "0")
    assert _proxy_config_from_env().upstream_tcp_keepalive_seconds == 0


@pytest.mark.parametrize("http2", [False, True])
def test_the_proxy_dials_every_upstream_through_keepalive_and_pinning(http2: bool) -> None:
    """Keepalive sits under the pin, so a pinned dial is probed too."""
    from fastapi.testclient import TestClient

    from headroom.proxy.server import create_app
    from headroom.proxy.upstream_pinning import PinnedAddressBackend

    config = ProxyConfig(http2=http2, optimize=False, cache_enabled=False, rate_limit_enabled=False)
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        for upstream in (proxy.http_client, proxy.http_client_h1):
            backend = upstream._transport._pool._network_backend
            assert isinstance(backend, PinnedAddressBackend)
            assert isinstance(backend._inner, KeepaliveNetworkBackend)


def test_the_proxy_leaves_sockets_alone_when_disabled() -> None:
    from fastapi.testclient import TestClient

    from headroom.proxy.server import create_app

    config = ProxyConfig(
        upstream_tcp_keepalive_seconds=0,
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
    )
    with TestClient(create_app(config)) as client:
        backend = client.app.state.proxy.http_client._transport._pool._network_backend
        assert not isinstance(backend._inner, KeepaliveNetworkBackend)

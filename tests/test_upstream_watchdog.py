"""Upstream watchdog: bounded waits where httpx's own timeouts do not reach.

httpcore's HTTP/2 connection waits on a stream-slot semaphore and a read lock
with no timeout, so one jammed shared connection parked every request (and the
/readyz probe) indefinitely. These tests stand in for that connection with a
transport that never answers: its handler awaits an Event nobody sets.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy import server as server_module
from headroom.proxy import upstream_watchdog
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app
from headroom.proxy.upstream_watchdog import (
    PHASE_BODY,
    PHASE_RESPONSE,
    PHASE_SEND,
    UpstreamClientGeneration,
    UpstreamClientRetirer,
    UpstreamWatchdogTimeout,
    close_upstream_client,
    install_upstream_watchdog,
)

URL = "https://api.anthropic.com/v1/messages"


class HangingTransport(httpx.AsyncBaseTransport):
    """Never answers, like a request parked on httpcore's stream semaphore."""

    def __init__(self, *, after_headers_sent: bool = False) -> None:
        self.calls = 0
        self.after_headers_sent = after_headers_sent

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.after_headers_sent:
            trace = request.extensions.get("trace")
            if trace is not None:
                await trace("http2.send_request_headers.complete", {})
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class OkTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(200, json={"ok": True})


class GatedStream(httpx.AsyncByteStream):
    """A healthy streaming body that yields its second chunk only when released."""

    def __init__(self, release: asyncio.Event, *, stall_forever: bool = False) -> None:
        self.release = release
        self.stall_forever = stall_forever
        self.closed = False

    async def __aiter__(self):  # type: ignore[override]
        yield b"data: one\n\n"
        await (asyncio.Event().wait() if self.stall_forever else self.release.wait())
        yield b"data: two\n\n"

    async def aclose(self) -> None:
        self.closed = True


class RoutingTransport(httpx.AsyncBaseTransport):
    """``/stream`` serves a gated stream, everything else hangs."""

    def __init__(self, release: asyncio.Event) -> None:
        self.release = release

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/stream":
            return httpx.Response(200, stream=GatedStream(self.release))
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _watched(transport: httpx.AsyncBaseTransport, deadline: float, **client_kwargs: Any):
    generation = UpstreamClientGeneration(0)
    fired: list[UpstreamClientGeneration] = []
    client = install_upstream_watchdog(
        httpx.AsyncClient(transport=transport, **client_kwargs),
        send_deadline=deadline,
        generation=generation,
        on_send_timeout=fired.append,
    )
    return client, generation, fired


def _proxy(monkeypatch, transports: list[httpx.AsyncBaseTransport], **config: Any):
    """A proxy whose client factory hands out ``transports`` in order."""
    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            upstream_watchdog_seconds=1,
            **config,
        )
    )
    built: list[httpx.AsyncClient] = []

    def factory(*, http2: bool) -> httpx.AsyncClient:
        # timeout=None leaves only the watchdog's own deadline in play.
        client = httpx.AsyncClient(transport=transports[len(built)], timeout=None)
        built.append(client)
        return client

    monkeypatch.setattr(proxy, "_new_upstream_client", factory)
    proxy._install_upstream_clients()
    return proxy, built


# --- transport ---------------------------------------------------------------


async def test_hung_send_fails_within_deadline_and_requests_recycle():
    transport = HangingTransport()
    client, generation, fired = _watched(transport, 0.2, timeout=None)

    started = time.monotonic()
    with pytest.raises(UpstreamWatchdogTimeout) as excinfo:
        await client.post(URL, json={"x": 1})
    elapsed = time.monotonic() - started

    assert 0.15 < elapsed < 2.0
    assert excinfo.value.phase == PHASE_SEND
    assert isinstance(excinfo.value, httpx.TimeoutException)
    assert fired == [generation]
    assert generation.in_flight == 0


async def test_send_deadline_never_undercuts_the_request_pool_and_connect_timeouts(monkeypatch):
    monkeypatch.setattr(upstream_watchdog, "NATIVE_TIMEOUT_MARGIN_SECONDS", 0.0)
    client, _, _ = _watched(HangingTransport(), 0.05, timeout=httpx.Timeout(0.25))

    started = time.monotonic()
    with pytest.raises(UpstreamWatchdogTimeout):
        await client.get(URL)
    # pool (0.25) + connect (0.25) outranks the 0.05s watchdog setting.
    assert time.monotonic() - started >= 0.45


async def test_wait_for_response_headers_is_bounded_but_does_not_recycle(monkeypatch):
    monkeypatch.setattr(upstream_watchdog, "NATIVE_TIMEOUT_MARGIN_SECONDS", 0.0)
    client, generation, fired = _watched(
        HangingTransport(after_headers_sent=True),
        30,
        timeout=httpx.Timeout(10, read=0.1, write=0.1),
    )

    started = time.monotonic()
    with pytest.raises(UpstreamWatchdogTimeout) as excinfo:
        await client.post(URL, json={})
    assert time.monotonic() - started < 2.0
    assert excinfo.value.phase == PHASE_RESPONSE
    # The request reached upstream; a slow answer is not evidence of a jam.
    assert fired == []
    assert generation.in_flight == 0


async def test_stalled_stream_chunk_is_bounded_by_the_read_timeout(monkeypatch):
    monkeypatch.setattr(upstream_watchdog, "NATIVE_TIMEOUT_MARGIN_SECONDS", 0.0)
    body = GatedStream(asyncio.Event(), stall_forever=True)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    client, generation, _ = _watched(
        httpx.MockTransport(handler), 30, timeout=httpx.Timeout(10, read=0.2)
    )
    chunks = []
    with pytest.raises(UpstreamWatchdogTimeout) as excinfo:
        async with client.stream("POST", URL, json={}) as response:
            async for chunk in response.aiter_raw():
                chunks.append(chunk)
    assert chunks == [b"data: one\n\n"]
    assert excinfo.value.phase == PHASE_BODY
    assert body.closed
    assert generation.in_flight == 0


async def test_long_healthy_stream_is_not_cut(monkeypatch):
    """The body deadline is per chunk: total stream length is unbounded."""
    monkeypatch.setattr(upstream_watchdog, "NATIVE_TIMEOUT_MARGIN_SECONDS", 0.0)

    class Steady(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[override]
            for _ in range(8):
                await asyncio.sleep(0.05)
                yield b"x"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Steady())

    client, _, _ = _watched(httpx.MockTransport(handler), 0.1, timeout=httpx.Timeout(10, read=0.15))
    async with client.stream("GET", URL) as response:
        assert await response.aread() == b"x" * 8


async def test_disabled_watchdog_leaves_the_client_untouched():
    client = httpx.AsyncClient(transport=OkTransport())
    original = client._transport
    install_upstream_watchdog(
        client,
        send_deadline=0,
        generation=UpstreamClientGeneration(0),
        on_send_timeout=lambda _g: None,
    )
    assert client._transport is original


async def test_hung_aclose_is_abandoned_within_its_timeout():
    class HungClose(OkTransport):
        async def aclose(self) -> None:
            await asyncio.Event().wait()

    client = httpx.AsyncClient(transport=HungClose())
    started = time.monotonic()
    await close_upstream_client(client, timeout=0.1)
    assert time.monotonic() - started < 1.0


# --- proxy recycle -------------------------------------------------------------


async def test_concurrent_timeouts_recycle_the_client_exactly_once(monkeypatch):
    hanging, fresh = HangingTransport(), OkTransport()
    proxy, built = _proxy(monkeypatch, [hanging, fresh, OkTransport()])
    old = proxy.http_client

    results = await asyncio.gather(
        *(proxy.http_client.post(URL, json={"n": n}) for n in range(8)),
        return_exceptions=True,
    )

    assert all(isinstance(r, UpstreamWatchdogTimeout) for r in results)
    assert hanging.calls == 8
    assert len(built) == 2  # the initial client and exactly one replacement
    assert proxy._upstream_generation is not None
    assert proxy._upstream_generation.number == 1
    assert proxy.http_client is not old

    # The old client drains (all its requests failed) and closes in the background.
    for _ in range(50):
        if old.is_closed:
            break
        await asyncio.sleep(0.02)
    assert old.is_closed

    # A normal request works afterwards, on the fresh client.
    response = await proxy.http_client.post(URL, json={})
    assert response.status_code == 200
    assert fresh.calls == 1


async def test_retry_moves_to_the_fresh_client(monkeypatch):
    hanging, fresh = HangingTransport(), OkTransport()
    proxy, built = _proxy(
        monkeypatch,
        [hanging, fresh],
        retry_max_attempts=2,
        retry_base_delay_ms=0,
        retry_max_delay_ms=0,
    )

    started = time.monotonic()
    response = await proxy._retry_request("POST", URL, {"x-api-key": "k"}, {"model": "m"})

    assert response.status_code == 200
    assert time.monotonic() - started < 5.0
    assert (hanging.calls, fresh.calls, len(built)) == (1, 1, 2)


async def test_healthy_stream_on_the_old_client_finishes_after_recycle(monkeypatch):
    release = asyncio.Event()
    proxy, _ = _proxy(monkeypatch, [RoutingTransport(release), OkTransport()])
    old = proxy.http_client
    stream_url = "https://api.anthropic.com/stream"

    response = await old.send(old.build_request("GET", stream_url), stream=True)
    chunks = response.aiter_raw()
    assert await chunks.__anext__() == b"data: one\n\n"

    with pytest.raises(UpstreamWatchdogTimeout):
        await old.post(URL, json={})
    assert proxy.http_client is not old

    await asyncio.sleep(0.05)
    assert not old.is_closed  # still serving the open stream
    release.set()
    assert await chunks.__anext__() == b"data: two\n\n"
    await response.aclose()

    for _ in range(50):
        if old.is_closed:
            break
        await asyncio.sleep(0.02)
    assert old.is_closed


async def test_retirer_closes_after_grace_even_if_a_stream_never_closes():
    generation = UpstreamClientGeneration(0)
    generation.enter()  # a leaked response that is never closed
    client = httpx.AsyncClient(transport=OkTransport())
    retirer = UpstreamClientRetirer(grace_seconds=0.1, close_timeout=0.1)
    await retirer.retire([client], generation)
    assert client.is_closed
    assert retirer.pending == 0


async def test_shutdown_closes_retired_clients_without_waiting_for_drain(monkeypatch):
    proxy, _ = _proxy(monkeypatch, [HangingTransport(), OkTransport()])
    old = proxy.http_client
    old_generation = proxy._upstream_generation
    assert old_generation is not None
    old_generation.enter()  # keeps the old generation from draining
    with pytest.raises(UpstreamWatchdogTimeout):
        await old.post(URL, json={})
    assert proxy._upstream_retirer.pending == 1

    started = time.monotonic()
    await proxy._upstream_retirer.shutdown()
    assert time.monotonic() - started < 5.0
    assert old.is_closed
    assert proxy._upstream_retirer.pending == 0


# --- HTTP surface ---------------------------------------------------------------


def _app_with_upstream(monkeypatch, transport_factory, **config: Any):
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
            upstream_watchdog_seconds=1,
            retry_max_attempts=1,
            **config,
        )
    )
    proxy = app.state.proxy
    monkeypatch.setattr(
        proxy,
        "_new_upstream_client",
        lambda *, http2: httpx.AsyncClient(transport=transport_factory(), timeout=None),
    )
    proxy._install_upstream_clients()
    return app, proxy


def test_hung_upstream_returns_504_in_the_anthropic_error_shape(monkeypatch):
    # The handler sets its own per-request timeouts; keep pool + connect small.
    monkeypatch.setattr(upstream_watchdog, "NATIVE_TIMEOUT_MARGIN_SECONDS", 0.0)
    transports = iter([HangingTransport(), OkTransport()])
    app, proxy = _app_with_upstream(
        monkeypatch, lambda: next(transports), connect_timeout_seconds=1
    )

    started = time.monotonic()
    response = TestClient(app).post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        content=json.dumps(
            {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ),
    )

    assert time.monotonic() - started < 10.0
    assert response.status_code == 504
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["code"] == "upstream_timeout"
    assert proxy._upstream_generation.number == 1


def test_readyz_is_not_ready_within_its_bound_while_upstream_hangs(monkeypatch):
    monkeypatch.delenv("HEADROOM_SKIP_UPSTREAM_CHECK", raising=False)
    monkeypatch.setattr(server_module, "READYZ_UPSTREAM_CHECK_DEADLINE_SECONDS", 0.5)
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )

    with TestClient(app) as test_client:
        # The probe's own HEAD timeout is 5s, so only the deadline can end it.
        test_client.app.state.proxy.http_client = httpx.AsyncClient(transport=HangingTransport())
        results: list[tuple[int, float, dict]] = []

        def probe() -> None:
            started = time.monotonic()
            response = test_client.get("/readyz")
            results.append((response.status_code, time.monotonic() - started, response.json()))

        # Two at once: the second waits on the probe lock, and is bounded too.
        threads = [threading.Thread(target=probe) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert len(results) == 2
    for status, elapsed, data in results:
        assert status == 503
        assert elapsed < 3.0
        assert data["ready"] is False
        assert data["checks"]["upstream"]["ready"] is False
        assert "did not finish" in data["checks"]["upstream"]["error"]

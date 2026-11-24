"""Tests for headroom.proxy.agy_retrieve.AgyRetrieveServer.

The retrieve server is a PLAIN-HTTP loopback listener that serves the same
FastAPI app (``create_app()``) as the HTTPS dispatch server — it *is*
``AgyDispatchServer(plain_http=True)``, so the hypercorn plumbing under test
here lives in :mod:`headroom.proxy.agy_dispatch`.  Its load-bearing property:
it shares the *process-global* compression store, so a marker stored on the
dispatch side resolves via ``GET /v1/retrieve/{hash}`` on this side.

All tests use ephemeral loopback ports; no TLS, no real network, no
``~/.headroom`` mutation beyond the in-memory process-global store (which is
reset around each test).
"""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.proxy import agy_dispatch
from headroom.proxy.agy_retrieve import AgyRetrieveServer


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/stats"),
        ("GET", "/health"),
        ("GET", "/openapi.json"),
        ("GET", "/debug/metrics"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/retrieve/stats"),
        ("OPTIONS", "/v1/retrieve"),
    ],
)
async def test_retrieve_listener_cannot_serve_other_proxy_api(monkeypatch, method, path):
    from fastapi import FastAPI

    app = FastAPI()
    visits = []

    @app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
    async def any_endpoint(path: str):
        visits.append(path)
        return {"visited": path}

    monkeypatch.setattr("headroom.proxy.server.create_app", lambda: app)
    async with AgyRetrieveServer() as srv:
        host, port = srv.address
        async with httpx.AsyncClient() as client:
            control = await client.get(f"http://{host}:{port}/v1/retrieve/stats")
            assert control.status_code == 200
            denied = await client.request(method, f"http://{host}:{port}{path}")
            assert denied.status_code == 404
        assert visits == ["v1/retrieve/stats"], "forbidden requests must never reach the proxy app"


@pytest.mark.parametrize(
    "host_header",
    [
        "attacker.invalid",
        "localhost.attacker.invalid",
        "127.0.0.1.attacker.invalid",
        "cloudcode-pa.googleapis.com",
    ],
)
async def test_retrieve_listener_rejects_non_loopback_host(monkeypatch, host_header):
    from fastapi import FastAPI

    app = FastAPI()
    visits = []

    @app.get("/v1/retrieve/stats")
    async def stats():
        visits.append(True)
        return {"stats": True}

    monkeypatch.setattr("headroom.proxy.server.create_app", lambda: app)
    async with AgyRetrieveServer() as srv:
        host, port = srv.address
        async with httpx.AsyncClient() as client:
            denied = await client.get(
                f"http://{host}:{port}/v1/retrieve/stats", headers={"host": host_header}
            )
            assert denied.status_code == 404
        assert visits == []


@pytest.fixture(autouse=True)
def _clean_compression_store():
    """Isolate the process-global compression store around each test."""
    reset_compression_store()
    yield
    reset_compression_store()


@pytest.mark.parametrize(
    "headers",
    [{"origin": "https://attacker.invalid"}, {"origin": "null"}, {"sec-fetch-site": "cross-site"}],
)
async def test_retrieve_listener_rejects_browser_cross_origin(headers):
    async with AgyRetrieveServer() as srv:
        host, port = srv.address
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://{host}:{port}/v1/retrieve/stats", headers=headers)
        assert response.status_code == 404
        assert "access-control-allow-origin" not in response.headers


async def test_retrieve_server_starts_on_loopback_plain_http() -> None:
    """Server binds loopback and answers plain HTTP (no TLS handshake)."""
    srv = AgyRetrieveServer(port=0)
    await srv.start()
    try:
        host, port = srv.address
        assert host == "127.0.0.1"
        assert isinstance(port, int) and port > 0

        # Plain HTTP (http://) must succeed — proving there is NO TLS layer.
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/v1/retrieve/stats")
        assert resp.status_code == 200
    finally:
        await srv.stop()


async def test_get_retrieve_returns_store_populated_content() -> None:
    """LOAD-BEARING: a hash stored via the process-global store resolves over
    plain HTTP from a SECOND create_app() — proving the cache is shared.

    This is exactly the dispatch-populates / retrieve-resolves contract: the
    HTTPS dispatch server stores markers into the same process-global singleton
    that this plain-HTTP listener serves.
    """
    original = '{"rows": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]}'
    compressed = '{"rows": "[Retrieve more]"}'

    # Populate the process-global store DIRECTLY (as the dispatch side would,
    # via the same get_compression_store() singleton) — the server is a
    # *separate* create_app() instance and must still see this entry.
    store = get_compression_store()
    hash_key = store.store(
        original=original,
        compressed=compressed,
        original_tokens=42,
        compressed_tokens=7,
        tool_name="search_api",
    )

    srv = AgyRetrieveServer(port=0)
    await srv.start()
    try:
        _, port = srv.address
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/v1/retrieve/{hash_key}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["hash"] == hash_key
        assert body["original_content"] == original
        assert body["tool_name"] == "search_api"
    finally:
        await srv.stop()


async def test_get_unknown_hash_returns_404() -> None:
    """An unknown marker hash returns 404 (not a 500/hang)."""
    srv = AgyRetrieveServer(port=0)
    await srv.start()
    try:
        _, port = srv.address
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/v1/retrieve/deadbeefdeadbeefdeadbeef")
        assert resp.status_code == 404
    finally:
        await srv.stop()


async def test_retrieve_server_binds_loopback_only() -> None:
    """The listener socket family/host must be loopback (127.0.0.1)."""
    srv = AgyRetrieveServer(port=0)
    await srv.start()
    try:
        host, _ = srv.address
        assert host == "127.0.0.1"
    finally:
        await srv.stop()


async def test_retrieve_server_clean_start_stop_no_leaked_server_tasks() -> None:
    """start()/stop() leaves no server-owned tasks (lifespan / connection).

    The only surviving task may be the FastAPI app's *periodic* TOIN-stats
    background task — an app-level concern that the production
    ``_start_agy_servers`` reaps at loop teardown (it cancels all pending tasks
    in its ``finally`` before ``loop.close()``).  This test mirrors that final
    sweep and asserts every leftover is cancellable (i.e. no task wedges the
    shutdown), and that the server's OWN lifespan task is gone.
    """
    loop = asyncio.get_running_loop()
    before = {t for t in asyncio.all_tasks(loop) if not t.done()}

    srv = AgyRetrieveServer(port=0)
    await srv.start()
    await srv.stop()
    assert srv._lifespan_task is None, "stop() must clear the lifespan task"

    await asyncio.sleep(0)
    after = {t for t in asyncio.all_tasks(loop) if not t.done()}
    leaked = after - before

    # Any leftover must be ONLY the app-level periodic stats task; no hypercorn
    # connection / lifespan task may survive stop().
    offending = [t for t in leaked if "_log_toin_stats_periodically" not in repr(t.get_coro())]
    assert not offending, f"retrieve server leaked server-owned tasks: {offending}"

    # Model the production loop-teardown sweep: every leftover cancels cleanly.
    for task in leaked:
        task.cancel()
    if leaked:
        await asyncio.gather(*leaked, return_exceptions=True)


async def test_retrieve_server_stop_idempotent() -> None:
    """stop() after stop() does not raise."""
    srv = AgyRetrieveServer(port=0)
    await srv.start()
    await srv.stop()
    await srv.stop()  # idempotent


def test_retrieve_server_address_raises_before_start() -> None:
    """address property raises RuntimeError before start()."""
    srv = AgyRetrieveServer(port=0)
    with pytest.raises(RuntimeError):
        _ = srv.address


async def test_start_raises_when_lifespan_startup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the hypercorn lifespan task fails during startup, start() surfaces
    that exception (instead of silently continuing on to bind a socket)."""
    import hypercorn.asyncio.run as hypercorn_run

    class _FailingLifespan:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def handle_lifespan(self) -> None:
            raise RuntimeError("lifespan startup boom")

        async def wait_for_startup(self) -> None:
            # Yield control so the handle_lifespan task (already scheduled by
            # loop.create_task) runs to completion — synchronously raising —
            # before this coroutine resumes and returns.
            await asyncio.sleep(0)

    monkeypatch.setattr(hypercorn_run, "Lifespan", _FailingLifespan)

    srv = AgyRetrieveServer(port=0)
    with pytest.raises(RuntimeError, match="lifespan startup boom"):
        await srv.start()

    # The failure must be surfaced before any socket gets bound.
    assert srv._server is None


async def test_start_continues_when_lifespan_task_completes_without_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the lifespan task is already ``done()`` by the time
    ``wait_for_startup()`` returns, but *without* an exception, start() must
    NOT raise — it continues on to bind the socket normally."""
    import hypercorn.asyncio.run as hypercorn_run

    class _InstantSucceedingLifespan:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def handle_lifespan(self) -> None:
            return None

        async def wait_for_startup(self) -> None:
            # Yield so the handle_lifespan task (scheduled by
            # loop.create_task) runs to completion synchronously — with no
            # exception — before this coroutine resumes and returns.
            await asyncio.sleep(0)

    monkeypatch.setattr(hypercorn_run, "Lifespan", _InstantSucceedingLifespan)

    srv = AgyRetrieveServer(port=0)
    await srv.start()  # must NOT raise: task is done(), but exception() is None
    try:
        assert srv._lifespan_task is not None
        assert srv._lifespan_task.done()
        assert srv._lifespan_task.exception() is None
        host, port = srv.address
        assert host == "127.0.0.1"
        assert isinstance(port, int) and port > 0
    finally:
        await srv.stop()


async def test_start_uses_so_exclusiveaddruse_on_non_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a non-POSIX ``os.name`` (e.g. Windows), the listener applies the
    exclusive-address-use socket option instead of SO_REUSEADDR — per the
    module docstring, plain SO_REUSEADDR on Windows would let a second
    process bind the same loopback port and intercept decrypted retrieve
    traffic."""

    class _OsNameShim:
        """Proxies the real ``os`` module except for ``.name``.

        We can't monkeypatch the real ``os.name`` attribute directly: pathlib
        (used transitively by ``create_app()``/hypercorn ``Config()`` during
        ``start()``) also reads ``os.name`` to pick ``WindowsPath`` vs.
        ``PosixPath`` and would break. Instead we rebind agy_dispatch's own
        module-level ``os`` reference (the shared plumbing this listener runs
        on) to this shim, leaving the real ``os`` module untouched.
        """

        def __init__(self, real_os: object, forced_name: str) -> None:
            self._real_os = real_os
            self.name = forced_name

        def __getattr__(self, item: str) -> object:
            return getattr(self._real_os, item)

    monkeypatch.setattr(agy_dispatch, "os", _OsNameShim(agy_dispatch.os, "nt"))
    # Real SO_EXCLUSIVEADDRUSE only exists on Windows; alias it to
    # SO_REUSEADDR's numeric value so the real setsockopt() syscall below
    # succeeds on this (POSIX) test host.
    monkeypatch.setattr(
        agy_dispatch.socket, "SO_EXCLUSIVEADDRUSE", socket.SO_REUSEADDR, raising=False
    )

    setsockopt_calls: list[tuple[socket.socket, int, int, int]] = []
    real_setsockopt = socket.socket.setsockopt

    def _spy_setsockopt(
        self: socket.socket, level: int, optname: int, value: int, *a: object, **kw: object
    ) -> None:
        setsockopt_calls.append((self, level, optname, value))
        real_setsockopt(self, level, optname, value, *a, **kw)

    monkeypatch.setattr(socket.socket, "setsockopt", _spy_setsockopt)

    class _FakeStartedServer:
        """Stand-in for the object asyncio.start_server() returns, so the
        forced (non-posix) code window doesn't have to drive real
        asyncio loop-internal connection machinery."""

        def __init__(self, sock: socket.socket) -> None:
            self.sockets = [sock]

        def close(self) -> None:
            self.sockets[0].close()

        async def wait_closed(self) -> None:
            return None

    async def _fake_start_server(
        _handler: object, sock: socket.socket | None = None, **_kw: object
    ) -> _FakeStartedServer:
        assert sock is not None
        return _FakeStartedServer(sock)

    monkeypatch.setattr(agy_dispatch.asyncio, "start_server", _fake_start_server)

    srv = AgyRetrieveServer(port=0)
    await srv.start()
    try:
        listener = srv._server.sockets[0]  # type: ignore[union-attr]
        calls_on_listener = [c for c in setsockopt_calls if c[0] is listener]
        # Exactly one setsockopt call was made on our listener socket, and it
        # went through the (non-posix) elif branch — the `if os.name ==
        # "posix"` branch never ran because we patched os.name to "nt".
        assert calls_on_listener == [(listener, socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]
        # The applied option is actually in effect on the real socket. Read
        # back a *non-zero* value rather than exactly 1: on macOS getsockopt()
        # reports SO_REUSEADDR's internal bitmask (4) while Linux echoes the 1
        # we set — both mean "enabled".
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) != 0
    finally:
        await srv.stop()


async def test_start_skips_sockopt_when_neither_posix_nor_exclusiveaddruse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``os.name`` isn't "posix" AND the platform lacks
    SO_EXCLUSIVEADDRUSE, neither socket-opt branch applies — the listener is
    still bound successfully, with no setsockopt call made at all."""

    class _OsNameShim:
        # See test_start_uses_so_exclusiveaddruse_on_non_posix for rationale:
        # we rebind agy_dispatch's own module-level `os` reference rather
        # than mutating the real `os` module (which pathlib etc. also read).
        def __init__(self, real_os: object, forced_name: str) -> None:
            self._real_os = real_os
            self.name = forced_name

        def __getattr__(self, item: str) -> object:
            return getattr(self._real_os, item)

    monkeypatch.setattr(agy_dispatch, "os", _OsNameShim(agy_dispatch.os, "nt"))
    monkeypatch.delattr(socket, "SO_EXCLUSIVEADDRUSE", raising=False)

    setsockopt_calls: list[tuple[socket.socket, int, int, int]] = []
    real_setsockopt = socket.socket.setsockopt

    def _spy_setsockopt(
        self: socket.socket, level: int, optname: int, value: int, *a: object, **kw: object
    ) -> None:
        setsockopt_calls.append((self, level, optname, value))
        real_setsockopt(self, level, optname, value, *a, **kw)

    monkeypatch.setattr(socket.socket, "setsockopt", _spy_setsockopt)

    srv = AgyRetrieveServer(port=0)
    await srv.start()
    try:
        listener = srv._server.sockets[0]  # type: ignore[union-attr]
        assert [c for c in setsockopt_calls if c[0] is listener] == []
        host, port = srv.address
        assert host == "127.0.0.1"
        assert isinstance(port, int) and port > 0
    finally:
        await srv.stop()


async def test_stop_swallows_lifespan_shutdown_exception() -> None:
    """stop() must not propagate an exception raised by
    ``lifespan.wait_for_shutdown()`` — it logs/ignores it and still tears
    down the rest of the server cleanly."""
    srv = AgyRetrieveServer(port=0)
    await srv.start()

    async def _boom() -> None:
        raise RuntimeError("shutdown boom")

    assert srv._lifespan is not None
    srv._lifespan.wait_for_shutdown = _boom  # type: ignore[method-assign]

    await srv.stop()  # must not raise despite wait_for_shutdown() failing

    assert srv._lifespan is None
    assert srv._server is None


async def test_stop_swallows_lifespan_task_cancel_exception() -> None:
    """stop() must not propagate an exception raised while awaiting the
    (just-cancelled) lifespan task — it cancels, swallows, and clears the
    reference regardless."""
    srv = AgyRetrieveServer(port=0)
    await srv.start()

    class _FakeCancelTask:
        def __init__(self) -> None:
            self.cancel_called = False

        def cancel(self) -> None:
            self.cancel_called = True

        def __await__(self) -> object:
            raise RuntimeError("await-after-cancel boom")

    fake_task = _FakeCancelTask()
    srv._lifespan_task = fake_task  # type: ignore[assignment]

    await srv.stop()  # must not raise despite awaiting the fake task failing

    assert fake_task.cancel_called is True
    assert srv._lifespan_task is None


async def test_async_context_manager_starts_and_stops() -> None:
    """Used as ``async with``, the server starts on __aenter__ and stops on
    __aexit__."""
    async with AgyRetrieveServer(port=0) as srv:
        assert isinstance(srv, AgyRetrieveServer)
        host, port = srv.address
        assert host == "127.0.0.1"
        assert isinstance(port, int) and port > 0

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/v1/retrieve/stats")
        assert resp.status_code == 200

    # __aexit__ ran stop(): lifespan task cleared, socket torn down.
    assert srv._lifespan_task is None
    with pytest.raises(RuntimeError):
        _ = srv.address

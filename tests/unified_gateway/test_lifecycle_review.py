"""Deterministic regression oracles for batch-4 lifecycle review findings."""

import asyncio
import json
import threading

import httpx
import pytest

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.credentials import CredentialBroker
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.execution import GatewayOperation
from headroom.proxy.gateway.runtime import GatewayRuntime
from tests.unified_gateway.test_lifecycle_ownership import Source
from tests.unified_gateway.test_ws_lifecycle import configuration, create, session


@pytest.mark.asyncio
async def test_cancelled_waiters_cannot_multiply_account_source_work():
    """Mutation caught: release a cancelled waiter without joining prior account work."""
    snapshot = GatewayConfigSnapshot.model_validate(configuration())
    source = Source(snapshot.credentials[0])
    broker = CredentialBroker({source.credential.id: source})
    waiters = []
    try:
        first = asyncio.create_task(broker.acquire(snapshot.routes[0]))
        waiters.append(first)
        await source.entered.wait()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        for _ in range(3):
            entered = asyncio.Event()

            async def acquire(started=entered):
                started.set()
                return await broker.acquire(snapshot.routes[0])

            waiter = asyncio.create_task(acquire())
            waiters.append(waiter)
            await entered.wait()
            assert len(broker._pending) == 1, "one actual acquisition per account until completion"
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        assert not broker._leases
    finally:
        for waiter in waiters:
            waiter.cancel()
        source.release.set()
        await asyncio.gather(*waiters, return_exceptions=True)
        await broker.aclose()
    assert source.closed == 1
    assert not broker._leases


def catalog_runtime(**limits):
    raw = configuration()
    raw["routes"][0]["catalog"]["source"] = "provider"
    raw["limits"]["shutdown_drain_seconds"] = 2
    raw["limits"].update(limits)
    return GatewayRuntime(
        GatewayConfigSnapshot.model_validate(raw),
        environ={"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret", "OPENAI_API_KEY": "secret"},
    )


@pytest.mark.asyncio
async def test_refresh_cancelled_before_start_releases_generation():
    """Mutation caught: release lives only in a coroutine never entered before cancellation."""
    runtime = catalog_runtime()
    entered = asyncio.Event()

    async def reader(*args):
        entered.set()
        return ()

    runtime.dependencies.metadata_reader = reader
    refresh = asyncio.create_task(runtime.refresh_catalog())
    revoke = asyncio.create_task(runtime.revoke(route_id=runtime.snapshot.routes[0].id))
    try:
        results = await asyncio.gather(refresh, revoke, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert results[1].applied
        assert not entered.is_set()
        await asyncio.gather(*tuple(runtime._cleanup_tasks), return_exceptions=True)
        assert not runtime._generation_users
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["drain", "complete"])
async def test_shutdown_denies_new_catalog_work(monkeypatch, phase):
    """Mutation caught: refresh creates a reader after shutdown has stopped admission."""
    runtime = catalog_runtime()
    generation = runtime.capture()
    owner = GatewayOperation(
        runtime,
        generation=generation,
        principal=generation.authenticator.authenticate({"authorization": "Bearer client-secret"}),
        route=generation.snapshot.routes[0],
        ingress_protocol="openai-responses",
        target_protocol="openai-responses",
        dispatch_plan="native",
    )
    calls = []

    async def reader(*args):
        calls.append(args)
        return ()

    runtime.dependencies.metadata_reader = reader
    stopped = asyncio.Event()
    original_stop = runtime.admission.shutdown

    async def stop():
        await original_stop()
        stopped.set()

    monkeypatch.setattr(runtime.admission, "shutdown", stop)
    shutting = asyncio.create_task(runtime.shutdown())
    try:
        await stopped.wait()
        if phase == "complete":
            await owner.close("cancelled")
            await shutting
        with pytest.raises(GatewayAuthorizationError) as caught:
            await runtime.refresh_catalog()
        assert caught.value.code == "gateway_shutting_down"
        assert not calls
        assert runtime.capture().catalog.revision == generation.catalog.revision
    finally:
        await owner.close("cancelled")
        await shutting


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["reload", "revoke"])
async def test_shutdown_fences_contended_generation_publication(monkeypatch, tmp_path, operation):
    """Mutation caught: readiness is checked before, but not inside, policy publication."""
    runtime = catalog_runtime()
    before = runtime.capture()
    path = tmp_path / "reload.json"
    path.write_text(before.snapshot.model_dump_json())
    publishing, stopping = asyncio.Event(), asyncio.Event()
    original_update, original_shutdown = runtime.admission.update_policy, runtime._shutdown

    async def update(*args, **kwargs):
        publishing.set()
        return await original_update(*args, **kwargs)

    async def shutdown():
        stopping.set()
        await original_shutdown()

    monkeypatch.setattr(runtime.admission, "update_policy", update)
    monkeypatch.setattr(runtime, "_shutdown", shutdown)
    async with runtime.admission._condition:
        mutation = asyncio.create_task(
            runtime.reload(path)
            if operation == "reload"
            else runtime.revoke(route_id=before.snapshot.routes[0].id)
        )
        await publishing.wait()
        shutting = asyncio.create_task(runtime.shutdown())
        await stopping.wait()
        assert not runtime.status().ready
    try:
        result = await mutation
        assert not result.applied
        assert result.error == "gateway_shutting_down"
        assert runtime.generation == before.number
        assert runtime.admission._generation == before.number
        assert runtime.snapshot.routes[0].enabled
    finally:
        await shutting


@pytest.mark.asyncio
@pytest.mark.parametrize("dependency", ["http_client", "broker"])
async def test_injected_dependency_outlives_owned_refresh(dependency):
    """Mutation caught: injected dependency closure waits for operations but not refreshes."""
    runtime = catalog_runtime(shutdown_cleanup_seconds=0.05)
    entered, released, cancelled, finished = (asyncio.Event() for _ in range(4))
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    broker = CredentialBroker({})
    injected = client if dependency == "http_client" else broker
    setattr(runtime.dependencies, dependency, injected)
    used_after_cancellation = []

    async def reader(generation, *args):
        assert getattr(generation, dependency) is injected
        entered.set()
        while not released.is_set():
            try:
                await released.wait()
            except asyncio.CancelledError:
                cancelled.set()
        assert not client.is_closed
        assert not broker._closed
        if dependency == "http_client":
            assert (
                await generation.http_client.get("https://fixture.invalid/models")
            ).status_code == 200
        used_after_cancellation.append(True)
        finished.set()
        return ()

    runtime.dependencies.metadata_reader = reader
    refresh = asyncio.create_task(runtime.refresh_catalog())
    await entered.wait()
    closed_by_runtime = False
    try:
        await runtime.shutdown()
        assert cancelled.is_set()
        assert not finished.is_set()
        assert runtime._generation_users == {1: 1}
        assert not client.is_closed
        assert not broker._closed
    finally:
        released.set()
        await asyncio.gather(refresh, return_exceptions=True)
        await asyncio.gather(*tuple(runtime._cleanup_tasks), return_exceptions=True)
        closed_by_runtime = client.is_closed if dependency == "http_client" else broker._closed
        await client.aclose()
        await broker.aclose()
    assert finished.is_set()
    assert used_after_cancellation == [True]
    assert closed_by_runtime
    assert client.is_closed
    assert broker._closed


@pytest.mark.asyncio
async def test_shutdown_fences_catalog_publication_after_lock_wait(monkeypatch):
    """Mutation caught: an already-created refresh publishes after shutdown begins."""
    runtime = catalog_runtime()
    before = runtime.capture()
    waiting, stopping = asyncio.Event(), asyncio.Event()
    lock = runtime._reload_lock

    class ResistantLock:
        async def __aenter__(self):
            waiting.set()
            while True:
                try:
                    await lock.acquire()
                    return self
                except asyncio.CancelledError:
                    pass

        async def __aexit__(self, *args):
            lock.release()

    async def reader(*args):
        return ()

    original_shutdown = runtime._shutdown

    async def shutdown():
        stopping.set()
        await original_shutdown()

    runtime.dependencies.metadata_reader = reader
    monkeypatch.setattr(runtime, "_reload_lock", ResistantLock())
    monkeypatch.setattr(runtime, "_shutdown", shutdown)
    async with lock:
        refresh = asyncio.create_task(runtime.refresh_catalog())
        await waiting.wait()
        shutting = asyncio.create_task(runtime.shutdown())
        await stopping.wait()
    try:
        result = (await asyncio.gather(refresh, return_exceptions=True))[0]
        assert isinstance(result, GatewayAuthorizationError)
        assert result.code == "gateway_shutting_down"
        assert runtime.capture().catalog.revision == before.catalog.revision
    finally:
        await shutting


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled", ["forward", "error", "close", "child"])
async def test_ws_deadline_and_cleanup_survive_stalled_peer_or_child(monkeypatch, stalled):
    """Mutations caught: unbounded forwarding/error writes, peer close, or child join."""
    raw = configuration()
    raw["limits"].update(request_deadline_seconds=0.08, shutdown_cleanup_seconds=0.05)
    entered, release = asyncio.Event(), asyncio.Event()

    async def block():
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass

    async def downstream(message):
        if (stalled in {"forward", "error"} and message["type"] == "websocket.send") or (
            stalled == "close" and message["type"] == "websocket.close"
        ):
            await block()

    async with session(monkeypatch, raw, downstream_send=downstream) as state:
        if stalled == "child":

            async def provider_receive():
                await block()
                raise StopAsyncIteration

            monkeypatch.setattr(state.upstream.received, "get", provider_receive)
        try:
            await create(state)
            await state.upstream.sent.get()
            if stalled in {"forward", "error"}:
                event = {"type": "response.output_text.delta", "delta": "hello"}
                if stalled == "error":
                    event = {"type": "error", "error": {"message": "private provider error"}}
                await state.upstream.received.put(json.dumps(event))
            else:
                if stalled == "child":
                    await entered.wait()
                await state.incoming.put({"type": "websocket.disconnect", "code": 1000})
            await entered.wait()
            # The production deadline must finish the session while the barrier
            # remains held. This timeout is only a non-cancelling failure watchdog.
            done, _ = await asyncio.wait({state.task}, timeout=1)
            assert state.task in done, "peer/child backpressure bypassed the deadline"
            assert not release.is_set()
            assert not state.runtime.active_work
            assert state.runtime.admission.active_count == 0
            assert state.runtime.admission.snapshot().unknown_charge_count == 1
            assert state.upstream.closed.is_set()
            assert state.upstream.connections == 1
            assert state.upstream.sent.empty()
        finally:
            release.set()
            await asyncio.gather(*tuple(state.runtime._cleanup_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_ws_retirement_shares_transport_cleanup_deadline(monkeypatch, tmp_path):
    """Mutation caught: generation retirement restarts the consumed cleanup budget."""
    raw = configuration()
    raw["limits"]["shutdown_cleanup_seconds"] = 0.5
    reader_entered, client_close_entered, release = (asyncio.Event() for _ in range(3))

    async def blocked(event):
        event.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass

    async with session(monkeypatch, raw) as state:
        original = state.runtime.capture().http_client
        original_close = original.aclose

        async def close_client():
            await blocked(client_close_entered)
            await original_close()

        async def receive():
            await blocked(reader_entered)
            raise StopAsyncIteration

        monkeypatch.setattr(original, "aclose", close_client)
        monkeypatch.setattr(state.upstream.received, "get", receive)
        try:
            await create(state)
            await state.upstream.sent.get()
            await reader_entered.wait()
            path = tmp_path / "reload.json"
            path.write_text(json.dumps(raw))
            assert (await state.runtime.reload(path)).applied
            await state.incoming.put({"type": "websocket.disconnect", "code": 1000})
            # Two held cleanup phases must share one 0.5s budget, not each
            # receive a fresh 0.5s allowance. No barrier is released to assist it.
            done, _ = await asyncio.wait({state.task}, timeout=0.8)
            assert state.task in done
            assert client_close_entered.is_set()
            assert state.runtime.admission.snapshot().unknown_charge_count == 1
            assert not state.runtime.active_work
        finally:
            release.set()
            await asyncio.gather(*tuple(state.runtime._cleanup_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_native_acquisition_is_owned_until_thread_finishes(monkeypatch):
    """Mutation caught: cancel the to_thread wrapper and close its still-running source."""
    snapshot = GatewayConfigSnapshot.model_validate(configuration())
    entered = asyncio.Event()
    release, finished = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    class NativeSource(Source):
        closed_after_native = False

        async def acquire(self, *, now):
            def worker():
                loop.call_soon_threadsafe(entered.set)
                release.wait()
                finished.set()

            await asyncio.to_thread(worker)
            self.release.set()
            return await super().acquire(now=now)

        async def aclose(self):
            self.closed_after_native = finished.is_set()
            await super().aclose()

    source = NativeSource(snapshot.credentials[0])
    broker = CredentialBroker({source.credential.id: source})
    task = asyncio.create_task(broker.acquire(snapshot.routes[0]))
    await entered.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    close_entered = asyncio.Event()
    original_close = broker._finish_close

    async def close():
        close_entered.set()
        await original_close()

    monkeypatch.setattr(broker, "_finish_close", close)
    closing = asyncio.create_task(broker.aclose())
    try:
        await close_entered.wait()
        assert len(broker._pending) == 1
        assert source.closed == 0
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.to_thread(finished.wait)
        await closing
    assert source.closed_after_native
    assert source.closed == 1
    assert not broker._leases


@pytest.mark.asyncio
async def test_ws_late_connection_is_closed_only_after_enter_finishes(monkeypatch):
    """Mutation caught: run context exit concurrently with cancellation-resistant enter."""
    raw = configuration()
    raw["limits"].update(request_deadline_seconds=0.08, shutdown_cleanup_seconds=0.05)
    enter_started, release_enter, enter_finished = (asyncio.Event() for _ in range(3))
    exit_observations: list[bool] = []

    class ResistantConnection:
        async def __aenter__(self):
            enter_started.set()
            while not release_enter.is_set():
                try:
                    await release_enter.wait()
                except asyncio.CancelledError:
                    pass
            enter_finished.set()
            return self

        async def __aexit__(self, *_args):
            exit_observations.append(enter_finished.is_set())

    connection = ResistantConnection()
    monkeypatch.setattr(
        "headroom.proxy.gateway.websocket.websocket_connection",
        lambda *_args, **_kwargs: connection,
    )
    async with session(monkeypatch, raw) as state:
        try:
            await create(state)
            await enter_started.wait()
            await state.incoming.put({"type": "websocket.disconnect", "code": 1000})
            done, _ = await asyncio.wait({state.task}, timeout=1)
            assert state.task in done, "session cleanup exceeded its bounded deadline"
            assert not enter_finished.is_set()
            release_enter.set()
            await asyncio.gather(*tuple(state.runtime._cleanup_tasks), return_exceptions=True)
            assert exit_observations == [True]
        finally:
            release_enter.set()

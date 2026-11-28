"""Publication, late source results, and retired generation ownership."""

import asyncio
import json

import pytest

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.credentials import CredentialBroker, CredentialLease, SecretHandle
from headroom.proxy.gateway.errors import GatewayCredentialUnavailable
from headroom.proxy.gateway.execution import GatewayOperation
from headroom.proxy.gateway.runtime import GatewayRuntime
from headroom.proxy.gateway.usage import conservative_cost_bound
from tests.unified_gateway.test_ws_lifecycle import configuration, create, session


class Source:
    def __init__(self, credential):
        self.credential = credential
        self.entered, self.release, self.exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.closed = 0

    async def acquire(self, *, now):
        self.entered.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                pass  # Model a late SDK result after cancellation.
        self.exited.set()
        return CredentialLease(
            self.credential.id,
            self.credential.provider,
            self.credential.id,
            self.credential.allowed_origins,
            self.credential.allowed_path_prefixes,
            None,
            1,
            SecretHandle("secret"),
        )

    async def invalidate(self, lease, reason):
        pass

    async def aclose(self):
        self.closed += 1


@pytest.mark.asyncio
async def test_revoked_broker_rejects_late_source_result_before_cache():
    snapshot = GatewayConfigSnapshot.model_validate(configuration())
    source = Source(snapshot.credentials[0])
    broker = CredentialBroker({source.credential.id: source})
    task = asyncio.create_task(broker.acquire(snapshot.routes[0]))
    await source.entered.wait()
    try:
        broker.revoke(source.credential.id)
    finally:
        source.release.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
    assert isinstance(result, GatewayCredentialUnavailable)
    assert not broker._leases


@pytest.mark.asyncio
async def test_cancelled_acquisition_never_publishes_late_result():
    snapshot = GatewayConfigSnapshot.model_validate(configuration())
    source = Source(snapshot.credentials[0])
    broker = CredentialBroker({source.credential.id: source})
    task = asyncio.create_task(broker.acquire(snapshot.routes[0]))
    await source.entered.wait()
    task.cancel()
    source.release.set()
    result = (await asyncio.gather(task, return_exceptions=True))[0]
    assert isinstance(result, asyncio.CancelledError)
    await source.exited.wait()
    assert not broker._leases


@pytest.mark.asyncio
async def test_broker_keeps_source_alive_until_late_acquire_finishes():
    snapshot = GatewayConfigSnapshot.model_validate(configuration())
    source = Source(snapshot.credentials[0])
    broker = CredentialBroker({source.credential.id: source})
    task = asyncio.create_task(broker.acquire(snapshot.routes[0]))
    await source.entered.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    close = asyncio.create_task(broker.aclose())
    try:
        done, _ = await asyncio.wait({close}, timeout=0.01)
        assert not done, "a source still executing acquisition cannot be closed"
        assert source.closed == 0
    finally:
        source.release.set()
        await close
    assert source.closed == 1
    assert not broker._leases


@pytest.mark.asyncio
async def test_idle_catalog_refresh_retains_captured_generation_until_result(monkeypatch, tmp_path):
    raw = configuration()
    raw["routes"][0]["catalog"]["source"] = "provider"
    entered, release = asyncio.Event(), asyncio.Event()
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.model_validate(raw),
        environ={"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret", "OPENAI_API_KEY": "secret"},
    )
    old = runtime.capture()

    async def metadata(*args):
        entered.set()
        await release.wait()
        assert not old.http_client.is_closed
        return ()

    runtime.dependencies.metadata_reader = metadata
    refresh = asyncio.create_task(runtime.refresh_catalog())
    await entered.wait()
    path = tmp_path / "reload.json"
    path.write_text(json.dumps(raw))
    try:
        assert (await runtime.reload(path)).applied
        assert not old.http_client.is_closed
    finally:
        release.set()
        await refresh
        await runtime.shutdown()
    assert old.http_client.is_closed


@pytest.mark.asyncio
async def test_retired_client_closes_only_after_captured_operation_releases(monkeypatch, tmp_path):
    clients = []

    class Client:
        closed = 0

        async def aclose(self):
            self.closed += 1

    def client(snapshot):
        result = Client()
        clients.append(result)
        return result

    monkeypatch.setattr("headroom.proxy.gateway.runtime.http_client", client)
    raw = configuration()
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.model_validate(raw),
        environ={"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret", "OPENAI_API_KEY": "secret"},
    )
    generation = runtime.capture()
    route = generation.snapshot.routes[0]
    operation = GatewayOperation(
        runtime,
        generation=generation,
        principal=GatewayPrincipal("local-app", frozenset({"inference"}), frozenset({route.id})),
        route=route,
        ingress_protocol="openai-responses",
        target_protocol="openai-responses",
        dispatch_plan="native",
    )
    await operation.start_attempt(
        route.credentials[0],
        conservative_cost_bound(route, {}, qualified_contracts=frozenset({"synthetic-ws-v1"})),
    )
    path = tmp_path / "reload.json"
    raw["routes"][0]["pricing"]["revision"] = "fixture-b"
    path.write_text(json.dumps(raw))
    assert (await runtime.reload(path)).applied
    assert clients[0].closed == 0
    assert operation.generation.number == 1
    assert runtime.capture().number == 2
    await operation.close("cancelled")
    assert clients[0].closed == 1
    assert clients[1].closed == 0
    await runtime.shutdown()
    await runtime.shutdown()
    assert [c.closed for c in clients] == [1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("check", ["transport", "reload", "revoke"])
async def test_runtime_closes_injected_transport_and_rejects_reload_after_shutdown(tmp_path, check):
    import httpx

    raw = configuration()
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.model_validate(raw),
        environ={"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret", "OPENAI_API_KEY": "secret"},
    )
    injected = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    runtime.dependencies.http_client = injected
    await runtime.shutdown()
    try:
        if check == "transport":
            assert injected.is_closed
        path = tmp_path / "reload.json"
        path.write_text(json.dumps(raw))
        result = (
            await runtime.revoke(route_id=raw["routes"][0]["id"])
            if check == "revoke"
            else await runtime.reload(path)
        )
        assert not result.applied
        assert runtime.generation == 1
    finally:
        await injected.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ["principal_id", "route_id", "account_id", "grant", "scope"])
async def test_revoke_cancels_active_websocket_and_idle_session(monkeypatch, tmp_path, selector):
    raw = configuration()
    if selector == "grant":
        spare = dict(raw["routes"][0], id="spare", public_model="spare")
        raw["routes"].append(spare)
        raw["client_auth"]["principals"][0]["routes"].append("spare")
    async with session(monkeypatch, raw) as state:
        await create(state)
        await asyncio.wait_for(state.upstream.sent.get(), 3)
        if selector in {"grant", "scope"}:
            raw = state.raw
            if selector == "grant":
                raw["client_auth"]["principals"][0]["routes"] = ["spare"]
            else:
                raw["client_auth"]["principals"][0]["scopes"] = ["models"]
            path = tmp_path / "reload.json"
            path.write_text(json.dumps(raw))
            assert (await state.runtime.reload(path)).applied
        else:
            await state.runtime.revoke(
                **{
                    selector: {
                        "principal_id": "local-app",
                        "route_id": "openai-native",
                        "account_id": "openai-api",
                    }[selector]
                }
            )
        done, _ = await asyncio.wait({state.task}, timeout=1)
        assert state.task in done
        assert state.upstream.closed.is_set()
        assert state.runtime.admission.snapshot().unknown_charge_count == 1
        assert not state.runtime.active_work


@pytest.mark.asyncio
async def test_revoke_before_first_turn_closes_idle_socket(monkeypatch):
    async with session(monkeypatch) as state:
        await state.runtime.revoke(principal_id="local-app")
        done, _ = await asyncio.wait({state.task}, timeout=1)
        assert state.task in done
        assert state.upstream.connections == 0


@pytest.mark.asyncio
async def test_route_revoke_cancels_ws_acquire_while_other_work_retains_broker(monkeypatch):
    raw = configuration()
    raw["routes"].append(dict(raw["routes"][0], id="spare", public_model="spare"))
    raw["client_auth"]["principals"][0]["routes"].append("spare")
    async with session(monkeypatch, raw) as state:
        generation = state.runtime.capture()
        source = Source(generation.snapshot.credentials[0])
        generation.broker._sources[source.credential.id] = source
        other = GatewayOperation(
            state.runtime,
            generation=generation,
            principal=generation.authenticator.authenticate(
                {"authorization": "Bearer client-secret"}
            ),
            route=generation.snapshot.routes[1],
            ingress_protocol="openai-responses",
            target_protocol="openai-responses",
            dispatch_plan="native",
        )
        try:
            await create(state)
            await source.entered.wait()
            await state.runtime.revoke(route_id="openai-native")
            done, _ = await asyncio.wait({state.task}, timeout=0.2)
            assert state.task in done, (
                "unrelated captured work must not keep a revoked acquisition alive"
            )
            assert state.upstream.sent.empty()
        finally:
            source.release.set()
            await other.close("cancelled")


@pytest.mark.asyncio
async def test_revoke_returns_applied_when_guarded_sdk_result_outlives_cleanup(monkeypatch):
    async with session(monkeypatch) as state:
        generation = state.runtime.capture()
        source = Source(generation.snapshot.credentials[0])
        generation.broker._sources[source.credential.id] = source
        await create(state)
        await source.entered.wait()
        try:
            result = await asyncio.wait_for(state.runtime.revoke(route_id="openai-native"), 1.5)
            assert result.applied
            assert state.runtime.admission.active_count == 0
            assert source.closed == 0
            assert not generation.broker._leases
        finally:
            source.release.set()
            await source.exited.wait()


@pytest.mark.asyncio
async def test_concurrent_revocations_cannot_restore_each_others_authority():
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.model_validate(configuration()),
        environ={"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret", "OPENAI_API_KEY": "secret"},
    )
    started = [asyncio.Event(), asyncio.Event()]

    async def revoke(index, **selector):
        started[index].set()
        return await runtime.revoke(**selector)

    try:
        async with runtime._reload_lock:
            first = asyncio.create_task(revoke(0, account_id="openai-api"))
            second = asyncio.create_task(revoke(1, principal_id="local-app"))
            await asyncio.gather(*(event.wait() for event in started))
        assert all(result.applied for result in await asyncio.gather(first, second))
        assert not runtime.snapshot.credentials[0].enabled
        assert not runtime.snapshot.client_auth.principals[0].enabled
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_revoked_route_cancels_catalog_source_work():
    raw = configuration()
    raw["routes"][0]["catalog"]["source"] = "provider"
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.model_validate(raw),
        environ={"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret", "OPENAI_API_KEY": "secret"},
    )
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def read(*args):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    runtime.dependencies.metadata_reader = read
    refresh = asyncio.create_task(runtime.refresh_catalog())
    try:
        await entered.wait()
        assert (await runtime.revoke(route_id="openai-native")).applied
        await asyncio.wait_for(cancelled.wait(), 0.2)
    finally:
        await runtime.shutdown()
        await asyncio.gather(refresh, return_exceptions=True)

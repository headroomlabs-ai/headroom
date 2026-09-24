"""Turn ownership through the actual WebSocket dispatcher, with event barriers."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocket

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.runtime import GatewayRuntime
from headroom.proxy.gateway.websocket import dispatch_native_responses_websocket


def configuration():
    raw = json.loads(
        (
            Path(__file__).parents[2]
            / "docs/proposals/unified-api-gateway/examples/gateway.api-keys.json"
        ).read_text()
    )
    raw["routes"] = raw["routes"][:1]
    raw["credentials"] = raw["credentials"][:1]
    raw["client_auth"]["principals"][0]["routes"] = [raw["routes"][0]["id"]]
    route = raw["routes"][0]
    route["pricing"] = {
        "input_usd_per_million": "1",
        "output_usd_per_million": "2",
        "cache_read_usd_per_million": "1",
        "cache_create_usd_per_million": "1",
        "revision": "fixture-a",
    }
    route["model_bounds"] = {
        "max_input_tokens": 10,
        "max_output_tokens": 10,
        "default_max_output_tokens": 10,
        "provider_contract": "synthetic-ws-v1",
    }
    raw["limits"] = {
        "request_deadline_seconds": 2,
        "websocket_idle_seconds": 2,
        "shutdown_drain_seconds": 0.05,
        "shutdown_cleanup_seconds": 0.5,
    }
    return raw


def completed(response_id="resp_a"):
    return json.dumps(
        {
            "type": "response.completed",
            "response": {"id": response_id, "usage": {"input_tokens": 3, "output_tokens": 2}},
        }
    )


class Upstream:
    def __init__(self):
        self.sent = asyncio.Queue()
        self.received = asyncio.Queue()
        self.closed = asyncio.Event()
        self.connections = 0

    async def __aenter__(self):
        self.connections += 1
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        self.closed.set()

    async def send(self, frame):
        await self.sent.put(frame)

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.received.get()
        if isinstance(event, BaseException):
            raise event
        return event


@asynccontextmanager
async def session(monkeypatch, raw=None, *, downstream_send=None):
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.model_validate(raw or configuration()),
        environ={
            "HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret",
            "OPENAI_API_KEY": "provider-secret",
        },
    )
    runtime.dependencies.qualified_cost_contracts = frozenset({"synthetic-ws-v1"})
    incoming, outgoing = asyncio.Queue(), asyncio.Queue()
    upstream = Upstream()
    monkeypatch.setattr("websockets.connect", lambda *args, **kwargs: upstream)
    headers = [(b"host", b"127.0.0.1"), (b"authorization", b"Bearer client-secret")]

    async def send(message):
        if downstream_send is not None:
            await downstream_send(message)
        await outgoing.put(message)

    socket = WebSocket(
        {
            "type": "websocket",
            "path": "/v1/responses",
            "headers": headers,
            "app": SimpleNamespace(state=SimpleNamespace(gateway_runtime=runtime)),
        },
        incoming.get,
        send,
    )
    # Authenticate using the same decoded headers used by Starlette's handshake middleware.
    socket.scope["gateway_principal"] = runtime.authenticator.authenticate(socket.headers)
    await incoming.put({"type": "websocket.connect"})
    task = asyncio.create_task(dispatch_native_responses_websocket(socket, None))
    assert (await asyncio.wait_for(outgoing.get(), 3))["type"] == "websocket.accept"
    state = SimpleNamespace(
        runtime=runtime,
        upstream=upstream,
        incoming=incoming,
        outgoing=outgoing,
        task=task,
        raw=raw or configuration(),
    )
    try:
        yield state
    finally:
        await incoming.put({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait({task}, timeout=3)
        if not task.done():
            task.cancel()
        finished = await asyncio.gather(task, return_exceptions=True)
        assert not isinstance(finished[0], Exception), finished[0]
        await runtime.shutdown()


async def create(state):
    frame = json.dumps(
        {
            "type": "response.create",
            "response": {"model": state.raw["routes"][0]["public_model"], "input": "hello"},
        }
    )
    await state.incoming.put({"type": "websocket.receive", "text": frame})
    return frame


async def message(state):
    result = await asyncio.wait_for(state.outgoing.get(), 3)
    return json.loads(result["text"]) if "text" in result else result


@pytest.mark.asyncio
async def test_second_turn_budget_is_priced_and_never_sent(monkeypatch):
    raw = configuration()
    raw["admission"] = {"budget_usd": "0.000030", "unknown_cost_policy": "block"}
    async with session(monkeypatch, raw) as state:
        await create(state)
        sent = asyncio.create_task(state.upstream.sent.get())
        error = asyncio.create_task(state.outgoing.get())
        done, _ = await asyncio.wait({sent, error}, timeout=3, return_when=asyncio.FIRST_COMPLETED)
        assert sent in done, error.result() if error.done() else "create was not forwarded"
        error.cancel()
        await asyncio.gather(error, return_exceptions=True)
        await state.upstream.received.put(completed())
        assert (await message(state))["type"] == "response.completed"
        await create(state)
        assert (await message(state))["error"]["code"] == "gateway_admission_budget"
        await asyncio.wait_for(state.task, 3)
        assert state.upstream.sent.empty()
        assert state.runtime.admission.snapshot().known_micro_usd == 7
        assert state.runtime.observability._totals["logical_requests"] == 2
        assert state.runtime.observability._totals["attempts"] == 1


@pytest.mark.asyncio
async def test_accepted_disconnect_retains_liability_and_never_replays(monkeypatch):
    async with session(monkeypatch) as state:
        await create(state)
        await asyncio.wait_for(state.upstream.sent.get(), 3)
        await state.upstream.received.put(ConnectionError("provider-secret"))
        await asyncio.wait_for(state.task, 3)
        ledger = state.runtime.admission.snapshot()
        assert ledger.unknown_charge_count == 1
        assert ledger.unresolved_micro_usd == 30
        assert state.runtime.observability._totals["attempts"] == 1
        assert state.runtime.admission.active_count == 0
        assert not state.runtime.active_work
        assert state.upstream.connections == 1
        assert state.upstream.sent.empty()


@pytest.mark.asyncio
async def test_shutdown_closes_idle_and_active_sessions(monkeypatch):
    async with session(monkeypatch) as state:
        await create(state)
        await asyncio.wait_for(state.upstream.sent.get(), 3)
        await state.upstream.received.put(completed())
        await message(state)
        assert state.runtime.active_work, "idle sockets must remain runtime-owned"
        await asyncio.wait_for(state.runtime.shutdown(), 2)
        await asyncio.wait_for(state.task, 1)
        assert state.upstream.closed.is_set()
        assert not state.runtime.active_work
        assert state.runtime.admission.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal", ["malformed", "response.failed", "response.incomplete", "response.cancelled"]
)
async def test_terminal_errors_finalize_one_failed_turn(monkeypatch, terminal):
    async with session(monkeypatch) as state:
        await create(state)
        await asyncio.wait_for(state.upstream.sent.get(), 3)
        await state.upstream.received.put(
            "{invalid" if terminal == "malformed" else json.dumps({"type": terminal})
        )
        await asyncio.wait_for(state.task, 3)
        assert state.runtime.observability._totals["logical_requests"] == 1
        assert state.runtime.observability._totals["attempts"] == 1
        assert state.runtime.admission.snapshot().unknown_charge_count == 1
        assert state.upstream.closed.is_set()


@pytest.mark.asyncio
async def test_turn_deadline_fires_without_provider_bytes(monkeypatch):
    raw = configuration()
    raw["limits"]["request_deadline_seconds"] = 0.08
    async with session(monkeypatch, raw) as state:
        await create(state)
        await asyncio.wait_for(state.upstream.sent.get(), 3)
        done, _ = await asyncio.wait({state.task}, timeout=0.7)
        assert state.task in done, "gateway must fire the deadline without test cancellation"
        assert state.runtime.admission.snapshot().unknown_charge_count == 1
        assert state.upstream.closed.is_set()


@pytest.mark.asyncio
async def test_second_cancellation_cannot_interrupt_session_accounting(monkeypatch):
    async with session(monkeypatch) as state:
        entered, release = asyncio.Event(), asyncio.Event()

        async def close(upstream, *args):
            entered.set()
            await release.wait()
            upstream.closed.set()

        monkeypatch.setattr(Upstream, "__aexit__", close)
        await create(state)
        await asyncio.wait_for(state.upstream.sent.get(), 3)
        await state.upstream.received.put(ConnectionError())
        await entered.wait()
        state.task.cancel()
        release.set()
        await asyncio.gather(state.task, return_exceptions=True)
        done, _ = await asyncio.wait(
            {asyncio.create_task(state.runtime._work_empty.wait())}, timeout=0.5
        )
        assert done, "session cleanup must finish without a second runtime shutdown"
        assert state.runtime.admission.active_count == 0
        assert state.runtime.admission.snapshot().unknown_charge_count == 1
        assert not state.runtime.active_work


@pytest.mark.asyncio
async def test_idle_session_expires_without_credentials_or_attempt(monkeypatch):
    raw = configuration()
    raw["limits"]["websocket_idle_seconds"] = 0.08
    async with session(monkeypatch, raw) as state:
        done, _ = await asyncio.wait({state.task}, timeout=0.5)
        assert state.task in done
        assert state.upstream.connections == 0
        assert not state.runtime.active_work
        assert state.runtime.observability._totals["attempts"] == 0

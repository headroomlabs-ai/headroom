"""Actual Uvicorn test process, with one explicit fake-server destination pin."""

import asyncio
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import uvicorn
from fastapi import Request

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.credential_sources.environment import EnvironmentCredentialSource
from headroom.proxy.gateway.egress import EgressPolicy
from headroom.proxy.gateway.execution import GatewayOperation
from headroom.proxy.gateway.lifecycle import GatewayServer
from headroom.proxy.gateway.transport import PinnedHTTPTransport, tls_context
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


def run(path: Path) -> None:
    snapshot = GatewayConfigSnapshot.load(path)
    app = create_app(ProxyConfig(gateway=snapshot, gateway_config_path=path))
    runtime = app.state.gateway_runtime
    observations = []
    counts = {"identity": 0}
    operations = []
    finished = {}
    arrivals = []
    reserve_barrier = asyncio.Event()
    acquire_entered, acquire_release = asyncio.Event(), asyncio.Event()
    queue_entered = asyncio.Event()
    shutdown_started = asyncio.Event()
    original_shutdown = runtime.shutdown

    async def shutdown_runtime():
        shutdown_started.set()
        await original_shutdown()

    runtime.shutdown = shutdown_runtime
    original_init = GatewayOperation.__init__
    original_finish = GatewayOperation._finish
    original_reserve = runtime.admission.reserve

    def capture_operation(operation, *args, **kwargs):
        original_init(operation, *args, **kwargs)
        operations.append(operation)

    async def finish_operation(operation, *args, **kwargs):
        try:
            return await original_finish(operation, *args, **kwargs)
        finally:
            finished[operation.id] = time.monotonic()

    async def reserve(request):
        arrivals.append({"principal": request.principal_id, "route": request.route_id})
        barrier_size = int(os.environ.get("GATEWAY_TEST_RESERVE_BARRIER", "0"))
        if barrier_size:
            if len(arrivals) >= barrier_size:
                reserve_barrier.set()
            await asyncio.wait_for(reserve_barrier.wait(), 5)
        pending = asyncio.create_task(original_reserve(request))
        asyncio.get_running_loop().call_soon(
            lambda: queue_entered.set() if runtime.admission.queued_count else None
        )
        return await pending

    GatewayOperation.__init__ = capture_operation
    GatewayOperation._finish = finish_operation
    runtime.admission.reserve = reserve
    now = [1000.0]
    original_acquire = EnvironmentCredentialSource.acquire

    async def acquire(source, *, now):
        counts["identity"] += 1
        acquire_entered.set()
        if os.environ.get("GATEWAY_TEST_ACQUIRE_BARRIER") == "1":
            await acquire_release.wait()
        return await original_acquire(source, now=now)

    EnvironmentCredentialSource.acquire = acquire

    def resolve(host, port):
        assert host in {"llm.internal.example", "api.anthropic.com"}
        assert host == httpx.URL(snapshot.routes[0].upstream_origin).host
        assert port in {httpx.URL(route.upstream_origin).port or 443 for route in snapshot.routes}
        return ("10.111.0.10",) if host == "llm.internal.example" else ("93.184.216.34",)

    class LocalFixtureTransport(PinnedHTTPTransport):
        async def handle_async_request(self, request):
            destination = request.extensions["gateway_destination"]
            assert destination.hostname in {"llm.internal.example", "api.anthropic.com"}
            assert destination.addresses == resolve(destination.hostname, destination.port)
            # The real policy/header boundary has approved the configured private
            # destination. Only this test transport maps that exact endpoint to
            # the fixture socket; TLS still verifies llm.internal.example.
            request.extensions["gateway_destination"] = replace(
                destination, addresses=("127.0.0.1",)
            )
            fixture_port = os.environ.get("GATEWAY_TEST_UPSTREAM_PORT")
            if fixture_port:
                assert destination.hostname == "api.anthropic.com" and destination.port == 443
                request.url = request.url.copy_with(port=int(fixture_port))
                request.extensions["gateway_destination"] = replace(
                    request.extensions["gateway_destination"],
                    port=int(fixture_port),
                    url=str(request.url),
                )
            fail_port = os.environ.pop("GATEWAY_TEST_CONNECT_FAIL_PORT", None)
            if fail_port:
                request.url = request.url.copy_with(port=int(fail_port))
                request.extensions["gateway_destination"] = replace(
                    request.extensions["gateway_destination"],
                    port=int(fail_port),
                    url=str(request.url),
                )
            return await super().handle_async_request(request)

    runtime.dependencies.egress_policy = EgressPolicy(resolver=resolve)
    runtime.dependencies.clock = lambda: now[0]
    runtime.dependencies.qualified_cost_contracts = frozenset({"synthetic-http-v1"})
    from headroom.proxy.gateway import websocket as gateway_websocket
    from headroom.proxy.gateway.transport import websocket_connection

    def fixture_websocket(destination, headers, context):
        assert destination.addresses == resolve(destination.hostname, destination.port)
        return websocket_connection(
            replace(destination, addresses=("127.0.0.1",)), headers, context
        )

    gateway_websocket.websocket_connection = fixture_websocket
    runtime.dependencies.http_client = httpx.AsyncClient(
        transport=LocalFixtureTransport(
            verify=tls_context(snapshot),
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        ),
        trust_env=False,
        follow_redirects=False,
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        response = await call_next(request)
        context = getattr(request.state, "gateway", None)
        if context is not None:
            observations.append(
                {
                    "generation": context.generation.number,
                    "catalog_revision": context.catalog.revision,
                    "tariff": context.route.pricing.revision if context.route.pricing else None,
                    **(
                        {
                            "account": context.account_selection.account_ref
                            if context.account_selection
                            else None
                        }
                        if os.environ.get("GATEWAY_TEST_RUNTIME_HTTP")
                        else {}
                    ),
                }
            )
        return response

    @app.get("/__test/probe")
    async def probe():
        return {
            **counts,
            "observations": observations,
            "totals": dict(runtime.observability._totals),
            "ledger": asdict(runtime.admission.snapshot()),
            "active": runtime.admission.active_count,
            "queued": runtime.admission.queued_count,
            "owned": len(runtime.active_work),
            "generation": runtime.generation,
            "reserve_arrivals": arrivals,
            "operations": [
                {
                    "principal": operation.principal.id,
                    "route": operation.route.id,
                    "generation": operation.generation.number,
                    "tariff": operation.route.pricing.revision if operation.route.pricing else None,
                    "deadline": operation.deadline,
                    "finished": finished.get(operation.id),
                    "terminal": operation.terminal,
                    "attempts": [asdict(attempt) for attempt in operation.attempts],
                }
                for operation in operations
            ],
        }

    @app.get("/__test/idle")
    async def idle():
        await asyncio.wait_for(runtime._work_empty.wait(), 10)
        return await probe()

    @app.post("/__test/cool/{account}/{quota}")
    async def cool(account: str, quota: str):
        import time

        runtime.router.cool_down(account, quota_key=quota, until=time.time() + 60)
        return {"cooled": True}

    @app.post("/__test/clock/{value}")
    async def advance(value: float):
        now[0] = value
        return {"now": now[0]}

    @app.post("/__test/expire-source")
    async def expire_source():
        broker = runtime.capture().broker
        for account, lease in tuple(broker._leases.items()):
            broker._leases[account] = replace(lease, expires_at=0)
        return {"expired": True}

    @app.post("/__test/acquire-entered")
    async def acquiring():
        await asyncio.wait_for(acquire_entered.wait(), 5)
        return {"entered": True}

    @app.post("/__test/queue-entered")
    async def queued():
        await asyncio.wait_for(queue_entered.wait(), 5)
        return {"queued": runtime.admission.queued_count}

    @app.post("/__test/release-acquire")
    async def release_acquire():
        acquire_release.set()
        return {"released": True}

    @app.post("/__test/shutdown")
    async def shutdown():
        await runtime.shutdown()
        return await probe()

    @app.post("/__test/shutdown-started")
    async def shutting_down():
        await asyncio.wait_for(shutdown_started.wait(), 5)
        return {"started": True}

    @app.post("/__test/stop")
    async def stop():
        runner.should_exit = True
        return {"stopping": True}

    @app.on_event("shutdown")
    async def close_fixture_client():
        await runtime.dependencies.http_client.aclose()

    app.router.routes.sort(
        key=lambda route: 0 if getattr(route, "path", "").startswith("/__test/") else 1
    )
    runner = GatewayServer(
        uvicorn.Config(
            app, host="127.0.0.1", port=snapshot.runtime.port, log_level="info", access_log=False
        ),
        runtime=runtime,
    )
    runner.run()


if __name__ == "__main__":
    run(Path(sys.argv[1]))

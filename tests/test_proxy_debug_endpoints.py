"""Tests for the loopback-only /debug/* introspection endpoints (Unit 5)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from headroom.proxy.debug_introspection import (
    collect_tasks,
)
from headroom.proxy.loopback_guard import (
    LOOPBACK_HOSTS,
    is_loopback_host,
    is_loopback_host_header,
    require_loopback,
)
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.warmup import WarmupRegistry
from headroom.proxy.ws_session_registry import (
    WebSocketSessionRegistry,
    WSSessionHandle,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    # Pin the simulated client address to loopback so the /debug/* guard
    # accepts the request. Without this, FastAPI's TestClient reports
    # the host as ``testclient`` and the guard correctly 404s us.
    # ``base_url`` pins the inbound ``Host:`` header to a loopback name
    # so the DNS-rebinding gate added in 2026-06 also passes.
    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 12345),
    ) as test_client:
        yield test_client


@pytest.fixture
def app_and_client():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 12345),
    ) as test_client:
        yield app, test_client


@pytest.fixture
def app_and_external_client():
    """TestClient that reports a non-loopback address (to exercise 404)."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("10.0.0.1", 54321),
    ) as test_client:
        yield app, test_client


@pytest.fixture
def app_and_rebinding_client():
    """TestClient that simulates a DNS-rebinding attack.

    The simulated TCP peer is loopback (``request.client.host`` passes
    the legacy IP check), but the inbound ``Host:`` header reads
    ``attacker.com`` — exactly what the browser sends after the
    attacker's DNS record flips to ``127.0.0.1``.
    """
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(
        app,
        base_url="http://attacker.com",
        client=("127.0.0.1", 12345),
    ) as test_client:
        yield app, test_client


# ---------------------------------------------------------------------------
# Loopback guard unit tests
# ---------------------------------------------------------------------------


def test_is_loopback_host_accepts_canonical_hosts():
    for host in LOOPBACK_HOSTS:
        assert is_loopback_host(host) is True
    # None (TestClient with no client info) is treated as loopback.
    assert is_loopback_host(None) is True


def test_is_loopback_host_rejects_external_hosts():
    assert is_loopback_host("10.0.0.1") is False
    assert is_loopback_host("192.168.1.100") is False
    assert is_loopback_host("8.8.8.8") is False


def test_is_loopback_host_accepts_ipv6_mapped_ipv4_loopback():
    # On Linux dual-stack sockets with IPV6_V6ONLY=0, an IPv4 loopback
    # connection arrives as ``::ffff:127.0.0.1``. The guard must treat
    # this as loopback or /debug/* silently 404s when the proxy binds
    # to ``::`` / ``0.0.0.0``.
    assert is_loopback_host("::ffff:127.0.0.1") is True


def test_is_loopback_host_rejects_ipv6_mapped_external_ipv4():
    assert is_loopback_host("::ffff:10.0.0.1") is False


def test_is_loopback_host_rejects_non_loopback_ipv6():
    assert is_loopback_host("2001:db8::1") is False


def test_is_loopback_host_rejects_malformed_input():
    assert is_loopback_host("not-an-ip") is False
    assert is_loopback_host("") is False


def test_require_loopback_raises_404_for_external_client():
    class _FakeClient:
        host = "10.0.0.1"

    class _FakeRequest:
        client = _FakeClient()

    with pytest.raises(HTTPException) as exc_info:
        require_loopback(_FakeRequest())  # type: ignore[arg-type]

    assert exc_info.value.status_code == 404
    # Privacy: 404 explicitly, not 403 — endpoints should be invisible.
    assert exc_info.value.status_code != 403


def test_require_loopback_accepts_loopback_client():
    class _FakeClient:
        host = "127.0.0.1"

    class _FakeRequest:
        client = _FakeClient()

    # Should not raise. ``headers`` is absent so the Host-header gate
    # falls back to the legacy IP-only behaviour for callers that
    # construct a bare request stub.
    require_loopback(_FakeRequest())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Host-header (DNS-rebinding) guard unit tests
# ---------------------------------------------------------------------------


def test_is_loopback_host_header_accepts_canonical_values():
    for value in (
        "127.0.0.1",
        "127.0.0.1:8787",
        "localhost",
        "localhost:8787",
        "LOCALHOST",
        "Localhost:8787",
        "[::1]",
        "[::1]:8787",
    ):
        assert is_loopback_host_header(value) is True, value


def test_is_loopback_host_header_rejects_external_names():
    for value in (
        "attacker.com",
        "attacker.com:8787",
        "evil.example",
        "10.0.0.1",
        "10.0.0.1:8787",
        "8.8.8.8",
    ):
        assert is_loopback_host_header(value) is False, value


def test_is_loopback_host_header_rejects_missing_and_malformed():
    assert is_loopback_host_header(None) is False
    assert is_loopback_host_header("") is False
    assert is_loopback_host_header("   ") is False
    # Unterminated bracketed IPv6
    assert is_loopback_host_header("[::1") is False
    # Hostname that merely contains a loopback substring
    assert is_loopback_host_header("localhost.attacker.com") is False


def test_require_loopback_blocks_dns_rebinding_host_header():
    """Loopback IP + ``Host: attacker.com`` is the rebinding signature."""

    class _FakeClient:
        host = "127.0.0.1"

    class _FakeHeaders:
        def get(self, key, default=None):
            if key.lower() == "host":
                return "attacker.com"
            return default

    class _FakeRequest:
        client = _FakeClient()
        headers = _FakeHeaders()

    with pytest.raises(HTTPException) as exc_info:
        require_loopback(_FakeRequest())  # type: ignore[arg-type]
    assert exc_info.value.status_code == 404


def test_require_loopback_accepts_loopback_host_header():
    class _FakeClient:
        host = "127.0.0.1"

    class _FakeHeaders:
        def get(self, key, default=None):
            if key.lower() == "host":
                return "127.0.0.1:8787"
            return default

    class _FakeRequest:
        client = _FakeClient()
        headers = _FakeHeaders()

    # Should not raise — both gates pass.
    require_loopback(_FakeRequest())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Serializer unit tests
# ---------------------------------------------------------------------------


def test_warmup_registry_to_dict_returns_registry_shape():
    """Serializer equivalent of the old collect_warmup helper.

    The helper was inlined at the /debug/warmup route handler in server.py
    (``registry.to_dict() if registry else {}``); this test preserves
    coverage of the registry's own serializer contract.
    """
    registry = WarmupRegistry()
    registry.kompress.mark_loaded(handle=object(), source_status="enabled")
    registry.memory_backend.mark_error("boom")

    payload = registry.to_dict()

    assert payload["kompress"]["status"] == "loaded"
    assert payload["memory_backend"]["status"] == "error"
    assert payload["memory_backend"]["error"] == "boom"
    # Raw handle must never leak into the serialized payload.
    assert "handle" not in payload["kompress"]


def test_ws_session_registry_snapshot_returns_registered_entries():
    """Serializer equivalent of the old collect_ws_sessions helper."""
    reg = WebSocketSessionRegistry()
    handle = WSSessionHandle(
        session_id="sess-debug-1",
        request_id="req-debug-1",
        client_addr="127.0.0.1:9999",
        upstream_url="wss://upstream/test",
    )
    reg.register(handle)

    payload = reg.snapshot()
    assert len(payload) == 1
    assert payload[0]["session_id"] == "sess-debug-1"
    assert payload[0]["request_id"] == "req-debug-1"


@pytest.mark.asyncio
async def test_collect_tasks_returns_current_tasks_with_metadata():
    async def _noop_task():
        await asyncio.sleep(0.05)

    task = asyncio.create_task(_noop_task(), name="debug-test-task")
    try:
        entries = collect_tasks()
        matching = [e for e in entries if e["name"] == "debug-test-task"]
        assert matching, "expected the named task to appear in collect_tasks output"
        entry = matching[0]
        assert entry["coro_qualname"] is not None
        # Privacy: no frame locals, no coroutine args.
        assert "locals" not in entry
        assert "cr_frame" not in entry
        assert "args" not in entry
        assert entry["stack_depth"] is None or isinstance(entry["stack_depth"], int)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass


@pytest.mark.asyncio
async def test_collect_tasks_derives_age_from_ws_registry_for_codex_relays():
    reg = WebSocketSessionRegistry()
    sid = "relay-sess-1"
    reg.register(
        WSSessionHandle(
            session_id=sid,
            request_id="req-relay-1",
            client_addr="127.0.0.1:1",
            upstream_url="wss://upstream",
        )
    )

    async def _long_relay():
        await asyncio.sleep(0.2)

    relay_task = asyncio.create_task(_long_relay(), name=f"codex-ws-c2u-{sid}")
    try:
        await asyncio.sleep(0.02)  # let some age accrue
        entries = collect_tasks(ws_registry=reg)
        named = [e for e in entries if e["name"] == f"codex-ws-c2u-{sid}"]
        assert named, "expected relay task in output"
        entry = named[0]
        assert entry["age_seconds"] is not None
        assert entry["age_seconds"] >= 0.0
    finally:
        relay_task.cancel()
        try:
            await relay_task
        except (asyncio.CancelledError, BaseException):
            pass


# ---------------------------------------------------------------------------
# HTTP endpoint tests (loopback)
# ---------------------------------------------------------------------------


def test_debug_tasks_returns_json_array_for_loopback(client):
    response = client.get("/debug/tasks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Each entry at least has name + coro_qualname fields.
    for entry in data:
        assert "name" in entry
        assert "coro_qualname" in entry


def test_debug_tasks_stack_depth_is_gated_behind_query(client):
    """Default response must not compute stack_depth (P3 Fix 29 perf gate).

    ``?stack=true`` opts into the synchronous ``Task.get_stack`` walk; the
    default stays cheap so snapshotting during a reconnect storm does
    not stall the event loop.
    """
    default = client.get("/debug/tasks")
    assert default.status_code == 200
    for entry in default.json():
        assert entry["stack_depth"] is None, (
            f"default /debug/tasks must not compute stack_depth; "
            f"got {entry['stack_depth']!r} for {entry.get('name')!r}"
        )

    with_stack = client.get("/debug/tasks?stack=true")
    assert with_stack.status_code == 200
    entries = with_stack.json()
    # At least one entry should have a computed depth (the TestClient
    # itself runs under a task). Some entries may still be None if
    # get_stack raised defensively — we only require that opting in
    # produces at least one integer result.
    integer_depths = [e["stack_depth"] for e in entries if isinstance(e["stack_depth"], int)]
    assert integer_depths, (
        f"expected at least one int stack_depth when ?stack=true; got entries={entries!r}"
    )


def test_debug_warmup_reports_registry_slots(client):
    response = client.get("/debug/warmup")
    assert response.status_code == 200
    data = response.json()
    # Registry surfaces all canonical slot names.
    assert "kompress" in data
    assert "magika" in data
    assert "memory_backend" in data
    assert "memory_embedder" in data
    assert "runtime" in data
    # Each slot has at least a status field.
    assert "status" in data["memory_backend"]
    assert data["runtime"]["anthropic_pre_upstream"]["resolved_concurrency"] >= 0
    assert data["runtime"]["websocket_sessions"]["active_relay_tasks"] == 0


def test_debug_ws_sessions_reports_live_session(app_and_client):
    app, client = app_and_client
    proxy = app.state.proxy
    assert proxy is not None, "create_app must wire app.state.proxy"

    sid = "sess-debug-http"
    proxy.ws_sessions.register(
        WSSessionHandle(
            session_id=sid,
            request_id="req-debug-http",
            client_addr="127.0.0.1:12345",
            upstream_url="wss://upstream/test",
        )
    )
    try:
        response = client.get("/debug/ws-sessions")
        assert response.status_code == 200
        data = response.json()
        matching = [entry for entry in data if entry["session_id"] == sid]
        assert matching, "expected live session in /debug/ws-sessions output"
        assert matching[0]["request_id"] == "req-debug-http"
    finally:
        proxy.ws_sessions.deregister(sid, cause="response_completed")

    # After cleanup the session is gone.
    response = client.get("/debug/ws-sessions")
    assert response.status_code == 200
    assert all(entry["session_id"] != sid for entry in response.json())


def test_debug_endpoints_do_not_mutate_state(client):
    # Call each endpoint 100 times and confirm the second read equals
    # the first — no accidental mutation from serialization.
    first_tasks = client.get("/debug/tasks").json()
    first_warmup = client.get("/debug/warmup").json()
    first_ws = client.get("/debug/ws-sessions").json()

    for _ in range(100):
        client.get("/debug/tasks")
        client.get("/debug/warmup")
        client.get("/debug/ws-sessions")

    # Warmup and ws-sessions are deterministic (no background work touches
    # them in this test config), so they must be identical.
    assert client.get("/debug/warmup").json() == first_warmup
    assert client.get("/debug/ws-sessions").json() == first_ws
    # Tasks may vary naturally, but the call itself never raises and the
    # shape never changes.
    new_tasks = client.get("/debug/tasks").json()
    assert isinstance(new_tasks, list)
    for entry in new_tasks:
        assert set(entry.keys()) == set(first_tasks[0].keys()) if first_tasks else True


def test_debug_tasks_does_not_leak_coro_locals(client):
    response = client.get("/debug/tasks")
    assert response.status_code == 200
    for entry in response.json():
        # Privacy check: the serializer must not leak coroutine locals,
        # frame state, or request bodies. Only name / qualname / age /
        # depth / done are allowed.
        assert set(entry.keys()) <= {
            "name",
            "coro_qualname",
            "age_seconds",
            "stack_depth",
            "done",
        }


# ---------------------------------------------------------------------------
# HTTP endpoint tests (non-loopback)
# ---------------------------------------------------------------------------


def test_debug_endpoints_return_404_for_non_loopback_client(app_and_external_client):
    _, client = app_and_external_client
    for path in ("/debug/tasks", "/debug/ws-sessions", "/debug/warmup"):
        response = client.get(path)
        assert response.status_code == 404, path
        # Must be 404, not 403 — invisible to scanners.
        assert response.status_code != 403


def test_debug_endpoints_block_dns_rebinding(app_and_rebinding_client):
    """Loopback client + ``Host: attacker.com`` must 404 like an external client.

    Regression for the DNS-rebinding gap: prior to 2026-06 the guard
    only checked ``request.client.host``, which a rebound browser
    passes trivially. Adding a ``Host:`` header allowlist closes that
    gap so a malicious site cannot read /debug/* over the user's
    loopback proxy via the wide-open CORS policy.
    """
    _, client = app_and_rebinding_client
    for path in ("/debug/tasks", "/debug/ws-sessions", "/debug/warmup"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.status_code != 403


def test_existing_health_routes_unchanged(client):
    # Invariant: Unit 5 must not regress the existing health endpoints.
    for path in ("/livez", "/readyz", "/health"):
        response = client.get(path)
        assert response.status_code == 200, path

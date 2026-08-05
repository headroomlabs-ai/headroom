"""Loopback-gating tests for state-mutating / content-leaking endpoints.

``/transformations/feed`` can return full prompt + completion bodies (when
``log_full_messages`` is on) and ``/cache/clear`` mutates server state. With the
default ``--host 0.0.0.0`` Docker bind, neither should be reachable by an
arbitrary network client — they are gated to the loopback interface via
``require_loopback`` (the same guard already used for ``/admin/*`` and
``/debug/*``). See #863.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from headroom.cache.backends import InMemoryBackend
from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.proxy.loopback_guard import is_ip_literal_host_header
from headroom.proxy.server import ProxyConfig, create_app

GATED = [
    ("get", "/transformations/feed"),
    ("post", "/cache/clear"),
]


def _make_app() -> FastAPI:
    return create_app(
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
        )
    )


def _loopback_client() -> TestClient:
    # A real loopback peer + a loopback Host header — passes both guard gates
    # (client-IP check and the DNS-rebinding Host-header check).
    return TestClient(_make_app(), base_url="http://127.0.0.1", client=("127.0.0.1", 12345))


def _seed_ccr_entry() -> str:
    reset_compression_store()
    store = get_compression_store(backend=InMemoryBackend())
    return store.store(
        "seeded-ccr-content",
        "<<ccr:seeded>>",
        original_tokens=3,
        compressed_tokens=1,
        tool_name="seeded-test",
    )


@pytest.mark.parametrize("method,path", GATED)
def test_non_loopback_caller_gets_404(method: str, path: str) -> None:
    # A vanilla TestClient presents client.host="testclient", which is not a
    # loopback IP, so the guard returns 404 (invisible, not 403).
    client = TestClient(_make_app())
    resp = client.request(method, path)
    assert resp.status_code == 404, resp.text


@pytest.mark.parametrize("method,path", GATED)
def test_loopback_caller_allowed(method: str, path: str) -> None:
    client = _loopback_client()
    resp = client.request(method, path)
    assert resp.status_code == 200, resp.text


# CCR data endpoints — cached session content, gated to 404 off-loopback (#1227).
def test_stats_lifetime_route_uses_dashboard_metadata_access_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS",
        "100.90.0.5/32",
    )
    app = _make_app()
    expected = {
        "requests": {"total": 7},
        "projects": {"headroom": {"requests": 3}},
        "persistence": {
            "enabled": True,
            "healthy": False,
            "error": "D:/private/proxy_savings.json: access denied",
        },
    }
    monkeypatch.setattr(
        app.state.proxy.metrics.savings_tracker,
        "lifetime_response",
        lambda: expected,
    )

    network = TestClient(app).get("/stats-lifetime")
    assert network.status_code == 200, network.text
    assert network.json() == {
        "requests": {"total": 7},
        "persistence": {
            "enabled": True,
            "healthy": False,
            "error": None,
        },
    }

    loopback = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 12345),
    ).get("/stats-lifetime")
    assert loopback.status_code == 200, loopback.text
    assert loopback.json() == expected

    trusted_dashboard = TestClient(
        app,
        base_url="http://100.82.0.2:8787",
        client=("100.90.0.5", 12345),
    ).get("/stats-lifetime")
    assert trusted_dashboard.status_code == 200, trusted_dashboard.text
    assert trusted_dashboard.json() == expected


CCR_GATED = [
    ("post", "/v1/retrieve"),
    ("get", "/v1/retrieve/stats"),
    ("get", "/v1/retrieve/somehash"),
    ("post", "/v1/retrieve/tool_call"),
    ("post", "/v1/compress"),
]


@pytest.mark.parametrize("method,path", CCR_GATED)
def test_ccr_non_loopback_gets_404(method: str, path: str) -> None:
    resp = TestClient(_make_app()).request(method, path, json={})
    assert resp.status_code == 404, resp.text


def test_ccr_retrieve_hash_route_blocks_valid_hash_for_non_loopback() -> None:
    ccr_hash = _seed_ccr_entry()
    try:
        loopback = _loopback_client()
        loopback_resp = loopback.get(f"/v1/retrieve/{ccr_hash}")
        assert loopback_resp.status_code == 200, loopback_resp.text
        assert loopback_resp.json()["original_content"] == "seeded-ccr-content"

        network_resp = TestClient(_make_app()).get(f"/v1/retrieve/{ccr_hash}")
        assert network_resp.status_code == 404, network_resp.text
    finally:
        reset_compression_store()


def test_dns_rebinding_host_header_rejected() -> None:
    # Loopback peer IP but an attacker-controlled Host header (the DNS-rebinding
    # shape) must still be rejected by the second gate.
    client = TestClient(_make_app(), base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    resp = client.get("/transformations/feed", headers={"host": "attacker.example"})
    assert resp.status_code == 404, resp.text


@pytest.mark.parametrize(
    "host_header",
    ["100.82.0.2", "100.82.0.2:8787", "[fd7a:115c:a1e0::2]", "[fd7a:115c:a1e0::2]:8787"],
)
def test_ip_literal_host_header_accepts_ip_addresses(host_header: str) -> None:
    assert is_ip_literal_host_header(host_header) is True


@pytest.mark.parametrize(
    "host_header",
    [None, "", "attacker.example", "localhost", "user@100.82.0.2", "100.82.0.2/path", "[fd7a::1"],
)
def test_ip_literal_host_header_rejects_non_addresses(host_header: str | None) -> None:
    assert is_ip_literal_host_header(host_header) is False


def _client(*, loopback: bool) -> TestClient:
    app = _make_app()
    if loopback:
        return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
    # Default TestClient presents client.host="testclient" — not loopback.
    return TestClient(app)


def test_health_config_block_is_loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """/health stays reachable for monitors but hides the `config` block (which
    echoes upstream API URLs + backend settings) from non-loopback callers."""
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")

    network = _client(loopback=False).get("/health")
    assert network.status_code == 200
    assert "config" not in network.json()
    # Basic health is still visible to monitors.
    assert network.json()["status"] in {"healthy", "unhealthy"}

    local = _client(loopback=True).get("/health")
    assert local.status_code == 200
    assert "config" in local.json()


def test_stats_per_request_metadata_is_loopback_only() -> None:
    """/stats keeps aggregate counters public but restricts per-request metadata
    (recent_requests / request_logs) and `config` to loopback callers."""
    network = _client(loopback=False).get("/stats")
    assert network.status_code == 200
    payload = network.json()
    assert "tokens" in payload  # aggregate counters still served
    assert "recent_requests" not in payload
    assert "request_logs" not in payload
    assert "config" not in payload

    local = _client(loopback=True).get("/stats").json()
    assert "recent_requests" in local
    assert "config" in local


def test_stats_metadata_served_to_trusted_gateway_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Containerized dashboards: a browser on the host reaches a bridge-network
    container via the gateway IP, so the peer isn't 127.0.0.1 and per-request
    metadata gets stripped. When the operator allow-lists the gateway CIDR via
    HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS, the peer is treated as
    loopback-equivalent and the metadata is served again."""
    gateway_ip = "172.18.0.1"  # typical docker/mocker bridge gateway
    app = _make_app()

    def _gateway_client() -> TestClient:
        # Loopback Host header (the operator browses http://127.0.0.1:8787) but
        # the peer IP is the container gateway, not loopback.
        return TestClient(app, base_url="http://127.0.0.1", client=(gateway_ip, 54321))

    # Without the allow-list, the gateway peer is untrusted → metadata stripped.
    monkeypatch.delenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", raising=False)
    stripped = _gateway_client().get("/stats").json()
    assert "recent_requests" not in stripped
    assert "config" not in stripped

    # Allow-list the gateway CIDR → peer trusted → metadata served.
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "172.18.0.0/16")
    served = _gateway_client().get("/stats").json()
    assert "recent_requests" in served
    assert "config" in served

    # DNS-rebinding defence still applies even for a trusted gateway peer: a
    # non-loopback Host header must be rejected.
    rebind = TestClient(app, base_url="http://attacker.example", client=(gateway_ip, 54321))
    payload = rebind.get("/stats").json()
    assert "recent_requests" not in payload


@pytest.mark.parametrize("cached", [False, True])
def test_dashboard_client_cidr_grants_stats_metadata_for_ip_literal_host(
    monkeypatch: pytest.MonkeyPatch,
    cached: bool,
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "100.90.0.5/32")
    app = _make_app()
    client = TestClient(
        app,
        base_url="http://100.82.0.2:8787",
        client=("100.90.0.5", 12345),
    )

    payload = client.get("/stats", params={"cached": int(cached)}).json()

    assert "recent_requests" in payload
    assert "request_logs" in payload
    assert "config" in payload


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "http://100.82.0.2:8787"},
        {"referer": "http://100.82.0.2:8787/dashboard"},
    ],
)
@pytest.mark.parametrize("cached", [False, True])
def test_dashboard_client_cidr_grants_stats_metadata_to_same_origin_browser(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], cached: bool
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "100.90.0.5/32")
    client = TestClient(
        _make_app(),
        base_url="http://100.82.0.2:8787",
        client=("100.90.0.5", 12345),
    )

    payload = client.get("/stats", params={"cached": int(cached)}, headers=headers).json()

    assert "recent_requests" in payload
    assert "request_logs" in payload
    assert "config" in payload


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "http://attacker.example"},
        {"referer": "http://attacker.example/dashboard"},
    ],
)
@pytest.mark.parametrize("cached", [False, True])
def test_dashboard_client_cidr_hides_stats_metadata_from_cross_origin_browser(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], cached: bool
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "100.90.0.5/32")
    client = TestClient(
        _make_app(),
        base_url="http://100.82.0.2:8787",
        client=("100.90.0.5", 12345),
    )

    response = client.get("/stats", params={"cached": int(cached)}, headers=headers)
    payload = response.json()

    assert response.status_code == 200
    assert "tokens" in payload
    assert "recent_requests" not in payload
    assert "request_logs" not in payload
    assert "config" not in payload


def test_dashboard_client_cidr_only_uses_forwarded_proto_from_trusted_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "100.90.0.5/32")
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "172.18.0.0/16")
    client = TestClient(
        _make_app(),
        base_url="http://100.82.0.2:8787",
        client=("172.18.0.1", 12345),
    )

    payload = client.get(
        "/stats",
        headers={
            "origin": "https://100.82.0.2:8787",
            "x-forwarded-for": "100.90.0.5",
            "x-forwarded-proto": "https",
        },
    ).json()

    assert "recent_requests" in payload
    assert "request_logs" in payload
    assert "config" in payload

    spoofed = (
        TestClient(
            _make_app(),
            base_url="http://100.82.0.2:8787",
            client=("100.90.0.5", 12345),
        )
        .get(
            "/stats",
            headers={
                "origin": "https://100.82.0.2:8787",
                "x-forwarded-proto": "https",
            },
        )
        .json()
    )

    assert "recent_requests" not in spoofed
    assert "request_logs" not in spoofed
    assert "config" not in spoofed


def test_dashboard_client_cidr_rejects_unlisted_clients_and_hostname_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "100.90.0.5/32")
    app = _make_app()

    unlisted = (
        TestClient(
            app,
            base_url="http://100.82.0.2:8787",
            client=("100.90.0.6", 12345),
        )
        .get("/stats")
        .json()
    )
    hostname = (
        TestClient(
            app,
            base_url="http://100.82.0.2:8787",
            client=("100.90.0.5", 12345),
        )
        .get("/stats", headers={"host": "attacker.example"})
        .json()
    )

    for payload in (unlisted, hostname):
        assert "recent_requests" not in payload
        assert "request_logs" not in payload
        assert "config" not in payload


def test_dashboard_client_cidr_only_accepts_forwarded_client_from_trusted_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "100.90.0.5/32")
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "172.18.0.0/16")
    app = _make_app()

    trusted = (
        TestClient(
            app,
            base_url="http://100.82.0.2:8787",
            client=("172.18.0.1", 12345),
        )
        .get("/stats", headers={"x-forwarded-for": "100.90.0.5"})
        .json()
    )
    forged = (
        TestClient(
            app,
            base_url="http://100.82.0.2:8787",
            client=("198.51.100.10", 12345),
        )
        .get("/stats", headers={"x-forwarded-for": "100.90.0.5"})
        .json()
    )

    assert "recent_requests" in trusted
    assert "recent_requests" not in forged


def test_dashboard_client_cidr_normalizes_ipv4_mapped_ipv6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "100.90.0.0/24")
    app = _make_app()
    payload = (
        TestClient(
            app,
            base_url="http://100.82.0.2:8787",
            client=("::ffff:100.90.0.5", 12345),
        )
        .get("/stats")
        .json()
    )

    assert "recent_requests" in payload


def test_dashboard_client_cidr_does_not_expand_other_management_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS", "100.90.0.5/32")
    client = TestClient(
        _make_app(),
        base_url="http://100.82.0.2:8787",
        client=("100.90.0.5", 12345),
    )

    health = client.get("/health")
    assert health.status_code == 200
    assert "config" not in health.json()
    assert client.get("/admin/upstream").status_code == 404
    assert client.get("/debug/tasks").status_code == 404
    assert client.post("/stats/reset").status_code == 404

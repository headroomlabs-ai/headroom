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

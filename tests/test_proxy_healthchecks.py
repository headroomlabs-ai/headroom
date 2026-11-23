import os
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, __version__, create_app


@pytest.fixture
def client(monkeypatch):
    # Skip the live upstream connectivity probe in unit tests — tests verify
    # the check logic separately (see test_readyz_upstream_check_* below).
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    # Loopback client/Host: /health serves the `config` block only to loopback
    # callers (network callers get the /readyz-shape body, no config).
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as test_client:
        yield test_client


def test_livez_reports_process_health(client):
    response = client.get("/livez")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "headroom-proxy"
    assert data["status"] == "healthy"
    assert data["alive"] is True
    assert data["version"] == __version__
    assert data["uptime_seconds"] >= 0


def test_readyz_reports_core_subsystem_checks(client):
    response = client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["status"] == "healthy"
    assert "config" not in data
    assert data["checks"]["startup"]["status"] == "healthy"
    assert data["checks"]["http_client"]["status"] == "healthy"
    assert data["checks"]["cache"]["status"] == "disabled"
    assert data["checks"]["rate_limiter"]["status"] == "disabled"
    assert data["checks"]["memory"]["status"] == "disabled"
    runtime = data["runtime"]
    assert runtime["anthropic_pre_upstream"]["resolved_concurrency"] == max(
        2, min(8, os.cpu_count() or 4)
    )
    assert runtime["anthropic_pre_upstream"]["source"] == "auto"
    assert runtime["anthropic_pre_upstream"]["acquire_timeout_seconds"] == 15.0
    assert runtime["anthropic_pre_upstream"]["compression_timeout_seconds"] == 30.0
    assert runtime["anthropic_pre_upstream"]["memory_context_timeout_seconds"] == 2.0
    assert runtime["anthropic_pre_upstream"]["codex_ws_gated"] is False
    assert runtime["websocket_sessions"]["active_sessions"] == 0
    assert runtime["websocket_sessions"]["active_relay_tasks"] == 0


def test_health_preserves_backwards_compatible_config_payload(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ready"] is True
    assert data["version"] == __version__
    config = data["config"]
    assert config["backend"] == "anthropic"
    assert config["optimize"] is False
    assert config["cache"] is False
    assert config["rate_limit"] is False
    assert config["memory"] is False
    assert config["learn"] is False
    assert config["code_graph"] is False
    assert config["savings_profile"] is None
    assert config["target_ratio"] is None
    assert config["max_items_after_crush"] == 50
    assert config["smart_crusher_with_compaction"] is None
    assert isinstance(config["pid"], int)
    # `headroom wrap` reads this on proxy reuse to warn on a mode mismatch.
    assert config["mode"] in ("token", "cache")


def test_health_reports_agent_savings_config():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        savings_profile="agent-90",
        target_ratio=0.10,
        compress_user_messages=True,
        compress_system_messages=True,
        protect_recent=2,
        protect_analysis_context=True,
        min_tokens_to_crush=120,
        max_items_after_crush=8,
        smart_crusher_with_compaction=False,
        accuracy_guard="strict",
    )
    app = create_app(config)

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    reported = response.json()["config"]
    assert reported["savings_profile"] == "agent-90"
    assert reported["target_ratio"] == 0.10
    assert reported["compress_user_messages"] is True
    assert reported["compress_system_messages"] is True
    assert reported["protect_recent"] == 2
    assert reported["protect_analysis_context"] is True
    assert reported["min_tokens_to_crush"] == 120
    assert reported["max_items_after_crush"] == 8
    assert reported["smart_crusher_with_compaction"] is False
    assert reported["accuracy_guard"] == "strict"


def test_health_includes_deployment_metadata_when_present(monkeypatch):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_DEPLOYMENT_PROFILE", "default")
    monkeypatch.setenv("HEADROOM_DEPLOYMENT_PRESET", "persistent-service")
    monkeypatch.setenv("HEADROOM_DEPLOYMENT_RUNTIME", "python")
    monkeypatch.setenv("HEADROOM_DEPLOYMENT_SUPERVISOR", "service")
    monkeypatch.setenv("HEADROOM_DEPLOYMENT_SCOPE", "user")

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["deployment"] == {
        "profile": "default",
        "preset": "persistent-service",
        "runtime": "python",
        "supervisor": "service",
        "scope": "user",
    }


def test_health_remains_200_when_proxy_is_not_ready(client):
    client.app.state.ready = False

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ready"] is False


def test_readyz_reports_memory_backend_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=True,
        memory_backend="local",
        memory_db_path=str(tmp_path / "headroom_memory.db"),
        memory_inject_tools=True,
        memory_inject_context=True,
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert data["checks"]["memory"]["status"] == "healthy"
    assert data["checks"]["memory"]["backend"] == "local"
    assert data["checks"]["memory"]["initialized"] is True


def test_readyz_initializes_qdrant_memory_backend(monkeypatch):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    from headroom.memory.backends import direct_mem0

    init_calls: list[str] = []

    class FakeDirectMem0Adapter:
        def __init__(self, config):
            self.config = config

        async def ensure_initialized(self):
            init_calls.append("initialized")

    monkeypatch.setattr(direct_mem0, "DirectMem0Adapter", FakeDirectMem0Adapter)

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=True,
        memory_backend="qdrant-neo4j",
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert init_calls == ["initialized"]
    assert data["checks"]["memory"]["status"] == "healthy"
    assert data["checks"]["memory"]["backend"] == "qdrant-neo4j"
    assert data["checks"]["memory"]["initialized"] is True


def test_shutdown_tolerates_stubbed_memory_handler(monkeypatch):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    with TestClient(app) as client:
        client.app.state.proxy.memory_handler = SimpleNamespace(
            health_status=lambda: {
                "enabled": False,
                "backend": None,
                "initialized": False,
                "native_tool": False,
                "bridge_enabled": False,
            }
        )
        response = client.get("/health")

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Upstream connectivity check tests
# ---------------------------------------------------------------------------


def test_readyz_upstream_check_disabled_by_env_var(monkeypatch):
    """HEADROOM_SKIP_UPSTREAM_CHECK=1 suppresses the probe and reports ready."""
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    with TestClient(app) as test_client:
        response = test_client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    # When the check is skipped the component is reported as "disabled"
    assert data["checks"]["upstream"]["enabled"] is False
    assert data["checks"]["upstream"]["ready"] is True


def test_readyz_upstream_check_failure_returns_503(monkeypatch):
    """A failed upstream probe makes /readyz return HTTP 503."""
    from unittest.mock import AsyncMock, patch

    import httpx

    monkeypatch.delenv("HEADROOM_SKIP_UPSTREAM_CHECK", raising=False)

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    # Patch the proxy's shared http_client.head so the probe uses the same
    # client as real traffic (which also means TLS/CA config is consistent).
    with TestClient(app) as test_client:
        with patch.object(
            test_client.app.state.proxy.http_client,
            "head",
            new=AsyncMock(side_effect=httpx.ConnectError("connection refused (test)")),
        ):
            response = test_client.get("/readyz")

    assert response.status_code == 503
    data = response.json()
    assert data["ready"] is False
    assert data["checks"]["upstream"]["ready"] is False
    assert data["checks"]["upstream"]["error"] == "ConnectError"


@pytest.mark.parametrize(
    ("openai_url", "anthropic_url", "expected_probe"),
    [
        (None, None, "https://api.anthropic.com"),
        ("https://llm.example/v1", None, "https://llm.example"),
        ("https://llm.example/v1", "https://claude.example/v1", "https://claude.example"),
    ],
)
def test_readyz_probes_the_configured_primary_upstream(
    monkeypatch, openai_url, anthropic_url, expected_probe
):
    from unittest.mock import AsyncMock

    monkeypatch.delenv("HEADROOM_SKIP_UPSTREAM_CHECK", raising=False)
    monkeypatch.delenv("ANTHROPIC_TARGET_API_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_TARGET_API_URL", raising=False)
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        openai_api_url=openai_url,
        anthropic_api_url=anthropic_url,
    )
    with TestClient(create_app(config)) as client:
        probe = AsyncMock(return_value=SimpleNamespace(status_code=200))
        monkeypatch.setattr(client.app.state.proxy.http_client, "head", probe)
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["checks"]["upstream"]["url"] == expected_probe
    probe.assert_awaited_once_with(expected_probe, timeout=5.0)


def test_readyz_does_not_expose_upstream_url_details(monkeypatch):
    from unittest.mock import AsyncMock

    import httpx

    monkeypatch.delenv("HEADROOM_SKIP_UPSTREAM_CHECK", raising=False)
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")
    url = "https://test-user:test-password@llm.example/test-route?token=test-query"
    config = ProxyConfig(openai_api_url=url, optimize=False)
    with TestClient(create_app(config)) as client:
        probe = AsyncMock(side_effect=httpx.ConnectError("Could not reach " + url))
        monkeypatch.setattr(client.app.state.proxy.http_client, "head", probe)
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["checks"]["upstream"]["url"] == "https://llm.example/<path>?<redacted>"
    assert response.json()["checks"]["upstream"]["error"] == "ConnectError"
    probe.assert_awaited_once_with(url, timeout=5.0)


def test_health_includes_upstream_check_result(monkeypatch):
    """/health always returns 200 but exposes the upstream check result."""
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "upstream" in data["checks"]
    upstream = data["checks"]["upstream"]
    assert "enabled" in upstream
    assert "ready" in upstream
    assert "status" in upstream


def test_health_reports_enabled_proxy_extensions():
    """`headroom wrap` reads this to carry the shared proxy's extensions over to a
    dedicated proxy it has to start beside it."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        proxy_extensions=["routemegood", "observability"],
    )
    app = create_app(config)

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        reported = client.get("/health").json()["config"]

    assert reported["proxy_extensions"] == ["observability", "routemegood"]

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from headroom.proxy.capabilities import DetachedModeError, build_capability_report
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.delenv("HEADROOM_DETACHED_PROFILE", raising=False)
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.delenv("HEADROOM_TOIN_BACKEND", raising=False)
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)


def _minimal_config(**overrides: Any) -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        subscription_tracking_enabled=False,
        **overrides,
    )


def test_capability_report_marks_stateless_as_detached() -> None:
    report = build_capability_report(_minimal_config(stateless=True))

    payload = report.to_dict()

    assert payload["detached"] is True
    assert payload["profile"] == "lenient"
    assert payload["local_state"]["available"] is False
    features = {item["feature"]: item for item in payload["features"]}
    assert features["proxy_request_handling"]["state"] == "full"
    assert features["compression"]["state"] == "disabled"
    assert features["dashboard_live_data"]["state"] == "degraded"
    assert features["session_aggregation"]["state"] == "disabled"


def test_strict_detached_profile_refuses_enabled_memory_without_state() -> None:
    config = _minimal_config(
        stateless=True,
        detached_profile="strict",
        memory_enabled=True,
    )

    with pytest.raises(DetachedModeError) as exc:
        create_app(config)

    violations = [item["feature"] for item in exc.value.report.to_dict()["strict_violations"]]
    assert violations == ["memory"]


def test_embedded_startup_honors_saved_detached_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom import settings_store

    monkeypatch.setattr(settings_store, "load", lambda: {"detached_profile": "silent"})

    app = create_app()

    assert app.state.capabilities["profile"] == "silent"


def test_capabilities_endpoint_health_stats_and_metrics_share_report() -> None:
    app = create_app(_minimal_config(stateless=True))

    with TestClient(app) as client:
        capabilities = client.get("/capabilities")
        health = client.get("/health")
        stats = client.get("/stats")
        metrics = client.get("/metrics")

    assert capabilities.status_code == 200
    capability_payload = capabilities.json()
    assert capability_payload["detached"] is True
    assert capability_payload["local_state"]["available"] is False
    assert "workspace_dir" not in capability_payload["local_state"]
    assert health.json()["capabilities"] == capability_payload
    assert stats.json()["capabilities"] == capability_payload
    assert 'headroom_feature_enabled{feature="proxy_request_handling"' in metrics.text
    assert 'headroom_feature_enabled{feature="session_aggregation"' in metrics.text


def test_loopback_capabilities_include_operator_workspace_path() -> None:
    app = create_app(_minimal_config(stateless=True))

    with TestClient(
        app,
        client=("127.0.0.1", 50000),
        headers={"host": "127.0.0.1"},
    ) as client:
        capabilities = client.get("/capabilities").json()
        health = client.get("/health").json()
        stats = client.get("/stats").json()

    expected = str(Path(os.environ["HEADROOM_WORKSPACE_DIR"]))
    assert capabilities["local_state"]["workspace_dir"] == expected
    assert health["capabilities"]["local_state"]["workspace_dir"] == expected
    assert stats["capabilities"]["local_state"]["workspace_dir"] == expected


def test_remote_health_never_exposes_workspace_path() -> None:
    app = create_app(_minimal_config(stateless=True, host="0.0.0.0"))

    with TestClient(
        app,
        client=("203.0.113.10", 50000),
        headers={"host": "proxy.example.test"},
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert str(Path(os.environ["HEADROOM_WORKSPACE_DIR"])) not in response.text
    assert "workspace_dir" not in response.json()["capabilities"]["local_state"]


def test_remote_capabilities_redact_workspace_paths_in_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = str(Path(os.environ["HEADROOM_WORKSPACE_DIR"]))
    import headroom.proxy.capabilities as capabilities_module

    monkeypatch.setattr(
        capabilities_module,
        "_probe_local_state",
        lambda _config: (False, f"PermissionError: {workspace}/private-file"),
    )
    app = create_app(_minimal_config(host="0.0.0.0"))

    with TestClient(
        app,
        client=("203.0.113.10", 50000),
        headers={"host": "proxy.example.test"},
    ) as client:
        response = client.get("/capabilities")

    assert workspace not in response.text
    assert "<workspace>/private-file" in response.text


def test_capabilities_do_not_expose_backend_url_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_TOIN_BACKEND", "redis://user:toin-secret@cache.internal:6379/0")
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "https://token:ccr-secret@ccr.internal/store")

    payload = build_capability_report(_minimal_config(stateless=True)).to_dict(
        include_workspace_dir=False
    )
    serialized = str(payload)
    features = {item["feature"]: item for item in payload["features"]}

    assert "toin-secret" not in serialized
    assert "ccr-secret" not in serialized
    assert "cache.internal" not in serialized
    assert "ccr.internal" not in serialized
    assert features["toin_tagging"]["backend"] == "redis"
    assert features["ccr_retrieval"]["backend"] == "https"


def test_stateless_startup_skips_file_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_file_logging() -> None:
        calls.append("file_logging")
        raise AssertionError("stateless startup must not install file logging")

    import headroom.proxy.server as server

    monkeypatch.setattr(server, "_setup_file_logging", fail_file_logging)

    app = create_app(_minimal_config(stateless=True))
    with TestClient(app) as client:
        response = client.get("/capabilities")

    assert response.status_code == 200
    assert calls == []

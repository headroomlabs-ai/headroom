from __future__ import annotations

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
    assert health.json()["capabilities"] == capability_payload
    assert stats.json()["capabilities"] == capability_payload
    assert 'headroom_feature_enabled{feature="proxy_request_handling"' in metrics.text
    assert 'headroom_feature_enabled{feature="session_aggregation"' in metrics.text


def test_stateless_startup_skips_file_logging_and_beacon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_file_logging() -> None:
        calls.append("file_logging")
        raise AssertionError("stateless startup must not install file logging")

    class FailingBeacon:
        def __init__(self, **kwargs: object) -> None:
            calls.append("beacon_init")
            raise AssertionError("stateless startup must not construct telemetry beacon")

    import headroom.proxy.server as server

    monkeypatch.setattr(server, "_setup_file_logging", fail_file_logging)
    monkeypatch.setattr("headroom.telemetry.beacon.TelemetryBeacon", FailingBeacon)

    app = create_app(_minimal_config(stateless=True))
    with TestClient(app) as client:
        response = client.get("/capabilities")

    assert response.status_code == 200
    assert calls == []

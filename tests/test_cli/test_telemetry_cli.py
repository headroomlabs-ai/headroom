from __future__ import annotations

import json

from click.testing import CliRunner

from headroom.cli.main import main


def test_telemetry_list_json_reports_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.delenv("HEADROOM_TELEMETRY", raising=False)
    monkeypatch.delenv("HEADROOM_OTEL_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("HEADROOM_LANGFUSE_ENABLED", raising=False)
    monkeypatch.delenv("HEADROOM_TOIN_BACKEND", raising=False)

    result = CliRunner().invoke(main, ["telemetry", "list", "--json"])

    assert result.exit_code == 0, result.output
    surfaces = {item["surface"]: item for item in json.loads(result.output)}
    assert surfaces["anonymous_beacon"]["status"] == "on"
    assert surfaces["anonymous_beacon"]["leaves_host_by_default"] is True
    assert surfaces["prometheus"]["status"] == "on"
    assert surfaces["prometheus"]["includes_prompt_content"] is False
    assert surfaces["savings_tracker"]["observe"].endswith("proxy_savings.json")
    assert surfaces["active_sessions"]["observe"].endswith("sessions")


def test_telemetry_list_json_reflects_env_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
    monkeypatch.setenv("HEADROOM_OTEL_METRICS_ENABLED", "true")
    monkeypatch.setenv("HEADROOM_OTEL_METRICS_ENDPOINT", "http://otel.local/v1/metrics")
    monkeypatch.setenv("HEADROOM_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("HEADROOM_TOIN_BACKEND", "none")
    monkeypatch.setenv("HEADROOM_CLUSTER_ENABLED", "true")
    monkeypatch.setenv("HEADROOM_CLUSTER_ID", "team-gamma")
    monkeypatch.setenv("HEADROOM_CLUSTER_DIR", str(tmp_path / "cluster"))

    result = CliRunner().invoke(main, ["telemetry", "list", "--json"])

    assert result.exit_code == 0, result.output
    surfaces = {item["surface"]: item for item in json.loads(result.output)}
    assert surfaces["anonymous_beacon"]["status"] == "off"
    assert surfaces["otel_metrics"]["status"] == "on"
    assert surfaces["otel_metrics"]["export"] == "http://otel.local/v1/metrics"
    assert surfaces["langfuse_tracing"]["status"] == "on"
    assert surfaces["toin"]["status"] == "off"
    assert surfaces["active_sessions"]["leaves_host_by_default"] is True
    assert "team-gamma" in surfaces["active_sessions"]["export"]


def test_telemetry_list_table_output() -> None:
    result = CliRunner().invoke(main, ["telemetry", "list"])

    assert result.exit_code == 0, result.output
    assert "Headroom Telemetry Surfaces" in result.output
    assert "anonymous_beacon" in result.output
    assert "prometheus" in result.output

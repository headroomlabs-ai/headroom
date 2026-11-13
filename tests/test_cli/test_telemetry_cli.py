from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


def test_telemetry_list_json_reports_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("HEADROOM_TELEMETRY", raising=False)
    monkeypatch.delenv("HEADROOM_BEACON", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    monkeypatch.delenv("HEADROOM_OTEL_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("HEADROOM_LANGFUSE_ENABLED", raising=False)
    monkeypatch.delenv("HEADROOM_TOIN_BACKEND", raising=False)

    result = CliRunner().invoke(main, ["telemetry", "list", "--json"])

    assert result.exit_code == 0, result.output
    surfaces = {item["surface"]: item for item in json.loads(result.output)}
    assert surfaces["anonymous_beacon"]["status"] == "on"
    assert surfaces["anonymous_beacon"]["leaves_host_by_default"] is True
    assert surfaces["anonymous_beacon"]["export"].startswith("https://")
    assert surfaces["local_telemetry"]["status"] == "off"
    assert surfaces["local_telemetry"]["leaves_host_by_default"] is False
    assert surfaces["prometheus"]["status"] == "on"
    assert surfaces["prometheus"]["includes_prompt_content"] is False
    assert surfaces["savings_tracker"]["observe"].endswith("proxy_savings.json")
    assert surfaces["active_sessions"]["observe"].endswith("sessions")
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


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
    assert surfaces["toin"]["status"] == "on"
    assert surfaces["active_sessions"]["leaves_host_by_default"] is True
    assert "team-gamma" in surfaces["active_sessions"]["export"]


def test_telemetry_list_table_output() -> None:
    result = CliRunner().invoke(main, ["telemetry", "list"])

    assert result.exit_code == 0, result.output
    assert "Headroom Telemetry Surfaces" in result.output
    assert "anonymous_beacon" in result.output
    assert "prometheus" in result.output


@pytest.mark.parametrize(
    ("beacon", "local", "upload_status", "local_status"),
    [("off", "on", "off", "on"), ("on", "off", "on", "off")],
)
def test_telemetry_list_separates_upload_and_local_switches(
    monkeypatch, beacon, local, upload_status, local_status
):
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.setenv("HEADROOM_BEACON", beacon)
    monkeypatch.setenv("HEADROOM_TELEMETRY", local)
    result = CliRunner().invoke(main, ["telemetry", "list", "--json"])
    assert result.exit_code == 0, result.output
    surfaces = {item["surface"]: item for item in json.loads(result.output)}
    assert surfaces["anonymous_beacon"]["status"] == upload_status
    assert surfaces["local_telemetry"]["status"] == local_status


def test_telemetry_list_reflects_in_memory_toin_and_prompt_logging(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("HEADROOM_TOIN_BACKEND", "none")
    monkeypatch.setenv("HEADROOM_LOG_MESSAGES", "true")

    result = CliRunner().invoke(main, ["telemetry", "list", "--json"])

    assert result.exit_code == 0, result.output
    surfaces = {item["surface"]: item for item in json.loads(result.output)}
    assert surfaces["toin"]["status"] == "on"
    assert "In-memory" in surfaces["toin"]["retention"]
    assert surfaces["proxy_log"]["includes_prompt_content"] is True

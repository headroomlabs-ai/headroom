"""Tests for the durable savings event ledger and the `headroom savings` CLI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from headroom import savings_ledger as L

UTC = timezone.utc


def _events_env(monkeypatch, tmp_path):
    path = tmp_path / "savings_events.jsonl"
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(path))
    return path


# --------------------------------------------------------------------------- #
# core ledger
# --------------------------------------------------------------------------- #


def test_unknown_model_uses_blended_fallback(monkeypatch, tmp_path):
    _events_env(monkeypatch, tmp_path)
    assert L.record_savings_event(tokens_before=1000, tokens_after=400, model=None, client="c")
    report = L.aggregate_savings()
    assert report.lifetime["tokens_saved"] == 600
    expected = round(600 * L.DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN, 6)
    assert report.lifetime["cost_usd"] == pytest.approx(expected)
    assert any(row["model"] == "unknown" for row in report.by_model)


def test_estimate_cost_unknown_short_circuits_to_fallback():
    assert L.estimate_cost_usd("unknown", 1000, fallback_rate=1e-6) == pytest.approx(0.001)
    assert L.estimate_cost_usd(L.UNKNOWN, 0) == 0.0


def test_free_model_is_not_billed_at_fallback(monkeypatch):
    """A known but 0-priced (free) model must cost $0, not the $3/M fallback.

    _estimate_compression_savings_usd returns a legitimate 0.0 for free models;
    the ledger must trust that rather than re-billing it at the blended rate.
    """
    monkeypatch.setattr(L, "_estimate_compression_savings_usd", lambda model, tokens: 0.0)

    assert L.estimate_cost_usd("free-local-model", 1_000_000) == 0.0


def test_priced_model_uses_litellm_estimate(monkeypatch):
    monkeypatch.setattr(L, "_estimate_compression_savings_usd", lambda model, tokens: tokens * 2e-6)

    assert L.estimate_cost_usd("some-model", 1_000_000) == pytest.approx(2.0)


def test_explicit_cost_is_honored(monkeypatch, tmp_path):
    _events_env(monkeypatch, tmp_path)
    L.record_savings_event(tokens_before=100, tokens_after=10, model="x", client="c", cost_usd=1.25)
    assert L.aggregate_savings().lifetime["cost_usd"] == pytest.approx(1.25)


def test_zero_or_negative_savings_not_recorded(monkeypatch, tmp_path):
    path = _events_env(monkeypatch, tmp_path)
    assert L.record_savings_event(tokens_before=100, tokens_after=100) is False
    assert L.record_savings_event(tokens_before=50, tokens_after=80) is False
    assert not path.exists() or path.read_text().strip() == ""
    assert L.aggregate_savings().lifetime["calls"] == 0


def test_breakdowns_aggregate_by_dimension(monkeypatch, tmp_path):
    _events_env(monkeypatch, tmp_path)
    L.record_savings_event(tokens_before=1000, tokens_after=300, model=None, client="claude-code")
    L.record_savings_event(tokens_before=500, tokens_after=200, model=None, client="claude-code")
    L.record_savings_event(
        tokens_before=2000, tokens_after=600, model="gpt", client="proxy", cost_usd=0.5
    )
    report = L.aggregate_savings()

    clients = {row["client"]: row for row in report.by_client}
    assert clients["claude-code"]["calls"] == 2
    assert clients["proxy"]["tokens_saved"] == 1400


def test_windows_today_week_last30(monkeypatch, tmp_path):
    _events_env(monkeypatch, tmp_path)
    now = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    L.record_savings_event(
        tokens_before=1000, tokens_after=500, model=None, client="c", timestamp=now
    )
    L.record_savings_event(
        tokens_before=1000,
        tokens_after=600,
        model=None,
        client="c",
        timestamp=now - timedelta(days=3),
    )
    L.record_savings_event(
        tokens_before=1000,
        tokens_after=700,
        model=None,
        client="c",
        timestamp=now - timedelta(days=20),
    )
    report = L.aggregate_savings(now=now)
    assert report.windows["today"]["tokens_saved"] == 500
    assert report.windows["last_7_days"]["tokens_saved"] == 500 + 400
    assert report.windows["last_30_days"]["tokens_saved"] == 500 + 400 + 300
    assert report.windows["last_30_days"]["calls"] == 3
    # 500 saved out of 1000 before today
    assert report.windows["today"]["savings_percent"] == pytest.approx(50.0)


def test_new_input_basis_pairs_compression_only_with_new_input(monkeypatch, tmp_path):
    _events_env(monkeypatch, tmp_path)
    # 400 saved, of which 100 is tool-schema deferral; 900 tokens newly entered
    # context. Compression-only 300 / (900 + 300) = 25%, same as /stats.
    L.record_savings_event(
        tokens_before=10_000,
        tokens_after=9_600,
        model=None,
        client="claude-code",
        new_input_tokens=900,
        deferred_tokens=100,
    )
    # An event without a cache breakdown (MCP tool, Bedrock) keeps writing the
    # old line and must not lend its savings to the new-input ratio.
    L.record_savings_event(tokens_before=1000, tokens_after=500, model=None, client="mcp")
    report = L.aggregate_savings()
    window = report.windows["today"]
    assert window["tokens_saved"] == 900
    assert window["new_input_tokens"] == 900
    assert window["new_input_savings_percent"] == pytest.approx(25.0)
    clients = {row["client"]: row for row in report.by_client}
    assert clients["mcp"]["new_input_tokens"] == 0
    assert clients["mcp"]["new_input_savings_percent"] == 0.0
    # The whole-wire ratio is untouched by the new field.
    assert window["savings_percent"] == pytest.approx(900 / 11_000 * 100, abs=0.1)


def test_savings_cli_prints_new_input_line_only_with_cache_data(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from headroom.cli.savings import savings

    _events_env(monkeypatch, tmp_path)
    L.record_savings_event(tokens_before=1000, tokens_after=500, model=None, client="mcp")
    out = CliRunner().invoke(savings, []).output
    assert "of new input" not in out
    L.record_savings_event(
        tokens_before=10_000, tokens_after=9_600, model=None, client="c", new_input_tokens=900
    )
    out = CliRunner().invoke(savings, []).output
    assert "of new input 30.8%" in out
    assert "newly entered context" in out


def test_retention_hard_capped_at_30_days(monkeypatch, tmp_path):
    _events_env(monkeypatch, tmp_path)
    now = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    L.record_savings_event(
        tokens_before=1000, tokens_after=500, model=None, client="c", timestamp=now
    )
    # 60 days old: within the requested 365-day window but past the 30-day cap.
    L.record_savings_event(
        tokens_before=1000,
        tokens_after=500,
        model=None,
        client="c",
        timestamp=now - timedelta(days=60),
    )
    # Caller asks for 365 days, but retention is hard-capped at 30.
    report = L.aggregate_savings(now=now, retention_days=365)
    assert report.lifetime["calls"] == 1
    assert report.windows["last_30_days"]["calls"] == 1


def test_appends_do_not_clobber_and_survive_restart(monkeypatch, tmp_path):
    path = _events_env(monkeypatch, tmp_path)
    for _ in range(5):
        L.record_savings_event(tokens_before=100, tokens_after=10, model=None, client="c")
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 5
    # aggregate_savings holds no in-memory state — it reads purely from disk,
    # so this also proves durability across a process restart.
    assert L.aggregate_savings().lifetime["calls"] == 5
    assert L.aggregate_savings().lifetime["tokens_saved"] == 5 * 90


def test_corrupt_lines_are_skipped(monkeypatch, tmp_path):
    path = _events_env(monkeypatch, tmp_path)
    L.record_savings_event(tokens_before=1000, tokens_after=400, model=None, client="c")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("not json\n\n")
    assert L.aggregate_savings().lifetime["calls"] == 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_reset_deletes_ledger(monkeypatch, tmp_path):
    pytest.importorskip("click")
    from click.testing import CliRunner

    from headroom.cli.savings import savings

    path = _events_env(monkeypatch, tmp_path)
    L.record_savings_event(tokens_before=1000, tokens_after=300, model=None, client="claude-code")
    assert path.exists()

    result = CliRunner().invoke(savings, ["--reset"])
    assert result.exit_code == 0
    assert "reset" in result.output.lower()
    assert not path.exists()

    # second reset on missing file is a no-op
    result2 = CliRunner().invoke(savings, ["--reset"])
    assert result2.exit_code == 0
    assert "Nothing to reset" in result2.output


def test_cli_empty_state(monkeypatch, tmp_path):
    pytest.importorskip("click")
    from click.testing import CliRunner

    from headroom.cli.savings import savings

    _events_env(monkeypatch, tmp_path)
    result = CliRunner().invoke(savings, [])
    assert result.exit_code == 0
    assert "No savings recorded yet." in result.output


def test_cli_renders_sections_and_json(monkeypatch, tmp_path):
    pytest.importorskip("click")
    from click.testing import CliRunner

    from headroom.cli.savings import savings

    _events_env(monkeypatch, tmp_path)
    L.record_savings_event(tokens_before=1000, tokens_after=300, model=None, client="claude-code")
    runner = CliRunner()

    result = runner.invoke(savings, [])
    assert result.exit_code == 0
    # No redundant top-line headline; the windows lead the output.
    assert "cost avoided" not in result.output
    assert "Today" in result.output and "Last 30 days" in result.output
    assert "Savings by client" in result.output and "claude-code" in result.output
    assert "Per-repo totals" not in result.output

    result_json = runner.invoke(savings, ["--json"])
    assert result_json.exit_code == 0
    payload = json.loads(result_json.output)
    assert payload["lifetime"]["tokens_saved"] == 700
    assert payload["windows"]["last_30_days"]["calls"] == 1
    assert "by_repo" not in payload


# --------------------------------------------------------------------------- #
# MCP tool path
# --------------------------------------------------------------------------- #


def test_mcp_compress_records_durable_event(monkeypatch, tmp_path):
    pytest.importorskip("mcp", reason="MCP SDK required")
    from headroom.ccr import mcp_server

    _events_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HEADROOM_MCP_CLIENT", "claude-code")

    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    server._record_savings({"original_tokens": 1000, "compressed_tokens": 250})

    report = L.aggregate_savings()
    assert report.lifetime["tokens_saved"] == 750
    assert {row["client"] for row in report.by_client} == {"claude-code"}


def test_mcp_record_savings_ignores_noop(monkeypatch, tmp_path):
    pytest.importorskip("mcp", reason="MCP SDK required")
    from headroom.ccr import mcp_server

    _events_env(monkeypatch, tmp_path)
    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    server._record_savings({"original_tokens": 500, "compressed_tokens": 500})
    assert L.aggregate_savings().lifetime["calls"] == 0


# --------------------------------------------------------------------------- #
# proxy path
# --------------------------------------------------------------------------- #


def test_proxy_record_request_appends_ledger_event(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import asyncio

    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "proxy_savings.json"))
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "savings_events.jsonl"))
    monkeypatch.setattr(
        "headroom.proxy.server.CostTracker._get_cache_prices",
        lambda self, model: (0.001, 0.0015, 0.002),
    )

    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        # identified harness -> recorded as that client
        asyncio.run(
            proxy.metrics.record_request(
                provider="openai",
                model="gpt-4o",
                input_tokens=120,
                output_tokens=24,
                tokens_saved=40,
                latency_ms=15.0,
                client="claude-code",
            )
        )
        # unidentified harness -> falls back to "proxy"
        asyncio.run(
            proxy.metrics.record_request(
                provider="openai",
                model="gpt-4o",
                input_tokens=80,
                output_tokens=10,
                tokens_saved=20,
                latency_ms=15.0,
            )
        )

    report = L.aggregate_savings()
    assert report.lifetime["tokens_saved"] == 60
    clients = {row["client"] for row in report.by_client}
    assert "claude-code" in clients
    assert "proxy" in clients
    assert any(row["model"] == "gpt-4o" for row in report.by_model)


# --------------------------------------------------------------------------- #
# retention cap + stale-schema reset
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_days", ["31", "60", "365"])
def test_cli_days_flag_capped_at_30(monkeypatch, tmp_path, bad_days):
    pytest.importorskip("click")
    from click.testing import CliRunner

    from headroom.cli.savings import savings

    _events_env(monkeypatch, tmp_path)
    result = CliRunner().invoke(savings, ["--days", bad_days])
    assert result.exit_code != 0
    assert "30" in result.output  # IntRange error mentions the allowed max


def test_zero_saving_request_keeps_its_place_in_the_new_input_denominator(monkeypatch, tmp_path):
    """100 saved on 100 new input, then 0 saved on 10,000 new input. The
    new-input basis is 100 / (10,100 + 100), under 1 percent, not the 50
    percent that only counting requests that saved would report. The
    saved-event figures do not see the second request at all."""
    _events_env(monkeypatch, tmp_path)
    assert L.record_savings_event(
        tokens_before=200, tokens_after=100, model=None, client="claude-code", new_input_tokens=100
    )
    assert L.record_savings_event(
        tokens_before=10_000,
        tokens_after=10_000,
        model=None,
        client="claude-code",
        new_input_tokens=10_000,
    )
    # Without a new-input figure a zero-saving request is still not an event.
    assert not L.record_savings_event(
        tokens_before=500, tokens_after=500, model=None, client="claude-code"
    )
    window = L.aggregate_savings().windows["today"]
    assert window["tokens_saved"] == 100
    assert window["new_input_tokens"] == 10_100
    assert window["new_input_savings_percent"] == pytest.approx(1.0, abs=0.05)
    assert window["savings_percent"] == pytest.approx(50.0)

"""Unverifiable historical prices must not enter the current priced headline."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from io import StringIO

import pytest

import headroom.proxy.savings_tracker as savings_tracker_module
from headroom.pricing.counterfactual import BASIS_CATALOG, BASIS_LIST
from headroom.proxy.savings_tracker import SavingsTracker


def _legacy_state():
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 5,
        "lifetime": {
            "requests": 1,
            "tokens_saved": 1000,
            "compression_savings_usd": 100.0,
            "total_input_tokens": 2000,
            "total_input_cost_usd": 0.5,
        },
        "history": [
            {
                "timestamp": timestamp,
                "total_tokens_saved": 1000,
                "compression_savings_usd": 100.0,
                "total_input_tokens": 2000,
                "total_input_cost_usd": 0.5,
            }
        ],
        "projects": {
            "sample": {
                "requests": 1,
                "tokens_saved": 1000,
                "compression_savings_usd": 100.0,
            }
        },
        "by_model": {
            "sample-model": {
                "requests": 1,
                "tokens_saved": 1000,
                "compression_savings_usd": 100.0,
            }
        },
    }


@pytest.mark.parametrize("schema", [5, 6])
def test_legacy_prices_are_excluded_without_losing_usage_or_audit_values(tmp_path, schema):
    path = tmp_path / "savings.json"
    state = _legacy_state()
    if schema == 6:
        state["schema_version"] = 6
        state["lifetime"]["compression_savings_list_usd"] = 100.0
        state["lifetime"]["savings_basis"] = BASIS_LIST
    path.write_text(json.dumps(state), encoding="utf-8", newline="\n")
    result = SavingsTracker(path=str(path)).history_response(history_mode="full")

    assert result["lifetime"]["compression_savings_usd"] == 0.0
    assert result["lifetime"]["legacy_compression_savings_usd"] == 100.0
    assert result["lifetime"]["requests"] == 1
    assert result["lifetime"]["tokens_saved"] == 1000
    assert result["lifetime"]["total_input_cost_usd"] == 0.5
    assert result["history"][0]["compression_savings_usd"] == 0.0
    assert result["history"][0]["legacy_compression_savings_usd"] == 100.0
    assert result["projects"]["sample"]["compression_savings_usd"] == 0.0
    assert result["projects"]["sample"]["legacy_compression_savings_usd"] == 100.0
    assert result["by_model"]["sample-model"]["compression_savings_usd"] == 0.0
    assert result["by_model"]["sample-model"]["legacy_compression_savings_usd"] == 100.0
    assert result["series"]["daily"][0]["compression_savings_usd_delta"] == 0.0


@pytest.mark.parametrize("quality", [BASIS_CATALOG, BASIS_LIST])
def test_new_priced_requests_do_not_launder_legacy_dollars_after_restart(tmp_path, quality):
    path = tmp_path / "savings.json"
    path.write_text(json.dumps(_legacy_state()), encoding="utf-8", newline="\n")
    tracker = SavingsTracker(path=str(path))
    tracker.record_request(
        model="sample-model",
        project="sample",
        input_tokens=1000,
        tokens_saved=100,
        tool_search_saved=10,
        total_input_cost_usd=0.55,
        estimated_savings_usd={
            "compression": 0.01,
            "tool_schema": 0.002,
            "compression_list": 0.1,
            "tool_schema_list": 0.02,
            "basis": quality,
        },
    )
    restarted = SavingsTracker(path=str(path))
    result = restarted.snapshot()

    assert result["lifetime"]["compression_savings_usd"] == pytest.approx(0.012)
    assert result["lifetime"]["tool_schema_savings_usd"] == pytest.approx(0.002)
    assert result["lifetime"]["legacy_compression_savings_usd"] == 100.0
    assert result["lifetime"]["requests"] == 2
    assert result["lifetime"]["tokens_saved"] == 1100
    assert result["lifetime"]["total_input_cost_usd"] == pytest.approx(0.55)
    assert result["projects"]["sample"]["compression_savings_usd"] == pytest.approx(0.012)
    assert result["by_model"]["sample-model"]["compression_savings_usd"] == pytest.approx(0.012)
    assert result["history"][-1]["compression_savings_usd"] == pytest.approx(0.012)
    assert result["history"][0]["legacy_compression_savings_usd"] == 100.0
    daily = list(csv.DictReader(StringIO(restarted.export_csv(series="daily"))))
    assert sum(float(row["compression_savings_usd_delta"]) for row in daily) == pytest.approx(0.012)
    assert float(daily[-1]["legacy_compression_savings_usd"]) == 100.0


@pytest.mark.parametrize("dollars", [100.0, -100.0])
def test_history_only_legacy_dollars_cannot_reseed_the_priced_lifetime(tmp_path, dollars):
    state = _legacy_state()
    state.pop("lifetime")
    state["history"][0]["compression_savings_usd"] = dollars
    state["history"][0]["tool_schema_savings_usd"] = dollars / 5
    path = tmp_path / "savings.json"
    path.write_text(json.dumps(state), encoding="utf-8", newline="\n")
    tracker = SavingsTracker(path=str(path))
    tracker.flush()
    result = SavingsTracker(path=str(path)).snapshot()["lifetime"]

    assert result["compression_savings_usd"] == 0.0
    assert result["legacy_compression_savings_usd"] == dollars
    assert result["legacy_tool_schema_savings_usd"] == dollars / 5
    assert result["tokens_saved"] == 1000


def test_signed_lifetime_audit_survives_older_checkpoint_flush_and_restart(tmp_path):
    state = _legacy_state()
    state["lifetime"]["compression_savings_usd"] = -100.0
    state["lifetime"]["tool_schema_savings_usd"] = -20.0
    state["history"][0]["compression_savings_usd"] = -50.0
    state["history"][0]["tool_schema_savings_usd"] = -10.0
    path = tmp_path / "savings.json"
    path.write_text(json.dumps(state), encoding="utf-8", newline="\n")

    tracker = SavingsTracker(path=str(path))
    loaded = tracker.snapshot()["lifetime"]
    assert loaded["legacy_compression_savings_usd"] == -100.0
    assert loaded["legacy_tool_schema_savings_usd"] == -20.0
    assert loaded["compression_savings_usd"] == 0.0
    tracker.flush()

    restarted = SavingsTracker(path=str(path)).history_response(history_mode="full")
    assert restarted["lifetime"]["legacy_compression_savings_usd"] == -100.0
    assert restarted["lifetime"]["legacy_tool_schema_savings_usd"] == -20.0
    assert restarted["history"][0]["legacy_compression_savings_usd"] == -50.0
    assert restarted["history"][0]["legacy_tool_schema_savings_usd"] == -10.0
    assert restarted["lifetime"]["tokens_saved"] == 1000


def test_priced_lifetime_recovers_missing_audits_independently_after_restart(tmp_path):
    path = tmp_path / "savings.json"
    tracker = SavingsTracker(path=str(path))
    tracker.record_request(
        model="sample-model",
        input_tokens=1000,
        tokens_saved=100,
        tool_search_saved=10,
        estimated_savings_usd={
            "compression": 0.01,
            "tool_schema": 0.002,
            "basis": BASIS_CATALOG,
        },
    )
    tracker.flush()
    state = json.loads(path.read_text(encoding="utf-8"))
    state["lifetime"].pop("legacy_compression_savings_usd")
    state["lifetime"]["legacy_tool_schema_savings_usd"] = -20.0
    state["history"][-1]["legacy_compression_savings_usd"] = 100.0
    state["history"][-1]["legacy_tool_schema_savings_usd"] = -10.0
    path.write_text(json.dumps(state), encoding="utf-8", newline="\n")

    partial = SavingsTracker(path=str(path))
    partial.flush()
    recovered = SavingsTracker(path=str(path)).snapshot()["lifetime"]
    assert recovered["legacy_compression_savings_usd"] == 100.0
    assert recovered["legacy_tool_schema_savings_usd"] == -20.0
    assert recovered["compression_savings_usd"] == pytest.approx(0.012)

    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["lifetime"].pop("legacy_tool_schema_savings_usd")
    path.write_text(json.dumps(saved), encoding="utf-8", newline="\n")
    missing_tool = SavingsTracker(path=str(path))
    missing_tool.flush()
    recovered_tool = SavingsTracker(path=str(path)).snapshot()["lifetime"]
    assert recovered_tool["legacy_compression_savings_usd"] == 100.0
    assert recovered_tool["legacy_tool_schema_savings_usd"] == -10.0
    assert recovered_tool["compression_savings_usd"] == pytest.approx(0.012)


def test_durable_lifetime_cost_does_not_mix_unversioned_and_new_prices(tmp_path):
    state = _legacy_state()
    state["lifetime_metrics"] = {
        "requests": {"total": 1},
        "cost": {
            "input_usd": 0.5,
            "compression_savings_usd": 100.0,
            "compression_savings_list_usd": 100.0,
            "savings_basis": BASIS_LIST,
        },
    }
    path = tmp_path / "savings.json"
    path.write_text(json.dumps(state), encoding="utf-8", newline="\n")
    tracker = SavingsTracker(path=str(path))
    tracker.record_lifetime_request(
        provider="openai",
        stack="codex",
        model="sample-model",
        input_tokens=1000,
        tokens_saved=100,
        input_usd=0.05,
        compression_savings_usd=0.01,
        compression_savings_list_usd=0.1,
        savings_basis=BASIS_CATALOG,
    )
    result = SavingsTracker(path=str(path)).lifetime_response()

    assert result["requests"]["total"] == 2
    assert result["cost"]["input_usd"] == pytest.approx(0.55)
    assert result["cost"]["compression_savings_usd"] == pytest.approx(0.01)
    assert result["cost"]["legacy_compression_savings_usd"] == 100.0


@pytest.mark.parametrize("legacy_dollars", [None, (100.0, 20.0), (-100.0, -20.0)])
@pytest.mark.parametrize("save_flush_every", [1, 3])
def test_compression_checkpoints_preserve_priced_buckets_and_audit_on_reload(
    tmp_path, monkeypatch, legacy_dollars, save_flush_every
):
    # Exercise the real no-LiteLLM fallback rather than depend on live catalogs.
    monkeypatch.setattr(savings_tracker_module, "LITELLM_AVAILABLE", False)
    hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hour -= timedelta(hours=2)
    path = tmp_path / "savings.json"
    legacy_compression, legacy_tool = legacy_dollars or (0.0, 0.0)
    if legacy_dollars is not None:
        state = _legacy_state()
        state["schema_version"] = 6
        state["lifetime"]["compression_savings_usd"] = legacy_compression
        state["lifetime"]["tool_schema_savings_usd"] = legacy_tool
        state["history"][0].update(
            timestamp=(hour - timedelta(hours=1)).isoformat(),
            compression_savings_usd=legacy_compression,
            tool_schema_savings_usd=legacy_tool,
        )
        path.write_text(json.dumps(state), encoding="utf-8", newline="\n")

    tracker = SavingsTracker(path=str(path), save_flush_every=save_flush_every)
    assert tracker.record_compression_savings(
        model="checkpoint-test-model", tokens_saved=1000, timestamp=hour
    )
    before = tracker.history_response(history_mode="full")
    checkpoint_dollars = before["history"][-1]["compression_savings_usd"]
    assert checkpoint_dollars > 0
    tracker.flush()

    restarted = SavingsTracker(path=str(path), save_flush_every=save_flush_every)
    after = restarted.history_response(history_mode="full")
    assert after["history"][-1]["compression_savings_usd"] == checkpoint_dollars
    assert after["lifetime"]["compression_savings_usd"] == checkpoint_dollars
    assert after["history"][-1]["legacy_compression_savings_usd"] == legacy_compression
    assert after["history"][-1]["legacy_tool_schema_savings_usd"] == legacy_tool

    restarted.record_request(
        model="checkpoint-test-model",
        input_tokens=1000,
        tokens_saved=200,
        tool_search_saved=20,
        estimated_savings_usd={"compression": 0.007, "tool_schema": 0.002},
        timestamp=hour + timedelta(hours=1),
    )
    restarted.flush()
    final_tracker = SavingsTracker(path=str(path))
    final = final_tracker.history_response(history_mode="full")
    assert final["lifetime"]["compression_savings_usd"] == pytest.approx(checkpoint_dollars + 0.009)
    assert final["history"][-1]["legacy_compression_savings_usd"] == legacy_compression
    assert final["history"][-1]["legacy_tool_schema_savings_usd"] == legacy_tool
    # The later request must not absorb savings from the earlier checkpoint.
    expected = pytest.approx([checkpoint_dollars, 0.009])
    assert [
        row["compression_savings_usd_delta"] for row in final["series"]["hourly"][-2:]
    ] == expected
    rows = list(csv.DictReader(StringIO(final_tracker.export_csv(series="hourly"))))
    assert [float(row["compression_savings_usd_delta"]) for row in rows[-2:]] == expected
    assert float(rows[-2]["legacy_compression_savings_usd"]) == legacy_compression
    assert float(rows[-2]["legacy_tool_schema_savings_usd"]) == legacy_tool


def test_current_estimate_quality_ignores_legacy_and_weakens_on_list_fallback(tmp_path):
    state = _legacy_state()
    state["schema_version"] = 6
    state["lifetime"].update(
        compression_savings_list_usd=100.0,
        savings_basis=BASIS_LIST,
    )
    path = tmp_path / "savings.json"
    path.write_text(json.dumps(state), encoding="utf-8", newline="\n")
    tracker = SavingsTracker(path=str(path))
    tracker.record_request(
        model="sample-model",
        input_tokens=1000,
        tokens_saved=100,
        estimated_savings_usd={"compression": 0.01, "basis": BASIS_CATALOG},
    )
    restarted = SavingsTracker(path=str(path))
    first = restarted.snapshot()["lifetime"]
    assert first["compression_savings_usd"] == pytest.approx(0.01)
    assert first["savings_basis"] == BASIS_CATALOG

    restarted.record_request(
        model="sample-model",
        input_tokens=1000,
        tokens_saved=100,
        estimated_savings_usd={"compression": 0.02, "basis": BASIS_LIST},
    )
    final = SavingsTracker(path=str(path)).snapshot()["lifetime"]
    assert final["compression_savings_usd"] == pytest.approx(0.03)
    assert final["legacy_compression_savings_usd"] == 100.0
    assert final["savings_basis"] == BASIS_LIST

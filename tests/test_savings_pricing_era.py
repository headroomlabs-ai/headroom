"""Unverifiable historical prices must not enter the current priced headline."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import StringIO

import pytest

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
    path = tmp_path / "savings.json"
    path.write_text(json.dumps(state), encoding="utf-8", newline="\n")
    result = SavingsTracker(path=str(path)).snapshot()["lifetime"]

    assert result["compression_savings_usd"] == 0.0
    assert result["legacy_compression_savings_usd"] == dollars
    assert result["tokens_saved"] == 1000


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

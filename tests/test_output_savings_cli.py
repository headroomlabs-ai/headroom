"""Smoke tests for the output-savings CLI and the outcome→ledger wiring."""

from __future__ import annotations

import json

from click.testing import CliRunner

from headroom.cli.main import main
from headroom.proxy.output_savings import SavingsRecorder, stratum_label


def test_output_savings_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    result = CliRunner().invoke(main, ["output-savings"])
    assert result.exit_code == 0
    assert "No output-savings data" in result.output


def test_output_savings_reports_estimate(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    # Seed a baseline + treatment observations directly via the ledger.
    from headroom.proxy.output_savings import SavingsLedger

    ledger = SavingsLedger()
    for _ in range(50):
        ledger.baseline.observe("opus|new_user_ask|s|tools", 1000)
    for _ in range(30):
        ledger.record("treatment", "opus|new_user_ask|s|tools", 700)
    ledger.save(tmp_path / "output_savings.json")

    result = CliRunner().invoke(main, ["output-savings"])
    assert result.exit_code == 0
    assert "ESTIMATED" in result.output
    assert "Reduction:" in result.output
    assert "30.0%" in result.output


def test_recorder_round_trips_via_labels(tmp_path):
    path = tmp_path / "savings.json"
    rec = SavingsRecorder(path, flush_every=1)
    # Baseline so the estimate has something to compare against.
    rec._ledger.baseline.observe("opus|new_user_ask|s|tools", 1000)
    labels = ["compress:smartcrush", stratum_label("treatment", "opus|new_user_ask|s|tools")]
    assert rec.record_from_labels(labels, output_tokens=600) is True
    assert rec.record_from_labels(["no-shaper-label"], output_tokens=999) is False
    est = rec.estimate()
    assert est.n_requests == 1
    assert est.tokens_saved == 400  # 1000 - 600

    # Persisted to disk.
    data = json.loads(path.read_text())
    assert "treatment" in data

"""Per-bucket output-shaping savings in the /stats-history rollup.

Covers the feature that lets a downstream dashboard stack output-shaping
savings as a distinct daily segment: SavingsTracker.record_request accepts a
per-request output_tokens_saved, accumulates it into each time bucket as
output_tokens_saved_delta / output_savings_usd_delta, and the read-only
SavingsRecorder.estimate_request_savings supplies that per-request number.
"""

from __future__ import annotations

from headroom.proxy.output_savings import (
    SavingsRecorder,
    stratum_key,
    stratum_label,
)
from headroom.proxy.savings_tracker import SavingsTracker


def test_record_request_buckets_output_shaping_savings(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "s.json"))

    # Request with both compression and output-shaping savings.
    tracker.record_request(
        model="claude-opus-4-8",
        input_tokens=1000,
        tokens_saved=100,
        output_tokens_saved=5000,
        timestamp="2026-03-27T09:00:00Z",
    )
    # Output-shaping-ONLY request (no compression) must still checkpoint, else
    # its output savings would be dropped from the rollup.
    tracker.record_request(
        model="claude-opus-4-8",
        input_tokens=1000,
        tokens_saved=0,
        output_tokens_saved=3000,
        timestamp="2026-03-27T09:30:00Z",
    )

    daily = tracker.history_response()["series"]["daily"]
    assert len(daily) == 1
    assert daily[0]["output_tokens_saved_delta"] == 8000
    assert daily[0]["output_savings_usd_delta"] > 0.0
    # Compression axis stays independent.
    assert daily[0]["tokens_saved"] == 100


def test_record_request_without_output_savings_is_backward_compatible(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "s.json"))
    tracker.record_request(
        model="gpt-4o",
        input_tokens=8192,
        tokens_saved=4096,
        timestamp="2026-03-27T09:00:00Z",
    )
    daily = tracker.history_response()["series"]["daily"]
    assert daily[0]["output_tokens_saved_delta"] == 0
    assert daily[0]["output_savings_usd_delta"] == 0.0


def _key() -> str:
    return stratum_key(turn_kind="code", input_tokens=8000, model="claude-opus-4-8", has_tools=True)


def test_estimate_request_savings_treatment_uses_baseline(tmp_path):
    rec = SavingsRecorder(str(tmp_path / "o.json"), flush_every=1)
    key = _key()
    for _ in range(5):
        rec._ledger.baseline.observe(key, 1000)  # baseline mean ~1000

    # Treatment request that emitted 600 -> saved ~400 vs the baseline.
    saved = rec.estimate_request_savings([stratum_label("treatment", key)], 600)
    assert saved == 400


def test_estimate_request_savings_zero_for_control_and_unknown(tmp_path):
    rec = SavingsRecorder(str(tmp_path / "o.json"), flush_every=1)
    key = _key()
    for _ in range(5):
        rec._ledger.baseline.observe(key, 1000)

    # Control arm is unshaped -> no attributable saving.
    assert rec.estimate_request_savings([stratum_label("control", key)], 600) == 0
    # No shaping label at all.
    assert rec.estimate_request_savings(["something-else"], 600) == 0
    # Treatment but output exceeded the baseline -> clamped to 0, never negative.
    assert rec.estimate_request_savings([stratum_label("treatment", key)], 5000) == 0

"""Per-bucket output-shaping savings in the /stats-history rollup.

Covers the feature that lets a downstream dashboard stack output-shaping
savings as a distinct daily segment: SavingsTracker.record_request accepts a
per-request output_tokens_saved, accumulates it into each time bucket as
output_tokens_saved_delta / output_savings_usd_delta, and the read-only
SavingsRecorder.estimate_request_savings supplies that per-request number.
"""

from __future__ import annotations

import pytest

from headroom.proxy import output_savings
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


@pytest.fixture
def l4_factors(monkeypatch):
    """Benchmark factors as an enterprise extension would register them."""
    monkeypatch.setattr(output_savings, "MODELLED_REDUCTION", {4: (0.2, 0.5)})


def test_shaped_request_without_a_baseline_is_priced_from_its_level_factors(tmp_path, l4_factors):
    rec = SavingsRecorder(str(tmp_path / "o.json"), flush_every=1)
    labels = [stratum_label("treatment", _key()), "output_shaper:verbosity:L4"]

    # Observed output is already shaped: saved = 800 * r / (1 - r).
    assert rec.request_savings_range(labels, 800) == (200, 800)


def test_baseline_and_holdout_outrank_the_factors(tmp_path, l4_factors):
    rec = SavingsRecorder(str(tmp_path / "o.json"), flush_every=1)
    key = _key()
    for _ in range(5):
        rec._ledger.baseline.observe(key, 1000)

    shaped = "output_shaper:verbosity:L4"
    assert rec.request_savings_range([stratum_label("treatment", key), shaped], 600) == (400, 400)
    assert rec.request_savings_range([stratum_label("control", key)], 600) == (0, 0)


def test_no_registered_factors_means_no_modelled_saving(tmp_path):
    """Plain OSS ships an empty factor table: nothing is booked."""
    rec = SavingsRecorder(str(tmp_path / "o.json"), flush_every=1)
    labels = [stratum_label("treatment", _key()), "output_shaper:verbosity:L4"]
    assert rec.request_savings_range(labels, 800) == (0, 0)


def test_lifetime_carries_the_output_savings_range_across_a_restart(tmp_path):
    path = str(tmp_path / "s.json")
    tracker = SavingsTracker(path=path)
    tracker.record_request(
        model="claude-opus-4-8",
        input_tokens=1000,
        tokens_saved=0,
        output_tokens_saved=200,
        output_tokens_saved_high=800,
        timestamp="2026-03-27T09:00:00Z",
    )
    tracker.record_request(  # a point estimate: high end equals low end
        model="claude-opus-4-8",
        input_tokens=1000,
        tokens_saved=0,
        output_tokens_saved=100,
        timestamp="2026-03-27T09:30:00Z",
    )

    lifetime = SavingsTracker(path=path).snapshot()["lifetime"]
    assert lifetime["output_tokens_saved"] == 300
    assert lifetime["output_tokens_saved_high"] == 900
    low_usd, high_usd = lifetime["output_savings_usd"], lifetime["output_savings_usd_high"]
    assert low_usd > 0
    assert high_usd == pytest.approx(low_usd * 3, rel=1e-3)


def test_treatment_request_that_was_not_shaped_saves_nothing(tmp_path, l4_factors):
    """The stratum label is attached before shaping is attempted; only the
    verbosity label proves the request was shaped."""
    rec = SavingsRecorder(str(tmp_path / "o.json"), flush_every=1)
    key = _key()
    for _ in range(5):
        rec._ledger.baseline.observe(key, 1000)

    assert rec.request_savings_range([stratum_label("treatment", key)], 600) == (0, 0)


def test_factors_are_not_booked_once_the_deployment_has_a_baseline(tmp_path, l4_factors):
    """A baseline learned for other traffic must not leave this request on
    benchmark factors while /stats reports the estimated tier."""
    rec = SavingsRecorder(str(tmp_path / "o.json"), flush_every=1)
    other = stratum_key(
        turn_kind="chat", input_tokens=500, model="claude-opus-4-8", has_tools=False
    )
    for _ in range(5):
        rec._ledger.baseline.observe(other, 1000)

    labels = [stratum_label("treatment", _key()), "output_shaper:verbosity:L4"]
    low, high = rec.request_savings_range(labels, 800)
    assert low == high  # a baseline point estimate, never the factor band


def test_fresh_tracker_lifetime_carries_the_range_keys(tmp_path):
    lifetime = SavingsTracker(path=str(tmp_path / "s.json")).snapshot()["lifetime"]
    assert lifetime["output_tokens_saved_high"] == 0
    assert lifetime["output_savings_usd_high"] == 0.0

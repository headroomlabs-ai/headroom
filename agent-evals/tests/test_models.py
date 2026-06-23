"""Tests for the core data-model contracts."""

from __future__ import annotations

import math

import pytest

from agent_evals.models import (
    ArmName,
    DeltaEstimate,
    EquivalenceVerdict,
    Pricing,
    RunSavings,
    TaskResult,
    TaskSavings,
)


def test_savings_from_token_counts_basic(pricing: Pricing) -> None:
    s = TaskSavings.from_token_counts(tokens_before=1000, tokens_after=400, pricing=pricing)
    assert s.tokens_saved == 600
    assert s.savings_percent == pytest.approx(60.0)
    assert s.ratio == pytest.approx(0.4)
    # cost = tokens / 1e6 * input_usd_per_1m (=2.0)
    assert s.cost_usd_before == pytest.approx(1000 / 1_000_000 * 2.0)
    assert s.cost_usd_after == pytest.approx(400 / 1_000_000 * 2.0)
    assert s.cost_usd_saved == pytest.approx(s.cost_usd_before - s.cost_usd_after)
    assert s.source == "headers"


def test_savings_zero_tokens_before_is_safe(pricing: Pricing) -> None:
    s = TaskSavings.from_token_counts(tokens_before=0, tokens_after=0, pricing=pricing)
    assert s.tokens_saved == 0
    assert s.savings_percent == 0.0
    assert s.ratio == 1.0
    assert s.cost_usd_saved == 0.0
    assert math.isfinite(s.ratio)


def test_savings_no_compression_ratio_one(pricing: Pricing) -> None:
    s = TaskSavings.from_token_counts(tokens_before=500, tokens_after=500, pricing=pricing)
    assert s.tokens_saved == 0
    assert s.ratio == pytest.approx(1.0)
    assert s.savings_percent == pytest.approx(0.0)


def test_savings_carries_flags(pricing: Pricing) -> None:
    s = TaskSavings.from_token_counts(
        tokens_before=100,
        tokens_after=90,
        pricing=pricing,
        transforms=["smart_crusher", "read_lifecycle"],
        cached=True,
        compression_failed=False,
        source="stats_delta",
    )
    assert s.transforms == ["smart_crusher", "read_lifecycle"]
    assert s.cached is True
    assert s.source == "stats_delta"


def test_task_result_cell_key() -> None:
    tr = TaskResult(task_id="t1", arm=ArmName.B_HEADROOM, run_index=3, resolved=True)
    assert tr.cell_key == ("t1", "b_headroom", 3)


def test_equivalence_verdict_roundtrip() -> None:
    v = EquivalenceVerdict(
        delta=DeltaEstimate(point=-0.5, ci_low=-1.9, ci_high=0.9, method="paired_bootstrap"),
        margin=2.0,
        verdict="equivalent",
    )
    again = EquivalenceVerdict.model_validate_json(v.model_dump_json())
    assert again.verdict == "equivalent"
    assert again.delta.ci_low == pytest.approx(-1.9)


def test_run_savings_optional_preserved_tokens() -> None:
    rs = RunSavings(cache_read_tokens=120, prefix_freeze_busts_avoided=3)
    assert rs.prefix_freeze_tokens_preserved is None


def test_arm_name_values_are_stable() -> None:
    # Journal keys depend on these string values; guard against accidental renames.
    assert ArmName.A1_PASSTHROUGH.value == "a1_passthrough"
    assert ArmName.B_HEADROOM.value == "b_headroom"

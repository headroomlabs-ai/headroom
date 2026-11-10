"""Unit tests for :mod:`perfkit.baseline` filename parsing + policy (T011)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.perfkit.baseline import (
    REASON_APPROVED_INTENDED_SHIFT,
    REASON_FIRST_RECORD,
    REASON_FIXTURE_CHANGE,
    REASON_SCHEMA_BUMP,
    BaselinePolicyDecision,
    assert_reason,
    baseline_path,
    content_hash12,
    evaluate_first_record,
    parse_baseline_filename,
)
from benchmarks.perfkit.schema import SCHEMA_VERSION


def test_filename_round_trip() -> None:
    path = baseline_path("/tmp/baselines", "G2", SCHEMA_VERSION, "abc1234")
    ref = parse_baseline_filename(path)
    assert ref.tier == "G2"
    assert ref.schema_version == SCHEMA_VERSION
    assert ref.short_sha == "abc1234"
    assert ref.content_hash12 == "000000000000"


def test_filename_with_nontrivial_hash_parses() -> None:
    p = Path("/tmp/baselines/G1.schema-1.0.0.deadbeef1234.abc1234.json")
    ref = parse_baseline_filename(p)
    assert ref.content_hash12 == "deadbeef1234"
    assert ref.identifier == "G1.abc1234.deadbeef1234"


def test_invalid_filename_raises() -> None:
    with pytest.raises(ValueError):
        parse_baseline_filename("/tmp/baselines/not-a-baseline.json")
    with pytest.raises(ValueError):
        parse_baseline_filename("/tmp/baselines/G1.bogus-hash.abc1234.json")


def test_content_hash12_ignores_recorded_at() -> None:
    a = {"metrics": {"x": 1}, "recorded_at": "2024-01-01T00:00:00+00:00"}
    b = {"metrics": {"x": 1}, "recorded_at": "2025-02-02T00:00:00+00:00"}
    assert content_hash12(a) == content_hash12(b)


def test_first_record_when_no_baselines(tmp_path: Path) -> None:
    decision = evaluate_first_record(tmp_path, "G1", "abc1234")
    assert not decision.exists
    assert decision.reason_for_action == REASON_FIRST_RECORD


def test_present_when_baseline_at_commit(tmp_path: Path) -> None:
    target = baseline_path(tmp_path, "G1", SCHEMA_VERSION, "abc1234")
    target.write_text(json.dumps({"metrics": {"per_transform": []}, "noise_floor_used": {}}))
    decision = evaluate_first_record(tmp_path, "G1", "abc1234")
    assert decision.exists is True
    assert decision.path == target


def test_fixture_change_when_commit_differs(tmp_path: Path) -> None:
    target = baseline_path(tmp_path, "G1", SCHEMA_VERSION, "abc1234")
    target.write_text(json.dumps({"metrics": {"per_transform": []}, "noise_floor_used": {}}))
    decision = evaluate_first_record(tmp_path, "G1", "different")
    assert not decision.exists
    assert decision.reason_for_action == REASON_FIXTURE_CHANGE


def test_assert_reason_validates_known_reasons() -> None:
    assert_reason(REASON_FIRST_RECORD)
    assert_reason(REASON_APPROVED_INTENDED_SHIFT)
    assert_reason(REASON_SCHEMA_BUMP)
    with pytest.raises(ValueError):
        assert_reason("not_a_real_reason")


def test_baseline_policy_decision_factories() -> None:
    m = BaselinePolicyDecision.missing(reason=REASON_FIRST_RECORD)
    assert not m.exists
    assert m.reason_for_action == REASON_FIRST_RECORD
    p = Path("/tmp/baselines/G1.schema-1.0.0.abc.json")
    present = BaselinePolicyDecision.present(p)
    assert present.exists
    assert present.path == p
    with pytest.raises(ValueError):
        BaselinePolicyDecision.missing(reason="bogus")


def test_noise_floor_used_round_trip(tmp_path: Path) -> None:
    """A baseline file preserves its ``noise_floor_used`` policy byte-for-byte."""

    payload = {
        "codeowner": "headroom-perf",
        "recorded_by": "kellen",
        "reason": REASON_FIRST_RECORD,
        "noise_floor_used": {"G1": {"default": 10.0}, "G2": {"default": 15.0}},
        "policy_limits_used": {"G2_hard_p99_s": 0.050, "G3_hard_p99_s": 0.050},
        "metrics": {"per_transform": ()},
    }
    target = baseline_path(tmp_path, "G2", SCHEMA_VERSION, "abc1234")
    target.write_text(json.dumps(payload))
    loaded = json.loads(target.read_text())
    assert loaded["noise_floor_used"] == payload["noise_floor_used"]
    assert loaded["policy_limits_used"] == payload["policy_limits_used"]

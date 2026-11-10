"""Unit tests for the canonical :mod:`perfkit.schema` dataclasses (T010)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.perfkit.artifact_io import read_artifact, write_artifact
from benchmarks.perfkit.schema import (
    SCHEMA_VERSION,
    Attribution,
    ColdWarmSplit,
    G1Metrics,
    G2Metrics,
    G3Metrics,
    G4Metrics,
    G5Metrics,
    G6Metrics,
    G6Verdict,
    HarnessOverhead,
    PerfArtifact,
    Trend,
    median,
    quantile,
    slope_and_r2,
    stdev_pct,
)


def _common_attribution() -> Attribution:
    return Attribution(
        stable_id="G1:/v1/messages:smart_crusher:apply",
        tier="G1",
        request_path="/v1/messages",
        stage=None,
        transform="smart_crusher.apply",
        function="apply",
    )


def test_schema_version_constant_is_one_zero():
    assert SCHEMA_VERSION == "1.0.0"


def test_g1_metrics_round_trip(tmp_path: Path) -> None:
    """A G1 artifact written + read back keeps the same shape (T010)."""

    metrics = G1Metrics(
        per_transform=(
            {
                "transform": "smart_crusher.apply",
                "fixture_hash": "abc",
                "p50_s": 0.0015,
                "p99_s": 0.002,
                "rps": 5000.0,
            },
        ),
        python_vs_rust=None,
    )
    art = PerfArtifact(
        schema_version=SCHEMA_VERSION,
        commit="cafef00d",
        tier="G1",
        run_id="r1",
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:00:01+00:00",
        attribution=_common_attribution(),
        metrics=metrics,
    )
    written = write_artifact(art, artifacts_dir=tmp_path)
    payload = read_artifact(written)
    assert payload["metrics"]["per_transform"][0]["transform"] == "smart_crusher.apply"
    assert payload["attribution"]["stable_id"] == art.attribution.stable_id


def test_g2_metrics_cold_warm_split_is_optional(tmp_path: Path) -> None:
    metrics = G2Metrics(
        per_stage=({"stage": "compression", "p99_s": 0.04},),
        cold_warm_split=ColdWarmSplit(cold_p99_s=0.06, warm_p99_s=0.040, cold_warm_ratio=1.5),
    )
    art = PerfArtifact(
        schema_version=SCHEMA_VERSION,
        commit="abc",
        tier="G2",
        run_id="r2",
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:00:01+00:00",
        attribution=_common_attribution(),
        metrics=metrics,
        cold_warm_split=ColdWarmSplit(cold_p99_s=0.06, warm_p99_s=0.040, cold_warm_ratio=1.5),
        harness_overhead=HarnessOverhead(
            driver_cpu_pct=2.5,
            proxy_overhead_p99_s=0.001,
            share_of_p99=0.025,
            sample_count=200,
        ),
    )
    json_path = write_artifact(art, artifacts_dir=tmp_path)
    payload = read_artifact(json_path)
    assert payload["cold_warm_split"]["cold_warm_ratio"] == 1.5
    assert payload["harness_overhead"]["share_of_p99"] == 0.025


def test_g3_metrics_byte_faithful_default_true() -> None:
    metrics = G3Metrics(
        per_route=({"request_path": "/v1/messages", "p99_s": 0.03, "byte_faithful": True},),
    )
    assert metrics.byte_faithful is True


def test_g4_metrics_carry_throughput() -> None:
    metrics = G4Metrics(
        per_scenario=({"scenario": "long-conversation", "concurrency": 4, "p99_s": 0.045},),
        overall_throughput_rps=240.0,
    )
    assert metrics.overall_throughput_rps == 240.0


def test_g5_metrics_low_confidence_signal() -> None:
    metrics = G5Metrics(
        sample_count=42,
        trends=({"metric": "rss_mb", "slope": 0.0, "r_squared": 0.0, "above_threshold": False},),
    )
    assert metrics.sample_count == 42  # 42 < 60 -> low_confidence flag elsewhere


def test_g6_verdict_serializes_through_metrics_block(tmp_path: Path) -> None:
    v = G6Verdict(
        tier="G2",
        metric="per_stage/compression/p99_s",
        baseline_value=0.040,
        candidate_value=0.044,
        delta_pct=10.0,
        threshold_pct=15.0,
        verdict="pass",
        attribution_id="G2:/v1/messages:compression:smart_crusher:apply",
    )
    metrics = G6Metrics(
        verdicts=(v,),
        effective_limits={"per_stage/compression/p99_s": 0.050},
        baseline_identity_ok=True,
    )
    art = PerfArtifact(
        schema_version=SCHEMA_VERSION,
        commit="ghi",
        tier="G6",
        run_id="r6",
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:00:01+00:00",
        attribution=Attribution(
            stable_id="G6:overall:-:-:-",
            tier="G6",
            request_path="overall",
            stage="-",
            transform="-",
            function="-",
        ),
        metrics=metrics,
        verdicts=(v,),
    )
    payload = read_artifact(write_artifact(art, artifacts_dir=tmp_path))
    assert payload["metrics"]["verdicts"][0]["verdict"] == "pass"


def test_quantile_handles_empty_and_boundaries() -> None:
    assert quantile([], 0.5) == 0.0
    assert quantile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert quantile([1.0, 2.0, 3.0], 1.0) == 3.0
    with pytest.raises(ValueError):
        quantile([1.0], -0.1)


@pytest.mark.parametrize("q", [0.0, 0.01, 0.5, 0.99, 1.0])
def test_quantile_returns_single_observation(q: float) -> None:
    assert quantile([0.01], q) == 0.01


def test_median_returns_zero_for_empty() -> None:
    assert median([]) == 0.0
    assert median([1, 3, 5]) == 3.0


def test_stdev_pct_is_zero_for_constant_or_short() -> None:
    assert stdev_pct([1.0]) == 0.0
    assert stdev_pct([7.0, 7.0, 7.0]) == 0.0


def test_slope_and_r2_perfect_linear() -> None:
    slope, r2 = slope_and_r2([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
    assert slope == pytest.approx(1.0, rel=1e-6)
    assert r2 == pytest.approx(1.0, rel=1e-6)


def test_slope_and_r2_timestamp_coordinates() -> None:
    samples = [(1_700_000_000.0 + i, 1.0 + 0.1 * i) for i in range(10)]
    assert slope_and_r2(samples) == pytest.approx((0.1, 1.0))


@pytest.mark.parametrize("offset", [0.0, 1_700_000_000.0])
def test_slope_and_r2_noisy_trend(offset: float) -> None:
    samples = [(offset + i, y) for i, y in enumerate([1.0, 2.0, 2.0, 4.0])]
    assert slope_and_r2(samples) == pytest.approx((0.9, 81 / 95))


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ([], (0.0, 0.0)),
        ([(0.0, 1.0)], (0.0, 0.0)),
        ([(0.0, 1.0), (0.0, 2.0)], (0.0, 0.0)),
        ([(0.0, 1.0), (1.0, 1.0)], (0.0, 1.0)),
    ],
)
def test_slope_and_r2_degenerate_samples(
    samples: list[tuple[float, float]], expected: tuple[float, float]
) -> None:
    assert slope_and_r2(samples) == expected


def test_trend_serialization() -> None:
    """Trend (frozen+slots) round-trips through ``asdict`` + JSON."""

    import dataclasses

    t = Trend(
        metric="rss_mb",
        slope=1.2,
        r_squared=0.9,
        sample_count=64,
        above_threshold=True,
        threshold=1.0,
        unit="MB/h",
    )
    blob = json.dumps(dataclasses.asdict(t))
    payload = json.loads(blob)
    assert payload["metric"] == "rss_mb"
    assert payload["above_threshold"] is True

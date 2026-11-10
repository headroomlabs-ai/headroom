"""Per-tier/per-metric noise floor table.

Per ``research.md`` §Decision 2 + ``contracts/regression-verdict.md`` §3.3.

The noise floor is the minimum delta (in percent) that perfkit considers
non-noise for a given metric. Improvements and regressions smaller than the
floor are flagged ``low_confidence`` (advisory only; the comparator still
records them but never changes the exit code on them).

The floor also publishes the absolute hard ceiling for latency-critical tiers
(G2 + G3 share the same 50 ms P99 objective, see FR-003 / spec SC-001).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Per-tier thresholds for meaningful regression (percent).
TIER_FLOOR_PCT: Mapping[str, Mapping[str, float]] = {
    # G1 microbench: tight transform-level numbers; 10% to match the
    # project testing convention (constitution §IV) while leaving headroom
    # for the Python↔Rust translation noise of pytest-benchmark/criterion.
    "G1": {
        "default": 10.0,
        "python_vs_rust": 15.0,
    },
    # G2 component: includes Python↔Rust crossings inside a single stage.
    "G2": {
        "default": 15.0,
        "cold_warm_split": 25.0,
    },
    # G3 service: full proxy + upstream mock - includes OS scheduling jitter.
    "G3": {
        "default": 15.0,
        "cold_warm_split": 25.0,
        "harness_overhead": 20.0,
    },
    # G4 workload: scripted concurrency; noise grows with concurrency.
    "G4": {
        "default": 15.0,
        "throughput": 10.0,
    },
    # G5 soak: trend-based; the floor is used for slope thresholds not
    # relative deltas.
    "G5": {
        "default": 30.0,
        "rss_mb_per_h": 1.0,
        "fd_per_h": 1.0,
        "p99_drift_pct_per_h": 5.0,
    },
    # G6 verifier: the verdict engine; threshold is informed by the
    # underlying tier metric.
    "G6": {
        "default": 0.0,  # G6 defers to the underlying tier's threshold
    },
}


# Hard ceilings (absolute values, not percent).
HARD_P99_CEILING_S: Mapping[str, float] = {
    "G2": 0.050,  # 50 ms
    "G3": 0.050,  # 50 ms
}


# Trend thresholds for G5 soak (per hour).
SOAK_TREND_THRESHOLDS: Mapping[str, tuple[float, str]] = {
    # metric -> (threshold value, unit string)
    "rss_mb_per_h": (1.0, "MB/h"),  # > 1 MB RSS growth per hour
    "fd_per_h": (1.0, "fds/h"),  # > 1 leaked fd per hour
    "p99_drift_pct_per_h": (5.0, "%/h"),  # > 5% P99 drift per hour
    "ccr_store_mb_per_h": (5.0, "MB/h"),  # CCR store growth (advisory only)
}


@dataclass(slots=True, frozen=True)
class NoiseFloor:
    """Resolved noise-floor policy for a single tier + metric pair."""

    tier: str
    metric: str
    floor_pct: float

    def delta_is_meaningful(self, delta_pct: float) -> bool:
        """``True`` iff ``delta_pct`` exceeds the floor in absolute terms."""

        return abs(delta_pct) >= self.floor_pct


def resolve_floor(tier: str, metric: str = "default") -> NoiseFloor:
    """Look up the floor; ``default`` is used when the metric has no entry."""

    by_tier = TIER_FLOOR_PCT.get(tier, {})
    floor = float(by_tier.get(metric, by_tier.get("default", 10.0)))
    return NoiseFloor(tier=tier, metric=metric, floor_pct=floor)


def hard_p99_ceiling_s(tier: str) -> float | None:
    """Return the absolute P99 hard ceiling for a tier (or ``None``)."""

    return HARD_P99_CEILING_S.get(tier)


def soak_trend_thresholds() -> Mapping[str, tuple[float, str]]:
    """Per-hour thresholds the G5 soak engine flags as leaks."""

    return dict(SOAK_TREND_THRESHOLDS)

"""Canonical perfkit artifact schema (``schema_version = "1.0.0"``).

Each tier (G1..G6) emits **one** artifact per run, persisted as
``benchmarks/artifacts/<commit_sha>/<tier>/<run_id>.json`` with a companion
Markdown summary. The schema mirrors ``contracts/json-artifact-schema.md``
sections 2-6 and keeps tier-specific metrics isolated in a ``metrics`` block so
later tiers can be added without breaking the existing JSON contract.

Conventions:

* Dataclasses are ``slots=True, frozen=True`` - mutable state is not allowed
  in a written artifact.
* All join keys (``commit``, ``fixture_hash``, ``schema_version``, ``tier``,
  ``attribution_id``) are strings, never missing.
* Floats carry an ``error_pct`` field where useful; times are seconds.
* Quantile math is intentionally trivial (sample reservoir). Nothing in perfkit
  ever imports numpy/pandas - the harness must work on a clean checkout.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "1.0.0"

# Tier identifiers — kept as constants so a typo at a call site is a NameError.
G1 = "G1"
G2 = "G2"
G3 = "G3"
G4 = "G4"
G5 = "G5"
G6 = "G6"
ALL_TIERS: tuple[str, ...] = (G1, G2, G3, G4, G5, G6)

# Provider identifiers used by G3/G4 mock-upstream fixtures.
PROVIDERS: tuple[str, ...] = (
    "anthropic",
    "openai_chat",
    "openai_responses",
    "bedrock",
    "vertex",
)


# ---------------------------------------------------------------------------
# Tier-specific metric blocks (kept narrow so cross-tier tests stay simple).
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class G1Metrics:
    """G1 microbench metrics.

    ``per_transform`` is the unit of attribution: one row per transform under
    measurement (e.g. ``smart_crusher.apply``). ``python_vs_rust`` is the
    optional Python-vs-Rust divergence record required by FR-020.
    """

    per_transform: tuple[Mapping[str, Any], ...]
    python_vs_rust: Mapping[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class G2Metrics:
    """G2 component metrics - one row per stage (compression, CCR lookup, ...)."""

    per_stage: tuple[Mapping[str, Any], ...]
    cold_warm_split: Mapping[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class G3Metrics:
    """G3 service metrics (loopback production proxy + mock upstream)."""

    per_route: tuple[Mapping[str, Any], ...]
    harness_overhead: Mapping[str, Any] | None = None
    cold_warm_split: Mapping[str, Any] | None = None
    byte_faithful: bool = True


@dataclass(slots=True, frozen=True)
class G4Metrics:
    """G4 workload metrics - scripted concurrency scenarios.

    The split tracks ``concurrency`` per scenario so a workload report can
    pair P99 at the declared concurrency with throughput (req/s).
    """

    per_scenario: tuple[Mapping[str, Any], ...]
    overall_throughput_rps: float = 0.0


@dataclass(slots=True, frozen=True)
class G5Metrics:
    """G5 soak metrics.

    Each ``trend`` block carries slope + R^2; ``sample_count`` is the
    governing low-confidence knob (<60 samples -> ``low_confidence=True``).
    """

    sample_count: int
    trends: tuple[Mapping[str, Any], ...]
    breaches: tuple[Mapping[str, Any], ...] = ()


@dataclass(slots=True, frozen=True)
class G6Verdict:
    """One per-tier verdict row emitted by the comparator (G6)."""

    tier: str
    metric: str
    baseline_value: float
    candidate_value: float
    delta_pct: float
    threshold_pct: float
    verdict: str  # one of: pass, regression, improvement, low_confidence, baseline_missing, baseline_invalidated
    attribution_id: str | None = None


@dataclass(slots=True, frozen=True)
class G6Metrics:
    """G6 verdict engine output."""

    verdicts: tuple[G6Verdict, ...]
    effective_limits: Mapping[str, float]
    baseline_identity_ok: bool
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Cold/warm split + harness overhead + attribution blocks (cross-tier).
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ColdWarmSplit:
    """Per-metric cold vs warm split (required at G3, optional at G2)."""

    cold_p99_s: float
    warm_p99_s: float
    cold_warm_ratio: float
    method: str = "first_request_vs_p50_of_following"


@dataclass(slots=True, frozen=True)
class HarnessOverhead:
    """A/B-subtracted proxy overhead vs loopback mock (FR-019 / §10)."""

    driver_cpu_pct: float
    proxy_overhead_p99_s: float
    share_of_p99: float  # 0..1
    sample_count: int


@dataclass(slots=True, frozen=True)
class Attribution:
    """Stable attribution identifier (``tier:request_path:stage:transform:function``).

    See contracts/data-model.md Entity 5. ``parents`` is the chain of
    coarser-grained identifiers pointing at this row from higher tiers, so a
    G1 report can be reached from a G4 report by walking ``parents``.
    """

    stable_id: str
    tier: str
    request_path: str
    stage: str | None
    transform: str | None
    function: str | None
    parents: tuple[str, ...] = ()

    def to_chain(self) -> tuple[str, ...]:
        """Return ``(parents..., stable_id)`` for ordered rendering."""

        return (*self.parents, self.stable_id)


# ---------------------------------------------------------------------------
# Top-level artifact.
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AttributionChain:
    """The list of stable IDs encountered while drilling down."""

    entries: tuple[str, ...]
    tier: str = G4  # populated for compatibility with Markdown rendering


@dataclass(slots=True, frozen=True)
class Trend:
    """Slope + R^2 on a metric over time."""

    metric: str
    slope: float
    r_squared: float
    sample_count: int
    above_threshold: bool
    threshold: float | None = None
    unit: str = ""


@dataclass(slots=True, frozen=True)
class PerfArtifact:
    """Canonical artifact (one per tier per run).

    The union of tier-specific metric blocks is represented by ``metrics``
    being one of the G1..G6 dataclasses; ``tier`` selects which slot is
    populated.  All other blocks are tier-agnostic.
    """

    schema_version: str
    commit: str
    tier: str
    run_id: str
    started_at: str
    finished_at: str
    attribution: Attribution
    metrics: Any  # one of G1Metrics..G6Metrics
    baseline_ref: str | None = None
    noise_floor_used: Mapping[str, float] | None = None
    policy_limits_used: Mapping[str, float] | None = None
    harness_overhead: HarnessOverhead | None = None
    cold_warm_split: ColdWarmSplit | None = None
    trends: tuple[Trend, ...] = ()
    verdicts: tuple[G6Verdict, ...] = ()
    low_confidence: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe dict (tuples -> lists, dataclasses -> dicts)."""

        d = asdict(self)
        return _jsonify(d)


# ---------------------------------------------------------------------------
# Helpers used by tier implementations.
# ---------------------------------------------------------------------------


def quantile(samples: Sequence[float], q: float) -> float:
    """Type-safe ``q``-quantile using ``statistics.quantiles`` (n=100 cut points).

    ``q`` must be in ``[0, 1]``. Returns ``0.0`` for an empty sequence — a
    sample reservoir with zero observations is a degenerate but legal state
    during warmup.
    """

    if not samples:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0,1], got {q!r}")
    if len(samples) == 1:
        return float(samples[0])
    if q in (0.0, 1.0):
        return float(min(samples) if q == 0.0 else max(samples))
    # statistics.quantiles returns n-1 cut points; n=100 gives percentile resolution.
    cuts = statistics.quantiles(samples, n=100, method="inclusive")
    bucket = max(1, min(100, int(q * 100)))
    return float(cuts[bucket - 1])


def median(samples: Sequence[float]) -> float:
    """Convenience wrapper: median is robust and the tier reports use it."""

    return float(statistics.median(samples)) if samples else 0.0


def stdev_pct(values: Sequence[float]) -> float:
    """Coefficient of variation as a percent (0 if constant or empty)."""

    if len(values) < 2:
        return 0.0
    mu = statistics.fmean(values)
    if mu == 0:
        return 0.0
    sd = statistics.stdev(values)
    return abs(sd / mu) * 100.0


def mean(values: Sequence[float]) -> float:
    """Mean (0.0 for empty, NaN-safe — arithmetic mean only)."""

    if not values:
        return 0.0
    return float(statistics.fmean(values))


def slope_and_r2(samples: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """OLS slope + R^2 of ``y ~ x``.

    ``samples`` is a sequence of ``(x, y)`` points. Returns ``(0.0, 0.0)`` for
    fewer than two points or zero variance in x (degenerate trend).
    """

    if len(samples) < 2:
        return 0.0, 0.0
    mean_x = statistics.fmean(x for x, _ in samples)
    mean_y = statistics.fmean(y for _, y in samples)
    # Center coordinates to avoid cancellation with timestamp-sized values.
    xs = [x - mean_x for x, _ in samples]
    ys = [y - mean_y for _, y in samples]
    denom = math.fsum(x * x for x in xs)
    if denom == 0:
        return 0.0, 0.0
    slope = math.fsum(x * y for x, y in zip(xs, ys)) / denom
    ss_tot = math.fsum(y * y for y in ys)
    if ss_tot == 0:
        return float(slope), 1.0
    ss_res = math.fsum((y - slope * x) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot
    return float(slope), float(max(0.0, min(1.0, r2)))


def is_finite(x: float) -> bool:
    """``True`` iff ``x`` is a real number (not NaN / inf)."""

    return isinstance(x, (int, float)) and not (
        isinstance(x, float) and (math.isnan(x) or math.isinf(x))
    )


def dominant(values: Sequence[str]) -> str | None:
    """Most common element (mode). ``None`` for an empty input."""

    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Internal: convert nested dataclasses / tuples into JSON-safe primitives.
# ---------------------------------------------------------------------------


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "__dict__") or hasattr(obj, "__slots__"):
        return _jsonify(asdict(obj) if hasattr(obj, "__dataclass_fields__") else vars(obj))
    return str(obj)

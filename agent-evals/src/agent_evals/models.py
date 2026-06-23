"""Core data models for agent-evals.

Pure pydantic v2 models — the contracts every other module is built against. No I/O and no
global time/randomness: anything time- or RNG-dependent is injected by the caller so runs
are reproducible (see spec §4.1).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ArmName(str, Enum):
    """The experiment arms. Headline accuracy claim = B_HEADROOM vs A1_PASSTHROUGH."""

    A0_DIRECT = "a0_direct"
    A1_PASSTHROUGH = "a1_passthrough"
    B_HEADROOM = "b_headroom"
    B_ABLATE = "b_ablate"


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ProxyMode(str, Enum):
    """How the Headroom proxy is launched for an arm. An ArmSpec with ``proxy_mode=None``
    means no proxy at all (A0 direct)."""

    OFF = "off"  # `headroom proxy --no-optimize` — proxy in path, compression disabled (A1)
    TOKEN = "token"  # `headroom proxy --mode token` — compression enabled (B)


class Pricing(BaseModel):
    """USD per 1M tokens for the pinned model snapshot. Injected via config — never a literal
    inside logic. Compression reduces prompt/input tokens, so input pricing drives cost deltas."""

    input_usd_per_1m: float = Field(ge=0.0)
    output_usd_per_1m: float = Field(default=0.0, ge=0.0)


class ArmSpec(BaseModel):
    """Static description of one arm. The Arm runtime (arms.py) turns this into a live proxy."""

    name: ArmName
    provider: Provider
    # None => A0 direct (no proxy launched). Otherwise the proxy is launched in this mode.
    proxy_mode: ProxyMode | None = None
    # Extra flags appended to the `headroom proxy` command for ablation arms,
    # e.g. ["--disable-kompress"] or ["--no-read-lifecycle"].
    proxy_flags: list[str] = Field(default_factory=list)
    label: str


class TaskSavings(BaseModel):
    """Per-task Layer-1 savings, parsed from the per-response ``x-headroom-*`` headers.

    Per spec §4.1/§5: ``tokens_before/after/saved`` come from the per-request response headers
    (``x-headroom-tokens-before``/``-after``/``-saved``). ``savings_percent`` and ``ratio`` are
    DERIVED (``x-headroom-savings-percent`` is batch-path-only). ``cost_*`` is DERIVED
    client-side from token counts x pinned pricing (no cost header exists). Cache/prefix-freeze
    metrics are run-level (see ``RunSavings``), not per task.
    """

    tokens_before: int = Field(ge=0)
    tokens_after: int = Field(ge=0)
    tokens_saved: int
    savings_percent: float
    ratio: float
    transforms: list[str] = Field(default_factory=list)
    cached: bool = False
    compression_failed: bool = False
    cost_usd_before: float
    cost_usd_after: float
    cost_usd_saved: float
    added_latency_ms: float = 0.0
    source: Literal["headers", "stats_delta"] = "headers"

    @classmethod
    def from_token_counts(
        cls,
        *,
        tokens_before: int,
        tokens_after: int,
        pricing: Pricing,
        transforms: list[str] | None = None,
        cached: bool = False,
        compression_failed: bool = False,
        added_latency_ms: float = 0.0,
        source: Literal["headers", "stats_delta"] = "headers",
    ) -> TaskSavings:
        """Build from raw token counts, deriving saved/percent/ratio/cost. Guards divide-by-zero."""

        saved = tokens_before - tokens_after
        pct = (saved / tokens_before * 100.0) if tokens_before else 0.0
        ratio = (tokens_after / tokens_before) if tokens_before else 1.0
        cost_before = tokens_before / 1_000_000 * pricing.input_usd_per_1m
        cost_after = tokens_after / 1_000_000 * pricing.input_usd_per_1m
        return cls(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=saved,
            savings_percent=pct,
            ratio=ratio,
            transforms=transforms or [],
            cached=cached,
            compression_failed=compression_failed,
            cost_usd_before=cost_before,
            cost_usd_after=cost_after,
            cost_usd_saved=cost_before - cost_after,
            added_latency_ms=added_latency_ms,
            source=source,
        )


class RunSavings(BaseModel):
    """Run-level (not per-task) savings, read from the ``/stats`` lifetime snapshot aggregate.

    These have no per-response header, so they cannot be attributed to a single task.
    Exact ``/stats`` leaf names are bound at parse time against the live payload."""

    cache_read_tokens: int = 0
    prefix_freeze_busts_avoided: int = 0
    prefix_freeze_tokens_preserved: int | None = None


class BenchTask(BaseModel):
    """A single benchmark task. ``payload`` carries benchmark-specific fields (issue, repo, tests…)."""

    task_id: str
    payload: dict = Field(default_factory=dict)


class RolloutResult(BaseModel):
    """Output of one harness rollout (one task, one arm, one run). Rollout only — not graded."""

    task_id: str
    arm: ArmName
    run_index: int
    prediction: str
    trajectory_path: Path
    savings: TaskSavings | None = None
    wall_ms: float = 0.0
    error: str | None = None


class GradeResult(BaseModel):
    """Execution-graded verdict for one task (from the official benchmark grader)."""

    task_id: str
    resolved: bool
    detail: dict = Field(default_factory=dict)


class TaskResult(BaseModel):
    """One journal cell: the joined rollout + grade for (task, arm, run)."""

    task_id: str
    arm: ArmName
    run_index: int
    resolved: bool
    savings: TaskSavings | None = None
    wall_ms: float = 0.0
    error: str | None = None

    @property
    def cell_key(self) -> tuple[str, str, int]:
        """Stable identity used by the resumable journal to skip completed cells."""

        return (self.task_id, self.arm.value, self.run_index)


class DeltaEstimate(BaseModel):
    """A point estimate of an accuracy/savings delta with a confidence interval."""

    point: float
    ci_low: float
    ci_high: float
    method: str


class EquivalenceVerdict(BaseModel):
    """Verdict of a TOST/non-inferiority test against a pre-registered margin (pp)."""

    delta: DeltaEstimate
    margin: float
    verdict: Literal["equivalent", "inferior", "inconclusive", "superior"]


class RunManifest(BaseModel):
    """The frozen, pinned description of one experiment — the reproducibility contract."""

    experiment_id: str
    created_at: datetime
    headroom_git_sha: str
    agent_evals_git_sha: str
    model_snapshot: str
    provider: Provider
    auth_mode: str
    benchmark: str
    benchmark_ref: str
    harness: str
    harness_version: str
    docker_digests: dict[str, str] = Field(default_factory=dict)
    arms: list[ArmSpec]
    k_runs: int
    temperature: float
    seeds: list[int]
    alpha: float
    margins: dict[str, float]
    pricing: Pricing

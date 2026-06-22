"""Phase-0 scorecard: aggregate :class:`TaskResult` cells into per-arm summaries and render.

Phase 0 has NO inferential statistics — paired bootstrap + TOST land in Phase 1. So we report
raw per-arm resolved rates and savings medians plus a single NAIVE point delta
(``B_HEADROOM`` resolved_rate minus ``A1_PASSTHROUGH`` resolved_rate) that is explicitly
labelled as having no confidence interval or equivalence verdict yet. We do not invent
statistics here.

Pure functions + pydantic models. The only I/O is rendering a rich table to an in-memory
string (no files, no network).
"""

from __future__ import annotations

from statistics import fmean, median

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from ..logging import get_logger
from ..models import ArmName, TaskResult, TaskSavings

logger = get_logger("report.scorecard")

# The headline accuracy claim is B_HEADROOM vs A1_PASSTHROUGH (see ArmName docstring).
_HEADLINE_TREATMENT = ArmName.B_HEADROOM
_HEADLINE_BASELINE = ArmName.A1_PASSTHROUGH

_STATS_NOTE = (
    "naive point delta; paired bootstrap + TOST verdict arrive in Phase 1 (CI/verdict: Phase 1)"
)
_SAVINGS_NOTE = (
    "savings medians are computed only over cells that reported Layer-1 savings; "
    "cells without savings are excluded from savings medians but still counted for resolved rate"
)


class ArmSummary(BaseModel):
    """Aggregated Phase-0 metrics for one arm.

    ``resolved_rate`` is the fraction of cells (task x run) that resolved. The savings medians
    are computed only over cells that carry a :class:`TaskSavings` (cells with ``savings is None``
    are skipped); when an arm has no savings at all these default to ``0.0``.
    """

    arm: ArmName
    label: str
    n_cells: int = Field(ge=0)
    n_tasks: int = Field(ge=0)
    resolved_rate: float = Field(ge=0.0, le=1.0)
    median_tokens_before: float = 0.0
    median_tokens_after: float = 0.0
    median_savings_percent: float = 0.0
    median_cost_saved: float = 0.0
    mean_added_latency_ms: float = 0.0


class Scorecard(BaseModel):
    """The full Phase-0 scorecard: one summary per arm plus the headline naive delta."""

    experiment_id: str
    arms: list[ArmSummary] = Field(default_factory=list)
    # B_HEADROOM resolved_rate - A1_PASSTHROUGH resolved_rate. None if either arm is absent.
    accuracy_delta_b_vs_a1: float | None = None
    savings_note: str = _SAVINGS_NOTE
    stats_note: str = _STATS_NOTE


def _summarize_arm(arm: ArmName, cells: list[TaskResult]) -> ArmSummary:
    """Aggregate one arm's cells into an :class:`ArmSummary`.

    ``cells`` is non-empty (callers only summarize arms that have at least one cell).
    """

    n_cells = len(cells)
    n_resolved = sum(1 for c in cells if c.resolved)
    resolved_rate = n_resolved / n_cells
    n_tasks = len({c.task_id for c in cells})

    # The arm label is carried by the cells indirectly only via ArmName; Phase-0 cells do not
    # carry the ArmSpec label, so fall back to the enum value as a stable, human-readable label.
    label = arm.value

    savings: list[TaskSavings] = [c.savings for c in cells if c.savings is not None]
    if savings:
        median_tokens_before = float(median(s.tokens_before for s in savings))
        median_tokens_after = float(median(s.tokens_after for s in savings))
        median_savings_percent = float(median(s.savings_percent for s in savings))
        median_cost_saved = float(median(s.cost_usd_saved for s in savings))
        mean_added_latency_ms = float(fmean(s.added_latency_ms for s in savings))
    else:
        median_tokens_before = 0.0
        median_tokens_after = 0.0
        median_savings_percent = 0.0
        median_cost_saved = 0.0
        mean_added_latency_ms = 0.0

    return ArmSummary(
        arm=arm,
        label=label,
        n_cells=n_cells,
        n_tasks=n_tasks,
        resolved_rate=resolved_rate,
        median_tokens_before=median_tokens_before,
        median_tokens_after=median_tokens_after,
        median_savings_percent=median_savings_percent,
        median_cost_saved=median_cost_saved,
        mean_added_latency_ms=mean_added_latency_ms,
    )


def build_scorecard(results: list[TaskResult], experiment_id: str) -> Scorecard:
    """Group ``results`` by arm, aggregate each, and compute the naive B-vs-A1 accuracy delta.

    Arms are emitted in the canonical :class:`ArmName` declaration order (so the rendered table is
    stable regardless of input ordering). Arms with zero cells are omitted entirely.
    """

    by_arm: dict[ArmName, list[TaskResult]] = {}
    for r in results:
        by_arm.setdefault(r.arm, []).append(r)

    # Stable, declaration-order emission; skip arms with no cells.
    summaries = [_summarize_arm(arm, by_arm[arm]) for arm in ArmName if arm in by_arm]

    rates = {s.arm: s.resolved_rate for s in summaries}
    treatment = rates.get(_HEADLINE_TREATMENT)
    baseline = rates.get(_HEADLINE_BASELINE)
    if treatment is None or baseline is None:
        accuracy_delta: float | None = None
        logger.info(
            "scorecard headline delta unavailable (missing arm)",
            extra={
                "fields": {
                    "experiment_id": experiment_id,
                    "treatment_present": treatment is not None,
                    "baseline_present": baseline is not None,
                }
            },
        )
    else:
        accuracy_delta = treatment - baseline

    return Scorecard(
        experiment_id=experiment_id,
        arms=summaries,
        accuracy_delta_b_vs_a1=accuracy_delta,
    )


def _fmt_pct(fraction: float) -> str:
    """Format a 0..1 fraction as a percentage string."""

    return f"{fraction * 100:.1f}%"


def render_scorecard(scorecard: Scorecard) -> str:
    """Render ``scorecard`` to a plain-text string via a recording rich Console.

    Produces a per-arm table plus a HEADLINE line carrying the savings note, the naive accuracy
    delta, and the stats note. No files are written; the string is built in memory.
    """

    table = Table(title=f"Phase-0 Scorecard — {scorecard.experiment_id}")
    table.add_column("arm", no_wrap=True)
    table.add_column("label", no_wrap=True)
    table.add_column("cells", justify="right")
    table.add_column("tasks", justify="right")
    table.add_column("resolved", justify="right")
    table.add_column("tok before", justify="right")
    table.add_column("tok after", justify="right")
    table.add_column("savings %", justify="right")
    table.add_column("cost saved", justify="right")
    table.add_column("+latency ms", justify="right")

    for summary in scorecard.arms:
        table.add_row(
            summary.arm.value,
            summary.label,
            str(summary.n_cells),
            str(summary.n_tasks),
            _fmt_pct(summary.resolved_rate),
            f"{summary.median_tokens_before:.0f}",
            f"{summary.median_tokens_after:.0f}",
            f"{summary.median_savings_percent:.1f}%",
            f"${summary.median_cost_saved:.6f}",
            f"{summary.mean_added_latency_ms:.1f}",
        )

    if scorecard.accuracy_delta_b_vs_a1 is None:
        delta_text = "n/a (B_HEADROOM or A1_PASSTHROUGH arm missing)"
    else:
        delta_pp = scorecard.accuracy_delta_b_vs_a1 * 100.0
        delta_text = f"{delta_pp:+.1f}pp (B_HEADROOM - A1_PASSTHROUGH resolved rate)"

    console = Console(record=True, width=120)
    console.print(table)
    # soft_wrap keeps each headline line intact (no width-driven mid-sentence newline), so the
    # full note strings remain contiguous and greppable in the exported text.
    console.print(f"HEADLINE accuracy delta: {delta_text}", soft_wrap=True)
    console.print(f"HEADLINE savings: {scorecard.savings_note}", soft_wrap=True)
    console.print(f"HEADLINE stats: {scorecard.stats_note}", soft_wrap=True)
    return console.export_text()

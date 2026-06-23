"""Unit tests for the Phase-0 scorecard (no I/O, no network, no subprocess)."""

from __future__ import annotations

import pytest

from agent_evals.models import ArmName, Pricing, TaskResult, TaskSavings
from agent_evals.report.scorecard import (
    ArmSummary,
    Scorecard,
    build_scorecard,
    render_scorecard,
)


def _savings(tokens_before: int, tokens_after: int, pricing: Pricing) -> TaskSavings:
    return TaskSavings.from_token_counts(
        tokens_before=tokens_before, tokens_after=tokens_after, pricing=pricing
    )


@pytest.fixture
def results(pricing: Pricing) -> list[TaskResult]:
    """Two tasks (t1, t2) x three arms (A0/A1/B) x k=2 runs, with known resolved booleans.

    Resolved booleans (cells):
      A0: t1[r0]=T t1[r1]=T t2[r0]=T t2[r1]=F  -> 3/4 = 0.75
      A1: t1[r0]=T t1[r1]=F t2[r0]=F t2[r1]=F  -> 1/4 = 0.25
      B : t1[r0]=T t1[r1]=T t2[r0]=T t2[r1]=F  -> 3/4 = 0.75

    Only the B arm carries TaskSavings. The four B cells have savings_percent
    {50, 50, 60, 40} -> median 50.0. tokens_before {1000,1000,1000,1000} -> median 1000.0;
    tokens_after {500,500,400,600} -> median 500.0.
    """

    out: list[TaskResult] = []

    # A0_DIRECT — no savings.
    a0_resolved = {(0, "t1"): True, (1, "t1"): True, (0, "t2"): True, (1, "t2"): False}
    for (run, task), resolved in a0_resolved.items():
        out.append(
            TaskResult(task_id=task, arm=ArmName.A0_DIRECT, run_index=run, resolved=resolved)
        )

    # A1_PASSTHROUGH — no savings.
    a1_resolved = {(0, "t1"): True, (1, "t1"): False, (0, "t2"): False, (1, "t2"): False}
    for (run, task), resolved in a1_resolved.items():
        out.append(
            TaskResult(task_id=task, arm=ArmName.A1_PASSTHROUGH, run_index=run, resolved=resolved)
        )

    # B_HEADROOM — with savings; latency 10ms each so mean_added_latency_ms == 10.0.
    b_cells = [
        ("t1", 0, True, 1000, 500),
        ("t1", 1, True, 1000, 500),
        ("t2", 0, True, 1000, 400),
        ("t2", 1, False, 1000, 600),
    ]
    for task, run, resolved, before, after in b_cells:
        sv = TaskSavings.from_token_counts(
            tokens_before=before, tokens_after=after, pricing=pricing, added_latency_ms=10.0
        )
        out.append(
            TaskResult(
                task_id=task,
                arm=ArmName.B_HEADROOM,
                run_index=run,
                resolved=resolved,
                savings=sv,
            )
        )

    return out


def _arm(scorecard: Scorecard, arm: ArmName) -> ArmSummary:
    for s in scorecard.arms:
        if s.arm == arm:
            return s
    raise AssertionError(f"arm {arm} not in scorecard")


def test_resolved_rates_per_arm(results: list[TaskResult]) -> None:
    sc = build_scorecard(results, experiment_id="exp-1")
    assert _arm(sc, ArmName.A0_DIRECT).resolved_rate == pytest.approx(0.75)
    assert _arm(sc, ArmName.A1_PASSTHROUGH).resolved_rate == pytest.approx(0.25)
    assert _arm(sc, ArmName.B_HEADROOM).resolved_rate == pytest.approx(0.75)


def test_cell_and_task_counts(results: list[TaskResult]) -> None:
    sc = build_scorecard(results, experiment_id="exp-1")
    b = _arm(sc, ArmName.B_HEADROOM)
    assert b.n_cells == 4
    assert b.n_tasks == 2


def test_median_savings_percent_b_arm(results: list[TaskResult]) -> None:
    sc = build_scorecard(results, experiment_id="exp-1")
    b = _arm(sc, ArmName.B_HEADROOM)
    # savings_percent across B cells = {50, 50, 60, 40} -> median 50.0
    assert b.median_savings_percent == pytest.approx(50.0)
    assert b.median_tokens_before == pytest.approx(1000.0)
    assert b.median_tokens_after == pytest.approx(500.0)
    assert b.mean_added_latency_ms == pytest.approx(10.0)
    # median cost saved: per-cell saved tokens {500,500,600,400} -> median 500 tokens
    # cost = 500/1e6 * 2.0 (pricing fixture input_usd_per_1m=2.0)
    assert b.median_cost_saved == pytest.approx(500 / 1_000_000 * 2.0)


def test_accuracy_delta_b_vs_a1(results: list[TaskResult]) -> None:
    sc = build_scorecard(results, experiment_id="exp-1")
    # B (0.75) - A1 (0.25) = 0.50
    assert sc.accuracy_delta_b_vs_a1 == pytest.approx(0.50)


def test_arm_without_savings_defaults_sanely(results: list[TaskResult]) -> None:
    sc = build_scorecard(results, experiment_id="exp-1")
    a1 = _arm(sc, ArmName.A1_PASSTHROUGH)
    # No savings on A1 cells: savings medians default to 0.0 and do not crash.
    assert a1.median_savings_percent == 0.0
    assert a1.median_tokens_before == 0.0
    assert a1.median_tokens_after == 0.0
    assert a1.median_cost_saved == 0.0
    assert a1.mean_added_latency_ms == 0.0


def test_accuracy_delta_none_when_b_missing(pricing: Pricing) -> None:
    # Only A0 + A1 present -> headline delta is undefined (B absent).
    only_baseline = [
        TaskResult(task_id="t1", arm=ArmName.A0_DIRECT, run_index=0, resolved=True),
        TaskResult(task_id="t1", arm=ArmName.A1_PASSTHROUGH, run_index=0, resolved=True),
    ]
    sc = build_scorecard(only_baseline, experiment_id="exp-2")
    assert sc.accuracy_delta_b_vs_a1 is None


def test_accuracy_delta_none_when_a1_missing(pricing: Pricing) -> None:
    only_treatment = [
        TaskResult(task_id="t1", arm=ArmName.B_HEADROOM, run_index=0, resolved=True),
    ]
    sc = build_scorecard(only_treatment, experiment_id="exp-3")
    assert sc.accuracy_delta_b_vs_a1 is None


def test_arms_emitted_in_declaration_order(results: list[TaskResult]) -> None:
    # Shuffle the input; output must still be canonical ArmName order.
    sc = build_scorecard(list(reversed(results)), experiment_id="exp-1")
    emitted = [s.arm for s in sc.arms]
    assert emitted == [ArmName.A0_DIRECT, ArmName.A1_PASSTHROUGH, ArmName.B_HEADROOM]


def test_empty_results_is_safe() -> None:
    sc = build_scorecard([], experiment_id="exp-empty")
    assert sc.arms == []
    assert sc.accuracy_delta_b_vs_a1 is None


def test_render_scorecard_returns_text(results: list[TaskResult]) -> None:
    sc = build_scorecard(results, experiment_id="exp-1")
    text = render_scorecard(sc)
    assert isinstance(text, str)
    assert text.strip()
    # Every arm's label appears in the rendered table.
    for arm in (ArmName.A0_DIRECT, ArmName.A1_PASSTHROUGH, ArmName.B_HEADROOM):
        assert arm.value in text
    # Headline lines: stats note + savings note + the naive delta in pp.
    assert sc.stats_note in text
    assert sc.savings_note in text
    assert "CI/verdict: Phase 1" in text
    assert "+50.0pp" in text
    assert "exp-1" in text


def test_render_scorecard_handles_missing_delta(pricing: Pricing) -> None:
    sc = build_scorecard(
        [TaskResult(task_id="t1", arm=ArmName.A1_PASSTHROUGH, run_index=0, resolved=True)],
        experiment_id="exp-4",
    )
    text = render_scorecard(sc)
    assert "n/a" in text
    assert sc.stats_note in text

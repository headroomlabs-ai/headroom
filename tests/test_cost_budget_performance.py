"""Incremental budget aggregation regression coverage (#3367)."""

from __future__ import annotations

from datetime import datetime

import pytest

from headroom.proxy.budget_basis_policy import COST_BASIS_ESTIMATED, COST_BASIS_MEASURED
from headroom.proxy.cost import CostEntry, CostTracker


def test_budget_breakdown_does_not_scan_reporting_history() -> None:
    tracker = CostTracker(budget_limit_usd=100.0)
    now = datetime.now()
    tracker._costs.extend(
        CostEntry(now, 1.0, COST_BASIS_MEASURED) for _ in range(tracker.MAX_COST_ENTRIES)
    )
    tracker._record_budget_cost(CostEntry(now, 2.0, COST_BASIS_MEASURED))

    class NonIterableHistory:
        def __iter__(self):
            raise AssertionError("budget check rescanned reporting history")

    tracker._costs = NonIterableHistory()  # type: ignore[assignment]

    assert tracker.check_budget() == (True, 98.0)


def test_budget_aggregate_preserves_basis_totals_and_policy() -> None:
    tracker = CostTracker(
        budget_limit_usd=10.0,
        estimated_basis_policy="ignore",
    )
    now = datetime.now()
    tracker._record_budget_cost(CostEntry(now, 3.0, COST_BASIS_MEASURED))
    tracker._record_budget_cost(CostEntry(now, 20.0, COST_BASIS_ESTIMATED))

    assert tracker.period_cost_breakdown() == {
        "period": "daily",
        "policy": "ignore",
        "total_usd": 23.0,
        "measured_usd": 3.0,
        "estimated_usd": 20.0,
        "estimated_pct": pytest.approx(87.0),
        "records": 2,
        "estimated_records": 1,
    }
    assert tracker.check_budget() == (True, 7.0)


@pytest.mark.parametrize(
    ("period", "now", "expired_at", "current_at"),
    [
        (
            "hourly",
            datetime(2026, 8, 31, 12),
            datetime(2026, 8, 31, 10),
            datetime(2026, 8, 31, 11, 30),
        ),
        (
            "daily",
            datetime(2026, 8, 31, 12),
            datetime(2026, 8, 30, 23, 59),
            datetime(2026, 8, 31, 1),
        ),
        (
            "monthly",
            datetime(2026, 8, 31, 12),
            datetime(2026, 7, 31, 23, 59),
            datetime(2026, 8, 1),
        ),
    ],
)
def test_budget_window_evicts_expired_entries_once(period, now, expired_at, current_at) -> None:
    tracker = CostTracker(budget_limit_usd=100.0, budget_period=period)
    tracker._record_budget_cost(CostEntry(expired_at, 40.0, COST_BASIS_MEASURED))
    tracker._record_budget_cost(CostEntry(current_at, 5.0, COST_BASIS_ESTIMATED))

    tracker._refresh_budget_window(now)
    assert tracker._budget_measured_usd == 0.0
    assert tracker._budget_estimated_usd == 5.0
    assert tracker._budget_estimated_records == 1
    assert len(tracker._budget_costs) == 1

    # Re-reading the same window is constant-time and must not subtract twice.
    tracker._refresh_budget_window(now)
    assert tracker._budget_estimated_usd == 5.0
    assert len(tracker._budget_costs) == 1


def test_budget_aggregate_keeps_existing_100k_record_cap() -> None:
    tracker = CostTracker(budget_limit_usd=1_000_000.0)
    tracker.MAX_COST_ENTRIES = 2
    now = datetime.now()
    tracker._record_budget_cost(CostEntry(now, 1.0, COST_BASIS_MEASURED))
    tracker._record_budget_cost(CostEntry(now, 2.0, COST_BASIS_ESTIMATED))
    tracker._record_budget_cost(CostEntry(now, 4.0, COST_BASIS_MEASURED))

    assert tracker.period_cost_breakdown()["total_usd"] == 6.0
    assert tracker.period_cost_breakdown()["records"] == 2
    assert tracker.period_cost_breakdown()["estimated_records"] == 1

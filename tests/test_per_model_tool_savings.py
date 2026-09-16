"""Per-model attribution must count tool-schema deferral, not just compression.

Reported against 0.36.0 with a per-model breakdown reading:

    Tokens saved: 625,277
      · messages       36,071
      · tool schemas  589,206
    Per-Model Breakdown
      <model-a>: ... 35,907 tokens saved
      <model-b>: ...      0 tokens saved
      <model-c>: ...    164 tokens saved
      <model-d>: ...      0 tokens saved

The rows sum to 36,071 — the *messages* line exactly. All 589,206 tokens of
tool-schema deferral, 94% of the headline, had no row to land in, so the
breakdown contradicted the total printed four lines above it and every
tool-heavy model reported "0 tokens saved".

Both surfaces had the same shape of bug and both are pinned here:

* ``perf/analyzer.py`` summed ``tokens_saved`` per model while its own headline
  summed ``tokens_saved + tool_saved``.
* ``savings_tracker`` had no per-model field for deferral at all, and
  ``prometheus_metrics.record_request`` received the figure but did not pass it
  down — while already folding it into the per-model *dollars*, so money and
  tokens disagreed on the same row.
"""

from __future__ import annotations

from headroom.perf.analyzer import PerfRecord, PerfReport, format_report
from headroom.proxy.savings_tracker import SavingsTracker, _normalize_by_model


def _record(model: str, *, before: int, saved: int, tool_saved: int) -> PerfRecord:
    return PerfRecord(
        timestamp="2026-08-16T00:00:00Z",
        request_id=f"req-{model}-{saved}-{tool_saved}",
        model=model,
        tokens_before=before,
        tokens_after=before - saved,
        tokens_saved=saved,
        tool_saved=tool_saved,
    )


# --------------------------------------------------------------------------- #
# CLI report (`headroom perf`)
# --------------------------------------------------------------------------- #
def test_per_model_rows_reconcile_with_the_headline() -> None:
    """The reported symptom: rows that do not add up to the total above them."""
    report = PerfReport(
        perf_records=[
            _record("model-a", before=100_000, saved=35_907, tool_saved=400_000),
            _record("model-b", before=50_000, saved=0, tool_saved=189_206),
            _record("model-c", before=20_000, saved=164, tool_saved=0),
        ]
    )

    text = format_report(report)

    # Headline is unchanged: 36,071 messages + 589,206 tool schemas.
    assert "Tokens saved: 625,277" in text

    # Every model's own tool savings now appear on its row.
    assert "model-a: 1 reqs, 435,907 tokens saved" in text
    assert "model-b: 1 reqs, 189,206 tokens saved" in text
    assert "model-c: 1 reqs, 164 tokens saved" in text


def test_a_tool_only_model_no_longer_reads_zero() -> None:
    """A model whose entire win is deferral used to render as saving nothing."""
    report = PerfReport(
        perf_records=[_record("tool-heavy", before=8_000, saved=0, tool_saved=120_000)]
    )

    text = format_report(report)

    assert "tool-heavy: 1 reqs, 120,000 tokens saved" in text
    # Denominator includes what was withheld — deferred schemas were never in
    # tokens_before — so the percent is 120,000/128,000, not 120,000/8,000.
    assert "(94%)" in text
    assert "· messages 0  · tool schemas 120,000" in text


def test_a_compression_only_model_keeps_its_single_line_shape() -> None:
    report = PerfReport(perf_records=[_record("plain", before=10_000, saved=2_500, tool_saved=0)])

    text = format_report(report)

    assert "plain: 1 reqs, 2,500 tokens saved (25%)" in text
    assert "tool schemas" not in text.split("Per-Model Breakdown")[1]


# --------------------------------------------------------------------------- #
# Dashboard / API (`savings_tracker`)
# --------------------------------------------------------------------------- #
def test_tracker_attributes_deferral_to_the_model(tmp_path) -> None:
    tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
    tracker.record_request(
        model="gpt-5-codex",
        input_tokens=10_000,
        tokens_saved=1_000,
        tool_search_saved=90_000,
    )

    entry = tracker.snapshot()["by_model"]["gpt-5-codex"]

    # The two layers stay separately addressable...
    assert entry["tokens_saved"] == 1_000
    assert entry["tool_tokens_saved"] == 90_000
    # ...and the combined figure is what the percent is computed from.
    assert entry["headline_tokens_saved"] == 91_000
    # 91,000 / (91,000 + 10,000)
    assert entry["savings_percent"] == 90.1


def test_tracker_default_is_unchanged_without_deferral(tmp_path) -> None:
    """Callers that pass no deferral must see exactly the old numbers."""
    tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
    tracker.record_request(model="claude-sonnet-4-6", input_tokens=9_000, tokens_saved=1_000)

    entry = tracker.snapshot()["by_model"]["claude-sonnet-4-6"]

    assert entry["tool_tokens_saved"] == 0
    assert entry["headline_tokens_saved"] == 1_000
    assert entry["savings_percent"] == 10.0


def test_state_written_before_this_field_existed_still_loads() -> None:
    """Backward compatibility: the key is simply absent in older state files."""
    normalized = _normalize_by_model(
        {
            "legacy-model": {
                "requests": 3,
                "tokens_saved": 500,
                "compression_savings_usd": 0.25,
                "total_input_tokens": 4_500,
                "total_input_cost_usd": 1.5,
            }
        }
    )

    assert normalized["legacy-model"]["tool_tokens_saved"] == 0
    assert normalized["legacy-model"]["tokens_saved"] == 500


# --------------------------------------------------------------------------- #
# Dashboard "Per-Model Token Savings" table (`cost.py`)
# --------------------------------------------------------------------------- #
def test_cost_tracker_per_model_counts_both_layers() -> None:
    from headroom.proxy.cost import CostTracker

    tracker = CostTracker()
    tracker.record_tokens("gpt-5-codex", 1_000, 9_000, tool_schema_saved=40_000)

    row = tracker.stats()["per_model"]["gpt-5-codex"]

    assert row["compression_tokens_saved"] == 1_000
    assert row["tool_tokens_saved"] == 40_000
    assert row["tokens_saved"] == 41_000
    # 41,000 / (41,000 + 9,000)
    assert row["reduction_pct"] == 82.0


def test_cost_tracker_shows_a_deferral_only_model_at_all() -> None:
    """Keying the loop off compression alone dropped such a model entirely."""
    from headroom.proxy.cost import CostTracker

    tracker = CostTracker()
    tracker.record_tokens("tool-only", 0, 2_000, tool_schema_saved=18_000)

    stats = tracker.stats()

    assert "tool-only" in stats["per_model"]
    assert stats["per_model"]["tool-only"]["tokens_saved"] == 18_000


def test_cost_tracker_totals_reconcile_with_the_rows() -> None:
    from headroom.proxy.cost import CostTracker

    tracker = CostTracker()
    tracker.record_tokens("model-a", 1_000, 5_000, tool_schema_saved=40_000)
    tracker.record_tokens("model-b", 500, 5_000, tool_schema_saved=0)

    stats = tracker.stats()

    assert stats["total_tokens_saved"] == sum(
        row["tokens_saved"] for row in stats["per_model"].values()
    )
    assert stats["total_compression_tokens_saved"] == 1_500
    assert stats["total_tool_tokens_saved"] == 40_000


def test_cost_tracker_default_call_is_unchanged() -> None:
    """Existing callers that pass no deferral keep the old numbers exactly."""
    from headroom.proxy.cost import CostTracker

    tracker = CostTracker()
    tracker.record_tokens("claude-sonnet-4-6", 2_500, 7_500)

    row = tracker.stats()["per_model"]["claude-sonnet-4-6"]

    assert row["tokens_saved"] == 2_500
    assert row["tool_tokens_saved"] == 0
    assert row["reduction_pct"] == 25.0


# --------------------------------------------------------------------------- #
# The seam itself
# --------------------------------------------------------------------------- #
def test_metrics_forwards_deferral_to_the_tracker(tmp_path) -> None:
    """`record_request` always received the figure; it just never passed it on.

    Pinning this at the seam rather than only at the destination: the tracker
    could be correct in isolation and the dashboard still read zero, which is
    exactly the state that shipped.
    """
    import asyncio

    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
    metrics = PrometheusMetrics(savings_tracker=tracker, stateless=True)

    asyncio.run(
        metrics.record_request(
            provider="openai",
            model="gpt-5-codex",
            input_tokens=6_000,
            output_tokens=100,
            tokens_saved=400,
            latency_ms=12.0,
            tool_search_saved=54_000,
        )
    )

    entry = tracker.snapshot()["by_model"]["gpt-5-codex"]

    assert entry["tokens_saved"] == 400
    assert entry["tool_tokens_saved"] == 54_000
    assert entry["headline_tokens_saved"] == 54_400

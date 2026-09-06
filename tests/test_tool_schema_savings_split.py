"""Tool-schema dollars must be recorded disjointly, not only folded.

Reported by a desktop consumer of the persisted savings state. Since the
attribution unification, ``record_request`` folds the tool-schema layer's
dollars into ``compression_savings_usd`` (lifetime, display_session, and every
history checkpoint) while ``total_tokens_saved`` on the same checkpoint stays
message-only. Any consumer deriving a $/token rate from a checkpoint therefore
reads a figure inflated by ``1 + tool/message``:

- Traffic: message compression 3,299,618 tokens next to 18,435,490 tokens of
  tool-schema deferral (ratio 5.59x) on one real install.
- Implied rate: a $4.99/M blended savings rate became $32.88/M after the fold,
  which exceeds the input list price of every model in the mix.
- Reader impact: consumers that accumulated ``compression_savings_usd`` before
  the fold shipped had the field's meaning widened underneath them.

Same resolution shape as the per-model token fix: keep the folded headline
exactly as it is, and record the layer disjointly beside it --
``tool_tokens_saved`` / ``tool_schema_savings_usd`` on lifetime,
display_session, and checkpoints, with matching ``_delta`` fields on rollup
buckets -- so message-only dollars are recoverable by subtraction.
"""

from __future__ import annotations

import json

import pytest

from headroom.proxy.savings_tracker import SavingsTracker, _normalize_history_entry

MODEL = "claude-opus-5"


def _tracker(tmp_path) -> SavingsTracker:
    return SavingsTracker(
        path=str(tmp_path / "proxy_savings.json"),
        max_history_points=100,
        max_history_age_days=30,
    )


def _record(
    tracker: SavingsTracker,
    *,
    tokens_saved: int = 100,
    tool_search_saved: int = 500,
    compression_usd: float = 1.0,
    tool_usd: float = 5.0,
    timestamp: str = "2026-08-21T09:10:00Z",
) -> None:
    tracker.record_request(
        model=MODEL,
        input_tokens=1_000,
        tokens_saved=tokens_saved,
        tool_search_saved=tool_search_saved,
        estimated_savings_usd={
            "compression": compression_usd,
            "tool_schema": tool_usd,
            "output_shaping": 0.0,
            "provider_cache": 0.0,
        },
        timestamp=timestamp,
    )


def test_lifetime_and_session_record_tool_schema_disjointly(tmp_path):
    tracker = _tracker(tmp_path)
    _record(tracker)

    snapshot = tracker.snapshot()
    for block in (snapshot["lifetime"], snapshot["display_session"]):
        # The folded headline is unchanged: compression + tool_schema.
        assert block["compression_savings_usd"] == 6.0
        # The disjoint record makes message-only dollars recoverable.
        assert block["tool_schema_savings_usd"] == 5.0
        assert block["compression_savings_usd"] - block["tool_schema_savings_usd"] == 1.0
        assert block["tool_tokens_saved"] == 500


def test_checkpoints_carry_the_layer_and_survive_reload(tmp_path):
    tracker = _tracker(tmp_path)
    _record(tracker)
    tracker.flush()

    point = tracker.snapshot()["history"][-1]
    assert point["tool_tokens_saved"] == 500
    assert point["tool_schema_savings_usd"] == 5.0
    # Checkpoint tokens stay message-only; the disjoint dollar field is what
    # lets a consumer pair a consistent numerator with that denominator.
    assert point["total_tokens_saved"] == 100

    # A fresh tracker on the same file must not zero the layer (the output
    # fields historically reset on every restart; these must not).
    reloaded = SavingsTracker(
        path=str(tmp_path / "proxy_savings.json"),
        max_history_points=100,
        max_history_age_days=30,
    )
    lifetime = reloaded.snapshot()["lifetime"]
    assert lifetime["tool_tokens_saved"] == 500
    assert lifetime["tool_schema_savings_usd"] == 5.0


def test_state_written_before_the_fields_existed_defaults_to_zero(tmp_path):
    path = tmp_path / "proxy_savings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "lifetime": {
                    "requests": 3,
                    "tokens_saved": 900,
                    "compression_savings_usd": 0.9,
                    "cache_read_tokens": 0,
                    "cache_savings_usd": 0.0,
                    "total_input_tokens": 5_000,
                    "total_input_cost_usd": 0.05,
                },
                "display_session": {},
                "history": [
                    {
                        "timestamp": "2026-08-20T10:00:00Z",
                        "total_tokens_saved": 900,
                        "compression_savings_usd": 0.9,
                        "total_input_tokens": 5_000,
                        "total_input_cost_usd": 0.05,
                    }
                ],
                "projects": {},
                "by_model": {},
            }
        )
    )

    tracker = SavingsTracker(path=str(path), max_history_points=100, max_history_age_days=30)
    lifetime = tracker.snapshot()["lifetime"]
    assert lifetime["tool_tokens_saved"] == 0
    assert lifetime["tool_schema_savings_usd"] == 0.0
    assert lifetime["tokens_saved"] == 900

    # Legacy tuple-shaped history entries normalize the same way.
    normalized = _normalize_history_entry(["2026-08-20T10:00:00Z", 900, 0.9])
    assert normalized is not None
    assert normalized["tool_tokens_saved"] == 0
    assert normalized["tool_schema_savings_usd"] == 0.0


def test_tool_only_request_appends_a_checkpoint(tmp_path):
    tracker = _tracker(tmp_path)
    # Deferral with zero message compression: real on tool-heavy traffic, and
    # previously invisible to the history because the append gate only looked
    # at message/cache/output deltas.
    _record(tracker, tokens_saved=0, compression_usd=0.0)

    history = tracker.snapshot()["history"]
    assert len(history) == 1
    assert history[-1]["tool_tokens_saved"] == 500
    assert history[-1]["tool_schema_savings_usd"] == 5.0
    assert history[-1]["total_tokens_saved"] == 0


def test_rollups_expose_tool_schema_deltas(tmp_path):
    tracker = _tracker(tmp_path)
    _record(tracker, timestamp="2026-08-21T09:10:00Z")
    _record(tracker, tool_search_saved=250, tool_usd=2.5, timestamp="2026-08-21T09:40:00Z")
    _record(tracker, tool_search_saved=1_000, tool_usd=10.0, timestamp="2026-08-21T10:05:00Z")

    hourly = tracker.history_response()["series"]["hourly"]
    assert [point["timestamp"] for point in hourly] == [
        "2026-08-21T09:00:00Z",
        "2026-08-21T10:00:00Z",
    ]

    first, second = hourly
    # Deltas start from a zero baseline, so the series' first checkpoint
    # contributes its full cumulative -- the same convention as every other
    # rollup delta in this module. Bucket one therefore carries 500 + 250.
    assert first["tool_tokens_saved_delta"] == 750
    assert first["tool_schema_savings_usd_delta"] == 7.5
    assert first["tool_tokens_saved"] == 750
    assert second["tool_tokens_saved_delta"] == 1_000
    assert second["tool_schema_savings_usd_delta"] == 10.0
    assert second["tool_tokens_saved"] == 1_750

    # The subtraction identity the split exists for: folded minus disjoint
    # equals message-only dollars, bucket by bucket (two $1 records, then one).
    assert first["compression_savings_usd_delta"] - first[
        "tool_schema_savings_usd_delta"
    ] == pytest.approx(2.0)
    assert second["compression_savings_usd_delta"] - second[
        "tool_schema_savings_usd_delta"
    ] == pytest.approx(1.0)


def test_rollup_csv_exports_the_delta_columns(tmp_path):
    tracker = _tracker(tmp_path)
    _record(tracker, timestamp="2026-08-21T09:10:00Z")
    _record(tracker, timestamp="2026-08-21T09:40:00Z")

    header = tracker.export_csv(series="hourly").splitlines()[0]
    assert "tool_tokens_saved_delta" in header
    assert "tool_schema_savings_usd_delta" in header

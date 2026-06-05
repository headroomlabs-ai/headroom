"""Tests for `headroom perf --format {json,csv}` (issue #595).

These tests exercise the formatter helpers directly so they don't need
to spin up the proxy or read real log files.
"""

from __future__ import annotations

import csv
import io
import json

from headroom.perf.analyzer import (
    PerfRecord,
    PerfReport,
    TransformRecord,
    format_csv,
    summary_dict,
)


def _make_report() -> PerfReport:
    return PerfReport(
        perf_records=[
            PerfRecord(
                timestamp="2026-06-01 10:00:00,000",
                request_id="hr_001",
                model="claude-opus-4-8",
                num_messages=10,
                tokens_before=1000,
                tokens_after=400,
                tokens_saved=600,
                cache_read=200,
                cache_write=50,
                cache_hit_pct=80,
                optimization_ms=43.1,
            ),
            PerfRecord(
                timestamp="2026-06-01 10:05:00,000",
                request_id="hr_002",
                model="claude-opus-4-8",
                num_messages=20,
                tokens_before=2000,
                tokens_after=600,
                tokens_saved=1400,
                cache_read=400,
                cache_write=100,
                cache_hit_pct=80,
                optimization_ms=55.0,
            ),
            PerfRecord(
                timestamp="2026-06-01 10:10:00,000",
                request_id="hr_003",
                model="gpt-5",
                num_messages=5,
                tokens_before=500,
                tokens_after=200,
                tokens_saved=300,
            ),
        ],
        transform_records=[
            TransformRecord(
                timestamp="2026-06-01 10:00:00,000",
                name="content_router",
                tokens_before=3000,
                tokens_after=1200,
                tokens_saved=1800,
            ),
        ],
        log_files_read=1,
        total_lines_parsed=42,
        requested_hours=24.0,
        oldest_kept_ts="2026-06-01 10:00:00,000",
        newest_kept_ts="2026-06-01 10:10:00,000",
    )


def test_summary_dict_top_level_aggregates() -> None:
    summary = summary_dict(_make_report())
    assert summary["window_hours"] == 24.0
    assert summary["total_requests"] == 3
    assert summary["total_tokens_before"] == 3500
    assert summary["total_tokens_after"] == 1200
    assert summary["tokens_saved"] == 2300
    # 2300 / 3500 == 65.7142...
    assert summary["savings_pct"] == 2300 / 3500 * 100
    # Only two of the three records carry cache info: read=600 write=150, total=750
    assert summary["cache_read_tokens"] == 600
    assert summary["cache_write_tokens"] == 150
    assert summary["cache_hit_pct"] == 600 / 750 * 100


def test_summary_dict_by_model_groups_by_model() -> None:
    summary = summary_dict(_make_report())
    by_model = summary["by_model"]
    assert set(by_model.keys()) == {"claude-opus-4-8", "gpt-5"}

    claude = by_model["claude-opus-4-8"]
    assert claude["requests"] == 2
    assert claude["tokens_before"] == 3000
    assert claude["tokens_after"] == 1000
    assert claude["tokens_saved"] == 2000

    gpt = by_model["gpt-5"]
    assert gpt["requests"] == 1
    assert gpt["tokens_saved"] == 300


def test_summary_dict_by_transform_groups_by_name() -> None:
    summary = summary_dict(_make_report())
    assert summary["by_transform"]["content_router"]["uses"] == 1
    assert summary["by_transform"]["content_router"]["tokens_saved"] == 1800


def test_summary_dict_round_trips_through_json() -> None:
    # Anything non-JSON-serialisable in the summary would blow up here.
    summary = summary_dict(_make_report())
    encoded = json.dumps(summary)
    decoded = json.loads(encoded)
    assert decoded["total_requests"] == 3


def test_summary_dict_handles_empty_report() -> None:
    summary = summary_dict(PerfReport())
    assert summary["total_requests"] == 0
    assert summary["total_tokens_before"] == 0
    assert summary["savings_pct"] == 0.0
    assert summary["cache_hit_pct"] == 0.0
    assert summary["by_model"] == {}
    assert summary["by_transform"] == {}


def test_format_csv_writes_header_and_rows() -> None:
    output = format_csv(_make_report())
    reader = csv.reader(io.StringIO(output))
    rows = list(reader)

    assert rows[0] == [
        "timestamp",
        "request_id",
        "model",
        "num_messages",
        "tokens_before",
        "tokens_after",
        "tokens_saved",
        "cache_read",
        "cache_write",
        "cache_hit_pct",
        "optimization_ms",
    ]
    assert len(rows) == 4  # header + 3 records
    first = rows[1]
    assert first[1] == "hr_001"
    assert first[2] == "claude-opus-4-8"
    assert first[4] == "1000"
    assert first[10] == "43"  # optimization_ms rounded


def test_format_csv_empty_report_writes_header_only() -> None:
    output = format_csv(PerfReport())
    reader = csv.reader(io.StringIO(output))
    rows = list(reader)
    assert len(rows) == 1  # header only
    assert rows[0][0] == "timestamp"

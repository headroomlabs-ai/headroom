"""Tests for `headroom perf --format {text,json,csv}` (issue #595)."""

from __future__ import annotations

import csv
import io
import json

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.perf import analyzer
from headroom.perf.analyzer import (
    PerfRecord,
    PerfReport,
    TransformRecord,
    build_perf_summary,
    perf_records_as_dicts,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _sample_report() -> PerfReport:
    """A small report with two models, cache numbers, and a transform."""
    return PerfReport(
        perf_records=[
            PerfRecord(
                timestamp="2026-06-05 10:00:00,000",
                request_id="hr_1",
                model="claude-sonnet-4.5",
                num_messages=10,
                tokens_before=1000,
                tokens_after=400,
                tokens_saved=600,
                cache_read=800,
                cache_write=200,
                cache_hit_pct=80,
                optimization_ms=12.0,
                transforms=["content_router"],
            ),
            PerfRecord(
                timestamp="2026-06-05 11:00:00,000",
                request_id="hr_2",
                model="claude-opus-4-8",
                num_messages=4,
                tokens_before=1000,
                tokens_after=600,
                tokens_saved=400,
                cache_read=200,
                cache_write=0,
                cache_hit_pct=100,
                optimization_ms=8.0,
                transforms=["content_router"],
            ),
        ],
        transform_records=[
            TransformRecord(
                timestamp="2026-06-05 10:00:00,000",
                name="content_router",
                tokens_before=2000,
                tokens_after=1000,
                tokens_saved=1000,
            ),
        ],
        log_files_read=1,
        total_lines_parsed=42,
        requested_hours=24.0,
        oldest_kept_ts="2026-06-05 10:00:00,000",
        newest_kept_ts="2026-06-05 11:00:00,000",
    )


# ---------------------------------------------------------------------------
# Pure builders
# ---------------------------------------------------------------------------


def test_build_perf_summary_totals_and_pct():
    summary = build_perf_summary(_sample_report())

    assert summary["total_requests"] == 2
    assert summary["total_tokens_before"] == 2000
    assert summary["total_tokens_after"] == 1000
    assert summary["tokens_saved"] == 1000
    # 1000 / 2000 == 50.0%
    assert summary["savings_pct"] == 50.0
    # cache: read 1000, write 200 -> 1000 / 1200 == 83.3%
    assert summary["cache_read_tokens"] == 1000
    assert summary["cache_write_tokens"] == 200
    assert summary["cache_hit_pct"] == 83.3
    assert summary["window_hours"] == 24.0


def test_build_perf_summary_by_model_and_transform():
    summary = build_perf_summary(_sample_report())

    models = {m["model"]: m for m in summary["by_model"]}
    assert set(models) == {"claude-sonnet-4.5", "claude-opus-4-8"}
    assert models["claude-sonnet-4.5"]["tokens_saved"] == 600
    assert models["claude-sonnet-4.5"]["savings_pct"] == 60.0
    assert models["claude-opus-4-8"]["savings_pct"] == 40.0

    assert summary["by_transform"][0]["transform"] == "content_router"
    assert summary["by_transform"][0]["tokens_saved"] == 1000
    assert summary["by_transform"][0]["uses"] == 1


def test_build_perf_summary_empty_report_no_zero_division():
    summary = build_perf_summary(PerfReport(requested_hours=168.0))
    assert summary["total_requests"] == 0
    assert summary["savings_pct"] == 0.0
    assert summary["cache_hit_pct"] == 0.0
    assert summary["by_model"] == []


def test_perf_records_as_dicts_roundtrips_fields():
    dicts = perf_records_as_dicts(_sample_report())
    assert len(dicts) == 2
    assert dicts[0]["request_id"] == "hr_1"
    assert dicts[0]["tokens_saved"] == 600
    # transforms stays a list for JSON consumers
    assert dicts[0]["transforms"] == ["content_router"]


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def _patch_report(monkeypatch, report: PerfReport) -> None:
    monkeypatch.setattr(analyzer, "parse_log_files", lambda last_n_hours=168.0: report)


def test_perf_json_format(runner, monkeypatch):
    _patch_report(monkeypatch, _sample_report())
    result = runner.invoke(main, ["perf", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["savings_pct"] == 50.0
    assert "by_model" in data
    assert data["total_requests"] == 2


def test_perf_json_raw_is_array(runner, monkeypatch):
    _patch_report(monkeypatch, _sample_report())
    result = runner.invoke(main, ["perf", "--format", "json", "--raw"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["request_id"] == "hr_1"


def test_perf_json_raw_preserves_client_field(runner, monkeypatch):
    report = _sample_report()
    report.perf_records[0].client = "codex"
    _patch_report(monkeypatch, report)

    result = runner.invoke(main, ["perf", "--format", "json", "--raw"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data[0]["client"] == "codex"


def test_parse_perf_line_preserves_client_field(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "proxy.log").write_text(
        "2026-06-10 10:00:00,000 - headroom.proxy - INFO - "
        "[hr_codex] PERF model=gpt-5 msgs=3 tok_before=1000 "
        "tok_after=90 tok_saved=910 cache_read=0 cache_write=0 "
        "cache_hit_pct=0 opt_ms=12 transforms=content_router client=codex\n"
    )
    monkeypatch.setattr(analyzer, "LOG_DIR", log_dir)

    report = analyzer.parse_log_files(last_n_hours=0)

    assert len(report.perf_records) == 1
    assert report.perf_records[0].client == "codex"


def test_perf_csv_by_model(runner, monkeypatch):
    _patch_report(monkeypatch, _sample_report())
    result = runner.invoke(main, ["perf", "--format", "csv"])
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(io.StringIO(result.output)))
    assert {r["model"] for r in rows} == {"claude-sonnet-4.5", "claude-opus-4-8"}
    sonnet = next(r for r in rows if r["model"] == "claude-sonnet-4.5")
    assert sonnet["tokens_saved"] == "600"


def test_perf_csv_raw_per_record(runner, monkeypatch):
    report = _sample_report()
    report.perf_records[0].client = "codex"
    _patch_report(monkeypatch, report)
    result = runner.invoke(main, ["perf", "--format", "csv", "--raw"])
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(io.StringIO(result.output)))
    assert len(rows) == 2
    assert rows[0]["request_id"] == "hr_1"
    assert rows[0]["client"] == "codex"
    # transforms flattened to a string cell
    assert rows[0]["transforms"] == "content_router"


def test_perf_text_default_unchanged(runner, monkeypatch):
    _patch_report(monkeypatch, _sample_report())
    result = runner.invoke(main, ["perf"])
    assert result.exit_code == 0, result.output
    assert "Headroom Performance Report" in result.output


def test_perf_rejects_unknown_format(runner, monkeypatch):
    _patch_report(monkeypatch, _sample_report())
    result = runner.invoke(main, ["perf", "--format", "xml"])
    assert result.exit_code != 0


def test_parse_perf_line_preserves_blank_client_field(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(analyzer, "LOG_DIR", logs_dir)
    (logs_dir / "proxy.log").write_text(
        "2026-06-10 10:00:00,000 - headroom.proxy - INFO - [req-blank] PERF "
        "model=gpt-5 msgs=1 tok_before=100 tok_after=50 tok_saved=50 "
        "cache_read=0 cache_write=0 cache_hit_pct=0 opt_ms=1 transforms=test client=\n",
        encoding="utf-8",
    )

    report = analyzer.parse_log_files(last_n_hours=0)

    assert len(report.perf_records) == 1
    assert report.perf_records[0].client == ""

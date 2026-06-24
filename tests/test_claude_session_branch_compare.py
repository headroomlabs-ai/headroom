from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks.claude_session_branch_compare import (
    BranchResult,
    _build_benchmark_command,
    _build_six_way_rows,
    _classify_delta,
    _ref_slug,
    build_compare_markdown,
    write_compare_report,
)


def _branch(label: str, ref: str, commit: str, total_cost: float) -> BranchResult:
    summaries = {
        "baseline": {
            "mode": "baseline",
            "total_cost_usd": total_cost + 1.0,
            "no_cache_total_cost_usd": total_cost + 5.0,
            "forwarded_input_tokens": 1_200,
            "cache_read_tokens": 800,
            "cache_write_tokens": 200,
            "regular_input_tokens": 400,
            "output_tokens": 120,
            "cache_bust_turns": 1,
            "ttl_expiry_turns": 2,
            "prompt_window_with_cache": 1_200,
            "prompt_window_without_cache_reads": 400,
        },
        "token": {
            "mode": "token",
            "total_cost_usd": total_cost,
            "no_cache_total_cost_usd": total_cost + 3.0,
            "forwarded_input_tokens": 900,
            "cache_read_tokens": 700,
            "cache_write_tokens": 150,
            "regular_input_tokens": 200,
            "output_tokens": 120,
            "cache_bust_turns": 4,
            "ttl_expiry_turns": 2,
            "prompt_window_with_cache": 900,
            "prompt_window_without_cache_reads": 200,
        },
        "cache": {
            "mode": "cache",
            "total_cost_usd": total_cost + 0.5,
            "no_cache_total_cost_usd": total_cost + 4.0,
            "forwarded_input_tokens": 950,
            "cache_read_tokens": 760,
            "cache_write_tokens": 180,
            "regular_input_tokens": 190,
            "output_tokens": 120,
            "cache_bust_turns": 1,
            "ttl_expiry_turns": 2,
            "prompt_window_with_cache": 950,
            "prompt_window_without_cache_reads": 190,
        },
    }
    return BranchResult(
        ref=ref,
        label=label,
        commit=commit,
        summary=f"{label} summary",
        dataset={
            "projects": 3,
            "sessions": 7,
            "requests": 80,
            "sampled_requests": 80,
            "sampling_note": "Most recent 10 turns per session",
        },
        observed={"cache_ratio_pct": 97.0},
        summaries=summaries,
        winners={
            "total_cost": "token",
            "no_cache_total_cost": "token",
            "window_with_cache": "token",
            "window_without_cache_reads": "cache",
        },
        output_dir=f"benchmark_results/{label}",
    )


def test_ref_slug_normalizes_refs() -> None:
    assert _ref_slug("upstream/main") == "upstream-main"
    assert _ref_slug("feature/cache.fix") == "feature-cache-fix"


def test_build_benchmark_command_includes_knobs() -> None:
    command = _build_benchmark_command(
        python_executable=sys.executable,
        script_path=Path("benchmarks") / "claude_session_mode_benchmark.py",
        root=Path.home() / ".claude" / "projects",
        output_dir=Path("benchmark_results") / "pr",
        max_sessions=5,
        recent_turns_per_session=200,
        cache_ttl_minutes=5,
        cache_write_multiplier=1.25,
        workers=1,
    )

    assert command[0] == sys.executable
    assert "--max-sessions" in command
    assert "--recent-turns-per-session" in command
    assert "--workers" in command


def test_build_compare_markdown_surfaces_branch_deltas() -> None:
    left = _branch("main", "upstream/main", "abc123456789", 12.0)
    right = _branch("pr", "HEAD", "def987654321", 11.0)

    markdown = build_compare_markdown(left, right)

    assert "Claude Session Branch Comparison" in markdown
    assert "`main`" not in markdown
    assert "main picks" not in markdown
    assert "Delta (pr - main)" in markdown
    assert "| token | Total Cost | $12.00 | $11.00 | $-1.00 |" in markdown


def test_write_compare_report_persists_payload(tmp_path: Path) -> None:
    left = _branch("main", "upstream/main", "abc123456789", 12.0)
    right = _branch("pr", "HEAD", "def987654321", 11.0)

    md_path, json_path, html_path = write_compare_report(tmp_path, left, right)

    assert md_path.exists()
    assert html_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["left"]["ref"] == "upstream/main"
    assert payload["right"]["label"] == "pr"
    assert payload["right_winners"]["total_cost"] == "token"


def test_branch_delta_classification_uses_metric_direction() -> None:
    assert _classify_delta("total_cost_usd", -1.0) == "assist"
    assert _classify_delta("cache_read_tokens", 10.0) == "assist"
    assert _classify_delta("cache_write_tokens", 5.0) == "harm"
    assert _classify_delta("output_tokens", 0.0) == "no_change"


def test_six_way_rows_cover_both_branches_and_modes() -> None:
    left = _branch("main", "upstream/main", "abc123456789", 12.0)
    right = _branch("pr", "HEAD", "def987654321", 11.0)

    rows = _build_six_way_rows(left, right)

    assert len(rows) == 6
    assert rows[0]["branch"] == "main"
    assert rows[0]["mode"] == "baseline"
    assert any(row["branch"] == "pr" and row["mode"] == "token" for row in rows)
    assert any(
        row["branch"] == "main"
        and row["mode"] == "token"
        and row["paid_input_delta_vs_branch_baseline"] == -200
        for row in rows
    )

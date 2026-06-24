#!/usr/bin/env python3
"""Compare Claude session mode simulations across two git refs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.claude_session_mode_benchmark import (
    IMPACT_DIRECTION,
    OUTPUT_JSON,
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    format_currency,
)

DEFAULT_OUTPUT_DIR = Path("benchmark_results") / "branch_compare"


@dataclass
class BranchResult:
    ref: str
    label: str
    commit: str
    summary: str
    dataset: dict[str, Any]
    observed: dict[str, Any]
    summaries: dict[str, dict[str, Any]]
    winners: dict[str, str]
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-ref", default="upstream/main")
    parser.add_argument("--right-ref", default="HEAD")
    parser.add_argument("--left-label", default="main")
    parser.add_argument("--right-label", default="pr")
    parser.add_argument("--root", type=Path, default=Path.home() / ".claude" / "projects")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--recent-turns-per-session", type=int, default=None)
    parser.add_argument("--cache-ttl-minutes", type=int, default=5)
    parser.add_argument("--cache-write-multiplier", type=float, default=1.25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use inside each worktree.",
    )
    parser.add_argument(
        "--keep-worktrees",
        action="store_true",
        help="Do not remove temporary worktrees after the comparison run.",
    )
    return parser.parse_args()


def _run_git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _ref_slug(ref: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in ref).strip("-").lower() or "ref"


def _branch_output_dir(base: Path, label: str) -> Path:
    return base / _ref_slug(label)


def _comparison_paths(base: Path) -> tuple[Path, Path, Path]:
    return (
        base / "claude_session_branch_compare.md",
        base / "claude_session_branch_compare.json",
        base / "claude_session_branch_compare.html",
    )


def _mode_metric(branch: BranchResult, mode: str, field: str) -> float:
    summary = branch.summaries[mode]
    if field == "no_cache_total_cost_usd":
        if "no_cache_total_cost_usd" in summary:
            value = summary["no_cache_total_cost_usd"]
        else:
            value = (
                float(summary["paid_input_cost_usd"])
                + (float(summary["cache_read_cost_usd"]) * 10.0)
                + float(summary["paid_output_cost_usd"])
            )
    elif field == "prompt_window_with_cache":
        value = float(summary["forwarded_input_tokens"])
    elif field == "prompt_window_without_cache_reads":
        value = float(summary["forwarded_input_tokens"]) - float(summary["cache_read_tokens"])
    else:
        value = summary[field]
    if isinstance(value, bool):
        return float(value)
    return float(value)


def _delta(left: float, right: float) -> float:
    return right - left


def _classify_delta(field: str, delta: float) -> str:
    direction = IMPACT_DIRECTION.get(field, "same")
    tolerance = 1e-9
    if abs(delta) <= tolerance:
        return "no_change"
    if direction == "lower":
        return "assist" if delta < 0 else "harm"
    if direction == "higher":
        return "assist" if delta > 0 else "harm"
    return "harm"


def _build_benchmark_command(
    python_executable: str,
    script_path: Path,
    root: Path,
    output_dir: Path,
    max_sessions: int | None,
    recent_turns_per_session: int | None,
    cache_ttl_minutes: int,
    cache_write_multiplier: float,
    workers: int,
) -> list[str]:
    command = [
        python_executable,
        str(script_path),
        "--root",
        str(root),
        "--output-dir",
        str(output_dir),
        "--cache-ttl-minutes",
        str(cache_ttl_minutes),
        "--cache-write-multiplier",
        str(cache_write_multiplier),
        "--workers",
        str(workers),
    ]
    if max_sessions is not None:
        command.extend(["--max-sessions", str(max_sessions)])
    if recent_turns_per_session is not None:
        command.extend(["--recent-turns-per-session", str(recent_turns_per_session)])
    return command


def _load_branch_result(
    repo_root: Path,
    ref: str,
    label: str,
    branch_output_dir: Path,
) -> BranchResult:
    payload = json.loads((branch_output_dir / OUTPUT_JSON).read_text(encoding="utf-8"))
    commit = _run_git(["rev-parse", ref], repo_root)
    summary = _run_git(["show", "-s", "--format=%s", ref], repo_root)
    return BranchResult(
        ref=ref,
        label=label,
        commit=commit,
        summary=summary,
        dataset=payload["dataset"],
        observed=payload["observed"],
        summaries=payload["summaries"],
        winners=payload["winners"],
        output_dir=str(branch_output_dir),
    )


def _run_branch_benchmark(
    repo_root: Path,
    ref: str,
    label: str,
    args: argparse.Namespace,
    worktree_root: Path,
) -> BranchResult:
    worktree_dir = worktree_root / _ref_slug(label)
    branch_output_dir = _branch_output_dir(args.output_dir, label)
    branch_output_dir.mkdir(parents=True, exist_ok=True)
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir)
    _run_git(["worktree", "add", "--detach", str(worktree_dir), ref], repo_root)
    try:
        command = _build_benchmark_command(
            python_executable=args.python,
            script_path=worktree_dir / "benchmarks" / "claude_session_mode_benchmark.py",
            root=args.root,
            output_dir=branch_output_dir,
            max_sessions=args.max_sessions,
            recent_turns_per_session=args.recent_turns_per_session,
            cache_ttl_minutes=args.cache_ttl_minutes,
            cache_write_multiplier=args.cache_write_multiplier,
            workers=args.workers,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(worktree_dir), env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        )
        subprocess.run(command, cwd=worktree_dir, check=True, env=env)
        return _load_branch_result(repo_root, ref, label, branch_output_dir)
    finally:
        if not args.keep_worktrees:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)],
                cwd=repo_root,
                check=True,
            )


def _winner_line(metric: str, left: BranchResult, right: BranchResult) -> str:
    left_winner = left.winners[metric]
    right_winner = right.winners[metric]
    if left_winner == right_winner:
        return f"- {metric}: both pick `{left_winner}`"
    return (
        f"- {metric}: `{left.label}` picks `{left_winner}`, `{right.label}` picks `{right_winner}`"
    )


def _build_six_way_rows(
    left: BranchResult, right: BranchResult
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for branch in (left, right):
        for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
            summary = branch.summaries[mode]
            cost_delta = _mode_metric(branch, mode, "total_cost_usd") - _mode_metric(
                branch, "baseline", "total_cost_usd"
            )
            window_delta = int(
                _mode_metric(branch, mode, "prompt_window_with_cache")
                - _mode_metric(branch, "baseline", "prompt_window_with_cache")
            )
            read_delta = int(
                _mode_metric(branch, mode, "cache_read_tokens")
                - _mode_metric(branch, "baseline", "cache_read_tokens")
            )
            write_delta = int(
                _mode_metric(branch, mode, "cache_write_tokens")
                - _mode_metric(branch, "baseline", "cache_write_tokens")
            )
            paid_input_delta = int(
                _mode_metric(branch, mode, "regular_input_tokens")
                - _mode_metric(branch, "baseline", "regular_input_tokens")
            )
            rows.append(
                {
                    "branch": branch.label,
                    "mode": mode,
                    "forwarded_input_tokens": int(summary["forwarded_input_tokens"]),
                    "cache_read_tokens": int(summary["cache_read_tokens"]),
                    "cache_write_tokens": int(summary["cache_write_tokens"]),
                    "regular_input_tokens": int(summary["regular_input_tokens"]),
                    "output_tokens": int(summary["output_tokens"]),
                    "total_cost_usd": float(summary["total_cost_usd"]),
                    "cost_delta_vs_branch_baseline": cost_delta,
                    "window_delta_vs_branch_baseline": window_delta,
                    "cache_read_delta_vs_branch_baseline": read_delta,
                    "cache_write_delta_vs_branch_baseline": write_delta,
                    "paid_input_delta_vs_branch_baseline": paid_input_delta,
                    "is_branch_winner": "yes" if branch.winners["total_cost"] == mode else "no",
                }
            )
    return rows


def build_compare_markdown(left: BranchResult, right: BranchResult) -> str:
    six_way_rows = _build_six_way_rows(left, right)
    lines = [
        "# Claude Session Branch Comparison",
        "",
        "## Branches",
        "",
        f"- {left.label}: `{left.ref}` @ `{left.commit[:12]}` - {left.summary}",
        f"- {right.label}: `{right.ref}` @ `{right.commit[:12]}` - {right.summary}",
        "",
        "## Dataset",
        "",
        f"- Projects: {right.dataset['projects']}",
        f"- Sessions: {right.dataset['sessions']}",
        f"- Requests: {right.dataset['requests']}",
        f"- Sampled requests: {right.dataset.get('sampled_requests', 0)}",
        f"- Sampling: {right.dataset.get('sampling_note', 'Full sessions')}",
        "",
        "## Winner Comparison",
        "",
        _winner_line("total_cost", left, right),
        _winner_line("no_cache_total_cost", left, right),
        _winner_line("window_with_cache", left, right),
        _winner_line("window_without_cache_reads", left, right),
        "",
        "## Six-Way Mode Matrix",
        "",
        "| Branch | Mode | Forwarded Input | Cache Read | Cache Write | Paid Input | Paid Output | Total Cost | Cost Δ vs Branch Baseline | Window Δ vs Branch Baseline | Winner |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *[
            "| "
            + " | ".join(
                [
                    str(row["branch"]),
                    str(row["mode"]),
                    f"{int(row['forwarded_input_tokens']):,}",
                    f"{int(row['cache_read_tokens']):,}",
                    f"{int(row['cache_write_tokens']):,}",
                    f"{int(row['regular_input_tokens']):,}",
                    f"{int(row['output_tokens']):,}",
                    format_currency(float(row["total_cost_usd"])),
                    format_currency(float(row["cost_delta_vs_branch_baseline"])),
                    f"{int(row['window_delta_vs_branch_baseline']):,}",
                    str(row["is_branch_winner"]),
                ]
            )
            + " |"
            for row in six_way_rows
        ],
        "",
        "## Mode Deltas",
        "",
        f"| Mode | Metric | {left.label} | {right.label} | Delta ({right.label} - {left.label}) | Classification |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    metrics = [
        ("total_cost_usd", "Total Cost", format_currency),
        ("no_cache_total_cost_usd", "No-Cache Total Cost", format_currency),
        ("forwarded_input_tokens", "Forwarded Input Tokens", lambda v: f"{int(v):,}"),
        ("cache_read_tokens", "Cache Read Tokens", lambda v: f"{int(v):,}"),
        ("cache_write_tokens", "Cache Write Tokens", lambda v: f"{int(v):,}"),
        ("cache_bust_turns", "Cache Bust Turns", lambda v: f"{int(v):,}"),
        ("ttl_expiry_turns", "TTL Expiry Turns", lambda v: f"{int(v):,}"),
        ("prompt_window_with_cache", "Window With Cache", lambda v: f"{int(v):,}"),
        (
            "prompt_window_without_cache_reads",
            "Window Without Cache Reads",
            lambda v: f"{int(v):,}",
        ),
    ]
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        for field, label, formatter in metrics:
            left_value = _mode_metric(left, mode, field)
            right_value = _mode_metric(right, mode, field)
            delta = _delta(left_value, right_value)
            delta_text = format_currency(delta) if "cost" in field else f"{int(delta):,}"
            classification = _classify_delta(field, delta)
            lines.append(
                f"| {mode} | {label} | {formatter(left_value)} | {formatter(right_value)} | {delta_text} | {classification} |"
            )
    return "\n".join(lines)


def build_compare_html(left: BranchResult, right: BranchResult) -> str:
    six_way_rows = []
    for row in _build_six_way_rows(left, right):
        six_way_rows.append(
            "<tr>"
            f"<td>{row['branch']}</td>"
            f"<td><span class='pill'>{row['mode']}</span></td>"
            f"<td>{int(row['forwarded_input_tokens']):,}</td>"
            f"<td>{int(row['cache_read_tokens']):,}</td>"
            f"<td>{int(row['cache_write_tokens']):,}</td>"
            f"<td>{int(row['regular_input_tokens']):,}</td>"
            f"<td>{int(row['output_tokens']):,}</td>"
            f"<td>{format_currency(float(row['total_cost_usd']))}</td>"
            f"<td>{format_currency(float(row['cost_delta_vs_branch_baseline']))}</td>"
            f"<td>{int(row['window_delta_vs_branch_baseline']):,}</td>"
            f"<td>{row['is_branch_winner']}</td>"
            "</tr>"
        )
    cards = []
    for branch in (left, right):
        cards.append(
            "<div class='card'>"
            f"<div class='eyebrow'>{branch.label}</div>"
            f"<h2>{branch.ref}</h2>"
            f"<p><code>{branch.commit[:12]}</code></p>"
            f"<p>{branch.summary}</p>"
            "<div class='winner-grid'>"
            f"<div><span>Total Cost</span><strong>{branch.winners['total_cost']}</strong></div>"
            f"<div><span>No Cache</span><strong>{branch.winners['no_cache_total_cost']}</strong></div>"
            f"<div><span>Window + Cache</span><strong>{branch.winners['window_with_cache']}</strong></div>"
            "<div><span>Window - Reads</span>"
            f"<strong>{branch.winners['window_without_cache_reads']}</strong></div>"
            "</div>"
            "</div>"
        )
    rows = []
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        for field, label in (
            ("total_cost_usd", "Total Cost"),
            ("no_cache_total_cost_usd", "No-Cache Total Cost"),
            ("forwarded_input_tokens", "Forwarded Input Tokens"),
            ("cache_read_tokens", "Cache Read Tokens"),
            ("cache_write_tokens", "Cache Write Tokens"),
            ("cache_bust_turns", "Cache Bust Turns"),
            ("prompt_window_with_cache", "Window With Cache"),
            ("prompt_window_without_cache_reads", "Window Without Cache Reads"),
        ):
            left_value = _mode_metric(left, mode, field)
            right_value = _mode_metric(right, mode, field)
            delta = _delta(left_value, right_value)
            is_cost = "cost" in field
            formatter = format_currency if is_cost else (lambda v: f"{int(v):,}")
            delta_text = format_currency(delta) if is_cost else f"{int(delta):,}"
            delta_class = "pos" if delta > 0 else "neg" if delta < 0 else "neutral"
            classification = _classify_delta(field, delta)
            rows.append(
                "<tr>"
                f"<td><span class='pill'>{mode}</span></td>"
                f"<td>{label}</td>"
                f"<td>{formatter(left_value)}</td>"
                f"<td>{formatter(right_value)}</td>"
                f"<td class='{delta_class}'>{delta_text}</td>"
                f"<td>{classification}</td>"
                "</tr>"
            )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Claude Session Branch Comparison</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --fg: #020617;
      --muted: #475569;
      --card: #ffffff;
      --line: #e2e8f0;
      --soft: #f1f5f9;
      --accent: #0f172a;
      --accent-soft: #e2e8f0;
      --good: #166534;
      --bad: #991b1b;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
      --radius: 16px;
      --font: "Geist", "Segoe UI", system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font); }}
    .shell {{ max-width: 1280px; margin: 0 auto; padding: 32px 16px 56px; }}
    .hero, .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 24px; }}
    .eyebrow {{ color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }}
    h1, h2 {{ margin: 0; letter-spacing: -0.03em; }}
    p {{ color: var(--muted); line-height: 1.5; }}
    .grid {{ display: grid; gap: 16px; margin-top: 16px; }}
    .two {{ grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
    .card {{ padding: 20px; }}
    .winner-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .winner-grid span {{ display: block; color: var(--muted); font-size: 12px; }}
    .winner-grid strong {{ display: block; margin-top: 4px; font-size: 16px; }}
    .table-card {{ margin-top: 16px; padding: 0; overflow: hidden; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ background: var(--soft); color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .pill {{
      display: inline-flex; align-items: center; border-radius: 999px; padding: 4px 10px;
      background: var(--accent-soft); color: var(--accent); font-size: 12px; font-weight: 600;
    }}
    .pos {{ color: var(--bad); font-weight: 600; }}
    .neg {{ color: var(--good); font-weight: 600; }}
    .neutral {{ color: var(--muted); }}
    code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Branch Comparison</div>
      <h1>Claude Session Mode Simulation</h1>
      <p>Same local Claude transcript corpus. Same simulation knobs. Two git refs. This report isolates code-level behavior changes between the branches.</p>
      <div class="grid two">
        {"".join(cards)}
      </div>
    </section>
    <section class="card table-card">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Branch</th>
              <th>Mode</th>
              <th>Forwarded Input</th>
              <th>Cache Read</th>
              <th>Cache Write</th>
              <th>Paid Input</th>
              <th>Paid Output</th>
              <th>Total Cost</th>
              <th>Cost Δ vs Branch Baseline</th>
              <th>Window Δ vs Branch Baseline</th>
              <th>Winner</th>
            </tr>
          </thead>
          <tbody>
            {"".join(six_way_rows)}
          </tbody>
        </table>
      </div>
    </section>
    <section class="card table-card">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Mode</th>
              <th>Metric</th>
              <th>{left.label}</th>
              <th>{right.label}</th>
              <th>Delta</th>
              <th>Classification</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows)}
          </tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>"""


def write_compare_report(
    output_dir: Path,
    left: BranchResult,
    right: BranchResult,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path, html_path = _comparison_paths(output_dir)
    md_path.write_text(build_compare_markdown(left, right), encoding="utf-8")
    html_path.write_text(build_compare_html(left, right), encoding="utf-8")
    payload = {
        "left": asdict(left),
        "right": asdict(right),
        "left_winners": left.winners,
        "right_winners": right.winners,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md_path, json_path, html_path


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if not args.output_dir.is_absolute():
        args.output_dir = (repo_root / args.output_dir).resolve()
    if not args.root.is_absolute():
        args.root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worktree_root = Path(tempfile.mkdtemp(prefix="headroom-branch-compare-"))
    try:
        left = _run_branch_benchmark(repo_root, args.left_ref, args.left_label, args, worktree_root)
        right = _run_branch_benchmark(
            repo_root, args.right_ref, args.right_label, args, worktree_root
        )
        md_path, json_path, html_path = write_compare_report(args.output_dir, left, right)
        print(f"Compared {left.label} ({left.ref}) vs {right.label} ({right.ref})")
        print(f"Markdown report: {md_path}")
        print(f"JSON report: {json_path}")
        print(f"HTML report: {html_path}")
        return 0
    finally:
        if args.keep_worktrees:
            print(f"Retained worktrees under {worktree_root}")
        else:
            shutil.rmtree(worktree_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

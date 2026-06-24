"""Report card generator for evaluation suite results.

Produces publishable Markdown, JSON, and HTML reports showing
baseline vs Headroom accuracy + token savings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkRunResult:
    """Result from a single benchmark run (unified across all runner types)."""

    name: str
    category: str  # "reasoning", "factual", "knowledge", "code", "qa", "tool_use", "lossless"
    tier: int

    # For lm-eval (standard benchmarks): baseline vs headroom scores
    baseline_score: float | None = None
    headroom_score: float | None = None
    delta: float | None = None

    # For before-after / compression-only: accuracy preservation rate
    accuracy_rate: float | None = None

    # Compression metrics (always present)
    avg_compression_ratio: float = 0.0
    tokens_saved: int = 0

    # Meta
    n_samples: int = 0
    model: str = ""
    metric_name: str = ""
    duration_seconds: float = 0.0
    passed: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "tier": self.tier,
            "n_samples": self.n_samples,
            "model": self.model,
            "metric": self.metric_name,
            "passed": self.passed,
            "avg_compression_ratio": round(self.avg_compression_ratio, 4),
            "tokens_saved": self.tokens_saved,
            "duration_seconds": round(self.duration_seconds, 2),
        }
        if self.baseline_score is not None:
            d["baseline_score"] = round(self.baseline_score, 4)
        if self.headroom_score is not None:
            d["headroom_score"] = round(self.headroom_score, 4)
        if self.delta is not None:
            d["delta"] = round(self.delta, 4)
        if self.accuracy_rate is not None:
            d["accuracy_rate"] = round(self.accuracy_rate, 4)
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class SuiteResult:
    """Complete evaluation suite results."""

    model: str
    tiers_run: list[int]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    benchmarks: list[BenchmarkRunResult] = field(default_factory=list)

    @property
    def standard_benchmarks(self) -> list[BenchmarkRunResult]:
        """Benchmarks with baseline vs headroom comparison."""
        return [b for b in self.benchmarks if b.baseline_score is not None]

    @property
    def compression_benchmarks(self) -> list[BenchmarkRunResult]:
        """Benchmarks measuring compression accuracy."""
        return [b for b in self.benchmarks if b.accuracy_rate is not None]

    @property
    def all_passed(self) -> bool:
        return all(b.passed for b in self.benchmarks)

    @property
    def pass_rate(self) -> float:
        if not self.benchmarks:
            return 0.0
        return sum(1 for b in self.benchmarks if b.passed) / len(self.benchmarks)

    @property
    def avg_delta(self) -> float:
        deltas = [b.delta for b in self.benchmarks if b.delta is not None]
        return sum(deltas) / len(deltas) if deltas else 0.0

    @property
    def avg_compression(self) -> float:
        ratios = [b.avg_compression_ratio for b in self.benchmarks if b.avg_compression_ratio > 0]
        return sum(ratios) / len(ratios) if ratios else 0.0

    @property
    def total_tokens_saved(self) -> int:
        return sum(b.tokens_saved for b in self.benchmarks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "timestamp": self.timestamp,
            "model": self.model,
            "tiers_run": self.tiers_run,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "summary": {
                "total_benchmarks": len(self.benchmarks),
                "passed": sum(1 for b in self.benchmarks if b.passed),
                "failed": sum(1 for b in self.benchmarks if not b.passed),
                "all_passed": self.all_passed,
                "avg_delta": round(self.avg_delta, 4),
                "avg_compression_ratio": round(self.avg_compression, 4),
                "total_tokens_saved": self.total_tokens_saved,
            },
            "benchmarks": [b.to_dict() for b in self.benchmarks],
        }


def generate_markdown(result: SuiteResult) -> str:
    """Generate publishable Markdown report card."""
    lines = [
        "## Headroom Accuracy Report Card",
        "",
        f"Model: `{result.model}` | Date: {result.timestamp[:10]} "
        f"| Suite Cost: ${result.total_cost_usd:.2f} "
        f"| Duration: {result.total_duration_seconds:.0f}s",
        "",
    ]

    # Standard benchmarks table
    standard = result.standard_benchmarks
    if standard:
        lines.extend(
            [
                '### Standard Benchmarks -- "No Accuracy Loss"',
                "",
                "| Benchmark | Category | N | Baseline | Headroom | Delta | Tokens Saved | Status |",
                "|-----------|----------|---|----------|----------|-------|--------------|--------|",
            ]
        )
        for b in standard:
            delta_str = f"{b.delta:+.3f}" if b.delta is not None else "N/A"
            baseline_str = f"{b.baseline_score:.3f}" if b.baseline_score is not None else "N/A"
            headroom_str = f"{b.headroom_score:.3f}" if b.headroom_score is not None else "N/A"
            comp_str = f"{b.avg_compression_ratio:.0%}" if b.avg_compression_ratio > 0 else "--"
            status = "PASS" if b.passed else "FAIL"
            lines.append(
                f"| {b.name} | {b.category} | {b.n_samples} | {baseline_str} | "
                f"{headroom_str} | {delta_str} | {comp_str} | {status} |"
            )
        lines.append("")

    # Compression benchmarks table
    compression = result.compression_benchmarks
    if compression:
        lines.extend(
            [
                '### Compression Benchmarks -- "Big Savings, Accuracy Preserved"',
                "",
                "| Benchmark | Category | N | Accuracy | Tokens Saved | Status |",
                "|-----------|----------|---|----------|--------------|--------|",
            ]
        )
        for b in compression:
            acc_str = f"{b.accuracy_rate:.1%}" if b.accuracy_rate is not None else "N/A"
            comp_str = f"{b.avg_compression_ratio:.0%}" if b.avg_compression_ratio > 0 else "--"
            status = "PASS" if b.passed else "FAIL"
            lines.append(
                f"| {b.name} | {b.category} | {b.n_samples} | {acc_str} | {comp_str} | {status} |"
            )
        lines.append("")

    # Verdict
    total = len(result.benchmarks)
    passed = sum(1 for b in result.benchmarks if b.passed)
    lines.extend(
        [
            f"**VERDICT: {passed}/{total} PASS** | "
            f"Avg delta: {result.avg_delta:+.3f} | "
            f"Avg savings: {result.avg_compression:.0%}",
            "",
        ]
    )

    return "\n".join(lines)


def generate_json(result: SuiteResult) -> str:
    """Generate JSON report for CI regression tracking."""
    return json.dumps(result.to_dict(), indent=2)


def generate_html(result: SuiteResult) -> str:
    """Generate HTML report for docs/presentations."""
    standard = result.standard_benchmarks
    compression = result.compression_benchmarks
    total = len(result.benchmarks)
    passed = sum(1 for b in result.benchmarks if b.passed)
    verdict_class = "pass" if result.all_passed else "fail"

    standard_rows = ""
    for b in standard:
        delta_str = f"{b.delta:+.3f}" if b.delta is not None else "N/A"
        baseline_str = f"{b.baseline_score:.3f}" if b.baseline_score is not None else "N/A"
        headroom_str = f"{b.headroom_score:.3f}" if b.headroom_score is not None else "N/A"
        comp_str = f"{b.avg_compression_ratio:.0%}" if b.avg_compression_ratio > 0 else "--"
        status = "PASS" if b.passed else "FAIL"
        status_class = "pass" if b.passed else "fail"
        standard_rows += f"""        <tr>
            <td>{b.name}</td><td>{b.category}</td><td>{b.n_samples}</td>
            <td>{baseline_str}</td><td>{headroom_str}</td><td>{delta_str}</td>
            <td>{comp_str}</td>
            <td><span class="badge {status_class}">{status}</span></td>
        </tr>\n"""

    compression_rows = ""
    for b in compression:
        acc_str = f"{b.accuracy_rate:.1%}" if b.accuracy_rate is not None else "N/A"
        comp_str = f"{b.avg_compression_ratio:.0%}" if b.avg_compression_ratio > 0 else "--"
        status = "PASS" if b.passed else "FAIL"
        status_class = "pass" if b.passed else "fail"
        compression_rows += f"""        <tr>
            <td>{b.name}</td><td>{b.category}</td><td>{b.n_samples}</td>
            <td>{acc_str}</td><td>{comp_str}</td>
            <td><span class="badge {status_class}">{status}</span></td>
        </tr>\n"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Headroom Accuracy Report Card</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 960px; margin: 40px auto; padding: 20px; background: #f8f9fa;
        }}
        h1 {{ color: #1a1a2e; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        .meta {{ color: #666; font-size: 14px; margin-bottom: 24px; }}
        .verdict {{
            font-size: 18px; font-weight: 700; padding: 16px 24px;
            border-radius: 8px; margin: 20px 0; display: inline-block;
        }}
        .verdict.pass {{ background: #dcfce7; color: #166534; }}
        .verdict.fail {{ background: #fee2e2; color: #991b1b; }}
        .metrics {{
            display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0;
        }}
        .metric {{
            background: white; padding: 16px 20px; border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        .metric .value {{ font-size: 24px; font-weight: 700; }}
        .metric .label {{ font-size: 12px; color: #666; margin-top: 4px; }}
        table {{
            width: 100%; border-collapse: collapse; background: white;
            border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            margin: 16px 0;
        }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; font-size: 13px; }}
        .badge {{
            display: inline-block; padding: 3px 10px; border-radius: 20px;
            font-size: 12px; font-weight: 600;
        }}
        .badge.pass {{ background: #dcfce7; color: #166534; }}
        .badge.fail {{ background: #fee2e2; color: #991b1b; }}
        footer {{
            margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd;
            color: #888; font-size: 13px;
        }}
    </style>
</head>
<body>
    <h1>Headroom Accuracy Report Card</h1>
    <div class="meta">
        Model: <code>{result.model}</code> |
        Date: {result.timestamp[:10]} |
        Cost: ${result.total_cost_usd:.2f} |
        Duration: {result.total_duration_seconds:.0f}s
    </div>

    <div class="verdict {verdict_class}">
        {passed}/{total} PASS &mdash;
        Avg delta: {result.avg_delta:+.3f} |
        Avg savings: {result.avg_compression:.0%}
    </div>

    <div class="metrics">
        <div class="metric"><div class="value">{total}</div><div class="label">Benchmarks</div></div>
        <div class="metric"><div class="value {verdict_class}">{passed}/{total}</div><div class="label">Passed</div></div>
        <div class="metric"><div class="value">{result.avg_delta:+.3f}</div><div class="label">Avg Delta</div></div>
        <div class="metric"><div class="value">{result.avg_compression:.0%}</div><div class="label">Avg Savings</div></div>
        <div class="metric"><div class="value">{result.total_tokens_saved:,}</div><div class="label">Tokens Saved</div></div>
    </div>

    {"<h2>Standard Benchmarks</h2>" if standard else ""}
    {"<table><tr><th>Benchmark</th><th>Category</th><th>N</th><th>Baseline</th><th>Headroom</th><th>Delta</th><th>Saved</th><th>Status</th></tr>" if standard else ""}
    {standard_rows}
    {"</table>" if standard else ""}

    {"<h2>Compression Benchmarks</h2>" if compression else ""}
    {"<table><tr><th>Benchmark</th><th>Category</th><th>N</th><th>Accuracy</th><th>Saved</th><th>Status</th></tr>" if compression else ""}
    {compression_rows}
    {"</table>" if compression else ""}

    <footer>Generated by <strong>Headroom Evaluation Framework</strong> | <code>pip install headroom-ai[evals]</code></footer>
</body>
</html>"""


def save_reports(result: SuiteResult, output_dir: str | Path) -> dict[str, Path]:
    """Save all report formats to a directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Markdown
    md_path = output_dir / "report_card.md"
    md_path.write_text(generate_markdown(result))
    paths["markdown"] = md_path

    # JSON
    json_path = output_dir / "results.json"
    json_path.write_text(generate_json(result))
    paths["json"] = json_path

    # HTML
    html_path = output_dir / "report.html"
    html_path.write_text(generate_html(result))
    paths["html"] = html_path

    return paths

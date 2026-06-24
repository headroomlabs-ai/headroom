"""Report generation for evaluation results."""

from headroom.evals.reports.report_card import (
    BenchmarkRunResult,
    SuiteResult,
    generate_html,
    generate_json,
    generate_markdown,
    save_reports,
)

__all__ = [
    "BenchmarkRunResult",
    "SuiteResult",
    "generate_html",
    "generate_json",
    "generate_markdown",
    "save_reports",
]

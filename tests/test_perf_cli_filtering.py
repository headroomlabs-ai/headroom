"""``headroom perf`` must surface CLI context-tool (RTK) lifetime savings.

RTK keeps its savings in its own counter, which never lands in ``proxy.log``,
so ``headroom perf`` used to omit them entirely — the report showed only
proxy-compression savings. These tests pin that the report (text + JSON)
includes the context-tool lifetime savings when available, and degrades
cleanly to proxy-only when the tool is absent.
"""

from __future__ import annotations

import headroom.proxy.helpers as helpers
from headroom.perf import analyzer

_FAKE_RTK = {
    "installed": True,
    "tool": "rtk",
    "label": "RTK",
    "lifetime": {"tokens_saved": 26_853_652, "commands": 8000, "savings_pct": 68.9},
}


def test_build_perf_summary_includes_cli_filtering(monkeypatch):
    monkeypatch.setattr(helpers, "_get_context_tool_stats", lambda: _FAKE_RTK)
    report = analyzer.parse_log_files(last_n_hours=0.0)
    summary = analyzer.build_perf_summary(report)

    assert summary["cli_filtering"] is not None
    assert summary["cli_filtering"]["tool"] == "rtk"
    assert summary["cli_filtering"]["tokens_saved"] == 26_853_652
    assert summary["cli_filtering"]["savings_pct"] == 68.9


def test_format_report_shows_cli_filtering(monkeypatch):
    monkeypatch.setattr(helpers, "_get_context_tool_stats", lambda: _FAKE_RTK)
    report = analyzer.parse_log_files(last_n_hours=0.0)
    text = analyzer.format_report(report)

    assert "RTK CLI Filtering" in text
    assert "26,853,652" in text


def test_perf_omits_cli_filtering_when_tool_absent(monkeypatch):
    monkeypatch.setattr(helpers, "_get_context_tool_stats", lambda: None)
    report = analyzer.parse_log_files(last_n_hours=0.0)

    summary = analyzer.build_perf_summary(report)
    assert summary["cli_filtering"] is None

    text = analyzer.format_report(report)
    assert "CLI Filtering" not in text

"""Tool-search / deferral savings must aggregate into Metrics and surface in the
reporting sinks — not live only in per-request tags (which every sink reading
metrics.* structurally missed: session summary, cost summary, all-layers total,
`headroom perf --json`)."""

from __future__ import annotations

import asyncio

from headroom.perf.analyzer import PerfRecord, PerfReport, build_perf_summary
from headroom.proxy.prometheus_metrics import PrometheusMetrics


def test_metrics_accumulates_tool_search_saved_apart_from_message() -> None:
    m = PrometheusMetrics()

    async def go() -> None:
        await m.record_request(
            provider="anthropic",
            model="claude-x",
            input_tokens=100,
            output_tokens=10,
            tokens_saved=0,
            latency_ms=1.0,
            tool_search_saved=1500,
        )
        await m.record_request(
            provider="anthropic",
            model="claude-x",
            input_tokens=100,
            output_tokens=10,
            tokens_saved=200,
            latency_ms=1.0,
            tool_search_saved=800,
        )

    asyncio.run(go())
    assert m.tokens_saved_total == 200  # message compression only
    assert m.tool_search_saved_total == 2300  # tool-schema layer, aggregated


def test_build_perf_summary_includes_tool_saved() -> None:
    report = PerfReport(
        perf_records=[
            PerfRecord(
                timestamp="t",
                request_id="r1",
                model="m",
                tokens_before=1000,
                tokens_after=900,
                tokens_saved=100,
                tool_saved=5000,
            ),
            PerfRecord(
                timestamp="t",
                request_id="r2",
                model="m",
                tokens_before=500,
                tokens_after=500,
                tokens_saved=0,
                tool_saved=3000,
            ),
        ]
    )
    summary = build_perf_summary(report)
    assert summary["tokens_saved"] == 100  # message
    assert summary["tool_saved"] == 8000  # tool-schema surfaced in json/csv sink

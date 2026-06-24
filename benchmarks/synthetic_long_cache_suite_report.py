#!/usr/bin/env python3
"""Run a long deterministic synthetic suite for cache and rewrite behavior."""

from __future__ import annotations

import copy
import html
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmarks.claude_session_mode_benchmark as bench
from benchmarks.claude_session_mode_benchmark import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    ReplayTurn,
    SessionReplay,
    determine_winners,
    format_currency,
    simulate_replays,
)

OUTPUT_DIR = Path("benchmark_results") / "synthetic_long_cache_suite"
MODEL = "claude-sonnet-4-6"
TTL_MINUTES = 5
TURNS_PER_SCENARIO = 400


class _FakeProvider:
    @staticmethod
    def get_context_limit(model: str) -> int:
        return 200_000


class _HistoryPressurePipeline:
    @staticmethod
    def apply(messages, **kwargs):  # noqa: ANN001
        rewritten = []
        total = len(messages)
        # Leave the latest two messages untouched; rewrite older tool results.
        # Token mode reprocesses full history, so prior-turn tool results become
        # compressed on later turns and can bust prefix cache. Cache mode only
        # processes the newly-appended delta, so it does not revisit older turns.
        protected_start = max(total - 2, 0)
        for index, message in enumerate(messages):
            content = message.get("content")
            if (
                index < protected_start
                and isinstance(content, list)
                and any(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in content
                )
            ):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        new_blocks.append({**block, "content": "[compressed-older-tool-result]"})
                    else:
                        new_blocks.append(copy.deepcopy(block))
                rewritten.append({**message, "content": new_blocks})
            else:
                rewritten.append(copy.deepcopy(message))
        return SimpleNamespace(messages=rewritten)


class _FakeProxy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(image_optimize=False)
        self.anthropic_provider = _FakeProvider()
        self.anthropic_pipeline = _HistoryPressurePipeline()


def _tool_result_payload(turn_number: int, scenario: str) -> str:
    return (f"{scenario}-tool-output-{turn_number} " * 80).strip()


def _build_stable_append_only() -> SessionReplay:
    base = datetime(2026, 3, 13, 1, 0, tzinfo=timezone.utc)
    turns: list[ReplayTurn] = []
    for index in range(TURNS_PER_SCENARIO):
        turns.append(
            ReplayTurn(
                session_id="stable-append-only",
                project_key="C--git-synthetic",
                decoded_project_path=r"C:\git\synthetic",
                request_id=f"stable-{index + 1:04d}",
                model=MODEL,
                timestamp=base + timedelta(minutes=index * 2),
                input_messages=[
                    {
                        "role": "user",
                        "content": f"Stable append-only turn {index + 1}. Summarize and continue.",
                    }
                ],
                assistant_message={"role": "assistant", "content": f"ok stable {index + 1}"},
                output_tokens=12,
            )
        )
    return SessionReplay(
        session_id="stable-append-only",
        project_key="C--git-synthetic",
        decoded_project_path=r"C:\git\synthetic",
        turns=turns,
    )


def _build_token_rewrite_pressure() -> SessionReplay:
    base = datetime(2026, 3, 14, 1, 0, tzinfo=timezone.utc)
    turns: list[ReplayTurn] = []
    for index in range(TURNS_PER_SCENARIO):
        turn_no = index + 1
        turns.append(
            ReplayTurn(
                session_id="token-rewrite-pressure",
                project_key="C--git-synthetic",
                decoded_project_path=r"C:\git\synthetic",
                request_id=f"rewrite-{turn_no:04d}",
                model=MODEL,
                timestamp=base + timedelta(minutes=index * 2),
                input_messages=[
                    {"role": "user", "content": f"Inspect tool output for turn {turn_no}."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"tool-{turn_no}",
                                "content": _tool_result_payload(turn_no, "rewrite"),
                            }
                        ],
                    },
                ],
                assistant_message={"role": "assistant", "content": f"ok rewrite {turn_no}"},
                output_tokens=14,
            )
        )
    return SessionReplay(
        session_id="token-rewrite-pressure",
        project_key="C--git-synthetic",
        decoded_project_path=r"C:\git\synthetic",
        turns=turns,
    )


def _build_ttl_resets() -> SessionReplay:
    base = datetime(2026, 3, 15, 1, 0, tzinfo=timezone.utc)
    turns: list[ReplayTurn] = []
    for index in range(TURNS_PER_SCENARIO):
        turns.append(
            ReplayTurn(
                session_id="ttl-resets",
                project_key="C--git-synthetic",
                decoded_project_path=r"C:\git\synthetic",
                request_id=f"ttl-{index + 1:04d}",
                model=MODEL,
                timestamp=base + timedelta(minutes=index * 7),
                input_messages=[
                    {
                        "role": "user",
                        "content": f"TTL reset turn {index + 1}. Continue the thread.",
                    }
                ],
                assistant_message={"role": "assistant", "content": f"ok ttl {index + 1}"},
                output_tokens=12,
            )
        )
    return SessionReplay(
        session_id="ttl-resets",
        project_key="C--git-synthetic",
        decoded_project_path=r"C:\git\synthetic",
        turns=turns,
    )


def _build_suite() -> list[SessionReplay]:
    return [
        _build_stable_append_only(),
        _build_token_rewrite_pressure(),
        _build_ttl_resets(),
    ]


def _scenario_label(session_id: str) -> str:
    return session_id.replace("-", " ").title()


def _run_suite() -> tuple[dict[str, dict[str, bench.ModeSummary]], dict[str, bench.ModeSummary]]:
    original_make_proxy = bench._make_proxy
    bench._make_proxy = lambda mode: _FakeProxy()
    try:
        per_scenario: dict[str, dict[str, bench.ModeSummary]] = {}
        suite = _build_suite()
        for replay in suite:
            _, summaries = simulate_replays([replay], cache_ttl_minutes=TTL_MINUTES)
            per_scenario[replay.session_id] = summaries
        _, aggregate = simulate_replays(suite, cache_ttl_minutes=TTL_MINUTES)
    finally:
        bench._make_proxy = original_make_proxy
    return per_scenario, aggregate


def _summary_payload(summary: bench.ModeSummary) -> dict[str, int | float | str]:
    return {
        "total_cost_usd": summary.total_cost_usd,
        "no_cache_total_cost_usd": summary.no_cache_total_cost_usd,
        "forwarded_input_tokens": summary.forwarded_input_tokens,
        "cache_bust_turns": summary.cache_bust_turns,
        "ttl_expiry_turns": summary.ttl_expiry_turns,
        "rewrite_turns": summary.rewrite_turns,
        "stable_replay_rewrite_turns": summary.stable_replay_rewrite_turns,
        "busting_rewrite_turns": summary.busting_rewrite_turns,
        "non_cache_eligible_rewrite_turns": summary.non_cache_eligible_rewrite_turns,
        "retroactive_rewrite_turns": summary.retroactive_rewrite_turns,
    }


def _write_report(
    per_scenario: dict[str, dict[str, bench.ModeSummary]],
    aggregate: dict[str, bench.ModeSummary],
) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "turns_per_scenario": TURNS_PER_SCENARIO,
        "total_turns": TURNS_PER_SCENARIO * len(per_scenario),
        "ttl_minutes": TTL_MINUTES,
        "scenarios": {
            session_id: {mode: _summary_payload(summary) for mode, summary in summaries.items()}
            for session_id, summaries in per_scenario.items()
        },
        "aggregate": {mode: _summary_payload(summary) for mode, summary in aggregate.items()},
        "aggregate_winners": determine_winners(aggregate),
    }
    json_path = OUTPUT_DIR / "synthetic_long_cache_suite.json"
    md_path = OUTPUT_DIR / "synthetic_long_cache_suite.md"
    html_path = OUTPUT_DIR / "synthetic_long_cache_suite.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Synthetic Long Cache Suite",
        "",
        f"- Turns per scenario: `{TURNS_PER_SCENARIO}`",
        f"- Total turns: `{TURNS_PER_SCENARIO * len(per_scenario)}`",
        f"- Cache TTL: `{TTL_MINUTES}` minutes",
        "",
        "## Scenarios",
        "",
        "1. `stable-append-only`: append-only conversation, no rewrite pressure",
        "2. `token-rewrite-pressure`: each turn adds a tool result; older tool results become compressible later",
        "3. `ttl-resets`: append-only conversation with >TTL gaps to force normal cache expiry",
        "",
    ]

    for session_id, summaries in per_scenario.items():
        winners = determine_winners(summaries)
        md_lines.extend(
            [
                f"## {_scenario_label(session_id)}",
                "",
                "| Mode | Cost | Forwarded Tokens | Cache Busts | TTL Expiry | Rewrites | Stable Replay Rewrites | Busting Rewrites | Retroactive Rewrites |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
            summary = summaries[mode]
            md_lines.append(
                f"| `{mode}` | {format_currency(summary.total_cost_usd)} | "
                f"{summary.forwarded_input_tokens:,} | {summary.cache_bust_turns} | "
                f"{summary.ttl_expiry_turns} | {summary.rewrite_turns} | "
                f"{summary.stable_replay_rewrite_turns} | {summary.busting_rewrite_turns} | "
                f"{summary.retroactive_rewrite_turns} |"
            )
        md_lines.extend(
            [
                "",
                f"- total cost winner: `{winners['total_cost']}`",
                f"- no-cache total cost winner: `{winners['no_cache_total_cost']}`",
                f"- window winner with cache counted: `{winners['window_with_cache']}`",
                "",
            ]
        )

    aggregate_winners = determine_winners(aggregate)
    md_lines.extend(
        [
            "## Aggregate",
            "",
            "| Mode | Cost | Forwarded Tokens | Cache Busts | TTL Expiry | Rewrites | Stable Replay Rewrites | Busting Rewrites | Retroactive Rewrites |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = aggregate[mode]
        md_lines.append(
            f"| `{mode}` | {format_currency(summary.total_cost_usd)} | "
            f"{summary.forwarded_input_tokens:,} | {summary.cache_bust_turns} | "
            f"{summary.ttl_expiry_turns} | {summary.rewrite_turns} | "
            f"{summary.stable_replay_rewrite_turns} | {summary.busting_rewrite_turns} | "
            f"{summary.retroactive_rewrite_turns} |"
        )
    md_lines.extend(
        [
            "",
            f"- total cost winner: `{aggregate_winners['total_cost']}`",
            f"- no-cache total cost winner: `{aggregate_winners['no_cache_total_cost']}`",
            f"- window winner if cache tokens count: `{aggregate_winners['window_with_cache']}`",
            f"- window winner if cache read tokens do not count: `{aggregate_winners['window_without_cache_reads']}`",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    scenario_sections: list[str] = []
    for session_id, summaries in per_scenario.items():
        rows = []
        for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
            summary = summaries[mode]
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(mode)}</code></td>"
                f"<td>{html.escape(format_currency(summary.total_cost_usd))}</td>"
                f"<td>{summary.forwarded_input_tokens:,}</td>"
                f"<td>{summary.cache_bust_turns}</td>"
                f"<td>{summary.ttl_expiry_turns}</td>"
                f"<td>{summary.rewrite_turns}</td>"
                f"<td>{summary.stable_replay_rewrite_turns}</td>"
                f"<td>{summary.busting_rewrite_turns}</td>"
                f"<td>{summary.retroactive_rewrite_turns}</td>"
                "</tr>"
            )
        scenario_sections.append(
            "<section class='card'>"
            f"<h2>{html.escape(_scenario_label(session_id))}</h2>"
            "<table><thead><tr><th>Mode</th><th>Cost</th><th>Forwarded Tokens</th><th>Cache Busts</th>"
            "<th>TTL Expiry</th><th>Rewrites</th><th>Stable Replay Rewrites</th>"
            "<th>Busting Rewrites</th><th>Retroactive Rewrites</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></section>"
        )

    aggregate_rows = []
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = aggregate[mode]
        aggregate_rows.append(
            "<tr>"
            f"<td><code>{html.escape(mode)}</code></td>"
            f"<td>{html.escape(format_currency(summary.total_cost_usd))}</td>"
            f"<td>{summary.forwarded_input_tokens:,}</td>"
            f"<td>{summary.cache_bust_turns}</td>"
            f"<td>{summary.ttl_expiry_turns}</td>"
            f"<td>{summary.rewrite_turns}</td>"
            f"<td>{summary.stable_replay_rewrite_turns}</td>"
            f"<td>{summary.busting_rewrite_turns}</td>"
            f"<td>{summary.retroactive_rewrite_turns}</td>"
            "</tr>"
        )

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Synthetic Long Cache Suite</title>"
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.55;color:#111827;background:#f8fafc}"
        "h1,h2{letter-spacing:-0.02em}"
        "code{background:#e5e7eb;padding:1px 4px;border-radius:4px}"
        "table{border-collapse:collapse;width:100%;margin:16px 0;background:white}"
        "th,td{border:1px solid #cbd5e1;padding:10px;text-align:left}"
        "th{background:#e2e8f0}"
        ".card{background:white;border:1px solid #cbd5e1;border-radius:16px;padding:24px;margin:18px 0;box-shadow:0 8px 24px rgba(15,23,42,.06)}"
        "</style></head><body>"
        "<h1>Synthetic Long Cache Suite</h1>"
        f"<div class='card'><p>Total turns: <code>{TURNS_PER_SCENARIO * len(per_scenario)}</code><br>"
        f"Turns per scenario: <code>{TURNS_PER_SCENARIO}</code><br>"
        f"Cache TTL: <code>{TTL_MINUTES}</code> minutes</p></div>"
        + "".join(scenario_sections)
        + "<section class='card'><h2>Aggregate</h2>"
        "<table><thead><tr><th>Mode</th><th>Cost</th><th>Forwarded Tokens</th><th>Cache Busts</th>"
        "<th>TTL Expiry</th><th>Rewrites</th><th>Stable Replay Rewrites</th>"
        "<th>Busting Rewrites</th><th>Retroactive Rewrites</th></tr></thead><tbody>"
        + "".join(aggregate_rows)
        + "</tbody></table></section></body></html>"
    )
    html_path.write_text(html_doc, encoding="utf-8")
    return md_path, json_path, html_path


def main() -> int:
    per_scenario, aggregate = _run_suite()
    md_path, json_path, html_path = _write_report(per_scenario, aggregate)
    print("Synthetic long cache suite")
    print(f"turns_per_scenario={TURNS_PER_SCENARIO}")
    print(f"total_turns={TURNS_PER_SCENARIO * len(per_scenario)}")
    for session_id, summaries in per_scenario.items():
        print(f"scenario={session_id}")
        for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
            summary = summaries[mode]
            print(
                f"  {mode}: cost={format_currency(summary.total_cost_usd)} "
                f"busts={summary.cache_bust_turns} ttl={summary.ttl_expiry_turns} "
                f"rewrites={summary.rewrite_turns} stable_rw={summary.stable_replay_rewrite_turns} "
                f"bust_rw={summary.busting_rewrite_turns} "
                f"forwarded={summary.forwarded_input_tokens}"
            )
    print("aggregate")
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = aggregate[mode]
        print(
            f"  {mode}: cost={format_currency(summary.total_cost_usd)} "
            f"busts={summary.cache_bust_turns} ttl={summary.ttl_expiry_turns} "
            f"rewrites={summary.rewrite_turns} stable_rw={summary.stable_replay_rewrite_turns} "
            f"bust_rw={summary.busting_rewrite_turns} "
            f"forwarded={summary.forwarded_input_tokens}"
        )
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

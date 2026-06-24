#!/usr/bin/env python3
"""Run a deterministic synthetic replay that forces token-mode cache busts."""

from __future__ import annotations

import copy
import html
import json
import sys
from datetime import datetime
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
    _apply_mode_to_messages,
    _cache_gap_within_ttl,
    determine_winners,
    format_currency,
    get_tokenizer,
    simulate_replays,
)

OUTPUT_DIR = Path("benchmark_results") / "synthetic_token_cache_bust"


class _FakeProvider:
    @staticmethod
    def get_context_limit(model: str) -> int:
        return 200_000


class _FakePipeline:
    @staticmethod
    def apply(messages, **kwargs):  # noqa: ANN001
        rewritten = []
        should_rewrite_history = len(messages) > 2
        for message in messages:
            content = message.get("content")
            if (
                should_rewrite_history
                and isinstance(content, list)
                and any(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in content
                )
            ):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        new_blocks.append({**block, "content": "[compressed-tool-result]"})
                    else:
                        new_blocks.append(block)
                rewritten.append({**message, "content": new_blocks})
            else:
                rewritten.append(copy.deepcopy(message))
        return SimpleNamespace(messages=rewritten)


class _FakeProxy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(image_optimize=False)
        self.anthropic_provider = _FakeProvider()
        self.anthropic_pipeline = _FakePipeline()


def _build_replay() -> SessionReplay:
    return SessionReplay(
        session_id="token-cache-bust",
        project_key="C--git-synthetic",
        decoded_project_path=r"C:\git\synthetic",
        turns=[
            ReplayTurn(
                session_id="token-cache-bust",
                project_key="C--git-synthetic",
                decoded_project_path=r"C:\git\synthetic",
                request_id="r1",
                model="claude-sonnet-4-6",
                timestamp=datetime.fromisoformat("2026-03-13T01:00:00+00:00"),
                input_messages=[
                    {"role": "user", "content": "Summarize this tool output"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-1",
                                "content": "X" * 800,
                            }
                        ],
                    },
                ],
                assistant_message={"role": "assistant", "content": "ok"},
                output_tokens=10,
            ),
            ReplayTurn(
                session_id="token-cache-bust",
                project_key="C--git-synthetic",
                decoded_project_path=r"C:\git\synthetic",
                request_id="r2",
                model="claude-sonnet-4-6",
                timestamp=datetime.fromisoformat("2026-03-13T01:02:00+00:00"),
                input_messages=[{"role": "user", "content": "What changed?"}],
                assistant_message={"role": "assistant", "content": "done"},
                output_tokens=12,
            ),
        ],
    )


def _build_bust_events(replay: SessionReplay) -> dict[str, list[dict[str, object]]]:
    events: dict[str, list[dict[str, object]]] = {
        "baseline": [],
        PROXY_MODE_TOKEN: [],
        PROXY_MODE_CACHE: [],
    }
    ttl_minutes = 5
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        proxy = None if mode == "baseline" else _FakeProxy()
        prefix_tracker = None if mode == "baseline" else bench.PrefixCacheTracker("anthropic")
        comp_cache = bench.CompressionCache() if mode == PROXY_MODE_TOKEN else None
        conversation: list[dict[str, object]] = []
        previous_original: list[dict[str, object]] | None = None
        previous_forwarded_context: list[dict[str, object]] | None = None
        previous_forwarded_request: list[dict[str, object]] | None = None
        previous_request_id: str | None = None
        previous_timestamp: datetime | None = None

        for turn in replay.turns:
            conversation.extend(copy.deepcopy(turn.input_messages))
            forwarded = _apply_mode_to_messages(
                proxy,
                mode,
                conversation,
                model=turn.model,
                prefix_tracker=prefix_tracker,
                comp_cache=comp_cache,
                previous_original_messages=previous_original,
                previous_forwarded_messages=previous_forwarded_context,
            )

            if previous_forwarded_request is not None and _cache_gap_within_ttl(
                turn.timestamp,
                previous_timestamp,
                ttl=bench.timedelta(minutes=ttl_minutes),
            ):
                prefix_preserved = (
                    len(forwarded) >= len(previous_forwarded_request)
                    and forwarded[: len(previous_forwarded_request)] == previous_forwarded_request
                )
                if not prefix_preserved:
                    divergent_index = next(
                        (
                            idx
                            for idx, (prev_msg, curr_msg) in enumerate(
                                zip(previous_forwarded_request, forwarded, strict=False)
                            )
                            if prev_msg != curr_msg
                        ),
                        min(len(previous_forwarded_request), len(forwarded)),
                    )
                    events[mode].append(
                        {
                            "request_id": turn.request_id,
                            "previous_request_id": previous_request_id,
                            "divergent_index": divergent_index,
                            "previous_forwarded": previous_forwarded_request,
                            "current_forwarded": forwarded,
                        }
                    )

            tokenizer = get_tokenizer(turn.model)
            if prefix_tracker is not None:
                bench._update_prefix_tracker(
                    prefix_tracker,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    messages=forwarded,
                    message_token_counts=[tokenizer.count_message(msg) for msg in forwarded],
                    original_messages=conversation,
                )

            conversation.append(copy.deepcopy(turn.assistant_message))
            previous_original = copy.deepcopy(conversation)
            previous_forwarded_context = copy.deepcopy(forwarded) + [
                copy.deepcopy(turn.assistant_message)
            ]
            previous_forwarded_request = copy.deepcopy(forwarded)
            previous_request_id = turn.request_id
            previous_timestamp = turn.timestamp

    return events


def _write_report(
    replay: SessionReplay,
    summaries: dict[str, bench.ModeSummary],
    winners: dict[str, str],
    events: dict[str, list[dict[str, object]]],
) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": replay.session_id,
        "requests": len(replay.turns),
        "summaries": {
            mode: {
                "total_cost_usd": summary.total_cost_usd,
                "cache_bust_turns": summary.cache_bust_turns,
                "rewrite_turns": summary.rewrite_turns,
                "retroactive_rewrite_turns": summary.retroactive_rewrite_turns,
                "forwarded_input_tokens": summary.forwarded_input_tokens,
            }
            for mode, summary in summaries.items()
        },
        "winners": winners,
        "events": events,
    }
    json_path = OUTPUT_DIR / "synthetic_token_cache_bust.json"
    md_path = OUTPUT_DIR / "synthetic_token_cache_bust.md"
    html_path = OUTPUT_DIR / "synthetic_token_cache_bust.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Synthetic Token Cache Bust Report",
        "",
        f"Session: `{replay.session_id}`",
        f"Requests: `{len(replay.turns)}`",
        "",
        "## Summary",
        "",
        "| Mode | Cost | Cache Busts | Rewrites | Retroactive Rewrites | Forwarded Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = summaries[mode]
        md_lines.append(
            f"| `{mode}` | {format_currency(summary.total_cost_usd)} | "
            f"{summary.cache_bust_turns} | {summary.rewrite_turns} | "
            f"{summary.retroactive_rewrite_turns} | {summary.forwarded_input_tokens} |"
        )
    md_lines.extend(
        [
            "",
            "## Winners",
            "",
            f"- total cost: `{winners['total_cost']}`",
            f"- no-cache total cost: `{winners['no_cache_total_cost']}`",
            f"- window with cache counted: `{winners['window_with_cache']}`",
            f"- window without cache reads: `{winners['window_without_cache_reads']}`",
            "",
            "## Cache Bust Events",
            "",
        ]
    )
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        md_lines.append(f"### `{mode}`")
        if not events[mode]:
            md_lines.append("")
            md_lines.append("- none")
            md_lines.append("")
            continue
        md_lines.append("")
        for event in events[mode]:
            md_lines.append(
                f"- request `{event['request_id']}` diverged from `{event['previous_request_id']}` "
                f"at message index `{event['divergent_index']}`"
            )
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    rows = []
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = summaries[mode]
        rows.append(
            "<tr>"
            f"<td>{html.escape(mode)}</td>"
            f"<td>{html.escape(format_currency(summary.total_cost_usd))}</td>"
            f"<td>{summary.cache_bust_turns}</td>"
            f"<td>{summary.rewrite_turns}</td>"
            f"<td>{summary.retroactive_rewrite_turns}</td>"
            f"<td>{summary.forwarded_input_tokens}</td>"
            "</tr>"
        )
    event_sections = []
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        section = [f"<h2>{html.escape(mode)}</h2>"]
        if not events[mode]:
            section.append("<p>none</p>")
        else:
            section.append("<ul>")
            for event in events[mode]:
                section.append(
                    "<li>"
                    f"request <code>{html.escape(str(event['request_id']))}</code> diverged from "
                    f"<code>{html.escape(str(event['previous_request_id']))}</code> at message index "
                    f"<code>{event['divergent_index']}</code>"
                    "</li>"
                )
            section.append("</ul>")
        event_sections.append("".join(section))

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Synthetic Token Cache Bust Report</title>"
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,sans-serif;margin:32px;line-height:1.5;}"
        "table{border-collapse:collapse;width:100%;margin:16px 0;}"
        "th,td{border:1px solid #d0d7de;padding:8px 10px;text-align:left;}"
        "th{background:#f6f8fa;}"
        "code{background:#f6f8fa;padding:1px 4px;border-radius:4px;}"
        "</style></head><body>"
        "<h1>Synthetic Token Cache Bust Report</h1>"
        f"<p>Session: <code>{html.escape(replay.session_id)}</code><br>Requests: <code>{len(replay.turns)}</code></p>"
        "<table><thead><tr><th>Mode</th><th>Cost</th><th>Cache Busts</th><th>Rewrites</th>"
        "<th>Retroactive Rewrites</th><th>Forwarded Tokens</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        "<h2>Winners</h2><ul>"
        f"<li>total cost: <code>{html.escape(winners['total_cost'])}</code></li>"
        f"<li>no-cache total cost: <code>{html.escape(winners['no_cache_total_cost'])}</code></li>"
        f"<li>window with cache counted: <code>{html.escape(winners['window_with_cache'])}</code></li>"
        f"<li>window without cache reads: <code>{html.escape(winners['window_without_cache_reads'])}</code></li>"
        "</ul><h2>Cache Bust Events</h2>" + "".join(event_sections) + "</body></html>"
    )
    html_path.write_text(html_doc, encoding="utf-8")
    return md_path, json_path, html_path


def main() -> int:
    original_make_proxy = bench._make_proxy
    bench._make_proxy = lambda mode: _FakeProxy()
    try:
        replay = _build_replay()
        dataset, summaries = simulate_replays([replay], cache_ttl_minutes=5)
        events = _build_bust_events(replay)
    finally:
        bench._make_proxy = original_make_proxy

    winners = determine_winners(summaries)
    md_path, json_path, html_path = _write_report(replay, summaries, winners, events)
    print("Synthetic token-cache-bust replay")
    print(f"requests={dataset.requests}")
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = summaries[mode]
        print(
            f"{mode}: cost={format_currency(summary.total_cost_usd)} "
            f"busts={summary.cache_bust_turns} "
            f"rewrites={summary.rewrite_turns} "
            f"retro_rw={summary.retroactive_rewrite_turns} "
            f"forwarded={summary.forwarded_input_tokens}"
        )
    print(f"winner_total_cost={winners['total_cost']}")
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

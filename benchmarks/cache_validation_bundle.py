#!/usr/bin/env python3
"""Generate a reproducible local cache-validation report bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmarks.claude_session_mode_benchmark as real_bench
import benchmarks.synthetic_long_cache_suite_report as long_suite
import benchmarks.synthetic_token_cache_bust_report as token_bust
from benchmarks.claude_session_mode_benchmark import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    _apply_mode_to_messages,
    _cache_gap_within_ttl,
    _rewrite_scope,
    build_dataset_and_observed_from_files,
    determine_winners,
    format_currency,
    get_tokenizer,
    load_session_replay,
    resolve_checkpoint_dir,
    select_session_files,
    simulate_session_files,
    trim_replay_to_recent_turns,
    write_report,
)
from headroom.cache.compression_cache import CompressionCache
from headroom.cache.prefix_tracker import PrefixCacheTracker

DEFAULT_OUTPUT_DIR = Path("benchmark_results") / "cache_validation_bundle"


def _excerpt_content(content: Any, *, max_chars: int) -> str:
    if isinstance(content, str):
        text = content.replace("\n", " ")
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    if isinstance(content, list):
        parts = []
        for block in content[:4]:
            if isinstance(block, dict):
                btype = str(block.get("type", "unknown"))
                bcontent = block.get("content", "")
                if isinstance(bcontent, str):
                    bcontent = bcontent.replace("\n", " ")
                    bcontent = bcontent[:max_chars] + ("..." if len(bcontent) > max_chars else "")
                parts.append(f"[{btype}] {bcontent}")
            else:
                parts.append(str(block)[:max_chars])
        return " | ".join(parts)
    return str(content)[:max_chars]


def _message_preview(msg: dict[str, Any], *, max_chars: int) -> dict[str, str]:
    return {
        "role": str(msg.get("role")),
        "content_excerpt": _excerpt_content(msg.get("content"), max_chars=max_chars),
    }


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _redact_text(value: str, *, prefix: str) -> str:
    return f"{prefix}-{_stable_hash(value)}"


def _redact_path(value: str) -> str:
    path = Path(value)
    suffix = path.suffix
    return f"path-{_stable_hash(value)}{suffix}"


def _git_output(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _runtime_metadata(repo_root: Path) -> dict[str, Any]:
    return {
        "git_sha": _git_output(["rev-parse", "HEAD"], repo_root),
        "git_dirty": bool(_git_output(["status", "--porcelain"], repo_root)),
        "python_version": sys.version,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def _corpus_fingerprint(
    *,
    root: Path,
    session_files: list[Path],
    max_sessions: int | None,
    recent_turns_per_session: int | None,
    cache_ttl_minutes: int,
) -> dict[str, Any]:
    normalized_files = [str(p.resolve()) for p in session_files]
    payload = {
        "root": str(root.resolve()),
        "session_files": normalized_files,
        "max_sessions": max_sessions,
        "recent_turns_per_session": recent_turns_per_session,
        "cache_ttl_minutes": cache_ttl_minutes,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "root": str(root.resolve()),
        "session_file_count": len(session_files),
        "session_files_sha256": digest,
        "max_sessions": max_sessions,
        "recent_turns_per_session": recent_turns_per_session,
        "cache_ttl_minutes": cache_ttl_minutes,
    }


def _collect_real_processed_events(
    *,
    root: Path,
    recent_turns_per_session: int | None,
    max_events_per_mode: int,
    ttl_minutes: int,
    max_chars: int,
    include_content: bool,
) -> dict[str, Any]:
    ttl = timedelta(minutes=ttl_minutes)
    events: list[dict[str, Any]] = []
    session_files = select_session_files(root)
    for mode in (PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        proxy = real_bench._make_proxy(mode)
        collected = 0
        for session_file in session_files:
            replay = load_session_replay(session_file)
            if replay is None:
                continue
            replay = trim_replay_to_recent_turns(replay, recent_turns_per_session)
            prefix_tracker = PrefixCacheTracker("anthropic")
            comp_cache = CompressionCache() if mode == PROXY_MODE_TOKEN else None
            conversation: list[dict[str, Any]] = []
            previous_original_context: list[dict[str, Any]] | None = None
            previous_forwarded_context: list[dict[str, Any]] | None = None
            previous_forwarded: list[dict[str, Any]] = []
            previous_timestamp = None
            pending = None
            for turn in replay.turns:
                tokenizer = get_tokenizer(turn.model)
                prior_context_message_count = len(conversation)
                conversation.extend(turn.input_messages)
                forwarded = _apply_mode_to_messages(
                    proxy,
                    mode,
                    conversation,
                    model=turn.model,
                    prefix_tracker=prefix_tracker,
                    comp_cache=comp_cache,
                    previous_original_messages=previous_original_context,
                    previous_forwarded_messages=previous_forwarded_context,
                )
                rewrite, retro = _rewrite_scope(
                    conversation,
                    forwarded,
                    stable_prefix_message_count=prior_context_message_count,
                )
                if rewrite:
                    prior_forwarded = (
                        pending.forwarded if pending is not None else previous_forwarded
                    )
                    prior_ts = pending.turn.timestamp if pending is not None else previous_timestamp
                    eligible = bool(
                        prior_ts is not None
                        and _cache_gap_within_ttl(turn.timestamp, prior_ts, ttl=ttl)
                        and prior_forwarded
                    )
                    prefix_preserved = None
                    first_diff_index = None
                    if eligible:
                        prefix_preserved = (
                            len(forwarded) >= len(prior_forwarded)
                            and forwarded[: len(prior_forwarded)] == prior_forwarded
                        )
                        if not prefix_preserved:
                            for idx, (a, b) in enumerate(zip(prior_forwarded, forwarded)):
                                if a != b:
                                    first_diff_index = idx
                                    break
                            if first_diff_index is None:
                                first_diff_index = min(len(prior_forwarded), len(forwarded))
                    events.append(
                        {
                            "mode": mode,
                            "session_id": replay.session_id
                            if include_content
                            else _redact_text(replay.session_id, prefix="session"),
                            "project": replay.decoded_project_path
                            if include_content
                            else _redact_path(replay.decoded_project_path),
                            "request_id": turn.request_id
                            if include_content
                            else _redact_text(turn.request_id, prefix="request"),
                            "timestamp": turn.timestamp.isoformat(),
                            "cache_eligible": eligible,
                            "prefix_preserved": prefix_preserved,
                            "retroactive_rewrite": retro,
                            "first_diff_index": first_diff_index,
                            "original_tail": [
                                _message_preview(m, max_chars=max_chars)
                                if include_content
                                else {
                                    "role": str(m.get("role")),
                                    "content_excerpt": "[redacted]",
                                }
                                for m in conversation[max(0, len(conversation) - 4) :]
                            ],
                            "forwarded_tail": [
                                _message_preview(m, max_chars=max_chars)
                                if include_content
                                else {
                                    "role": str(m.get("role")),
                                    "content_excerpt": "[redacted]",
                                }
                                for m in forwarded[max(0, len(forwarded) - 4) :]
                            ],
                        }
                    )
                    collected += 1
                    if collected >= max_events_per_mode:
                        break
                if pending is not None:
                    previous_forwarded = copy.deepcopy(pending.forwarded)
                    previous_timestamp = pending.turn.timestamp
                real_bench._update_prefix_tracker(
                    prefix_tracker,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    messages=forwarded,
                    message_token_counts=[tokenizer.count_message(msg) for msg in forwarded],
                    original_messages=conversation,
                )

                class Pending:
                    pass

                pending = Pending()
                pending.turn = turn
                pending.forwarded = forwarded
                conversation.append(turn.assistant_message)
                previous_original_context = copy.deepcopy(conversation)
                previous_forwarded_context = copy.deepcopy(forwarded) + [
                    copy.deepcopy(turn.assistant_message)
                ]
            if collected >= max_events_per_mode:
                break
    return {"events": events}


def _write_processed_event_reports(
    output_dir: Path, payload: dict[str, Any]
) -> tuple[Path, Path, Path]:
    out_dir = output_dir / "real_processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "real_processed_rewrite_report.json"
    md_path = out_dir / "real_processed_rewrite_report.md"
    html_path = out_dir / "real_processed_rewrite_report.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Real Processed Rewrite Report",
        "",
        "Local-only report from real Claude transcript replays. Do not commit.",
        "",
    ]
    for mode in (PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        mode_events = [e for e in payload["events"] if e["mode"] == mode]
        md.extend([f"## `{mode}`", ""])
        if not mode_events:
            md.extend(["No rewrite events captured.", ""])
            continue
        for i, e in enumerate(mode_events, start=1):
            md.extend(
                [
                    f"### Event {i}",
                    "",
                    f"- session: `{e['session_id']}`",
                    f"- request: `{e['request_id']}`",
                    f"- cache eligible: `{e['cache_eligible']}`",
                    f"- prefix preserved: `{e['prefix_preserved']}`",
                    f"- retroactive rewrite: `{e['retroactive_rewrite']}`",
                    f"- first diff index: `{e['first_diff_index']}`",
                    "",
                    "**Original Tail**",
                    "",
                ]
            )
            for msg in e["original_tail"]:
                md.append(f"- `{msg['role']}`: {msg['content_excerpt']}")
            md.extend(["", "**Forwarded Tail**", ""])
            for msg in e["forwarded_tail"]:
                md.append(f"- `{msg['role']}`: {msg['content_excerpt']}")
            md.extend(["", ""])
    md_path.write_text("\n".join(md), encoding="utf-8")

    sections = []
    for mode in (PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        mode_events = [e for e in payload["events"] if e["mode"] == mode]
        cards = []
        for i, e in enumerate(mode_events, start=1):
            orig = "".join(
                f"<li><code>{html.escape(str(m['role']))}</code>: "
                f"{html.escape(str(m['content_excerpt']))}</li>"
                for m in e["original_tail"]
            )
            fwd = "".join(
                f"<li><code>{html.escape(str(m['role']))}</code>: "
                f"{html.escape(str(m['content_excerpt']))}</li>"
                for m in e["forwarded_tail"]
            )
            cards.append(
                "<div class='event'>"
                f"<h3>Event {i}</h3>"
                f"<p><strong>session</strong>: <code>{html.escape(e['session_id'])}</code><br>"
                f"<strong>request</strong>: <code>{html.escape(e['request_id'])}</code><br>"
                f"<strong>cache eligible</strong>: <code>{e['cache_eligible']}</code><br>"
                f"<strong>prefix preserved</strong>: <code>{e['prefix_preserved']}</code><br>"
                f"<strong>retroactive rewrite</strong>: <code>{e['retroactive_rewrite']}</code><br>"
                f"<strong>first diff index</strong>: <code>{e['first_diff_index']}</code></p>"
                f"<div class='cols'><div><h4>Original Tail</h4><ul>{orig}</ul></div>"
                f"<div><h4>Forwarded Tail</h4><ul>{fwd}</ul></div></div>"
                "</div>"
            )
        sections.append(
            f"<section class='card'><h2>{html.escape(mode)}</h2>"
            + ("".join(cards) if cards else "<p>No rewrite events captured.</p>")
            + "</section>"
        )

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Real Processed Rewrite Report</title>"
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.55;color:#111827;background:#f8fafc}"
        ".card,.event{background:white;border:1px solid #cbd5e1;border-radius:16px;padding:20px;margin:18px 0;box-shadow:0 8px 24px rgba(15,23,42,.06)}"
        ".cols{display:grid;grid-template-columns:1fr 1fr;gap:20px} code{background:#e5e7eb;padding:1px 4px;border-radius:4px} ul{padding-left:20px}"
        "</style></head><body>"
        "<h1>Real Processed Rewrite Report</h1>"
        "<div class='card'><p>Local-only report from real Claude transcript replays. Do not commit.</p></div>"
        + "".join(sections)
        + "</body></html>"
    )
    html_path.write_text(html_doc, encoding="utf-8")
    return md_path, json_path, html_path


def _write_index(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    dataset: dict[str, Any],
    observed: dict[str, Any],
    summaries: dict[str, Any],
    winners: dict[str, str],
    metadata: dict[str, Any],
    corpus: dict[str, Any],
    processed_paths: tuple[Path, Path, Path],
    token_bust_paths: tuple[Path, Path, Path],
    long_suite_paths: tuple[Path, Path, Path],
) -> tuple[Path, Path]:
    md_path = output_dir / "index.md"
    html_path = output_dir / "index.html"
    md_lines = [
        "# Cache Validation Bundle",
        "",
        "This bundle is reproducible on another machine with local Claude transcript data in `~/.claude/projects`.",
        "",
        "## Configuration",
        "",
        f"- root: `{args.root}`",
        f"- output dir: `{args.output_dir}`",
        f"- recent turns per session: `{args.recent_turns_per_session}`",
        f"- workers: `{args.workers}`",
        f"- cache TTL minutes: `{args.cache_ttl_minutes}`",
        f"- cache write multiplier: `{args.cache_write_multiplier}`",
        f"- max real processed events per mode: `{args.max_real_events_per_mode}`",
        f"- include transcript content: `{args.include_content}`",
        "",
        "## Reproducibility",
        "",
        f"- git sha: `{metadata['git_sha']}`",
        f"- git dirty: `{metadata['git_dirty']}`",
        f"- python: `{metadata['implementation']}`",
        f"- platform: `{metadata['platform']}`",
        f"- corpus session file count: `{corpus['session_file_count']}`",
        f"- corpus fingerprint: `{corpus['session_files_sha256']}`",
        "",
        "## Real Corpus Summary",
        "",
        f"- projects: `{dataset['projects']}`",
        f"- sessions: `{dataset['sessions']}`",
        f"- requests: `{dataset['requests']}`",
        f"- observed total cost: `{format_currency(observed['total_cost_usd'])}`",
        f"- winner by total cost: `{winners['total_cost']}`",
        "",
        "| Mode | Total Cost | Cache Busts | Busting Rewrites | Stable Replay Rewrites | Rewrites | Retroactive Rewrites | TTL Expiry | Forwarded Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = summaries[mode]
        md_lines.append(
            f"| `{mode}` | {format_currency(summary['total_cost_usd'])} | {summary['cache_bust_turns']} | "
            f"{summary['busting_rewrite_turns']} | {summary['stable_replay_rewrite_turns']} | "
            f"{summary['rewrite_turns']} | {summary['retroactive_rewrite_turns']} | "
            f"{summary['ttl_expiry_turns']} | {summary['forwarded_input_tokens']:,} |"
        )
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `cache_bust_turns` and `busting_rewrite_turns` are the hard-failure metrics for Anthropic prefix caching.",
            "- `stable_replay_rewrite_turns` indicates replay of previously-forwarded bytes that still preserves cache prefix stability.",
            "- `retroactive_rewrite_turns` is descriptive only; it does not imply a cache break by itself.",
            "- `ttl_expiry_turns` is workload timing context, not compression correctness.",
            "",
            "## Artifacts",
            "",
            f"- real corpus summary markdown: [real/{real_bench.OUTPUT_MD}](real/{real_bench.OUTPUT_MD})",
            f"- real corpus summary html: [real/{real_bench.OUTPUT_HTML}](real/{real_bench.OUTPUT_HTML})",
            f"- real processed markdown: [real_processed/{processed_paths[0].name}](real_processed/{processed_paths[0].name})",
            f"- real processed html: [real_processed/{processed_paths[2].name}](real_processed/{processed_paths[2].name})",
            f"- synthetic token bust markdown: [synthetic_token_bust/{token_bust_paths[0].name}](synthetic_token_bust/{token_bust_paths[0].name})",
            f"- synthetic token bust html: [synthetic_token_bust/{token_bust_paths[2].name}](synthetic_token_bust/{token_bust_paths[2].name})",
            f"- synthetic long suite markdown: [synthetic_long_suite/{long_suite_paths[0].name}](synthetic_long_suite/{long_suite_paths[0].name})",
            f"- synthetic long suite html: [synthetic_long_suite/{long_suite_paths[2].name}](synthetic_long_suite/{long_suite_paths[2].name})",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    rows = []
    for mode in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        summary = summaries[mode]
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(mode)}</code></td>"
            f"<td>{html.escape(format_currency(summary['total_cost_usd']))}</td>"
            f"<td>{summary['cache_bust_turns']}</td>"
            f"<td>{summary['busting_rewrite_turns']}</td>"
            f"<td>{summary['stable_replay_rewrite_turns']}</td>"
            f"<td>{summary['rewrite_turns']}</td>"
            f"<td>{summary['retroactive_rewrite_turns']}</td>"
            f"<td>{summary['ttl_expiry_turns']}</td>"
            f"<td>{summary['forwarded_input_tokens']:,}</td>"
            "</tr>"
        )
    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Cache Validation Bundle</title>"
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.55;color:#111827;background:#f8fafc}"
        ".card{background:white;border:1px solid #cbd5e1;border-radius:16px;padding:24px;margin:18px 0;box-shadow:0 8px 24px rgba(15,23,42,.06)}"
        "table{border-collapse:collapse;width:100%;margin:16px 0;background:white}"
        "th,td{border:1px solid #cbd5e1;padding:10px;text-align:left}th{background:#e2e8f0}"
        "code{background:#e5e7eb;padding:1px 4px;border-radius:4px}"
        "</style></head><body>"
        "<h1>Cache Validation Bundle</h1>"
        "<div class='card'>"
        f"<p><strong>root</strong>: <code>{html.escape(str(args.root))}</code><br>"
        f"<strong>recent turns per session</strong>: <code>{html.escape(str(args.recent_turns_per_session))}</code><br>"
        f"<strong>workers</strong>: <code>{args.workers}</code><br>"
        f"<strong>cache TTL minutes</strong>: <code>{args.cache_ttl_minutes}</code><br>"
        f"<strong>include transcript content</strong>: <code>{args.include_content}</code></p>"
        "</div>"
        "<div class='card'><h2>Reproducibility</h2>"
        f"<p><strong>git sha</strong>: <code>{html.escape(str(metadata['git_sha']))}</code><br>"
        f"<strong>git dirty</strong>: <code>{metadata['git_dirty']}</code><br>"
        f"<strong>python</strong>: <code>{html.escape(str(metadata['implementation']))}</code><br>"
        f"<strong>platform</strong>: <code>{html.escape(str(metadata['platform']))}</code><br>"
        f"<strong>corpus session file count</strong>: <code>{corpus['session_file_count']}</code><br>"
        f"<strong>corpus fingerprint</strong>: <code>{html.escape(str(corpus['session_files_sha256']))}</code></p>"
        "</div>"
        "<div class='card'><h2>Real Corpus Summary</h2>"
        f"<p>projects: <code>{dataset['projects']}</code><br>"
        f"sessions: <code>{dataset['sessions']}</code><br>"
        f"requests: <code>{dataset['requests']}</code><br>"
        f"observed total cost: <code>{html.escape(format_currency(observed['total_cost_usd']))}</code><br>"
        f"winner by total cost: <code>{html.escape(winners['total_cost'])}</code></p>"
        "<table><thead><tr><th>Mode</th><th>Total Cost</th><th>Cache Busts</th><th>Busting Rewrites</th>"
        "<th>Stable Replay Rewrites</th><th>Rewrites</th>"
        "<th>Retroactive Rewrites</th><th>TTL Expiry</th><th>Forwarded Tokens</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        "<p><strong>Interpretation</strong>: <code>cache_bust_turns</code> and "
        "<code>busting_rewrite_turns</code> are the hard-failure metrics. "
        "<code>stable_replay_rewrite_turns</code> is acceptable stable replay. "
        "<code>retroactive_rewrite_turns</code> is descriptive only. "
        "<code>ttl_expiry_turns</code> is workload timing context.</p></div>"
        "<div class='card'><h2>Artifacts</h2><ul>"
        f"<li><a href='real/{real_bench.OUTPUT_HTML}'>Real corpus summary HTML</a></li>"
        f"<li><a href='real/{real_bench.OUTPUT_MD}'>Real corpus summary Markdown</a></li>"
        f"<li><a href='real_processed/{processed_paths[2].name}'>Real processed rewrite HTML</a></li>"
        f"<li><a href='real_processed/{processed_paths[0].name}'>Real processed rewrite Markdown</a></li>"
        f"<li><a href='synthetic_token_bust/{token_bust_paths[2].name}'>Synthetic token-bust HTML</a></li>"
        f"<li><a href='synthetic_token_bust/{token_bust_paths[0].name}'>Synthetic token-bust Markdown</a></li>"
        f"<li><a href='synthetic_long_suite/{long_suite_paths[2].name}'>Synthetic long suite HTML</a></li>"
        f"<li><a href='synthetic_long_suite/{long_suite_paths[0].name}'>Synthetic long suite Markdown</a></li>"
        "</ul></div></body></html>"
    )
    html_path.write_text(html_doc, encoding="utf-8")
    return md_path, html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=real_bench.DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recent-turns-per-session", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--cache-ttl-minutes", type=int, default=real_bench.DEFAULT_CACHE_TTL_MINUTES
    )
    parser.add_argument("--cache-write-multiplier", type=float, default=1.25)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-real-events-per-mode", type=int, default=8)
    parser.add_argument("--content-excerpt-chars", type=int, default=220)
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="Include real transcript-derived content excerpts in the processed event reports.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=real_bench.DEFAULT_OUTPUT_DIR / real_bench.CHECKPOINT_DIRNAME,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.getLogger("headroom.transforms").setLevel(logging.WARNING)
    logging.getLogger("headroom.proxy").setLevel(logging.WARNING)

    session_files = select_session_files(args.root, max_sessions=args.max_sessions)
    if not session_files:
        print(f"No Claude session replays found under {args.root}")
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    metadata = _runtime_metadata(repo_root)
    corpus = _corpus_fingerprint(
        root=args.root,
        session_files=session_files,
        max_sessions=args.max_sessions,
        recent_turns_per_session=args.recent_turns_per_session,
        cache_ttl_minutes=args.cache_ttl_minutes,
    )
    dataset, observed = build_dataset_and_observed_from_files(
        session_files,
        cache_write_multiplier=args.cache_write_multiplier,
        recent_turns_per_session=args.recent_turns_per_session,
    )
    checkpoint_base = output_dir / "checkpoints" / corpus["session_files_sha256"]
    checkpoint_dir = resolve_checkpoint_dir(
        checkpoint_base,
        recent_turns_per_session=args.recent_turns_per_session,
        cache_ttl_minutes=args.cache_ttl_minutes,
    )

    real_output_dir = output_dir / "real"
    summaries = simulate_session_files(
        session_files,
        dataset,
        cache_ttl_minutes=args.cache_ttl_minutes,
        cache_write_multiplier=args.cache_write_multiplier,
        workers=args.workers,
        checkpoint_dir=checkpoint_dir,
        recent_turns_per_session=args.recent_turns_per_session,
    )
    real_md, real_json, real_html = write_report(real_output_dir, dataset, observed, summaries)

    processed_payload = _collect_real_processed_events(
        root=args.root,
        recent_turns_per_session=args.recent_turns_per_session,
        max_events_per_mode=args.max_real_events_per_mode,
        ttl_minutes=args.cache_ttl_minutes,
        max_chars=args.content_excerpt_chars,
        include_content=args.include_content,
    )
    processed_paths = _write_processed_event_reports(output_dir, processed_payload)

    token_bust.OUTPUT_DIR = output_dir / "synthetic_token_bust"
    token_bust_replay = token_bust._build_replay()
    original_make_proxy = token_bust.bench._make_proxy
    token_bust.bench._make_proxy = lambda mode: token_bust._FakeProxy()
    try:
        _, token_bust_summaries = token_bust.simulate_replays(
            [token_bust_replay],
            cache_ttl_minutes=token_bust.TTL_MINUTES if hasattr(token_bust, "TTL_MINUTES") else 5,
        )
        token_bust_events = token_bust._build_bust_events(token_bust_replay)
    finally:
        token_bust.bench._make_proxy = original_make_proxy
    token_bust_paths = token_bust._write_report(
        token_bust_replay,
        token_bust_summaries,
        determine_winners(token_bust_summaries),
        token_bust_events,
    )

    long_suite.OUTPUT_DIR = output_dir / "synthetic_long_suite"
    per_scenario, aggregate = long_suite._run_suite()
    long_suite_paths = long_suite._write_report(per_scenario, aggregate)

    bundle_payload = {
        "config": {
            "root": _redact_path(str(args.root.resolve())),
            "output_dir": _redact_path(str(output_dir.resolve())),
            "recent_turns_per_session": args.recent_turns_per_session,
            "workers": args.workers,
            "cache_ttl_minutes": args.cache_ttl_minutes,
            "cache_write_multiplier": args.cache_write_multiplier,
            "max_sessions": args.max_sessions,
            "max_real_events_per_mode": args.max_real_events_per_mode,
            "content_excerpt_chars": args.content_excerpt_chars,
            "include_content": args.include_content,
            "checkpoint_dir": _redact_path(str(checkpoint_dir.resolve())),
        },
        "runtime": metadata,
        "corpus": corpus,
        "real": {
            "dataset": asdict(dataset),
            "observed": asdict(observed),
            "summaries": {mode: asdict(summary) for mode, summary in summaries.items()},
            "winners": determine_winners(summaries),
            "paths": {
                "markdown": str(real_md),
                "json": str(real_json),
                "html": str(real_html),
            },
        },
        "processed_real": {
            "events": processed_payload["events"],
            "paths": {
                "markdown": str(processed_paths[0]),
                "json": str(processed_paths[1]),
                "html": str(processed_paths[2]),
            },
        },
        "synthetic_token_bust": {
            "paths": {
                "markdown": str(token_bust_paths[0]),
                "json": str(token_bust_paths[1]),
                "html": str(token_bust_paths[2]),
            }
        },
        "synthetic_long_suite": {
            "paths": {
                "markdown": str(long_suite_paths[0]),
                "json": str(long_suite_paths[1]),
                "html": str(long_suite_paths[2]),
            }
        },
    }
    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(bundle_payload, indent=2), encoding="utf-8")

    index_md, index_html = _write_index(
        output_dir,
        args=args,
        dataset=asdict(dataset),
        observed=asdict(observed),
        summaries={mode: asdict(summary) for mode, summary in summaries.items()},
        winners=determine_winners(summaries),
        metadata=metadata,
        corpus=corpus,
        processed_paths=processed_paths,
        token_bust_paths=token_bust_paths,
        long_suite_paths=long_suite_paths,
    )

    print(f"Index markdown: {index_md}")
    print(f"Index html: {index_html}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Trace and report concrete cache-busting turns from local Claude session replays."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_OUTPUT_DIR = Path("benchmark_results") / "cache_bust_trace"


@dataclass
class BustEvent:
    branch: str
    mode: str
    session_id: str
    project: str
    request_id: str
    timestamp: str
    first_diff_index: int | None
    prev_len: int
    curr_len: int
    prev_msg: dict[str, Any] | None
    curr_msg: dict[str, Any] | None
    prev_tail: list[dict[str, Any]]
    curr_tail: list[dict[str, Any]]
    retroactive_rewrite: bool


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


def _first_diff_index(prev: list[dict[str, Any]], curr: list[dict[str, Any]]) -> int | None:
    for i, (a, b) in enumerate(zip(prev, curr)):
        if a != b:
            return i
    if len(prev) != len(curr):
        return min(len(prev), len(curr))
    return None


def _trace_branch(
    repo_root: Path,
    ref: str,
    label: str,
    *,
    recent_turns_per_session: int,
    max_events_per_mode: int = 10,
) -> list[BustEvent]:
    worktree_root = Path(tempfile.mkdtemp(prefix="headroom-bust-trace-"))
    worktree_dir = worktree_root / _ref_slug(label)
    _run_git(["worktree", "add", "--detach", str(worktree_dir), ref], repo_root)
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(worktree_dir)
        code = """
import copy, json
from datetime import timedelta
from pathlib import Path
import importlib.util
import os
import sys

module_path = Path(os.environ['BUST_TRACE_SCRIPT'])
spec = importlib.util.spec_from_file_location('branch_benchmark', module_path)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

PROXY_MODE_CACHE = mod.PROXY_MODE_CACHE
PROXY_MODE_TOKEN = mod.PROXY_MODE_TOKEN
PrefixCacheTracker = mod.PrefixCacheTracker
_apply_mode_to_messages = mod._apply_mode_to_messages
_cache_gap_within_ttl = mod._cache_gap_within_ttl
_rewrite_scope = mod._rewrite_scope
get_tokenizer = mod.get_tokenizer
load_session_replay = mod.load_session_replay
select_session_files = mod.select_session_files
trim_replay_to_recent_turns = mod.trim_replay_to_recent_turns
_make_proxy = mod._make_proxy
from headroom.cache.compression_cache import CompressionCache

ROOT = Path.home() / '.claude' / 'projects'
TTL = timedelta(minutes=5)
recent_turns_per_session = int(__import__('os').environ['BUST_TRACE_RECENT'])
max_events_per_mode = int(__import__('os').environ['BUST_TRACE_MAX'])

def first_diff_index(prev, curr):
    for i, (a, b) in enumerate(zip(prev, curr)):
        if a != b:
            return i
    if len(prev) != len(curr):
        return min(len(prev), len(curr))
    return None

def trace_mode(mode):
    proxy = _make_proxy(mode)
    session_files = select_session_files(ROOT)
    events = []
    for session_file in session_files:
        replay = load_session_replay(session_file)
        if replay is None:
            continue
        replay = trim_replay_to_recent_turns(replay, recent_turns_per_session)
        prefix_tracker = PrefixCacheTracker('anthropic')
        comp_cache = CompressionCache() if mode == PROXY_MODE_TOKEN else None
        conversation = []
        conversation_token_total = 0
        previous_forwarded = []
        previous_original_context = None
        previous_forwarded_context = None
        previous_timestamp = None
        pending = None
        for turn in replay.turns:
            tokenizer = get_tokenizer(turn.model)
            turn_input_token_total = sum(tokenizer.count_message(msg) for msg in turn.input_messages)
            prior_context_message_count = len(conversation)
            conversation.extend(turn.input_messages)
            raw_input_tokens = conversation_token_total + turn_input_token_total
            forwarded = _apply_mode_to_messages(
                proxy, mode, conversation,
                model=turn.model, prefix_tracker=prefix_tracker, comp_cache=comp_cache,
                previous_original_messages=previous_original_context,
                previous_forwarded_messages=previous_forwarded_context,
            )
            if pending is not None:
                eligible = _cache_gap_within_ttl(pending.turn.timestamp, previous_timestamp, ttl=TTL)
                if eligible and previous_forwarded:
                    prefix_preserved = (
                        len(pending.forwarded) >= len(previous_forwarded)
                        and pending.forwarded[: len(previous_forwarded)] == previous_forwarded
                    )
                    if not prefix_preserved:
                        idx = first_diff_index(previous_forwarded, pending.forwarded)
                        _, retro = _rewrite_scope(
                            pending.request_messages,
                            pending.forwarded,
                            stable_prefix_message_count=max(len(previous_forwarded) - 1, 0),
                        )
                        events.append({
                            'mode': mode,
                            'session_id': replay.session_id,
                            'project': replay.decoded_project_path,
                            'request_id': pending.turn.request_id,
                            'timestamp': pending.turn.timestamp.isoformat(),
                            'first_diff_index': idx,
                            'prev_len': len(previous_forwarded),
                            'curr_len': len(pending.forwarded),
                            'prev_msg': previous_forwarded[idx] if idx is not None and idx < len(previous_forwarded) else None,
                            'curr_msg': pending.forwarded[idx] if idx is not None and idx < len(pending.forwarded) else None,
                            'prev_tail': previous_forwarded_context[-4:] if previous_forwarded_context else [],
                            'curr_tail': pending.request_messages[-4:],
                            'retroactive_rewrite': retro,
                        })
                        if len(events) >= max_events_per_mode:
                            return events
                previous_forwarded = copy.deepcopy(pending.forwarded)
                previous_timestamp = pending.turn.timestamp
            try:
                prefix_tracker.update_from_response(
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    messages=forwarded,
                    message_token_counts=[tokenizer.count_message(msg) for msg in forwarded],
                    original_messages=conversation,
                )
            except TypeError:
                prefix_tracker.update_from_response(
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    messages=forwarded,
                    message_token_counts=[tokenizer.count_message(msg) for msg in forwarded],
                )
            class Pending: pass
            pending = Pending()
            pending.turn = turn
            pending.request_messages = copy.deepcopy(conversation)
            pending.forwarded = forwarded
            conversation.append(turn.assistant_message)
            conversation_token_total = raw_input_tokens + tokenizer.count_message(turn.assistant_message)
            previous_original_context = copy.deepcopy(conversation)
            previous_forwarded_context = copy.deepcopy(forwarded) + [copy.deepcopy(turn.assistant_message)]
    return events

print(json.dumps({
    'token': trace_mode(PROXY_MODE_TOKEN),
    'cache': trace_mode(PROXY_MODE_CACHE),
}, indent=2))
"""
        env["BUST_TRACE_RECENT"] = str(recent_turns_per_session)
        env["BUST_TRACE_MAX"] = str(max_events_per_mode)
        script_path = worktree_dir / "benchmarks" / "claude_session_mode_benchmark.py"
        if not script_path.exists():
            script_path = repo_root / "benchmarks" / "claude_session_mode_benchmark.py"
        env["BUST_TRACE_SCRIPT"] = str(script_path)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=worktree_dir,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        payload = json.loads(completed.stdout)
        events: list[BustEvent] = []
        for mode in ("token", "cache"):
            for item in payload[mode]:
                events.append(BustEvent(branch=label, **item))
        return events
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"trace failed for {label} ({ref})\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}"
        ) from exc
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_dir)],
            cwd=repo_root,
            check=True,
        )


def _render_markdown(events: list[BustEvent], recent_turns_per_session: int) -> str:
    lines = [
        "# Cache Bust Trace Report",
        "",
        f"- Sampling: most recent {recent_turns_per_session} turns per session",
        "",
    ]
    for branch in ("main", "pr"):
        lines.append(f"## {branch}")
        lines.append("")
        branch_events = [e for e in events if e.branch == branch]
        for mode in ("token", "cache"):
            lines.append(f"### {mode}")
            mode_events = [e for e in branch_events if e.mode == mode]
            if not mode_events:
                lines.append("")
                lines.append("- No bust events captured.")
                lines.append("")
                continue
            for event in mode_events:
                lines.append("")
                lines.append(
                    f"- `{event.project}` `{event.session_id}` `{event.request_id}` "
                    f"{event.timestamp} diff_index={event.first_diff_index} "
                    f"retroactive={event.retroactive_rewrite}"
                )
        lines.append("")
    return "\n".join(lines)


def _render_html(events: list[BustEvent], recent_turns_per_session: int) -> str:
    sections = []
    for branch in ("main", "pr"):
        rows = []
        branch_events = [e for e in events if e.branch == branch]
        for mode in ("token", "cache"):
            mode_events = [e for e in branch_events if e.mode == mode]
            if not mode_events:
                rows.append(
                    f"<tr><td>{mode}</td><td colspan='6'>No bust events captured.</td></tr>"
                )
                continue
            for event in mode_events:
                rows.append(
                    "<tr>"
                    f"<td>{mode}</td>"
                    f"<td>{event.project}</td>"
                    f"<td>{event.session_id}</td>"
                    f"<td>{event.request_id}</td>"
                    f"<td>{event.timestamp}</td>"
                    f"<td>{event.first_diff_index}</td>"
                    f"<td>{event.retroactive_rewrite}</td>"
                    "</tr>"
                )
        sections.append(
            f"<section><h2>{branch}</h2><table><thead><tr>"
            "<th>Mode</th><th>Project</th><th>Session</th><th>Request</th>"
            "<th>Timestamp</th><th>First Diff</th><th>Retroactive</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cache Bust Trace Report</title>
  <style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
    .shell {{ max-width: 1280px; margin: 0 auto; padding: 32px 16px 48px; }}
    h1, h2 {{ letter-spacing: -0.02em; }}
    section {{ background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; white-space: nowrap; }}
    th {{ background: #f1f5f9; }}
  </style>
</head>
<body>
  <div class="shell">
    <h1>Cache Bust Trace Report</h1>
    <p>Most recent {recent_turns_per_session} turns per session.</p>
    {"".join(sections)}
  </div>
</body>
</html>"""


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    recent_turns_per_session = 200
    events = _trace_branch(
        repo_root, "upstream/main", "main", recent_turns_per_session=recent_turns_per_session
    )
    events.extend(
        _trace_branch(repo_root, "HEAD", "pr", recent_turns_per_session=recent_turns_per_session)
    )

    md_path = output_dir / "cache_bust_trace.md"
    json_path = output_dir / "cache_bust_trace.json"
    html_path = output_dir / "cache_bust_trace.html"
    md_path.write_text(_render_markdown(events, recent_turns_per_session), encoding="utf-8")
    json_path.write_text(
        json.dumps([asdict(event) for event in events], indent=2), encoding="utf-8"
    )
    html_path.write_text(_render_html(events, recent_turns_per_session), encoding="utf-8")
    print(md_path)
    print(json_path)
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

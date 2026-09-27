#!/usr/bin/env python3
"""Phase 0 token census — run this yourself against your own session logs.

    python3 scripts/census.py --root ~/.claude/projects

Groups by **tool name first** (`Read`, `Grep`, `WebFetch`, ...), with detected
content type as a secondary column — not content-type-first, which would hide
the actual go/kill question: whether the tools Headroom already excludes by
name (see `headroom/config.py`) matter in practice. A JSON config file read
through `Read` shows up under the `file_read` tool-name bucket, not
`mcp_json`, because that's the bucket whose behavior an edge would change.

Reuses `benchmarks/claude_session_mode_benchmark.py`'s `load_session_replay()`
for JSONL parsing rather than re-deriving it — it already handles this exact
Claude Code transcript shape correctly, including project-path decoding and
multi-line-per-turn grouping. This script only adds the token-census logic on
top: bucketing, lifetime-cost weighting, and cost estimation with a built-in
fallback price table for when LiteLLM is missing or doesn't know the model.

Fixed task set (Phase 0's other deliverable): SWE-bench Verified task IDs can
be pinned into a manifest here if the dataset host is reachable; the 10
own-repo tasks with written pass checks have to come from you — see
`README.md`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_SRC))

from benchmarks.claude_session_mode_benchmark import (  # noqa: E402
    DEFAULT_ROOT,
    ReplayTurn,
    SessionReplay,
    load_session_replay,
)
from headroom_edge_layer.buckets import (  # noqa: E402
    BUCKET_FILE_READ,
    BUCKET_GREP,
    BUCKET_WEB,
    classify_tool_name,
)
from headroom_edge_layer.cache_safety import estimate_tokens  # noqa: E402
from headroom_edge_layer.message_utils import get_block_text  # noqa: E402

try:
    from headroom.proxy.cost import CostTracker
except Exception:  # noqa: BLE001 - census must still run without a full install
    CostTracker = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fallback pricing. CostTracker.estimate_cost() depends on LiteLLM's pricing
# DB and returns None when LiteLLM is missing/incompatible (it doesn't
# install on Python 3.14+) or the model isn't in its database — this census
# must fail loudly on that (a visible "unpriced" row), never silently
# record $0. Prices are USD per 1M tokens, approximate, and meant to be
# extended, not trusted as authoritative — the litellm path is preferred
# whenever it works.
# ---------------------------------------------------------------------------
_FALLBACK_PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (0.8, 4.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "o1": (15.0, 60.0),
}
_CACHE_READ_MULTIPLIER = 0.1
_CACHE_WRITE_5M_MULTIPLIER = 1.25
_CACHE_WRITE_1H_MULTIPLIER = 2.0


def _fallback_price_key(model: str) -> str | None:
    lower = model.lower()
    for key in _FALLBACK_PRICES_PER_1M:
        if key in lower:
            return key
    return None


@dataclass
class CostEstimate:
    usd: float | None
    source: str  # "litellm" | "fallback" | "unpriced"


def estimate_request_cost(
    cost_tracker: Any,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_5m_tokens: int,
    cache_write_1h_tokens: int,
) -> CostEstimate:
    total_cache_write = cache_write_5m_tokens + cache_write_1h_tokens
    if cost_tracker is not None:
        real = cost_tracker.estimate_cost(
            model,
            input_tokens,
            output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=total_cache_write,
        )
        if real is not None:
            return CostEstimate(usd=real, source="litellm")

    key = _fallback_price_key(model)
    if key is None:
        return CostEstimate(usd=None, source="unpriced")
    input_price, output_price = _FALLBACK_PRICES_PER_1M[key]
    usd = (
        input_tokens * input_price
        + cache_read_tokens * input_price * _CACHE_READ_MULTIPLIER
        + cache_write_5m_tokens * input_price * _CACHE_WRITE_5M_MULTIPLIER
        + cache_write_1h_tokens * input_price * _CACHE_WRITE_1H_MULTIPLIER
    ) / 1_000_000 + output_tokens * output_price / 1_000_000
    return CostEstimate(usd=usd, source="fallback")


# ---------------------------------------------------------------------------
# Supplementary scan for usage.cache_creation's 5m/1h split, which
# load_session_replay() does not carry on ReplayTurn (it only keeps the
# aggregate cache_creation_input_tokens). Keyed by request_id so it merges
# cleanly with the turns load_session_replay() already reconstructed —
# this is intentionally the only place this script re-reads the raw JSONL.
# ---------------------------------------------------------------------------
def _read_cache_creation_splits(session_file: Path) -> dict[str, tuple[int, int]]:
    splits: dict[str, tuple[int, int]] = {}
    try:
        with session_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "assistant" or not event.get("requestId"):
                    continue
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                usage = message.get("usage")
                if not isinstance(usage, dict):
                    continue
                creation = usage.get("cache_creation")
                if not isinstance(creation, dict):
                    continue
                write_5m = int(creation.get("ephemeral_5m_input_tokens", 0) or 0)
                write_1h = int(creation.get("ephemeral_1h_input_tokens", 0) or 0)
                if write_5m or write_1h:
                    splits[str(event["requestId"])] = (write_5m, write_1h)
    except OSError:
        pass
    return splits


# ---------------------------------------------------------------------------
# Lightweight content-type guess for the secondary column. Deliberately not
# the full Rust-backed detector (headroom/transforms/content_detector.py) —
# this is a census, not a production compression decision, and staying
# dependency-light keeps this script runnable standalone.
# ---------------------------------------------------------------------------
def _guess_content_type(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if stripped[0] in "{[":
        try:
            json.loads(stripped)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass
    if stripped.startswith("diff --git") or re.search(r"^@@ -\d+", stripped, re.MULTILINE):
        return "diff"
    if re.search(r"<html|<!DOCTYPE html", stripped, re.IGNORECASE):
        return "html"
    if re.search(r"^\s*(def |class |function |import |#include)", stripped, re.MULTILINE):
        return "source_code"
    if re.search(r"^[^:\n]+:\d+:", stripped, re.MULTILINE):
        return "search_results"
    return "text"


def _extract_path(tool_input: Any) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in ("file_path", "path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass
class BucketStats:
    fresh_tokens: int = 0
    lifetime_weighted_tokens: float = 0.0
    occurrences: int = 0
    repeat_reads: int = 0
    oversized_outputs: int = 0
    content_types: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class CensusResult:
    bucket_stats: dict[str, BucketStats] = field(default_factory=lambda: defaultdict(BucketStats))
    sessions_scanned: int = 0
    sessions_with_turns: int = 0
    total_real_cost_usd: float = 0.0
    unpriced_requests: int = 0
    fallback_priced_requests: int = 0
    total_fresh_tokens: int = 0
    total_lifetime_weighted_tokens: float = 0.0


def census_session(replay: SessionReplay, result: CensusResult, cost_tracker: Any, splits: dict[str, tuple[int, int]]) -> None:
    total_turns = len(replay.turns)
    if total_turns == 0:
        return
    result.sessions_with_turns += 1
    tool_use_index: dict[str, tuple[str, Any]] = {}
    seen_read_paths: dict[str, set[str]] = defaultdict(set)

    for turn_index, turn in enumerate(replay.turns):
        turns_left = total_turns - turn_index - 1
        weight = 1.25 + 0.1 * turns_left

        for message in turn.input_messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if message.get("role") == "user" and isinstance(content, str) and content.strip():
                tokens = estimate_tokens(content)
                _record(result, "other", tokens, weight, content_type="user_text")
                continue
            if message.get("role") != "user" or not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_use_id = block.get("tool_use_id")
                tool_name, tool_input = tool_use_index.get(tool_use_id, (None, None)) if isinstance(
                    tool_use_id, str
                ) else (None, None)
                text = get_block_text(block) or ""
                if not text:
                    continue
                bucket = classify_tool_name(tool_name)
                tokens = estimate_tokens(text)
                content_type = _guess_content_type(text)
                is_repeat = False
                is_oversized = False
                if bucket == BUCKET_FILE_READ:
                    path = _extract_path(tool_input)
                    if path:
                        if path in seen_read_paths and text in seen_read_paths[path]:
                            is_repeat = True
                        seen_read_paths[path].add(text)
                elif bucket == BUCKET_GREP:
                    is_oversized = text.count("\n") + 1 > 100
                elif bucket == BUCKET_WEB:
                    is_oversized = tokens > 2000
                _record(
                    result,
                    bucket,
                    tokens,
                    weight,
                    content_type=content_type,
                    repeat=is_repeat,
                    oversized=is_oversized,
                )

        # Register this turn's tool_use calls for the NEXT turn's tool_results
        # to resolve against — input_messages only ever carries the tail
        # since the previous turn, never the calling tool_use block itself.
        assistant_content = turn.assistant_message.get("content")
        if isinstance(assistant_content, list):
            for block in assistant_content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_use_id = block.get("id")
                    if isinstance(tool_use_id, str):
                        tool_use_index[tool_use_id] = (block.get("name", ""), block.get("input"))

        write_5m, write_1h = splits.get(turn.request_id, (turn.observed_cache_write_tokens, 0))
        estimate = estimate_request_cost(
            cost_tracker,
            model=turn.model,
            input_tokens=turn.observed_input_tokens,
            output_tokens=turn.output_tokens,
            cache_read_tokens=turn.observed_cache_read_tokens,
            cache_write_5m_tokens=write_5m,
            cache_write_1h_tokens=write_1h,
        )
        if estimate.source == "unpriced":
            result.unpriced_requests += 1
        elif estimate.source == "fallback":
            result.fallback_priced_requests += 1
            result.total_real_cost_usd += estimate.usd or 0.0
        else:
            result.total_real_cost_usd += estimate.usd or 0.0


def _record(
    result: CensusResult,
    bucket: str,
    tokens: int,
    weight: float,
    *,
    content_type: str,
    repeat: bool = False,
    oversized: bool = False,
) -> None:
    stats = result.bucket_stats[bucket]
    stats.fresh_tokens += tokens
    stats.lifetime_weighted_tokens += tokens * weight
    stats.occurrences += 1
    stats.content_types[content_type] += tokens
    if repeat:
        stats.repeat_reads += 1
    if oversized:
        stats.oversized_outputs += 1
    result.total_fresh_tokens += tokens
    result.total_lifetime_weighted_tokens += tokens * weight


def format_report(result: CensusResult) -> str:
    lines: list[str] = []
    lines.append(f"Sessions scanned: {result.sessions_scanned}")
    lines.append(f"Sessions with at least one turn: {result.sessions_with_turns}")
    lines.append(f"Total estimated real cost (USD): {result.total_real_cost_usd:.4f}")
    if result.unpriced_requests:
        lines.append(
            f"WARNING: {result.unpriced_requests} requests could not be priced "
            "(unknown model, no LiteLLM, and no fallback table entry) — cost total "
            "above UNDERCOUNTS. Add the model to _FALLBACK_PRICES_PER_1M."
        )
    if result.fallback_priced_requests:
        lines.append(
            f"NOTE: {result.fallback_priced_requests} requests priced via the "
            "built-in fallback table, not LiteLLM — treat as approximate."
        )
    lines.append("")
    lines.append("| Bucket | Share of fresh input tokens | Share of lifetime cost | "
                  "Repeat reads / oversized outputs | Top content types |")
    lines.append("|---|---|---|---|---|")

    for bucket, stats in sorted(
        result.bucket_stats.items(), key=lambda kv: kv[1].lifetime_weighted_tokens, reverse=True
    ):
        fresh_share = (
            stats.fresh_tokens / result.total_fresh_tokens if result.total_fresh_tokens else 0.0
        )
        lifetime_share = (
            stats.lifetime_weighted_tokens / result.total_lifetime_weighted_tokens
            if result.total_lifetime_weighted_tokens
            else 0.0
        )
        top_types = sorted(stats.content_types.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_types_str = ", ".join(f"{name} ({tokens}t)" for name, tokens in top_types)
        lines.append(
            f"| {bucket} | {fresh_share:.1%} | {lifetime_share:.1%} | "
            f"{stats.repeat_reads} repeats / {stats.oversized_outputs} oversized | {top_types_str} |"
        )

    file_read_grep_web_share = sum(
        stats.lifetime_weighted_tokens
        for bucket, stats in result.bucket_stats.items()
        if bucket in (BUCKET_FILE_READ, BUCKET_GREP, BUCKET_WEB)
    ) / (result.total_lifetime_weighted_tokens or 1)
    lines.append("")
    lines.append(
        f"file_read + grep + web = {file_read_grep_web_share:.1%} of lifetime cost. "
        "Gate: below ~25% means the skipped-content edge is small — move effort "
        "toward output-side savings and model routing before building Phase 2 further."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    if not args.root.exists():
        print(f"No such directory: {args.root}", file=sys.stderr)
        return 1

    cost_tracker = CostTracker() if CostTracker is not None else None
    result = CensusResult()
    session_files = sorted(args.root.glob("*/*.jsonl"))
    if args.max_sessions:
        session_files = session_files[: args.max_sessions]

    for session_file in session_files:
        replay = load_session_replay(session_file)
        result.sessions_scanned += 1
        if replay is None:
            continue
        splits = _read_cache_creation_splits(session_file)
        census_session(replay, result, cost_tracker, splits)

    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

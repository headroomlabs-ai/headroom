"""Edge 3 — grep results, with the enumeration guard.

Parses `path:line:content` lines (ripgrep/grep -n default), groups by file,
and keeps every file path with its match count — only surplus lines beyond
a per-file cap are elided. Never prunes when the command asks for file lists
or counts, or when the intent looks like an enumeration task ("rename every
usage"): silently missing hits on those tasks is exactly the failure mode
the protocol calls out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .cache_safety import (
    SavingsLedgerEntry,
    already_compressed,
    cap_input,
    estimate_tokens,
    format_elided_marker,
    store_original,
)
from .config import EdgeLayerConfig
from .scoring import Scorer

_LINE_RE = re.compile(r"^(?P<path>[^\n:]+):(?P<line>\d+):(?P<content>.*)$")

# Flags that mean "the model wants the file list/count itself, not the match
# text" — pruning match lines would be harmless here since none are kept
# anyway, but skip entirely to stay obviously correct.
_ENUMERATION_FLAGS = ("-l", "--files-with-matches", "-c", "--count")

_ENUMERATION_INTENT_WORDS = (
    "rename",
    "replace",
    "every",
    "all usages",
    "all references",
    "every usage",
    "every reference",
)


@dataclass
class _FileGroup:
    path: str
    lines: list[tuple[int, str, int]]  # (original_index, raw_line, match_line_no)


def _is_enumeration(*, tool_input: dict[str, Any], intent_query: str) -> bool:
    command = str(tool_input.get("command", "")) if isinstance(tool_input, dict) else ""
    args = command.split() if command else []
    if isinstance(tool_input, dict):
        raw_flags = tool_input.get("flags") or tool_input.get("args") or []
        if isinstance(raw_flags, list):
            args.extend(str(a) for a in raw_flags)
    if any(flag in args for flag in _ENUMERATION_FLAGS):
        return True
    intent_lower = intent_query.lower()
    return any(word in intent_lower for word in _ENUMERATION_INTENT_WORDS)


def _parse_groups(text: str) -> tuple[list[_FileGroup], dict[int, str]]:
    """Returns (file groups in first-seen order, {line_index: raw_line} for unmatched lines)."""
    groups: dict[str, _FileGroup] = {}
    order: list[str] = []
    unmatched: dict[int, str] = {}
    for index, raw_line in enumerate(text.split("\n")):
        match = _LINE_RE.match(raw_line)
        if match is None:
            unmatched[index] = raw_line
            continue
        path = match.group("path")
        if path not in groups:
            groups[path] = _FileGroup(path=path, lines=[])
            order.append(path)
        groups[path].lines.append((index, raw_line, int(match.group("line"))))
    return [groups[p] for p in order], unmatched


def compress_grep_result(
    text: str,
    *,
    tool_name: str,
    tool_call_id: str | None,
    tool_input: dict[str, Any],
    intent_query: str,
    config: EdgeLayerConfig,
    store: Any,
    scorer: Scorer,
    request_id: str,
) -> tuple[str, SavingsLedgerEntry | None]:
    if already_compressed(text):
        return text, None
    capped = cap_input(text, max_chars=config.max_input_chars_per_edge)
    if capped is None:
        return text, None
    if _is_enumeration(tool_input=tool_input, intent_query=intent_query):
        return text, None

    groups, unmatched = _parse_groups(text)
    if not groups:
        return text, None

    cap = config.grep_per_file_cap
    total_before = estimate_tokens(text)
    any_pruned = False

    # index -> line(s) to emit at that original position; kept in original
    # order below by sorting on the index. A pruned group's marker rides
    # along on the last kept line's entry rather than a synthetic new index,
    # so it always lands right after that file's kept lines and never before
    # a later file's lines that happen to share no index collision.
    output_lines: dict[int, str] = dict(unmatched)
    for group in groups:
        if len(group.lines) <= cap:
            for index, raw_line, _ in group.lines:
                output_lines[index] = raw_line
            continue

        any_pruned = True
        contents = [raw_line for _, raw_line, _ in group.lines]
        scores = scorer.score(intent_query, contents)
        ranked = sorted(range(len(group.lines)), key=lambda i: scores[i], reverse=True)
        kept_positions = set(ranked[:cap])
        kept_indices = [group.lines[i][0] for i in sorted(kept_positions)]

        elided_lines = [
            group.lines[i][1] for i in range(len(group.lines)) if i not in kept_positions
        ]
        elided_text = "\n".join(elided_lines)
        hash_key = store_original(
            store,
            elided_text,
            elided_text,
            config_version=config.config_version,
            salt=f"grep:{group.path}",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            query_context=intent_query,
        )
        marker = format_elided_marker(
            count=len(group.lines), unit="lines", kept=len(kept_positions), hash_key=hash_key
        )

        # Only the kept lines get an output entry — elided originals are
        # simply never added — and the file path plus true match count are
        # never dropped: the marker's `count=` names it even when every
        # match line but one is elided.
        line_by_index = {index: raw_line for index, raw_line, _ in group.lines}
        for index in kept_indices:
            output_lines[index] = line_by_index[index]
        last_kept_index = kept_indices[-1]
        output_lines[last_kept_index] = output_lines[last_kept_index] + f"\n{marker}"

    if not any_pruned:
        return text, None

    compressed = "\n".join(output_lines[i] for i in sorted(output_lines))
    tokens_after = estimate_tokens(compressed)
    entry = SavingsLedgerEntry(
        edge="grep",
        bucket="search",
        tool_name=tool_name,
        tokens_before=total_before,
        tokens_after=tokens_after,
        hash_key=None,
        request_id=request_id,
    )
    return compressed, entry

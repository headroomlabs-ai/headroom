"""Edge 4a/4b — file reads: repeat reads and reads-after-edits only.

Deliberately does NOT build 4c (outlines for 800+ line files) — the protocol
gates that behind a live A/B showing no rise in failed edits, which this
sandbox can't run.

Both sub-edges are pure functions of the *current* message list: the client
resends raw history every turn (cache-safety rule #1), so there is no need
for cross-request state — the same earlier occurrence of a file's content is
either still present in this turn's messages or it isn't, and that's exactly
what "still in context" means. Compares file content after stripping the
CLI's cat--n-style line-number prefix, since that format is not a stable
contract across agent-client versions ("Claude Code prefixes read lines with
line numbers, and that format can change between versions").
"""

from __future__ import annotations

import copy
import difflib
import re
from dataclasses import dataclass
from typing import Any

from .cache_safety import (
    SavingsLedgerEntry,
    already_compressed,
    cap_input,
    estimate_tokens,
    format_unchanged_marker,
    store_original,
)
from .config import EdgeLayerConfig
from .message_utils import collect_tool_use_index, get_block_text, set_block_text

_LINE_NUM_PREFIX = re.compile(r"^\s*\d+\t")


def _strip_line_numbers(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(_LINE_NUM_PREFIX.sub("", line) for line in lines)


def _extract_path(tool_input: Any) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in ("file_path", "path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass
class _ReadOccurrence:
    message_index: int
    block_index: int
    tool_call_id: str
    path: str
    text: str


def _collect_read_occurrences(
    messages: list[dict[str, Any]],
    *,
    tool_use_index: dict[str, tuple[str, Any]],
    read_tool_names: frozenset[str],
) -> list[_ReadOccurrence]:
    occurrences: list[_ReadOccurrence] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str) or tool_use_id not in tool_use_index:
                continue
            tool_name, tool_input = tool_use_index[tool_use_id]
            if tool_name not in read_tool_names:
                continue
            path = _extract_path(tool_input)
            if not path:
                continue
            text = get_block_text(block)
            if text is None or already_compressed(text):
                continue
            occurrences.append(
                _ReadOccurrence(
                    message_index=message_index,
                    block_index=block_index,
                    tool_call_id=tool_use_id,
                    path=path,
                    text=text,
                )
            )
    return occurrences


def compress_file_reads(
    messages: list[dict[str, Any]],
    *,
    config: EdgeLayerConfig,
    store: Any,
    request_id: str,
) -> tuple[list[dict[str, Any]], list[SavingsLedgerEntry]]:
    """Returns (possibly-mutated deep copy of messages, ledger entries).

    Only mutates `tool_result` blocks belonging to a read-tool call; every
    other message/block — including any assistant `thinking` content that
    happens to sit alongside them — is passed through byte-identical.
    """
    tool_use_index = collect_tool_use_index(messages)
    occurrences = _collect_read_occurrences(
        messages, tool_use_index=tool_use_index, read_tool_names=config.read_tool_names
    )
    if len(occurrences) < 2:
        return messages, []

    by_path: dict[str, list[_ReadOccurrence]] = {}
    for occ in occurrences:
        by_path.setdefault(occ.path, []).append(occ)

    ledger: list[SavingsLedgerEntry] = []
    mutations: dict[tuple[int, int], str] = {}

    for path, occs in by_path.items():
        if len(occs) < 2:
            continue
        for later, earlier in zip(occs[1:], occs[:-1]):
            capped = cap_input(later.text, max_chars=config.max_input_chars_per_edge)
            if capped is None:
                continue
            earlier_stripped = _strip_line_numbers(earlier.text)
            later_stripped = _strip_line_numbers(later.text)

            if earlier_stripped == later_stripped:
                marker, hash_key = _unchanged_marker(
                    later, earlier, config=config, store=store, request_id=request_id
                )
                mutations[(later.message_index, later.block_index)] = marker
                ledger.append(
                    SavingsLedgerEntry(
                        edge="file_read_repeat",
                        bucket="file_read",
                        tool_name="Read",
                        tokens_before=estimate_tokens(later.text),
                        tokens_after=estimate_tokens(marker),
                        hash_key=hash_key,
                        request_id=request_id,
                    )
                )
                continue

            diff_text = "\n".join(
                difflib.unified_diff(
                    earlier_stripped.split("\n"),
                    later_stripped.split("\n"),
                    fromfile=f"{path} (message {earlier.message_index})",
                    tofile=f"{path} (message {later.message_index})",
                    lineterm="",
                )
            )
            if not diff_text:
                continue
            if len(diff_text) >= config.file_read_diff_max_ratio * max(1, len(later.text)):
                # Diff isn't small enough to trust; leave the full read alone.
                continue

            hash_key = store_original(
                store,
                later.text,
                diff_text,
                config_version=config.config_version,
                salt=f"file_read_diff:{path}:{later.message_index}",
                tool_name="Read",
                tool_call_id=later.tool_call_id,
                ttl=config.file_read_marker_ttl_seconds,
            )
            replacement = (
                f"[Diff since message {earlier.message_index}, {path}:\n{diff_text}\n"
                f"Retrieve original: hash={hash_key}]"
            )
            mutations[(later.message_index, later.block_index)] = replacement
            ledger.append(
                SavingsLedgerEntry(
                    edge="file_read_diff",
                    bucket="file_read",
                    tool_name="Read",
                    tokens_before=estimate_tokens(later.text),
                    tokens_after=estimate_tokens(replacement),
                    hash_key=hash_key,
                    request_id=request_id,
                )
            )

    if not mutations:
        return messages, []

    result = copy.deepcopy(messages)
    for (message_index, block_index), new_text in mutations.items():
        block = result[message_index]["content"][block_index]
        set_block_text(block, new_text)
    return result, ledger


def _unchanged_marker(
    later: _ReadOccurrence,
    earlier: _ReadOccurrence,
    *,
    config: EdgeLayerConfig,
    store: Any,
    request_id: str,
) -> tuple[str, str]:
    # Re-store on every occurrence so the entry's TTL keeps getting refreshed
    # — a marker replayed on turn 40 of a session must not point at an entry
    # that expired at the 1800s default (cache-safety rule #5).
    hash_key = store_original(
        store,
        later.text,
        later.text,
        config_version=config.config_version,
        salt=f"file_read_unchanged:{later.path}",
        tool_name="Read",
        tool_call_id=later.tool_call_id,
        ttl=config.file_read_marker_ttl_seconds,
    )
    line_count = later.text.count("\n") + 1
    marker = format_unchanged_marker(
        origin_index=earlier.message_index,
        path=later.path,
        line_range=f"1-{line_count}",
        hash_key=hash_key,
    )
    return marker, hash_key

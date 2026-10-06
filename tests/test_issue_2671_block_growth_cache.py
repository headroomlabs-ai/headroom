"""Comprehensive regression for #2671's block-growing Anthropic histories.

The provider writes cache entries only at explicit breakpoints and searches at
most 20 block boundaries backwards on the next request.  Consequently:

* a pure append must advance the breakpoint to the newest block;
* a rewritten tail must anchor at the last byte-stable leading block;
* both shapes must retain one conversation lineage across turns;
* different tools/thinking profiles must never share that lineage, because
  Anthropic renders those segments before messages in its cache key.

The small cache oracle below models those write/lookback rules.  It catches a
green-but-inert implementation: merely moving a marker in a unit-built message
is insufficient unless the real resolve -> normalize -> record sequence carries
the previous turn's state forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from headroom.cache.prefix_tracker import (
    RELATION_BLOCK_APPEND,
    RELATION_BLOCK_REWRITE_TAIL,
    RELATION_DIVERGED,
    PrefixFreezeConfig,
    SessionTrackerStore,
    _strip_cache_control,
    classify_history_relation,
    extract_cache_stable_delta,
    normalize_message_cache_control,
    overlay_cached_prefix,
    segment_fingerprint,
)


def _text(text: str, *, cache: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def _message(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": blocks}]


def _pure_append(total: int) -> list[dict[str, Any]]:
    return _message([_text(f"block-{index}") for index in range(total)])


def _rewritten_tail(
    turn: int,
    churn_blocks: int,
    *,
    stable_blocks: int = 30,
    instruction: str = "instruction: summarize",
) -> list[dict[str, Any]]:
    blocks = [_text(f"stable-{index}") for index in range(stable_blocks)]
    blocks += [_text(f"turn-{turn}-changing-{index}") for index in range(churn_blocks)]
    # The captured production shape keeps a two-block identity suffix pinned at
    # the end while the blocks immediately before it are rewritten.
    blocks += [_text(instruction), _text("fixed end-of-transcript reminder")]
    return _message(blocks)


def _breakpoint(messages: list[dict[str, Any]]) -> tuple[int, int]:
    found = [
        (message_index, block_index)
        for message_index, message in enumerate(messages)
        if isinstance(message.get("content"), list)
        for block_index, block in enumerate(message["content"])
        if isinstance(block, dict) and "cache_control" in block
    ]
    assert len(found) == 1
    return found[0]


@dataclass
class _AnthropicBreakpointCache:
    """Deterministic model of Anthropic's explicit-breakpoint cache lookup."""

    entries: dict[str, int] = field(default_factory=dict)
    lookback_blocks: int = 20

    @staticmethod
    def _blocks(messages: list[dict[str, Any]]) -> list[Any]:
        blocks: list[Any] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                blocks.extend(_strip_cache_control(content))
        return blocks

    @staticmethod
    def _key(blocks: list[Any], end: int) -> str:
        return json.dumps(blocks[: end + 1], sort_keys=True, separators=(",", ":"))

    def request(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """Return simulated ``(cache_read_blocks, cache_write_blocks)``."""
        _, breakpoint = _breakpoint(messages)
        blocks = self._blocks(messages)
        read = 0
        first = max(0, breakpoint - self.lookback_blocks + 1)
        for candidate in range(breakpoint, first - 1, -1):
            key = self._key(blocks, candidate)
            if key in self.entries:
                read = self.entries[key]
                break
        written_prefix = breakpoint + 1
        write = max(0, written_prefix - read)
        self.entries[self._key(blocks, breakpoint)] = written_prefix
        return read, write


def _record(tracker, original, forwarded, *, read=0, write=10_000):  # noqa: ANN001
    tracker.update_from_response(
        cache_read_tokens=read,
        cache_write_tokens=write,
        messages=forwarded,
        original_messages=original,
    )


def test_classifier_separates_pure_append_from_rewritten_tail() -> None:
    append = classify_history_relation(_pure_append(35), _pure_append(30))
    rewrite = classify_history_relation(_rewritten_tail(2, 5), _rewritten_tail(1, 3))

    assert append.kind == RELATION_BLOCK_APPEND
    assert append.stable_prefix_blocks == 30
    assert rewrite.kind == RELATION_BLOCK_REWRITE_TAIL
    assert rewrite.stable_prefix_blocks == 30
    assert rewrite.stable_suffix_blocks == 2


def test_rewritten_tail_requires_a_real_previous_divergence() -> None:
    """The #2702 bug classified a pure append as a rewritten tail."""
    previous = _pure_append(30)
    current = _pure_append(31)
    relation = classify_history_relation(current, previous)

    assert relation.kind == RELATION_BLOCK_APPEND
    assert relation.stable_prefix_blocks == relation.previous_block_count


def test_rewritten_tail_requires_a_two_block_identity_suffix() -> None:
    """Sibling sub-calls sharing a transcript and generic reminder must split."""
    previous = _rewritten_tail(1, 3, instruction="instruction: summarize")
    sibling = _rewritten_tail(2, 5, instruction="instruction: title")

    assert classify_history_relation(sibling, previous).kind == RELATION_DIVERGED


def test_lineage_survives_rewritten_tail_growth_and_delivers_previous_state() -> None:
    store = SessionTrackerStore(PrefixFreezeConfig(min_cached_tokens=0))
    first_tracker = None

    for turn, churn in enumerate((3, 5, 8, 11), start=1):
        original = _rewritten_tail(turn, churn)
        tracker = store.resolve_tracker("shared", "anthropic", messages=original)
        first_tracker = first_tracker or tracker
        assert tracker is first_tracker
        previous = tracker.get_last_forwarded_messages()
        if turn > 1:
            assert previous, "lineage match must deliver the previous forwarded request"
        forwarded = normalize_message_cache_control(original, previous)
        _record(tracker, original, forwarded)

    assert store.active_sessions == 1
    assert first_tracker._turn_number == 4


def test_sibling_rewritten_tail_streams_do_not_ping_pong() -> None:
    store = SessionTrackerStore()
    seen = {}
    for turn, churn in enumerate((3, 5, 8), start=1):
        for instruction in ("instruction: summarize", "instruction: title"):
            original = _rewritten_tail(turn, churn, instruction=instruction)
            tracker = store.resolve_tracker("shared", "anthropic", messages=original)
            seen.setdefault(instruction, tracker)
            assert tracker is seen[instruction]
            forwarded = normalize_message_cache_control(
                original, tracker.get_last_forwarded_messages()
            )
            _record(tracker, original, forwarded)

    assert seen["instruction: summarize"] is not seen["instruction: title"]


def test_cache_affinity_splits_identical_histories_with_different_tools() -> None:
    store = SessionTrackerStore()
    history = _pure_append(30)
    shell = segment_fingerprint({"model": "claude", "tools": [{"name": "shell"}]})
    search = segment_fingerprint({"model": "claude", "tools": [{"name": "search"}]})

    shell_tracker = store.resolve_tracker(
        "shared", "anthropic", messages=history, cache_affinity=shell
    )
    search_tracker = store.resolve_tracker(
        "shared", "anthropic", messages=history, cache_affinity=search
    )

    assert search_tracker is not shell_tracker
    assert (
        store.resolve_tracker("shared", "anthropic", messages=history, cache_affinity=shell)
        is shell_tracker
    )


def test_cache_affinity_ignores_only_cache_directive_movement() -> None:
    base = {
        "model": "claude",
        "tools": [{"name": "shell", "cache_control": {"type": "ephemeral"}}],
    }
    moved = {"model": "claude", "tools": [{"name": "shell"}]}
    changed = {"model": "claude", "tools": [{"name": "search"}]}

    assert segment_fingerprint(base) == segment_fingerprint(moved)
    assert segment_fingerprint(base) != segment_fingerprint(changed)


def test_pure_append_replays_forwarded_blocks_and_advances_breakpoint() -> None:
    previous_original = _pure_append(30)
    previous_forwarded = _message([_text(f"C-{index}") for index in range(30)])
    current = _pure_append(34)

    overlaid = overlay_cached_prefix(current, current, previous_original, previous_forwarded)
    normalized = normalize_message_cache_control(overlaid, previous_forwarded)

    assert [block["text"] for block in normalized[0]["content"][:30]] == [
        f"C-{index}" for index in range(30)
    ]
    assert [block["text"] for block in normalized[0]["content"][30:]] == [
        f"block-{index}" for index in range(30, 34)
    ]
    assert _breakpoint(normalized) == (0, 33)


def test_whole_message_delta_path_cannot_discard_appended_blocks() -> None:
    """Block appends require a splice, never an empty whole-message delta."""
    previous = _pure_append(30)

    assert extract_cache_stable_delta(_pure_append(34), previous, previous) is None


def test_cache_oracle_proves_pure_append_chains_without_rewrites() -> None:
    oracle = _AnthropicBreakpointCache()
    previous = None
    outcomes = []
    for total in (30, 34, 38, 43):
        current = _pure_append(total)
        forwarded = normalize_message_cache_control(current, previous)
        outcomes.append(oracle.request(forwarded))
        previous = forwarded

    assert outcomes == [(0, 30), (30, 4), (34, 4), (38, 5)]


def test_cache_oracle_proves_rewritten_tail_stops_perpetual_full_writes() -> None:
    oracle = _AnthropicBreakpointCache()
    previous = None
    outcomes = []
    breakpoints = []
    for turn, churn in enumerate((3, 5, 8, 11), start=1):
        current = _rewritten_tail(turn, churn)
        forwarded = normalize_message_cache_control(current, previous)
        breakpoints.append(_breakpoint(forwarded)[1])
        outcomes.append(oracle.request(forwarded))
        previous = forwarded

    # Cold turn writes its varying tail.  Turn two establishes the new stable
    # boundary; subsequent turns read it and perform no repeated full write.
    assert breakpoints == [34, 29, 29, 29]
    assert outcomes[0] == (0, 35)
    assert outcomes[1] == (0, 30)
    assert outcomes[2:] == [(30, 0), (30, 0)]


def test_relocation_kill_switch_restores_newest_block(monkeypatch) -> None:  # noqa: ANN001
    previous = normalize_message_cache_control(_rewritten_tail(1, 3))
    monkeypatch.setenv("HEADROOM_STABLE_BOUNDARY_BREAKPOINT", "0")
    current = _rewritten_tail(2, 5)

    forwarded = normalize_message_cache_control(current, previous)

    assert _breakpoint(forwarded) == (0, len(current[0]["content"]) - 1)

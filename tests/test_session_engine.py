"""Unit tests for the shared session-turn engine (headroom/proxy/session_engine).

The engine is the single cache-management brain for the proxy request paths
and the sidecar /v1/compress path; these tests pin its two freeze policies
and the overlay finalization directly, without an HTTP harness.
"""

from __future__ import annotations

import json

import pytest

from headroom.cache.compression_cache import CompressionCache
from headroom.proxy.session_engine import (
    FREEZE_POLICY_CONFIRMED_CLAMP,
    FREEZE_POLICY_REPLAYABLE,
    finalize_turn,
    prepare_turn,
)


def _tool_msg(content: str, call_id: str = "c1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _history_with_cached_tool(
    cache: CompressionCache, original: str, compressed: str
) -> list[dict]:
    """A 3-message history whose tool result has a cached compressed form."""
    cache.store_compressed(cache.content_hash(original), compressed, tokens_saved=10)
    return [
        {"role": "user", "content": "get items"},
        {"role": "assistant", "content": "calling"},
        _tool_msg(original),
    ]


# --------------------------------------------------------------------------- #
# prepare_turn: freeze policies                                                #
# --------------------------------------------------------------------------- #


def test_sidecar_policy_freezes_full_replayable_prefix() -> None:
    cache = CompressionCache()
    messages = _history_with_cached_tool(cache, "ORIGINAL " * 100, "[compressed]")
    messages.append({"role": "user", "content": "next"})

    prep = prepare_turn(cache, messages, policy=FREEZE_POLICY_REPLAYABLE)
    # user, assistant, cached tool are all stable; the trailing message is
    # always excluded by compute_frozen_count.
    assert prep.frozen_message_count == 3
    # The swap replaced the tool result with its cached compressed form.
    assert prep.pipeline_input[2]["content"] == "[compressed]"
    # The caller's list is never mutated.
    assert messages[2]["content"].startswith("ORIGINAL")


def test_sidecar_policy_explicit_pin_wins_when_larger() -> None:
    cache = CompressionCache()
    messages = [
        {"role": "user", "content": "a"},
        _tool_msg("never seen before " * 50),  # not in cache -> derived stops here
        {"role": "user", "content": "next"},
    ]
    derived = cache.compute_frozen_count(messages)
    assert derived == 1  # only the leading plain message
    prep = prepare_turn(cache, messages, policy=FREEZE_POLICY_REPLAYABLE, explicit_frozen=2)
    assert prep.frozen_message_count == 2


def test_sidecar_policy_derived_wins_when_explicit_smaller() -> None:
    cache = CompressionCache()
    messages = _history_with_cached_tool(cache, "ORIGINAL " * 100, "[compressed]")
    messages.append({"role": "user", "content": "next"})
    prep = prepare_turn(cache, messages, policy=FREEZE_POLICY_REPLAYABLE, explicit_frozen=1)
    assert prep.frozen_message_count == 3


def test_proxy_policy_clamps_by_cache_count() -> None:
    """Provider says 5 messages are cached, but local state can only replay 3:
    freezing past the replayable bound would forward raw bytes."""
    cache = CompressionCache()
    messages = _history_with_cached_tool(cache, "ORIGINAL " * 100, "[compressed]")
    messages.append(_tool_msg("uncached " * 50, "c2"))
    messages.append({"role": "user", "content": "next"})

    prep = prepare_turn(cache, messages, policy=FREEZE_POLICY_CONFIRMED_CLAMP, tracker_frozen=5)
    assert prep.frozen_message_count == 3


def test_proxy_policy_clamps_by_tracker() -> None:
    """Local state could replay 3, but the provider only confirmed 1: content
    past the confirmed prefix stays compressible (the #327 posture)."""
    cache = CompressionCache()
    messages = _history_with_cached_tool(cache, "ORIGINAL " * 100, "[compressed]")
    messages.append({"role": "user", "content": "next"})
    prep = prepare_turn(cache, messages, policy=FREEZE_POLICY_CONFIRMED_CLAMP, tracker_frozen=1)
    assert prep.frozen_message_count == 1


def test_proxy_policy_none_tracker_freezes_nothing() -> None:
    cache = CompressionCache()
    messages = _history_with_cached_tool(cache, "ORIGINAL " * 100, "[compressed]")
    prep = prepare_turn(cache, messages, policy=FREEZE_POLICY_CONFIRMED_CLAMP, tracker_frozen=None)
    assert prep.frozen_message_count == 0


def test_unknown_policy_rejected() -> None:
    cache = CompressionCache()
    with pytest.raises(ValueError):
        prepare_turn(cache, [], policy="wat")


def test_prepare_marks_frozen_tool_results_stable() -> None:
    cache = CompressionCache()
    original = "ORIGINAL " * 100
    messages = _history_with_cached_tool(cache, original, "[compressed]")
    messages.append({"role": "user", "content": "next"})
    prepare_turn(cache, messages, policy=FREEZE_POLICY_REPLAYABLE)
    assert cache.content_hash(original) in cache._stable_hashes


# --------------------------------------------------------------------------- #
# finalize_turn: overlay + recount hook                                        #
# --------------------------------------------------------------------------- #


def _prev_pair() -> tuple[list[dict], list[dict]]:
    prev_original = [
        {"role": "user", "content": "ORIGINAL " * 100},
        {"role": "assistant", "content": "ok"},
    ]
    prev_returned = [
        {"role": "user", "content": "[returned-form]"},
        {"role": "assistant", "content": "ok"},
    ]
    return prev_original, prev_returned


def test_finalize_replays_previous_returned_prefix() -> None:
    prev_original, prev_returned = _prev_pair()
    current = prev_original + [{"role": "user", "content": "next"}]
    # The pipeline "drifted": it emitted the raw original for message 0.
    drifted = [dict(m) for m in current]

    counted: list[int] = []

    def _count(msgs: list[dict]) -> int:
        counted.append(len(json.dumps(msgs)))
        return 42

    turn = finalize_turn(drifted, current, prev_original, prev_returned, count_tokens=_count)
    assert turn.replayed
    assert turn.messages[0]["content"] == "[returned-form]"
    assert turn.messages[-1]["content"] == "next"
    assert turn.tokens == 42
    assert len(counted) == 1


def test_finalize_noop_without_prev_snapshots() -> None:
    current = [{"role": "user", "content": "hi"}]
    calls: list[int] = []
    turn = finalize_turn(current, current, [], [], count_tokens=lambda m: calls.append(1) or 1)
    assert not turn.replayed
    assert turn.messages == current
    assert turn.tokens is None
    assert not calls  # count_tokens only runs when the overlay fired


def test_finalize_count_hook_failure_falls_back() -> None:
    prev_original, prev_returned = _prev_pair()
    current = prev_original + [{"role": "user", "content": "next"}]

    def _boom(_msgs: list[dict]) -> int:
        raise RuntimeError("tokenizer down")

    turn = finalize_turn(
        [dict(m) for m in current], current, prev_original, prev_returned, count_tokens=_boom
    )
    assert turn.replayed
    assert turn.tokens is None


def test_finalize_declines_inflating_replay_by_default() -> None:
    prev_original, prev_returned = _prev_pair()
    current = prev_original + [{"role": "user", "content": "next"}]
    # The pipeline recompressed message 0 SMALLER than the returned form.
    recompressed = [
        {"role": "user", "content": "[t]"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    turn = finalize_turn(recompressed, current, prev_original, prev_returned)
    assert not turn.replayed
    assert turn.messages == recompressed


def test_finalize_confirmed_floor_replays_confirmed_bytes() -> None:
    prev_original, prev_returned = _prev_pair()
    current = prev_original + [{"role": "user", "content": "next"}]
    recompressed = [
        {"role": "user", "content": "[t]"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    turn = finalize_turn(
        recompressed, current, prev_original, prev_returned, confirmed_frozen_count=2
    )
    assert turn.replayed
    assert turn.messages[0]["content"] == "[returned-form]"
    assert turn.messages[-1]["content"] == "next"


# --------------------------------------------------------------------------- #
# OpenAI proxy token-path migration: formula identity + marking benefit.       #
# --------------------------------------------------------------------------- #


def test_replayable_without_pin_equals_bare_cache_count() -> None:
    """The OpenAI proxy token path historically froze on compute_frozen_count
    alone; REPLAYABLE with no explicit pin must be formula-identical, so its
    migration onto the engine is a pure extraction."""
    cache = CompressionCache()
    messages = _history_with_cached_tool(cache, "AAAA " * 50, "[c1]")
    messages.append(_tool_msg("uncached content", call_id="c2"))
    messages.append({"role": "user", "content": "next"})

    prep = prepare_turn(cache, messages, policy=FREEZE_POLICY_REPLAYABLE)
    assert prep.frozen_message_count == cache.compute_frozen_count(messages)
    # And that count stops at the uncached tool_result (index 3).
    assert prep.frozen_message_count == 3


def test_marking_preserves_freeze_across_entry_eviction() -> None:
    """The one real benefit mark_stable_from_messages adds on the migrated
    path: an in-prefix tool_result stays stable via `_stable_hashes` even
    after its compressed ENTRY is evicted by the per-cache LRU, so the frozen
    count does not collapse at that position on the next turn."""
    cache = CompressionCache(max_entries=100)
    original = "BBBB " * 50
    messages = _history_with_cached_tool(cache, original, "[c1]")
    messages.append({"role": "user", "content": "next"})

    prep = prepare_turn(cache, messages, policy=FREEZE_POLICY_REPLAYABLE)
    assert prep.frozen_message_count == 3  # tool in prefix, marked stable

    # Simulate entry LRU turnover: the compressed entry disappears.
    h = cache.content_hash(original)
    with cache._lock:
        cache._cache.pop(h, None)

    # Without marking, the frozen count would collapse to 2 here; the
    # stable-hash record keeps the position frozen.
    assert cache.compute_frozen_count(messages) == 3

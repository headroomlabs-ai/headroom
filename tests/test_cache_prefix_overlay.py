# ruff: noqa: E402 — test sections import after helper/setup code by design.
"""overlay_cached_prefix: freeze must forward the CACHED (compressed) bytes.

The freeze path can emit the agent's ORIGINAL bytes for a frozen message, but
the provider cached whatever we FORWARDED last turn (the compressed form).
Forwarding original then mismatches the cached prefix and busts the prompt cache
(observed: 100% of misses were this ``prefix_change``, ~56% of all cache-writes).
``overlay_cached_prefix`` replays the previously-forwarded prefix byte-identical
so the cache still hits — in BOTH proxy modes.
"""

from headroom.cache.prefix_tracker import overlay_cached_prefix


def M(role, text):
    return {"role": role, "content": text}


# Previous turn: 2 messages. Original was big; we FORWARDED the compressed form,
# so that compressed form is what the provider cached.
PREV_ORIG = [M("user", "READ foo.py:\n<2000 original lines>"), M("assistant", "ok")]
PREV_FWD = [M("user", "READ foo.py:\n<compressed>"), M("assistant", "ok")]
# This turn: agent appended one new message (append-only growth).
CUR_ORIG = PREV_ORIG + [M("user", "grep result:\n<800 original lines>")]
# What apply() produced in the buggy freeze path: ORIGINAL bytes for the frozen
# prefix (== PREV_ORIG) + compressed new tail.
OPTIMIZED_BUGGY = [PREV_ORIG[0], PREV_ORIG[1], M("user", "grep result:\n<compressed>")]


def test_replays_cached_compressed_prefix_byte_identical():
    out = overlay_cached_prefix(OPTIMIZED_BUGGY, CUR_ORIG, PREV_ORIG, PREV_FWD)
    # The frozen prefix now equals what the provider cached (compressed), NOT the
    # agent's original bytes → cache hits instead of busting.
    assert out[:2] == PREV_FWD
    assert out[:2] != PREV_ORIG
    # This turn's compressed tail is preserved.
    assert out[2] == OPTIMIZED_BUGGY[2]
    assert len(out) == len(CUR_ORIG)


def test_is_a_noop_relative_to_cache_when_already_correct():
    # If the freeze path already forwarded the compressed (cached) prefix, the
    # overlay reproduces exactly that — idempotent.
    already_correct = [PREV_FWD[0], PREV_FWD[1], M("user", "grep result:\n<compressed>")]
    out = overlay_cached_prefix(already_correct, CUR_ORIG, PREV_ORIG, PREV_FWD)
    assert out == already_correct


def test_not_append_only_returns_unchanged():
    # An early message changed → previous forwarded bytes may not correspond to
    # the same positions; do NOT overlay (accept a possible bust over corruption).
    changed = [M("user", "TOTALLY DIFFERENT"), PREV_ORIG[1], M("user", "x")]
    out = overlay_cached_prefix(OPTIMIZED_BUGGY, changed, PREV_ORIG, PREV_FWD)
    assert out == OPTIMIZED_BUGGY


def test_no_previous_state_returns_unchanged():
    assert overlay_cached_prefix(OPTIMIZED_BUGGY, CUR_ORIG, None, None) == OPTIMIZED_BUGGY
    assert overlay_cached_prefix(OPTIMIZED_BUGGY, CUR_ORIG, [], []) == OPTIMIZED_BUGGY


def test_forwarded_count_mismatch_returns_unchanged():
    # Defensive: not exactly one forwarded message per original → bail.
    assert (
        overlay_cached_prefix(OPTIMIZED_BUGGY, CUR_ORIG, PREV_ORIG, PREV_FWD[:1]) == OPTIMIZED_BUGGY
    )


def test_shorter_current_or_optimized_returns_unchanged():
    assert overlay_cached_prefix([M("user", "x")], [M("user", "x")], PREV_ORIG, PREV_FWD) == [
        M("user", "x")
    ]


def test_cache_hit_property_prefix_matches_last_forward():
    # The invariant that guarantees a cache hit: forwarded[:n] this turn ==
    # forwarded[:n] last turn (== what the provider cached).
    out = overlay_cached_prefix(OPTIMIZED_BUGGY, CUR_ORIG, PREV_ORIG, PREV_FWD)
    n = len(PREV_FWD)
    assert out[:n] == PREV_FWD  # exact byte-identical prefix → provider cache hit


# ============================================================================
# OpenAI function-calling frozen-count: tool_calls must be counted (Kimi bug)
# ============================================================================
# _estimate_message_tokens only counted `content` + Anthropic content-blocks,
# never OpenAI top-level `tool_calls`. So a function-calling assistant turn
# (content None, command in tool_calls) estimated to ~0, the frozen-prefix
# estimate overshot the real cache boundary, and the NEWEST delta got frozen —
# giving OpenAI/Kimi tool harnesses ~zero compression. These lock in the fix.
import json as _json

from headroom.cache.prefix_tracker import PrefixCacheTracker, PrefixFreezeConfig


def _openai_asst(cmd):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "bash", "arguments": _json.dumps({"command": cmd})},
            }
        ],
    }


def test_estimate_counts_openai_tool_calls():
    est = PrefixCacheTracker._estimate_message_tokens
    cmd = "cd /tmp/core && cat suma/apps/underwriting/followup/service.py"
    with_calls = est([_openai_asst(cmd)])[0]
    # empty content + no tool_calls counted => only the +20 overhead (~5 tok)
    bare = est([{"role": "assistant", "content": None}])[0]
    assert with_calls > bare + 5, (with_calls, bare)  # the command is now counted
    # legacy function_call shape too
    fc = est(
        [
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "bash", "arguments": _json.dumps({"command": cmd})},
            }
        ]
    )[0]
    assert fc > bare + 5, (fc, bare)


def test_frozen_count_leaves_openai_tool_delta_mutable():
    # A tool-based turn: cached prefix (system+task+prior tool obs) then a NEW
    # assistant tool_call + its observation. After update_from_response reports
    # the prefix cached, the frozen count must NOT swallow the newest delta.
    trk = PrefixCacheTracker("openai", PrefixFreezeConfig(min_cached_tokens=10))
    msgs = [
        {"role": "system", "content": "s" * 400},
        {"role": "user", "content": "task " * 200},
        _openai_asst("cd /tmp/core && rg -n foo ."),
        {"role": "tool", "tool_call_id": "c1", "content": "hit\n" * 300},  # cached prefix ends here
        _openai_asst("cd /tmp/core && cat foo.py"),  # NEW delta (assistant)
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "code\n" * 400,
        },  # NEW delta (observation)
    ]
    counts = PrefixCacheTracker._estimate_message_tokens(msgs)
    # cache_read ~= the first 4 messages' real tokens (prefix cached)
    cached_prefix_tokens = sum(counts[:4])
    trk.update_from_response(
        cache_read_tokens=cached_prefix_tokens,
        cache_write_tokens=0,
        messages=msgs,
        message_token_counts=counts,
    )
    frozen = trk.get_frozen_message_count()
    # must freeze ~the cached prefix (<=4), NOT the whole 6 (which would freeze
    # the newest observation delta and block all compression).
    assert frozen <= 4, f"frozen={frozen} swallowed the delta (len={len(msgs)})"

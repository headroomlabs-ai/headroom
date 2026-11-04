"""Reproduce the residual cache-bust: client MOVING cache_control defeats the
prefix overlay.

Real clients (Claude Code, litellm) move the cache_control breakpoint to the
newest message every turn — so a message that was marked last turn is unmarked
this turn (its dict bytes change). The first overlay fix compared *raw* message
dicts for its append-only guard, so a moved marker in the frozen prefix made the
guard fail → the overlay skipped the replay → the raw freeze forwarded ORIGINAL
bytes over the cached COMPRESSED prefix → partial bust (the ~42% residual seen
on the a10 run, with prefix_change=0).

These tests pin the exact scenario, prove the content-only guard fixes it, and
document the remaining piece (marker accumulation > 4 → needs stable placement).
"""

from headroom.cache.prefix_tracker import (
    PrefixCacheTracker,
    PrefixFreezeConfig,
    overlay_cached_prefix,
)


def M(role, text, cc=False):
    m = {"role": role, "content": text}
    if cc:
        m["cache_control"] = {"type": "ephemeral"}
    return m


def _toklen(m):
    return max(1, len(str(m.get("content", ""))))


def _compress(m):
    c = str(m.get("content", ""))
    return {**m, "content": c[: max(1, len(c) // 2)]}


def _freeze(original, frozen):
    # content_router freeze model: frozen prefix = ORIGINAL bytes, rest compressed.
    return [(original[i] if i < frozen else _compress(original[i])) for i in range(len(original))]


# ── Unit reproduction ────────────────────────────────────────────────────────
# Last turn we forwarded the compressed prefix; the client had marked msg1.
PREV_ORIG = [M("user", "READ foo:\n<big>"), M("assistant", "ok", cc=True)]
PREV_FWD = [M("user", "READ foo:\n<compressed>"), M("assistant", "ok", cc=True)]
# This turn the client MOVED the marker off msg1 onto the new last message (msg2).
CUR_ORIG = [M("user", "READ foo:\n<big>"), M("assistant", "ok"), M("user", "grep:\n<big>", cc=True)]
# Freeze forwarded ORIGINAL bytes for the frozen prefix + compressed tail.
OPTIMIZED = [M("user", "READ foo:\n<big>"), M("assistant", "ok"), M("user", "grep:\n<compressed>")]


def test_marker_move_would_fail_a_raw_dict_guard():
    # This is the exact condition the old (raw) guard tripped on: the frozen
    # prefix differs ONLY because cache_control moved off msg1.
    assert CUR_ORIG[:2] != PREV_ORIG
    # ...but with cache_control stripped, the content is an append-only extension.
    from headroom.cache.prefix_tracker import _strip_cache_control

    assert _strip_cache_control(CUR_ORIG[:2]) == _strip_cache_control(PREV_ORIG)


def test_overlay_skips_inflating_moved_marker_replay():
    out = overlay_cached_prefix(OPTIMIZED, CUR_ORIG, PREV_ORIG, PREV_FWD)
    # Moving cache_control must not exempt a larger replay candidate from the
    # no-inflation bound, even when content-only history alignment succeeds.
    assert out == OPTIMIZED


# ── Cross-turn: client moves the marker every turn, provider keys on full bytes ─
def _client_convo(t):
    msgs = [{"role": "user", "content": f"turn-{k}:" + "X" * 300} for k in range(1, t + 1)]
    msgs[-1] = {**msgs[-1], "cache_control": {"type": "ephemeral"}}  # mark ONLY the newest
    return msgs


def _cache_read(fwd, prev_fwd):
    # cache_control-AWARE (worst case): a moved marker changes the block's bytes,
    # so it breaks the byte-identical prefix.
    if not prev_fwd:
        return 0
    matched = 0
    for a, b in zip(fwd, prev_fwd):
        if a == b:
            matched += _toklen(a)
        else:
            break
    return matched


def _drive(use_overlay, turns=5):
    tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(min_cached_tokens=0))
    prev_fwd = None
    results = []
    last_fwd = None
    for t in range(1, turns + 1):
        cur = _client_convo(t)
        frozen = tracker.get_frozen_message_count()
        fwd = _freeze(cur, frozen)
        if use_overlay:
            fwd = overlay_cached_prefix(
                fwd,
                cur,
                tracker.get_last_original_messages(),
                tracker.get_last_forwarded_messages(),
            )
        exp = sum(_toklen(m) for m in prev_fwd) if prev_fwd else 0
        act = _cache_read(fwd, prev_fwd)
        results.append((exp, act))
        counts = [_toklen(m) for m in fwd]
        tracker.update_from_response(
            act, sum(counts) - act, fwd, message_token_counts=counts, original_messages=cur
        )
        prev_fwd = fwd
        last_fwd = fwd
    return results, last_fwd


def test_moving_marker_busts_without_overlay():
    results, _ = _drive(use_overlay=False)
    assert any(exp > act for exp, act in results[1:]), "moving marker should bust the raw freeze"


def test_moving_marker_no_bust_with_overlay():
    results, _ = _drive(use_overlay=True)
    for exp, act in results[1:]:
        assert act >= exp, f"cache bust under moved marker: expected {exp} read {act}"


# ── fix-2: Headroom owns cache_control placement (realistic block content) ────
from headroom.cache.prefix_tracker import (  # noqa: E402
    _strip_cache_control,
    normalize_message_cache_control,
)


def B(role, text, cc=False):
    """Anthropic block-style message (cache_control lives on a content block)."""
    blk = {"type": "text", "text": text}
    if cc:
        blk["cache_control"] = {"type": "ephemeral"}
    return {"role": role, "content": [blk]}


def _markers(messages):
    return sum(
        1
        for m in messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and "cache_control" in b
    )


def test_normalize_strips_all_and_keeps_one_on_last():
    # 5 accumulated markers (the pile-up the overlay would produce).
    msgs = [
        B("user", "a", cc=True),
        B("assistant", "b", cc=True),
        B("user", "c", cc=True),
        B("user", "d", cc=True),
        B("user", "e", cc=True),
    ]
    out = normalize_message_cache_control(msgs)
    assert _markers(out) == 1  # bounded — no >4 error
    assert "cache_control" in out[-1]["content"][-1]  # on the last block
    assert _strip_cache_control(out) == _strip_cache_control(msgs)  # content untouched


def test_normalize_stays_bounded_across_many_turns():
    """The accumulation that would 400 Anthropic is now capped at 1 every turn."""
    conv = []
    forwarded = []
    for t in range(1, 12):
        conv = conv + [B("user", f"turn-{t}", cc=True)]  # client marks the newest
        forwarded = normalize_message_cache_control(conv)
        assert _markers(forwarded) <= 4  # never exceeds Anthropic's limit
    assert _markers(forwarded) == 1  # exactly one, on the last message


def test_normalize_is_noop_when_no_block_markers():
    plain = [B("user", "a"), B("assistant", "b")]  # no cache_control
    out = normalize_message_cache_control(plain)
    # places exactly one breakpoint (so the prefix gets cached), content stable
    assert _markers(out) == 1
    assert _strip_cache_control(out) == _strip_cache_control(plain)


# ── fix-3 (#2375): consolidation must not silently drop the client's ttl ─────


def B_ttl(role, text, ttl):
    """Block-style message whose marker carries an explicit ttl (1h caching)."""
    blk = {"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": ttl}}
    return {"role": role, "content": [blk]}


def test_normalize_preserves_ttl_of_newest_marker():
    """A 1h-ttl client must not be silently downgraded to the 5m default."""
    msgs = [B("user", "a", cc=True), B_ttl("user", "b", "1h")]
    out = normalize_message_cache_control(msgs)
    assert _markers(out) == 1
    assert out[-1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_normalize_newest_marker_wins_over_stale_ttl():
    # Older replayed markers still carry 1h, but the client's NEWEST marker has
    # no ttl — the client switched back to the default; don't resurrect 1h.
    msgs = [B_ttl("user", "a", "1h"), B_ttl("assistant", "b", "1h"), B("user", "c", cc=True)]
    out = normalize_message_cache_control(msgs)
    assert _markers(out) == 1
    assert out[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_normalize_ttl_survives_many_turns():
    """The #2375 scenario: ttl held for one turn, gone on every later turn."""
    conv = []
    for t in range(1, 8):
        conv = conv + [B_ttl("user", f"turn-{t}", "1h")]  # client always asks 1h
        conv = normalize_message_cache_control(conv)
        assert _markers(conv) == 1
        assert conv[-1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# ── fix-4: mirror the CLIENT's marker positions (20-block lookback chain) ────
# Anthropic resolves each breakpoint by walking back at most ~20 content
# blocks. Agentic clients keep a marker on the previous turn's newest message
# as the read anchor; collapsing to a single newest-block marker breaks the
# chain on tool-heavy turns. With client_messages provided, normalize must
# keep exactly the client's positions.


def test_normalize_mirrors_client_marker_positions():
    client = [
        B("user", "a", cc=True),
        B("assistant", "b"),
        B("user", "c", cc=True),  # read anchor (previous newest)
        B("assistant", "d"),
        B("user", "e", cc=True),  # newest
    ]
    # Forwarded form: replay leftovers piled markers onto other messages too.
    merged = [
        B("user", "a", cc=True),
        B("assistant", "b", cc=True),
        B("user", "c", cc=True),
        B("assistant", "d", cc=True),
        B("user", "e", cc=True),
    ]
    out = normalize_message_cache_control(merged, client_messages=client)
    assert _markers(out) == 3  # exactly the client's three, not one
    for idx in (0, 2, 4):
        assert "cache_control" in out[idx]["content"][-1], idx
    for idx in (1, 3):
        assert _markers([out[idx]]) == 0, idx
    assert _strip_cache_control(out) == _strip_cache_control(merged)


def test_normalize_mirror_preserves_per_position_ttl():
    client = [B_ttl("user", "a", "1h"), B("user", "b", cc=True)]
    merged = [B("user", "a", cc=True), B("user", "b", cc=True)]
    out = normalize_message_cache_control(merged, client_messages=client)
    assert out[0]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert out[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_normalize_mirror_bounded_across_many_turns():
    """Client moves its pair of markers forward; forwarded stays at client count."""
    conv = []
    for t in range(1, 12):
        conv = conv + [B("user", f"turn-{t}")]
        # Client marks the newest and second-newest marked position (CC pattern).
        client = [dict(m) for m in conv]
        client[-1] = B("user", f"turn-{t}", cc=True)
        if len(client) >= 2:
            client[-2] = B(client[-2]["role"], client[-2]["content"][0]["text"], cc=True)
        # Forwarded side accumulated replay leftovers everywhere.
        merged = [B(m["role"], m["content"][0]["text"], cc=True) for m in client]
        out = normalize_message_cache_control(merged, client_messages=client)
        assert _markers(out) == min(2, len(client))
        assert "cache_control" in out[-1]["content"][-1]


def test_normalize_falls_back_when_counts_mismatch():
    client = [B("user", "a", cc=True)]  # transform changed message count
    merged = [B("user", "a", cc=True), B("user", "b", cc=True)]
    out = normalize_message_cache_control(merged, client_messages=client)
    assert _markers(out) == 1  # legacy consolidation
    assert "cache_control" in out[-1]["content"][-1]


def test_normalize_falls_back_when_client_has_no_markers():
    client = [B("user", "a"), B("user", "b")]
    merged = [B("user", "a", cc=True), B("user", "b")]
    out = normalize_message_cache_control(merged, client_messages=client)
    assert _markers(out) == 1  # legacy: still place one so the prefix caches
    assert "cache_control" in out[-1]["content"][-1]


def test_normalize_mirror_skips_string_content_positions():
    # Client marked message 0; forwarded counterpart is string-content (cannot
    # carry a marker) — the position is skipped, the rest still mirror.
    client = [B("user", "a", cc=True), B("user", "b", cc=True)]
    merged = [{"role": "user", "content": "a"}, B("user", "b", cc=True)]
    out = normalize_message_cache_control(merged, client_messages=client)
    assert _markers(out) == 1
    assert "cache_control" in out[1]["content"][-1]


# ── fix-5: block-level mirroring (multiple client markers in one message) ────


def test_normalize_mirrors_multiple_markers_within_one_message():
    """A 2-message request where the client marks TWO blocks of message 0
    (Claude Code's pattern on large first messages) keeps all three markers."""
    big = {
        "role": "user",
        "content": [
            {"type": "text", "text": "part-1", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "part-2"},
            {"type": "text", "text": "part-3", "cache_control": {"type": "ephemeral"}},
        ],
    }
    client = [big, B("user", "follow-up", cc=True)]
    # Forwarded form: an extra replay leftover on message 1's sibling... use
    # identical structure with markers everywhere to prove selective stripping.
    merged = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "part-1", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "part-2", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "part-3", "cache_control": {"type": "ephemeral"}},
            ],
        },
        B("user", "follow-up", cc=True),
    ]
    out = normalize_message_cache_control(merged, client_messages=client)
    assert _markers(out) == 3
    assert "cache_control" in out[0]["content"][0]
    assert "cache_control" not in out[0]["content"][1]
    assert "cache_control" in out[0]["content"][2]
    assert "cache_control" in out[1]["content"][-1]
    assert _strip_cache_control(out) == _strip_cache_control(merged)


def test_normalize_mirror_clamps_out_of_range_block_index():
    # Client marked block 2; forwarded message only has 1 block (transform
    # merged content) — marker falls back to the last block, not dropped.
    client = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
                {"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}},
            ],
        },
        B("user", "tail", cc=True),
    ]
    merged = [B("user", "abc-merged"), B("user", "tail", cc=True)]
    out = normalize_message_cache_control(merged, client_messages=client)
    assert _markers(out) == 2
    assert "cache_control" in out[0]["content"][-1]
    assert "cache_control" in out[1]["content"][-1]


def test_normalize_mirror_scalar_only_content_is_left_unchanged():
    """Client marks a message whose forwarded counterpart carries only scalar
    blocks: nothing can hold a marker and nothing was stripped, so the input
    comes back unchanged (identity, not a copy)."""
    client = [B("user", "a", cc=True)]
    merged = [{"role": "user", "content": ["scalar-only"]}]
    out = normalize_message_cache_control(merged, client_messages=client)
    assert out is merged
    assert _markers(out) == 0


def test_normalize_mirror_scalar_target_still_strips_leftovers():
    # Message 0's forwarded content is scalar-only (marker unplaceable) but a
    # replay leftover on message 1 still gets stripped, and message 1 keeps
    # its client marker.
    client = [B("user", "a", cc=True), B("user", "b", cc=True)]
    merged = [
        {"role": "user", "content": ["scalar-only"]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "left-over", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "b"},
            ],
        },
    ]
    out = normalize_message_cache_control(merged, client_messages=client)
    assert _markers(out) == 1
    assert "cache_control" not in out[1]["content"][0]
    assert "cache_control" in out[1]["content"][-1]

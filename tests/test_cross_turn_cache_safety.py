"""Cross-turn cache-safety invariant — the test class that catches cache busts.

Why the +150%-cache_create / +41%-cost bug slipped through: every prior cache
test was SINGLE-turn and used a fake tracker, so nobody exercised the real
multi-turn invariant that actually governs prompt-cache cost:

    Across append-only turns, the forwarded prefix must stay BYTE-IDENTICAL to
    what was forwarded (and cached) last turn — otherwise the provider re-creates
    the whole suffix (a cache bust) instead of reading it.

This simulates the provider's prefix cache (longest byte-identical leading run of
messages = cache_read; the rest = cache_create) and drives the REAL
``PrefixCacheTracker`` + the freeze model + ``overlay_cached_prefix`` over several
turns. It asserts the invariant directly, and proves the guard is load-bearing:
WITHOUT the overlay the freeze forwards the agent's original bytes and busts every
turn; WITH it the prefix stays stable.
"""

from headroom.cache.prefix_tracker import (
    PrefixCacheTracker,
    PrefixFreezeConfig,
    overlay_cached_prefix,
)


def _toklen(m) -> int:
    return max(1, len(str(m.get("content", ""))))


def _compress(m):
    """Deterministic stand-in for a real compressor (kompress is deterministic
    per content via the result cache): shrink the content by half."""
    c = str(m.get("content", ""))
    return {**m, "content": c[: max(1, len(c) // 2)]}


def _apply_freeze(original, frozen_count):
    """Faithful model of pipeline.apply()'s freeze: the frozen prefix is
    forwarded as the agent's ORIGINAL bytes; everything else is compressed.
    (Mirrors content_router.py: `result_slots[i] = message` for i < frozen.)"""
    return [
        (original[i] if i < frozen_count else _compress(original[i])) for i in range(len(original))
    ]


def _provider_cache_read(forwarded, prev_forwarded):
    """Longest byte-identical leading run of messages the provider can serve from
    cache, in tokens. A single differing message breaks the prefix (bust)."""
    if not prev_forwarded:
        return 0
    matched = 0
    for a, b in zip(forwarded, prev_forwarded):
        if a == b:
            matched += _toklen(a)
        else:
            break
    return matched


def _drive_turns(*, use_overlay: bool, turns: int = 5):
    """Return per-turn (expected_cache_read, actual_cache_read). A bust is any
    turn where actual < expected (the previously-cached prefix wasn't reused)."""
    # min_cached_tokens=0 so freeze activates from turn 2 regardless of size.
    tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(min_cached_tokens=0))
    convo: list[dict] = []
    prev_forwarded: list[dict] | None = None
    out = []
    for t in range(1, turns + 1):
        # Append-only growth: one new large tool output per turn.
        convo = convo + [{"role": "user", "content": f"tool-output-turn-{t}:" + "X" * 400}]

        frozen = tracker.get_frozen_message_count()
        forwarded = _apply_freeze(convo, frozen)
        if use_overlay:
            forwarded = overlay_cached_prefix(
                forwarded,
                convo,
                tracker.get_last_original_messages(),
                tracker.get_last_forwarded_messages(),
            )

        expected_read = sum(_toklen(m) for m in prev_forwarded) if prev_forwarded else 0
        actual_read = _provider_cache_read(forwarded, prev_forwarded)
        out.append((expected_read, actual_read))

        counts = [_toklen(m) for m in forwarded]
        write = sum(counts) - actual_read
        tracker.update_from_response(
            actual_read, write, forwarded, message_token_counts=counts, original_messages=convo
        )
        prev_forwarded = forwarded
    return out


def test_freeze_busts_cache_every_turn_without_overlay():
    """Proves the test is load-bearing: the raw freeze path busts the cache."""
    results = _drive_turns(use_overlay=False)
    # From turn 2 on, a hit was expected but the prefix broke (actual < expected).
    busts = [exp > act for (exp, act) in results[1:]]
    assert any(busts), "expected the un-fixed freeze path to bust the prefix cache"


def test_overlay_keeps_prefix_byte_identical_no_bust():
    """The fix: every turn reuses the full previously-cached prefix — no bust."""
    results = _drive_turns(use_overlay=True)
    for exp, act in results[1:]:
        assert act >= exp, (
            f"cache bust: expected to read {exp} cached tokens but only read {act} "
            "— forwarded prefix diverged from last turn"
        )


def test_cache_create_stays_bounded_to_the_delta_with_overlay():
    """Cost proxy: with the fix, per-turn cache_create ≈ the new delta only, not
    the whole re-created prefix (which is what drove +150% cache_create)."""
    tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(min_cached_tokens=0))
    convo: list[dict] = []
    prev_forwarded: list[dict] | None = None
    creates = []
    for t in range(1, 6):
        convo = convo + [{"role": "user", "content": f"turn-{t}:" + "X" * 400}]
        frozen = tracker.get_frozen_message_count()
        forwarded = overlay_cached_prefix(
            _apply_freeze(convo, frozen),
            convo,
            tracker.get_last_original_messages(),
            tracker.get_last_forwarded_messages(),
        )
        read = _provider_cache_read(forwarded, prev_forwarded)
        counts = [_toklen(m) for m in forwarded]
        create = sum(counts) - read
        creates.append(create)
        tracker.update_from_response(
            read, create, forwarded, message_token_counts=counts, original_messages=convo
        )
        prev_forwarded = forwarded
    # Steady-state cache_create per turn should be ~one delta message, NOT growing
    # with conversation length. Assert the last turn creates no more than the
    # first (which had no cache to reuse).
    assert creates[-1] <= creates[0] + 1


def test_background_improvement_never_busts_warm_cache_and_lands_on_cold():
    """The #3379 regression and its fix, end to end: a background improvement
    to an already-forwarded message must NOT change warm-cache bytes (the
    confirmed floor overrides it), and must land the moment the provider
    count collapses (cold cache re-baseline) - so long-session growth is
    bounded by the TTL-lapse cadence instead of unbounded pinning."""
    tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(min_cached_tokens=0))
    convo: list[dict] = []
    prev_forwarded: list[dict] | None = None
    improved_from_turn = 4

    def pipeline(original, frozen, turn):
        out = _apply_freeze(original, frozen)
        if turn >= improved_from_turn:
            # Background compression later found a much better form for the
            # very first message (this PR's stated trigger).
            out = [{**out[0], "content": str(out[0]["content"])[:4]}] + out[1:]
        return out

    for t in range(1, 9):
        convo = convo + [{"role": "user", "content": f"tool-output-turn-{t}:" + "X" * 400}]
        frozen = tracker.get_frozen_message_count()
        forwarded = overlay_cached_prefix(
            pipeline(convo, frozen, t),
            convo,
            tracker.get_last_original_messages(),
            tracker.get_last_forwarded_messages(),
            confirmed_frozen_count=frozen,
        )
        expected = sum(_toklen(m) for m in prev_forwarded) if prev_forwarded else 0
        actual = _provider_cache_read(forwarded, prev_forwarded)
        assert actual >= expected, f"turn {t}: warm-cache bust ({actual} < {expected})"
        counts = [_toklen(m) for m in forwarded]
        tracker.update_from_response(
            actual,
            sum(counts) - actual,
            forwarded,
            message_token_counts=counts,
            original_messages=convo,
        )
        prev_forwarded = forwarded

    # Cold cache: the provider count collapses, the floor collapses with it,
    # and the accumulated improvement finally reaches the wire.
    convo = convo + [{"role": "user", "content": "tool-output-turn-9:" + "X" * 400}]
    cold = overlay_cached_prefix(
        pipeline(convo, 0, 9),
        convo,
        tracker.get_last_original_messages(),
        tracker.get_last_forwarded_messages(),
        confirmed_frozen_count=0,
    )
    assert len(str(cold[0]["content"])) < len(str(prev_forwarded[0]["content"])), (
        "cold-cache turn must re-baseline: the background improvement lands"
    )


def test_prefix_growth_bounded_and_rebaselines_on_the_cold_turn():
    """The pinned-prefix cost of the confirmed floor, measured.

    While the cache stays warm the confirmed prefix is pinned to its
    first-forwarded form, so later compression improvements to it do not land -
    a real context-window cost, not just a cache benefit. This asserts the two
    bounds that keep it from becoming #3026-style growth:

    1. Warm turns cannot compound: per-turn growth of the forwarded request is
       exactly the new message, never the re-inflation of the pinned prefix.
    2. The first cold turn re-baselines to precisely what the fresh pipeline
       would forward, so the pinned overhead is bounded by the TTL-lapse
       cadence rather than accumulating for the life of the session.

    Worst case for the floor: background compression keeps finding a better
    form for EVERY message it has already seen, improving each turn.
    """
    tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(min_cached_tokens=0))
    convo: list[dict] = []
    prev_forwarded: list[dict] | None = None

    def pipeline(original, frozen, turn):
        out = _apply_freeze(original, frozen)
        # Later turns find better forms for everything already forwarded.
        keep = max(4, 200 // turn)
        return [{**m, "content": str(m["content"])[:keep]} for m in out[:-1]] + out[-1:]

    warm_ratios = []
    for t in range(1, 9):
        convo = convo + [{"role": "user", "content": f"tool-output-turn-{t}:" + "X" * 400}]
        frozen = tracker.get_frozen_message_count()
        fresh = pipeline(convo, frozen, t)
        forwarded = overlay_cached_prefix(
            fresh,
            convo,
            tracker.get_last_original_messages(),
            tracker.get_last_forwarded_messages(),
            confirmed_frozen_count=frozen,
        )
        total = sum(_toklen(m) for m in forwarded)
        ideal = sum(_toklen(m) for m in fresh)
        if prev_forwarded is not None:
            prev_total = sum(_toklen(m) for m in prev_forwarded)
            assert total - prev_total <= _toklen(forwarded[-1]), (
                f"turn {t}: forwarded grew by {total - prev_total} tokens, more than the "
                f"{_toklen(forwarded[-1])}-token delta message - the prefix compounded"
            )
            warm_ratios.append(total / ideal)
        actual = _provider_cache_read(forwarded, prev_forwarded)
        counts = [_toklen(m) for m in forwarded]
        tracker.update_from_response(
            actual,
            sum(counts) - actual,
            forwarded,
            message_token_counts=counts,
            original_messages=convo,
        )
        prev_forwarded = forwarded

    # Measured on this worst-case model, forwarded/fresh over turns 2-8:
    # 1.35, 1.84, 2.33, 2.83, 3.35, 3.88, 4.35 - the pinned prefix does widen
    # against an ideal pipeline while the cache stays warm (the cost the floor
    # buys the cache hit with). What the two assertions bound is the shape:
    # each warm turn adds only its own delta, and the cold turn below zeroes
    # the gap outright.
    # Cold turn: the provider count collapses, the floor collapses with it, and
    # the forwarded request is exactly the fresh pipeline's output again.
    convo = convo + [{"role": "user", "content": "tool-output-turn-9:" + "X" * 400}]
    fresh_cold = pipeline(convo, 0, 9)
    cold = overlay_cached_prefix(
        fresh_cold,
        convo,
        tracker.get_last_original_messages(),
        tracker.get_last_forwarded_messages(),
        confirmed_frozen_count=0,
    )
    assert sum(_toklen(m) for m in cold) == sum(_toklen(m) for m in fresh_cold), (
        "cold turn must re-baseline to the fresh pipeline's output"
    )
    assert warm_ratios[-1] > warm_ratios[0], (
        "harness sanity: the warm-turn model must actually be improving the "
        "prefix, otherwise the re-baseline assertion proves nothing"
    )

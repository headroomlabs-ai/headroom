"""A turn that diverges mid-history (rewind, compaction request) starts a new
lineage, but the prefix it shares with an existing lineage was cached as that
lineage's forwarded bytes. The new lineage must replay those bytes, or the whole
context is re-written at the cache-write price.
"""

from __future__ import annotations

from headroom.cache.prefix_tracker import (
    PrefixFreezeConfig,
    SessionTrackerStore,
    overlay_cached_prefix,
)


def _u(t: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": t}]}


def _a(t: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": t}]}


def _store() -> SessionTrackerStore:
    return SessionTrackerStore(PrefixFreezeConfig())


def _turn(store, msgs, forwarded=None, reply="ok", affinity="aff"):
    tracker = store.resolve_tracker("s", "anthropic", messages=msgs, cache_affinity=affinity)
    fwd = list(forwarded or msgs)
    tracker.update_from_response(
        cache_read_tokens=5000,
        cache_write_tokens=100,
        messages=fwd + [_a(reply)],
        original_messages=list(msgs) + [_a(reply)],
    )
    return tracker


def _history(n: int) -> tuple[list[dict], list[dict]]:
    """n alternating messages; assistant turns compress to a short form."""
    orig, fwd = [], []
    for i in range(n):
        if i % 2 == 0:
            orig.append(_u(f"q{i}"))
            fwd.append(_u(f"q{i}"))
        else:
            orig.append(_a(f"long tool output {i} " + "x" * 200))
            fwd.append(_a(f"short {i}"))
    return orig, fwd


def test_rewind_seeds_new_lineage_with_shared_compressed_prefix():
    store = _store()
    orig, fwd = _history(13)
    t1 = _turn(store, orig, forwarded=fwd, reply="A")

    rewound = orig[:10] + [_u("a different question")]
    t2 = store.resolve_tracker("s", "anthropic", messages=rewound, cache_affinity="aff")

    assert t2 is not t1
    assert t2.get_last_original_messages() == orig[:10]
    assert t2.get_last_forwarded_messages() == fwd[:10]
    replayed = overlay_cached_prefix(
        list(rewound), rewound, t2.get_last_original_messages(), t2.get_last_forwarded_messages()
    )
    assert replayed[:10] == fwd[:10]
    assert replayed[10] == rewound[10]


def test_compaction_request_after_recap_replays_prefix():
    store = _store()
    orig, fwd = _history(11)  # ends with a user message
    _turn(store, orig, forwarded=fwd, reply="A")
    recap = orig + [_a("A"), _u("The user stepped away and is coming back. Recap...")]
    _turn(store, recap, forwarded=fwd + [_a("A"), recap[-1]], reply="R")

    # Compaction drops the final reply and appends its instruction to the last user turn.
    last = {
        "role": "user",
        "content": orig[-1]["content"]
        + [{"type": "text", "text": "CRITICAL: Respond with TEXT ONLY."}],
    }
    compaction = orig[:-1] + [last]
    t = store.resolve_tracker("s", "anthropic", messages=compaction, cache_affinity="aff")

    assert t.get_last_original_messages() == orig[:-1]
    assert t.get_last_forwarded_messages() == fwd[:-1]


def test_seed_does_not_touch_donor_or_merge_lineages():
    store = _store()
    orig, fwd = _history(13)
    t1 = _turn(store, orig, forwarded=fwd, reply="A")
    t2 = store.resolve_tracker(
        "s", "anthropic", messages=orig[:10] + [_u("other")], cache_affinity="aff"
    )

    assert t2 is not t1
    assert t1.get_last_original_messages() == orig + [_a("A")]
    assert t1.get_last_forwarded_messages() == fwd + [_a("A")]


def test_no_seed_across_cache_affinity():
    store = _store()
    orig, fwd = _history(13)
    _turn(store, orig, forwarded=fwd, reply="A")
    t2 = store.resolve_tracker(
        "s", "anthropic", messages=orig[:10] + [_u("x")], cache_affinity="other"
    )
    assert t2.get_last_original_messages() == []


def test_no_seed_for_short_shared_prefix():
    store = _store()
    orig, fwd = _history(13)
    _turn(store, orig, forwarded=fwd, reply="A")
    t2 = store.resolve_tracker(
        "s", "anthropic", messages=orig[:5] + [_u("x")], cache_affinity="aff"
    )
    assert t2.get_last_original_messages() == []


def test_no_seed_when_donor_forwarded_is_not_one_to_one():
    store = _store()
    orig, fwd = _history(13)
    _turn(store, orig, forwarded=fwd + [_u("injected")], reply="A")
    t2 = store.resolve_tracker(
        "s", "anthropic", messages=orig[:10] + [_u("x")], cache_affinity="aff"
    )
    assert t2.get_last_original_messages() == []


def test_seed_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HEADROOM_DIVERGENCE_SEED", "0")
    store = _store()
    orig, fwd = _history(13)
    _turn(store, orig, forwarded=fwd, reply="A")
    t2 = store.resolve_tracker(
        "s", "anthropic", messages=orig[:10] + [_u("x")], cache_affinity="aff"
    )
    assert t2.get_last_original_messages() == []


def _with_system(msgs: list[dict]) -> list[dict]:
    """The handler resolves lineages on [synthetic system message, *messages]
    but records the turn's history without it (anthropic.py session_messages)."""
    return [{"role": "system", "content": "You are Claude Code."}, *msgs]


def test_rewind_seeds_when_lookup_carries_the_system_message():
    store = _store()
    orig, fwd = _history(13)
    t1 = store.resolve_tracker("s", "anthropic", messages=_with_system(orig), cache_affinity="aff")
    t1.update_from_response(
        cache_read_tokens=5000,
        cache_write_tokens=100,
        messages=fwd + [_a("A")],
        original_messages=orig + [_a("A")],
    )
    rewound = orig[:10] + [_u("a different question")]
    t2 = store.resolve_tracker(
        "s", "anthropic", messages=_with_system(rewound), cache_affinity="aff"
    )
    assert t2 is not t1
    assert t2.get_last_forwarded_messages() == fwd[:10]

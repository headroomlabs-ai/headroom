"""#4548: reasoning dropped on a cold turn stays dropped on the next warm turn."""

import contextvars
from types import SimpleNamespace

from headroom.transforms.thinking_compactor import compact_reasoning_openai_chat


class _HalfKompress:
    def compress(self, text, allow_download=False):
        words = text.split()
        return SimpleNamespace(compressed=" ".join(words[: len(words) // 2]))


def _history(n, tag):
    msgs = []
    for k in range(n):
        msgs.append({"role": "user", "content": f"q{k}"})
        reasoning = " ".join(f"{tag}{k}w{j}" for j in range(60))
        msgs.append({"role": "assistant", "content": f"a{k}", "reasoning_content": reasoning})
    return msgs


def test_cold_drop_reports_dropped_upto():
    out, stats = compact_reasoning_openai_chat(
        _history(3, "c"), kompress=None, keep_last_turns=1, drop=True
    )
    assert [m.get("reasoning_content") for m in out[1::2]][:2] == ["", ""]
    assert stats["dropped_upto"] == 2


def test_warm_turn_after_cold_reuses_cold_prefix():
    cold, stats = compact_reasoning_openai_chat(
        _history(3, "w"), kompress=None, keep_last_turns=1, drop=True
    )
    warm, _ = compact_reasoning_openai_chat(
        _history(4, "w"),
        kompress=_HalfKompress(),
        keep_last_turns=1,
        drop=False,
        drop_first_turns=stats["dropped_upto"],
    )
    assert warm[:4] == cold[:4]  # previously dropped turns byte-identical
    assert 0 < len(warm[5]["reasoning_content"].split()) < 60  # newer turn Kompressed
    assert len(warm[7]["reasoning_content"].split()) == 60  # active turn kept


def test_no_sticky_floor_is_unchanged():
    out, stats = compact_reasoning_openai_chat(
        _history(3, "u"), kompress=_HalfKompress(), keep_last_turns=1
    )
    assert all(len(m["reasoning_content"].split()) == 30 for m in out[1:4:2])
    assert stats["dropped_upto"] == 0


def test_inline_think_spans_stay_dropped():
    think = "<think>" + " ".join(f"t{j}" for j in range(60)) + "</think>"
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": think + "a0"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": think + "a1"},
    ]
    out, _ = compact_reasoning_openai_chat(
        msgs, kompress=_HalfKompress(), keep_last_turns=1, drop_first_turns=1
    )
    assert out[1]["content"] == "a0"
    assert out[3]["content"] == think + "a1"


def _tracker_with_floor(floor):
    from headroom.cache.prefix_tracker import PrefixCacheTracker

    tracker = PrefixCacheTracker("openai")
    tracker.update_from_response(cache_read_tokens=0, cache_write_tokens=0, messages=[])
    tracker.reasoning_drop_upto = floor
    return tracker


def test_floor_survives_shared_round_trip_and_warm_prefix_matches():
    from headroom.cache.prefix_tracker import SessionTrackerStore

    cold, stats = compact_reasoning_openai_chat(
        _history(3, "r"), kompress=None, keep_last_turns=1, drop=True
    )
    store = SessionTrackerStore()
    meta, table = store.encode_shared("k", _tracker_with_floor(stats["dropped_upto"]))
    restored = store.decode_shared(meta, table)
    assert restored.reasoning_drop_upto == 2
    warm, _ = compact_reasoning_openai_chat(
        _history(4, "r"),
        kompress=_HalfKompress(),
        keep_last_turns=1,
        drop_first_turns=restored.reasoning_drop_upto,
    )
    assert warm[:4] == cold[:4]


def test_old_metadata_without_floor_decodes_to_zero():
    from headroom.cache.prefix_tracker import SessionTrackerStore

    store = SessionTrackerStore()
    meta, table = store.encode_shared("k", _tracker_with_floor(3))
    meta["state"] = meta["state"][:5]
    meta["turns"] = [[t, s[:5]] for t, s in meta["turns"]]
    assert store.decode_shared(meta, table).reasoning_drop_upto == 0


def test_resumed_turn_restores_its_own_floor():
    tracker = _tracker_with_floor(2)
    tracker._turn_states[1] = tracker._turn_state()
    tracker.reasoning_drop_upto = 3  # a cold fork raised it
    tracker._resume_turn(1)
    assert tracker.reasoning_drop_upto == 2


def test_request_floor_ignores_a_concurrent_fork_raising_the_live_floor():
    from headroom.cache.prefix_tracker import _resolved_turn

    tracker = _tracker_with_floor(2)
    tracker._turn_states[1] = tracker._turn_state()
    request_a = contextvars.copy_context()
    request_a.run(_resolved_turn.set, (tracker, 1, tracker._turn_states[1]))
    tracker.reasoning_drop_upto = 3  # sibling fork went cold after A resolved

    floor = request_a.run(tracker.request_reasoning_drop_floor)
    assert floor == 2
    out, _ = compact_reasoning_openai_chat(
        _history(4, "f"), kompress=_HalfKompress(), keep_last_turns=1, drop_first_turns=floor
    )
    assert len(out[5]["reasoning_content"].split()) == 30  # Kompressed, not dropped


def test_cold_request_records_raised_floor_through_response():
    from headroom.cache.prefix_tracker import _resolved_turn

    tracker = _tracker_with_floor(0)
    tracker._turn_states[1] = tracker._turn_state()
    request = contextvars.copy_context()
    request.run(_resolved_turn.set, (tracker, 1, tracker._turn_states[1]))

    cold, stats = compact_reasoning_openai_chat(
        _history(3, "r"), kompress=None, keep_last_turns=1, drop=True
    )
    floor = max(request.run(tracker.request_reasoning_drop_floor), stats["dropped_upto"])
    request.run(tracker.raise_reasoning_drop_floor, floor)
    request.run(tracker.update_from_response, 0, 0, cold)

    assert tracker._turn_states[1][5] == 2
    tracker._resume_turn(1)
    warm, _ = compact_reasoning_openai_chat(
        _history(4, "r"),
        kompress=_HalfKompress(),
        keep_last_turns=1,
        drop=False,
        drop_first_turns=tracker.request_reasoning_drop_floor(),
    )
    assert [m.get("reasoning_content") for m in warm[1::2]][:2] == ["", ""]
    assert warm[:5] == cold[:5]


def test_cold_request_floor_write_isolated_from_sibling_through_response():
    from headroom.cache.prefix_tracker import _resolved_turn

    tracker = _tracker_with_floor(2)
    tracker._turn_states[1] = tracker._turn_state()
    request_a = contextvars.copy_context()
    request_a.run(_resolved_turn.set, (tracker, 1, tracker._turn_states[1]))
    tracker.reasoning_drop_upto = 3  # sibling fork mutated live floor

    floor = request_a.run(tracker.request_reasoning_drop_floor)
    assert floor == 2
    request_a.run(tracker.update_from_response, 0, 0, [])

    assert tracker._turn_states[1][5] == 2
    tracker._resume_turn(1)
    out, _ = compact_reasoning_openai_chat(
        _history(4, "s"),
        kompress=_HalfKompress(),
        keep_last_turns=1,
        drop_first_turns=tracker.request_reasoning_drop_floor(),
    )
    assert len(out[5]["reasoning_content"].split()) == 30  # Kompressed, not dropped

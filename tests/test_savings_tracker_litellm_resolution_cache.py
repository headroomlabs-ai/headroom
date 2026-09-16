"""Regression: `_resolve_litellm_model`'s cache must be bounded (PR #2860 review).

A plain unbounded dict cache keyed by a client-controlled model string is a
memory-retention path on a request-facing proxy: a caller can grow it without
limit by sending a new model name on every request. The fix uses a bounded
`functools.lru_cache`. These tests pin the three properties that actually
matter, independent of the litellm pricing behavior covered elsewhere:

- repeated resolution of the same unresolvable model only probes litellm once
- the cache never grows past its bound, no matter how many distinct model
  names get resolved
- an evicted name is transparently re-probed (never silently wrong or stuck)
  rather than growing the cache further
"""

from __future__ import annotations

import types

from headroom.proxy import savings_tracker as st


def _fake_litellm_always_unresolvable(probe_calls: dict[str, int]) -> types.SimpleNamespace:
    """A fake litellm where every model is unpriced and unresolvable.

    `cost_per_token` always raises — exactly what a real custom/local model
    litellm has never heard of does — which is the call this cache exists to
    memoize (see the comment above `_resolve_litellm_model` in
    savings_tracker.py: that raise is also where real litellm prints its
    noisy "Provider List" banner, #2851).
    """

    def cost_per_token(*, model, prompt_tokens, completion_tokens):
        probe_calls[model] = probe_calls.get(model, 0) + 1
        raise RuntimeError("unknown model")

    return types.SimpleNamespace(model_cost={}, cost_per_token=cost_per_token)


def test_resolve_litellm_model_probes_unknown_model_once(monkeypatch):
    probe_calls: dict[str, int] = {}
    monkeypatch.setattr(
        st, "_get_litellm_module", lambda: _fake_litellm_always_unresolvable(probe_calls)
    )

    for _ in range(5):
        resolved = st._resolve_litellm_model("widget-local-model")
        assert resolved == "widget-local-model"

    assert probe_calls == {"widget-local-model": 1}


def test_resolve_litellm_model_cache_is_bounded(monkeypatch):
    probe_calls: dict[str, int] = {}
    monkeypatch.setattr(
        st, "_get_litellm_module", lambda: _fake_litellm_always_unresolvable(probe_calls)
    )

    extra_beyond_bound = 50
    for i in range(st._MODEL_RESOLUTION_CACHE_MAXSIZE + extra_beyond_bound):
        st._resolve_litellm_model(f"widget-local-model-{i}")

    info = st._resolve_litellm_model.cache_info()
    assert info.maxsize == st._MODEL_RESOLUTION_CACHE_MAXSIZE
    # However many distinct names were resolved, the cache itself never
    # grows past its bound -- this is the actual memory-retention fix.
    assert info.currsize == st._MODEL_RESOLUTION_CACHE_MAXSIZE


def test_resolve_litellm_model_evicted_name_reprobes(monkeypatch):
    probe_calls: dict[str, int] = {}
    monkeypatch.setattr(
        st, "_get_litellm_module", lambda: _fake_litellm_always_unresolvable(probe_calls)
    )

    st._resolve_litellm_model("seed-model")
    assert probe_calls["seed-model"] == 1

    # Push exactly `maxsize` new distinct names through without ever touching
    # "seed-model" again -- LRU eviction must push it out to make room.
    for i in range(st._MODEL_RESOLUTION_CACHE_MAXSIZE):
        st._resolve_litellm_model(f"filler-model-{i}")

    # A resolvable name being evicted is not a correctness bug (it just
    # re-probes) -- the assertion that matters is that it *does* re-probe
    # rather than silently reusing a slot it no longer legitimately owns.
    st._resolve_litellm_model("seed-model")
    assert probe_calls["seed-model"] == 2

"""Regression: caches keyed by the request's ``model`` string stay bounded (#4479).

``model`` comes straight off a proxied request body, so a client that sends a
new name on every request must not be able to grow a process-wide cache
without limit. #4407 bounded the per-model maps in the cost and savings
trackers; these tests pin the same property for the three remaining caches:

- ``TokenizerRegistry._cache`` (``headroom/tokenizers/registry.py``)
- the beacon's ``_model_cache`` behind ``_public_model``
  (``headroom/telemetry/session.py``)
- ``_resolved_model_cache`` behind ``resolve_litellm_model``
  (``headroom/pricing/litellm_pricing.py``)

Each is least-recently-used: a name in steady use is never evicted, and an
evicted name is recomputed on its next use rather than answered wrongly.
"""

from __future__ import annotations

import pytest

from headroom.pricing import litellm_pricing as lp
from headroom.telemetry import session
from headroom.tokenizers.estimator import EstimatingTokenCounter
from headroom.tokenizers.registry import TokenizerRegistry
from headroom.utils import remember_bounded

# --------------------------------------------------------------------------
# remember_bounded
# --------------------------------------------------------------------------


def test_remember_bounded_evicts_least_recently_inserted():
    cache: dict[str, int] = {}
    for i in range(5):
        remember_bounded(cache, f"k{i}", i, maxsize=3)

    assert list(cache) == ["k2", "k3", "k4"]


def test_remember_bounded_reinsert_moves_key_to_newest():
    cache: dict[str, int] = {}
    for i in range(3):
        remember_bounded(cache, f"k{i}", i, maxsize=3)

    remember_bounded(cache, "k0", 0, maxsize=3)  # a hit re-records the key
    remember_bounded(cache, "k3", 3, maxsize=3)

    assert list(cache) == ["k2", "k0", "k3"]


def test_remember_bounded_nonpositive_maxsize_stores_nothing():
    cache: dict[str, int] = {"stale": 1}
    remember_bounded(cache, "k", 1, maxsize=0)

    assert cache == {}


def test_remember_bounded_tolerates_a_dict_resized_by_another_thread():
    class ResizedDuringIteration(dict):
        def __iter__(self):
            raise RuntimeError("dictionary changed size during iteration")

    cache = ResizedDuringIteration({"k0": 0, "k1": 1})
    remember_bounded(cache, "k2", 2, maxsize=2)

    # The eviction is skipped for this call rather than raising on the request
    # path; the next insert that can iterate trims the cache again.
    assert dict.__getitem__(cache, "k2") == 2


# --------------------------------------------------------------------------
# TokenizerRegistry._cache
# --------------------------------------------------------------------------


@pytest.fixture
def clean_tokenizer_cache():
    TokenizerRegistry.clear_cache()
    yield
    TokenizerRegistry.clear_cache()


def test_tokenizer_registry_cache_is_bounded(clean_tokenizer_cache):
    cap = TokenizerRegistry._CACHE_MAXSIZE
    for i in range(cap + 50):
        TokenizerRegistry.get(f"client-model-{i}-" + "m" * 1_000)

    assert len(TokenizerRegistry._cache) == cap


def test_tokenizer_registry_estimation_fallback_is_bounded(clean_tokenizer_cache):
    cap = TokenizerRegistry._CACHE_MAXSIZE
    for i in range(cap + 10):
        tokenizer = TokenizerRegistry.get(f"client-model-{i}", backend="no-such-backend")
        assert isinstance(tokenizer, EstimatingTokenCounter)

    assert len(TokenizerRegistry._cache) == cap


def test_tokenizer_registry_keeps_a_model_in_steady_use(clean_tokenizer_cache):
    cap = TokenizerRegistry._CACHE_MAXSIZE
    hot = TokenizerRegistry.get("gpt-4o")
    for i in range(cap + 50):
        TokenizerRegistry.get(f"client-model-{i}")
        assert TokenizerRegistry.get("gpt-4o") is hot

    oldest = TokenizerRegistry.get("client-model-0")
    assert "client-model-0:auto" in TokenizerRegistry._cache
    assert oldest is TokenizerRegistry.get("client-model-0")


# --------------------------------------------------------------------------
# telemetry.session._model_cache
# --------------------------------------------------------------------------


@pytest.fixture
def clean_public_model_cache():
    session._model_cache.clear()
    yield
    session._model_cache.clear()


def test_public_model_cache_is_bounded(clean_public_model_cache):
    cap = session._MODEL_CACHE_MAXSIZE
    for i in range(cap + 50):
        assert session._public_model(f"custom-{i}-" + "m" * 1_000) is None

    assert len(session._model_cache) == cap


def test_public_model_evicted_name_is_recomputed(clean_public_model_cache, monkeypatch):
    calls: list[str] = []
    real_pin = session.pin_litellm_cost_map_local

    def counting_pin() -> None:
        calls.append("probe")
        real_pin()

    monkeypatch.setattr(session, "pin_litellm_cost_map_local", counting_pin)

    cap = session._MODEL_CACHE_MAXSIZE
    session._public_model("seed-model")
    for i in range(cap):
        session._public_model(f"custom-{i}")
    probes_before = len(calls)

    assert session._public_model("seed-model") is None
    assert len(calls) == probes_before + 1
    assert len(session._model_cache) == cap


# --------------------------------------------------------------------------
# litellm_pricing._resolved_model_cache
# --------------------------------------------------------------------------


@pytest.fixture
def counted_resolution(monkeypatch):
    lp._resolved_model_cache.clear()
    calls: dict[str, int] = {}

    def fake_uncached(model: str) -> str:
        calls[model] = calls.get(model, 0) + 1
        return f"resolved/{model}"

    monkeypatch.setattr(lp, "_resolve_litellm_model_uncached", fake_uncached)
    monkeypatch.delenv("HEADROOM_MODEL_ALIAS_MAP", raising=False)
    yield calls
    lp._resolved_model_cache.clear()


def test_resolved_model_cache_is_bounded(counted_resolution):
    cap = lp._RESOLVED_MODEL_CACHE_MAXSIZE
    for i in range(cap + 50):
        assert lp.resolve_litellm_model(f"custom-{i}") == f"resolved/custom-{i}"

    assert len(lp._resolved_model_cache) == cap


def test_resolved_model_cache_is_lru(counted_resolution):
    cap = lp._RESOLVED_MODEL_CACHE_MAXSIZE
    lp.resolve_litellm_model("hot-model")
    lp.resolve_litellm_model("cold-model")
    for i in range(cap):
        lp.resolve_litellm_model(f"custom-{i}")
        lp.resolve_litellm_model("hot-model")

    assert counted_resolution["hot-model"] == 1

    assert lp.resolve_litellm_model("cold-model") == "resolved/cold-model"
    assert counted_resolution["cold-model"] == 2

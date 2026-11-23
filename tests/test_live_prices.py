"""A running proxy prices models newer than litellm's bundled table.

Enterprise launchers pin litellm to its bundled table, which lacked
``claude-opus-5-5``, so savings for it fell back to a flat default rate. The
proxy now merges LiteLLM's current table in the background.
"""

from __future__ import annotations

import http.client
import io
import json
import urllib.request

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.pricing import live_prices  # noqa: E402
from headroom.pricing.counterfactual import (  # noqa: E402
    BASIS_CATALOG,
    CacheMix,
    Region,
    price_savings,
)
from headroom.pricing.litellm_pricing import _DEEPSEEK_V4_PRICING, litellm  # noqa: E402

NEW_MODEL = "claude-headroom-test-future-9"
NEW_ROW = {
    "litellm_provider": "anthropic",
    "mode": "chat",
    "input_cost_per_token": 4e-06,
    "output_cost_per_token": 2e-05,
    "cache_read_input_token_cost": 2e-07,
    "cache_creation_input_token_cost": 5e-06,
    "cache_creation_input_token_cost_above_1hr": 8e-06,
}


@pytest.fixture
def isolated_catalog(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    yield
    live_prices._clear_price_caches()


def _serve(monkeypatch, table: dict) -> list[str]:
    calls: list[str] = []

    def fake_urlopen(request, timeout=None, context=None):
        calls.append(request.full_url)
        return io.BytesIO(json.dumps(table).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _price_new_model():
    warm = CacheMix.from_usage(cache_read_tokens=15_000, uncached_input_tokens=10)
    return price_savings(
        1_000, model=NEW_MODEL, mix=warm, region=Region.PREFIX, provider="anthropic"
    )


def test_refresh_prices_a_model_the_bundled_table_lacks(monkeypatch, isolated_catalog):
    before = _price_new_model()  # memoises the miss, as a live proxy would
    table = dict(litellm.model_cost)
    table[NEW_MODEL] = NEW_ROW
    _serve(monkeypatch, table)

    assert live_prices.refresh_litellm_prices() >= 1

    after = _price_new_model()
    assert before.usd == 0.0
    assert after.basis == BASIS_CATALOG
    assert after.usd == pytest.approx(1_000 * 2e-07)
    # The same table again changes nothing: the daily refresh settles.
    assert live_prices.refresh_litellm_prices() == 0


def test_refresh_keeps_headroom_deepseek_off_peak_rates(monkeypatch, isolated_catalog):
    model, ours = next(iter(_DEEPSEEK_V4_PRICING.items()))
    table = dict(litellm.model_cost)
    for key in (model, f"deepseek/{model}"):
        table[key] = {**table.get(key, {}), "input_cost_per_token": 1.0}  # upstream peak
    _serve(monkeypatch, table)

    live_prices.refresh_litellm_prices()

    for key in (model, f"deepseek/{model}"):
        assert litellm.model_cost[key]["input_cost_per_token"] == ours["input_cost_per_token"]


def test_a_truncated_download_keeps_the_bundled_table(monkeypatch, isolated_catalog):
    _serve(monkeypatch, {NEW_MODEL: NEW_ROW})

    assert live_prices.refresh_litellm_prices() is None
    assert NEW_MODEL not in litellm.model_cost


def test_a_truncated_body_keeps_the_bundled_table(monkeypatch, isolated_catalog):
    def fake_urlopen(request, timeout=None, context=None):
        raise http.client.IncompleteRead(b"{")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert live_prices.refresh_litellm_prices() is None


def test_offline_never_fetches(monkeypatch):
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.delenv("HEADROOM_PRICE_REFRESH", raising=False)
    calls = _serve(monkeypatch, {})

    assert not live_prices.is_price_refresh_enabled()
    assert live_prices.refresh_litellm_prices() is None
    assert calls == []

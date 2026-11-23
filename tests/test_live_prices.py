"""A running proxy prices models newer than litellm's bundled table.

Enterprise launchers pin litellm to its bundled table, which lacked
``claude-opus-5-5``, so savings for it fell back to a flat default rate. The
proxy now merges LiteLLM's current table in the background.
"""

from __future__ import annotations

import asyncio
import copy
import http.client
import io
import json
import threading
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
    """Undo a merge: the price table and the provider name sets litellm keeps."""
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    name_sets = {name: set(v) for name, v in vars(litellm).items() if isinstance(v, set)}
    by_provider = {k: copy.copy(v) for k, v in litellm.models_by_provider.items()}
    yield
    for name, saved in name_sets.items():
        live = getattr(litellm, name)
        live.clear()
        live.update(saved)
    litellm.models_by_provider.clear()
    litellm.models_by_provider.update(by_provider)
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


def test_refresh_drops_a_rate_upstream_retired(monkeypatch, isolated_catalog):
    retired = "input_cost_per_token_above_200k_tokens"
    litellm.model_cost[NEW_MODEL] = {**NEW_ROW, retired: 8e-06}
    table = dict(litellm.model_cost)
    table[NEW_MODEL] = NEW_ROW
    _serve(monkeypatch, table)

    live_prices.refresh_litellm_prices()

    assert retired not in litellm.model_cost[NEW_MODEL]


def test_refresh_prices_an_alias_at_its_target_rate(monkeypatch, isolated_catalog):
    alias = "claude-headroom-test-latest"
    table = dict(litellm.model_cost)
    table[NEW_MODEL] = NEW_ROW
    table[alias] = {"alias_of": NEW_MODEL}
    _serve(monkeypatch, table)

    live_prices.refresh_litellm_prices()

    assert litellm.model_cost[alias]["cache_read_input_token_cost"] == 2e-07
    assert "alias_of" not in litellm.model_cost[alias]


def test_a_deepseek_alias_takes_headroom_off_peak_rate(monkeypatch, isolated_catalog):
    model, ours = next(iter(_DEEPSEEK_V4_PRICING.items()))
    target = f"deepseek/{model}"
    alias = "deepseek/deepseek-headroom-test-latest"
    table = dict(litellm.model_cost)
    table[target] = {**table.get(target, {}), "input_cost_per_token": 1.0}  # upstream peak
    table[alias] = {"alias_of": target}
    _serve(monkeypatch, table)

    live_prices.refresh_litellm_prices()

    for field in ("input_cost_per_token", "output_cost_per_token"):
        assert litellm.model_cost[alias][field] == litellm.model_cost[target][field]
    assert litellm.model_cost[alias]["input_cost_per_token"] == ours["input_cost_per_token"]


def test_cancelling_the_refresh_does_not_wait_for_a_stuck_download(monkeypatch):
    monkeypatch.setenv("HEADROOM_PRICE_REFRESH", "on")
    started, release = threading.Event(), threading.Event()

    def stuck_refresh():
        started.set()
        release.wait(30)

    monkeypatch.setattr(live_prices, "refresh_litellm_prices", stuck_refresh)

    async def run_and_cancel():
        task = asyncio.create_task(live_prices.refresh_periodically())
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)

    try:
        asyncio.run(run_and_cancel())
        workers = [t for t in threading.enumerate() if t.name == "headroom-price-refresh"]
        assert workers and all(t.daemon for t in workers)  # abandoned at exit, never joined
    finally:
        release.set()


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

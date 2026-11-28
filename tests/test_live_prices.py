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
import sys
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


def _add_old_row(row: dict) -> None:
    """List ``NEW_MODEL`` at ``row``, as a table bundled before a refresh would."""
    from headroom.pricing.litellm_pricing import invalidate_litellm_lookup_caches

    litellm.model_cost[NEW_MODEL] = row
    litellm.add_known_models({NEW_MODEL: row})
    invalidate_litellm_lookup_caches()
    live_prices._clear_price_caches()


def _input_rate(model: str, provider: str) -> object:
    """``model``'s input price from ``litellm.get_model_info``, or the error it raised."""
    try:
        return litellm.get_model_info(model, custom_llm_provider=provider)["input_cost_per_token"]
    except Exception as exc:  # noqa: BLE001 - litellm raises a bare Exception when unmapped
        return exc


class _Overlap:
    """Pause one reader thread at a point the test chooses, while a refresh runs.

    Every wait is bounded, the reader is always released, and a reader that
    never reached the pause point is reported, so no test hangs on it.
    """

    def __init__(self, read) -> None:
        self.paused, self._resume = threading.Event(), threading.Event()
        self.results: list[object] = []

        def run() -> None:
            try:
                self.results.append(read())
            except Exception as exc:  # noqa: BLE001 - reported to the test, which asserts on it
                self.results.append(exc)

        self.reader = threading.Thread(target=run)

    def pause_here(self) -> None:
        """Call from the code under test; pauses only the reader thread."""
        if threading.current_thread() is self.reader:
            self.paused.set()
            self._resume.wait(5)

    def refresh_during_read(self, models: dict) -> bool:
        """Start the read, refresh once it pauses, then let it finish. False if it never paused."""
        self.reader.start()
        try:
            reached = self.paused.wait(5)
            if reached:
                live_prices.merge_litellm_prices(models)
        finally:
            self._resume.set()
            self.reader.join(5)
        assert not self.reader.is_alive()
        return reached


def test_a_resolution_overlapping_a_refresh_is_not_reused_after_it(monkeypatch, isolated_catalog):
    """A name resolved against the old table is not served once the new one is published.

    The pause sits in Headroom's resolver, after it resolved against the old
    table and before it caches the answer, so this holds on every supported
    LiteLLM.
    """
    from headroom.pricing import litellm_pricing
    from headroom.pricing.litellm_pricing import get_model_pricing, resolve_litellm_model

    resolve = litellm_pricing._resolve_litellm_model_uncached
    overlap = _Overlap(lambda: resolve_litellm_model(NEW_MODEL.upper()))

    def resolve_then_pause(model):
        resolved = resolve(model)
        overlap.pause_here()
        return resolved

    monkeypatch.setattr(litellm_pricing, "_resolve_litellm_model_uncached", resolve_then_pause)

    assert overlap.refresh_during_read({NEW_MODEL: NEW_ROW})
    assert overlap.results == [NEW_MODEL.upper()]  # the old table lacked it
    assert resolve_litellm_model(NEW_MODEL.upper()) == NEW_MODEL
    pricing = get_model_pricing(NEW_MODEL.upper())
    assert pricing is not None
    assert pricing.input_cost_per_1m == pytest.approx(NEW_ROW["input_cost_per_token"] * 1e6)


def test_rates_read_during_a_refresh_are_not_reused_after_it(monkeypatch, isolated_catalog):
    """Rates resolved from a row a refresh then replaced are not served afterwards.

    The pause sits in Headroom's rate resolver, after it read the old row and
    before the result is cached.
    """
    from headroom.pricing import counterfactual

    old_rate, new_rate = 9e-06, NEW_ROW["input_cost_per_token"]
    _add_old_row({**NEW_ROW, "input_cost_per_token": old_rate})
    read_row = counterfactual._catalog_row
    overlap = _Overlap(lambda: counterfactual.resolve_rates(NEW_MODEL).uncached)

    def read_then_pause(model):
        row = read_row(model)
        overlap.pause_here()
        return row

    monkeypatch.setattr(counterfactual, "_catalog_row", read_then_pause)

    assert overlap.refresh_during_read({NEW_MODEL: NEW_ROW})
    assert overlap.results == [pytest.approx(old_rate)]
    assert counterfactual.resolve_rates(NEW_MODEL).uncached == pytest.approx(new_rate)


def _inside(function_name: str) -> bool:
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == function_name:
            return True
        frame = frame.f_back
    return False


class _PausingRow(dict):
    """A catalog row that records, or pauses on, LiteLLM's reads while it memoises model info.

    LiteLLM fetches the row afresh several times while building the info it
    memoises, so only a pause after its last read leaves the build holding the
    old row. A dry run of the same lookup counts the reads.
    """

    overlap: _Overlap | None = None
    pause_at: int | None = None
    reads = 0

    def _read(self, key) -> None:
        if not _inside("_cached_get_model_info"):
            return
        self.reads += 1
        if self.reads == self.pause_at and self.overlap is not None:
            self.overlap.pause_here()

    def __getitem__(self, key):
        self._read(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._read(key)
        return super().get(key, default)


def test_a_budget_estimate_after_an_overlapping_lookup_uses_the_refreshed_rate(isolated_catalog):
    """LiteLLM's model-info cache cannot keep a rate a refresh replaced.

    LiteLLM 1.101 memoises model info (``_cached_get_model_info``) without
    keying it on the table, so a lookup that read the old row and finished
    after the refresh cleared that cache stored the old rate again, and budget
    estimates kept charging it. The pause sits inside that memoised build,
    after its last read of the row; a LiteLLM without that cache has nothing
    to go stale.
    """
    from headroom.pricing.litellm_pricing import invalidate_litellm_lookup_caches
    from headroom.proxy.cost import CostTracker

    old_rate, new_rate = 9e-06, NEW_ROW["input_cost_per_token"]
    old_row = _PausingRow({**NEW_ROW, "input_cost_per_token": old_rate})
    _add_old_row(old_row)

    def estimate() -> float | None:
        return CostTracker().estimate_cost(NEW_MODEL, input_tokens=1_000_000, output_tokens=0)

    estimate()  # dry run: how many reads does the memoised build make?
    if not old_row.reads:
        pytest.skip("this LiteLLM has no _cached_get_model_info to go stale")
    old_row.pause_at, old_row.reads = old_row.reads, 0
    invalidate_litellm_lookup_caches()
    live_prices._clear_price_caches()
    overlap = _Overlap(estimate)
    old_row.overlap = overlap

    assert overlap.refresh_during_read({NEW_MODEL: NEW_ROW})
    old_row.overlap = None
    assert estimate() == pytest.approx(new_rate * 1_000_000)


def test_a_refresh_does_not_break_a_lookup_iterating_the_table(isolated_catalog):
    """A refresh landing while LiteLLM iterates the table leaves that lookup priced.

    LiteLLM 1.101 resolves a mixed-case name by rebuilding a lowercase index
    that iterates ``litellm.model_cost``; growing that dict in place, even in
    one ``dict.update``, failed the iteration and the model read as unmapped.
    The rebuild then stores an index of the replaced table, which Headroom's
    lookups do not use. A key whose ``lower()`` blocks pauses the rebuild;
    LiteLLM versions that index the table natively never call it, and have no
    such overlap.
    """
    from headroom.pricing.litellm_pricing import get_model_pricing, resolve_litellm_model

    overlap = _Overlap(lambda: _input_rate("GPT-4O", "openai"))

    class PausingKey(str):
        def lower(self) -> str:
            overlap.pause_here()
            return str(self).lower()

    litellm.model_cost[PausingKey("headroom-test-pause")] = {"input_cost_per_token": 0.0}
    litellm.utils._invalidate_model_cost_lowercase_map()

    if not overlap.refresh_during_read({NEW_MODEL: NEW_ROW}):
        assert not isinstance(overlap.results[0], Exception), overlap.results
        pytest.skip("this LiteLLM does not rebuild its name index from the table in Python")
    assert overlap.results == [pytest.approx(2.5e-06)]
    assert resolve_litellm_model(NEW_MODEL.upper()) == NEW_MODEL
    pricing = get_model_pricing(NEW_MODEL.upper())
    assert pricing is not None
    assert pricing.input_cost_per_1m == pytest.approx(NEW_ROW["input_cost_per_token"] * 1e6)


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


MINIMAX_BARE, MINIMAX_PREFIXED = "MiniMax-M3", "minimax/MiniMax-M3"


@pytest.fixture
def minimax_prefixed_only(monkeypatch, isolated_catalog):
    """Startup on a table listing MiniMax-M3 only under ``minimax/`` at $0.60/M input."""
    from headroom.pricing import litellm_pricing

    monkeypatch.setattr(litellm_pricing, "_mirrored_bare_rows", set())
    litellm.model_cost.pop(MINIMAX_BARE, None)
    litellm.model_cost[MINIMAX_PREFIXED] = {**NEW_ROW, "input_cost_per_token": 6e-07}


def _refresh_minimax_prefixed_to_1_20(monkeypatch):
    table = {k: v for k, v in litellm.model_cost.items() if k != MINIMAX_BARE}
    table[MINIMAX_PREFIXED] = {**NEW_ROW, "input_cost_per_token": 1.2e-06}
    _serve(monkeypatch, table)
    live_prices.refresh_litellm_prices()


def test_a_bare_row_headroom_made_follows_its_refreshed_prefixed_row(
    monkeypatch, minimax_prefixed_only
):
    from headroom.pricing.litellm_pricing import _register_minimax_pricing, get_model_pricing

    _register_minimax_pricing()
    _refresh_minimax_prefixed_to_1_20(monkeypatch)

    for name in (MINIMAX_BARE, MINIMAX_PREFIXED):
        assert get_model_pricing(name).input_cost_per_1m == pytest.approx(1.20)


def test_a_bare_row_litellm_supplied_keeps_its_own_rate(monkeypatch, minimax_prefixed_only):
    from headroom.pricing.litellm_pricing import _register_minimax_pricing, get_model_pricing

    litellm.model_cost[MINIMAX_BARE] = {**NEW_ROW, "input_cost_per_token": 9e-07}
    _register_minimax_pricing()
    _refresh_minimax_prefixed_to_1_20(monkeypatch)

    assert get_model_pricing(MINIMAX_BARE).input_cost_per_1m == pytest.approx(0.90)
    assert get_model_pricing(MINIMAX_PREFIXED).input_cost_per_1m == pytest.approx(1.20)


def test_cancelling_the_refresh_does_not_wait_for_a_stuck_download(monkeypatch):
    monkeypatch.setenv("HEADROOM_PRICE_REFRESH", "on")
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
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

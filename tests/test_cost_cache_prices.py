"""Reconcile reported cache slices with model-specific LiteLLM billing.

Explicit catalog rates win independently. Missing rates use the canonical
calculator, whose fallback differs across supported versions and models;
unavailable calculations do not invent Headroom provider-wide prices.
Counterfactual savings and cache-card provenance remain separate from billing,
and inferred writes cannot double-charge the uncached input bucket.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from headroom.log_safety import WarnOnce
from headroom.pricing.counterfactual import resolve_rates
from headroom.proxy.cost import CostTracker


@pytest.fixture(autouse=True)
def clear_rate_cache():
    resolve_rates.cache_clear()
    yield
    resolve_rates.cache_clear()


def _patch_litellm(monkeypatch: pytest.MonkeyPatch, model_cost: dict) -> None:
    resolve_rates.cache_clear()
    monkeypatch.setattr(
        "headroom.pricing.counterfactual._litellm",
        lambda: SimpleNamespace(model_cost=model_cost),
    )
    monkeypatch.setattr(
        "headroom.pricing.litellm_pricing.resolve_litellm_model",
        lambda model: model,
    )


class TestGetCachePrices:
    def test_uses_explicit_litellm_cache_fields_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_litellm(
            monkeypatch,
            {
                "m": {
                    "input_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": 1e-7,
                    "cache_creation_input_token_cost": 1.25e-6,
                    "litellm_provider": "anthropic",
                }
            },
        )
        assert CostTracker()._get_cache_prices("m") == (1e-7, 1.25e-6, 2e-6, 1e-6)

    def test_missing_cache_fields_priced_at_zero_not_full_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A model with an input price but no cache fields: exactly the 1600+
        # long-tail models the bug affects. Before the fix both slices billed at
        # the full uncached rate; now they fail closed at $0, reconciling with
        # what litellm.cost_per_token books for the same slice.
        _patch_litellm(
            monkeypatch,
            {"m": {"input_cost_per_token": 5e-7, "litellm_provider": "mistral"}},
        )
        assert CostTracker()._get_cache_prices("m") == (0.0, 0.0, 0.0, 5e-7)

    def test_only_the_missing_field_is_zeroed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Explicit read cost present, write cost absent: keep the real read,
        # fail closed on the write.
        _patch_litellm(
            monkeypatch,
            {
                "m": {
                    "input_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": 3e-7,
                    "litellm_provider": "openai",
                }
            },
        )
        assert CostTracker()._get_cache_prices("m") == (3e-7, 0.0, 0.0, 1e-6)

    def test_no_input_price_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_litellm(monkeypatch, {"m": {"litellm_provider": "anthropic"}})
        assert CostTracker()._get_cache_prices("m") is None

    @pytest.mark.parametrize(
        "litellm_provider",
        ["bedrock", "bedrock_converse", "mistral", "fireworks_ai", "cohere_chat", None, "made-up"],
    )
    def test_heterogeneous_and_unknown_providers_do_not_inherit_anthropic_rates(
        self, monkeypatch: pytest.MonkeyPatch, litellm_provider
    ) -> None:
        # Regression: a missing cache field must never be back-filled with
        # Anthropic's 0.1x/1.25x economics. Bedrock fronts Anthropic, Amazon,
        # Meta, Mistral, Cohere and others; unknown OpenAI-compatible providers
        # are likewise not Anthropic-priced.
        uncached = 4e-6
        info = {"input_cost_per_token": uncached}
        if litellm_provider is not None:
            info["litellm_provider"] = litellm_provider
        _patch_litellm(monkeypatch, {"m": info})
        cache_read, cache_write, cache_write_1h, got_uncached = CostTracker()._get_cache_prices("m")
        assert got_uncached == uncached
        assert cache_read == 0.0
        assert cache_write == 0.0
        assert cache_write_1h == 0.0
        # Explicitly not the Anthropic multipliers the old fallback would apply.
        assert cache_read != pytest.approx(uncached * 0.1)
        assert cache_write != pytest.approx(uncached * 1.25)

    def test_missing_cache_slices_match_canonical_zero_pricing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The missing cache slices are priced at $0, the same amount the canonical
        litellm.cost_per_token path books for a slice LiteLLM cannot price, so the
        cache breakdown reconciles with the primary accounting instead of diverging.
        """
        _patch_litellm(
            monkeypatch,
            {"m": {"input_cost_per_token": 5e-7, "litellm_provider": "fireworks_ai"}},
        )
        cr_price, cw_price, _, uncached_price = CostTracker()._get_cache_prices("m")
        cr_tokens, cw_tokens = 20_000, 5_000
        # Cache-slice cost contributed to totals() is exactly $0, matching canonical.
        assert cr_tokens * cr_price + cw_tokens * cw_price == 0.0
        assert uncached_price == 5e-7

    def test_totals_and_stats_charge_only_known_slices(self, monkeypatch):
        _patch_litellm(monkeypatch, {"m": {"input_cost_per_token": 5e-7}})
        tracker = CostTracker()
        tracker.record_tokens(
            model="m",
            tokens_saved=0,
            tokens_sent=26_000,
            cache_read_tokens=20_000,
            cache_write_tokens=5_000,
            uncached_tokens=1_000,
        )
        assert tracker.totals() == (26_000, 0.0005)
        assert tracker.stats()["total_input_cost_usd"] == 0.0005

    def test_missing_read_preserves_explicit_write_and_ttl_tier(self, monkeypatch):
        _patch_litellm(
            monkeypatch,
            {
                "m": {
                    "input_cost_per_token": 1e-6,
                    "input_cost_per_token_above_200k_tokens": 2e-6,
                    "cache_creation_input_token_cost": 1.25e-6,
                    "cache_creation_input_token_cost_above_200k_tokens": 2.5e-6,
                    "cache_creation_input_token_cost_above_1hr": 2e-6,
                }
            },
        )
        assert CostTracker()._get_cache_prices("m") == (0.0, 1.25e-6, 2e-6, 1e-6)
        assert CostTracker()._get_cache_prices("m", long_context=True) == (
            0.0,
            2.5e-6,
            4e-6,
            2e-6,
        )

    def test_savings_provider_ratio_remains_separate_from_billed_prices(self, monkeypatch):
        _patch_litellm(monkeypatch, {"m": {"input_cost_per_token": 1e-6}})
        assert resolve_rates("m", provider="anthropic").read == 1e-7
        assert CostTracker()._get_cache_prices("m")[0] == 0.0

    def test_missing_write_price_keeps_live_zone_and_tool_savings(self, monkeypatch):
        _patch_litellm(monkeypatch, {"m": {"input_cost_per_token": 1e-6}})
        tracker = CostTracker()
        tracker.record_tokens(
            model="m",
            tokens_saved=10_000,
            tokens_sent=10_000,
            cache_write_tokens=10_000,
            tool_schema_saved=5_000,
        )
        stats = tracker.stats()
        assert stats["total_input_cost_usd"] == 0.0
        assert stats["cache_aware_savings_usd"] == pytest.approx(0.01)
        assert stats["tool_savings_usd"] == pytest.approx(0.005)

    @pytest.mark.parametrize("cache_read_price", [None, "invalid"])
    def test_unknown_read_rate_is_not_advertised_as_catalog_free_cache(
        self, monkeypatch, cache_read_price
    ):
        from headroom.proxy.cost import build_prefix_cache_stats
        from headroom.proxy.prometheus_metrics import PrometheusMetrics

        _patch_litellm(
            monkeypatch,
            {
                "gpt-unknown": {
                    "input_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": cache_read_price,
                }
            },
        )
        tracker = CostTracker()
        tracker._tokens_sent_by_model["gpt-unknown"] = 1_000_000
        monkeypatch.setattr(tracker, "_get_list_price", lambda model: 1.0)
        metrics = PrometheusMetrics()
        metrics.cache_by_provider["openai"].update(
            {
                "requests": 1,
                "hit_requests": 1,
                "cache_read_tokens": 1_000_000,
            }
        )
        row = build_prefix_cache_stats(metrics, tracker)["by_provider"]["openai"]
        assert row["cache_pricing_source"] == "provider_default"
        assert row["savings_usd"] == pytest.approx(0.5)

    def test_canonical_litellm_cache_only_cost_matches_dashboard(self, monkeypatch):
        import litellm

        from headroom.pricing.litellm_pricing import resolve_litellm_model

        # Keep the actual canonical calculator, with a deterministic catalog
        # row: the shared test suite enriches real model rows with cache rates.
        model = "openai/headroom-missing-cache-rate-regression"
        monkeypatch.setitem(
            litellm.model_cost,
            model,
            {
                "input_cost_per_token": 5e-7,
                "output_cost_per_token": 1e-6,
                "litellm_provider": "openai",
                "mode": "chat",
            },
        )
        resolved = resolve_litellm_model(model)
        info = litellm.model_cost[resolved]
        assert info.get("cache_read_input_token_cost") is None
        assert info.get("cache_creation_input_token_cost") is None
        canonical, _ = litellm.cost_per_token(
            model=resolved,
            prompt_tokens=10_000,
            completion_tokens=0,
            cache_read_input_tokens=10_000,
        )
        tracker = CostTracker()
        tracker.record_tokens(
            model=model, tokens_saved=0, tokens_sent=10_000, cache_read_tokens=10_000
        )
        assert tracker.totals()[1] == round(canonical, 4)

    @pytest.mark.parametrize("long_context", [False, True])
    def test_missing_catalog_rates_use_the_model_specific_canonical_calculation(
        self, monkeypatch, long_context
    ):
        resolve_rates.cache_clear()
        monkeypatch.setattr(
            "headroom.pricing.litellm_pricing.resolve_litellm_model", lambda model: model
        )
        calls = []

        def canonical(**kwargs):
            calls.append(kwargs)
            factor = 2 if kwargs["prompt_tokens"] > 200_000 else 1
            return (
                factor
                * (
                    kwargs.get("cache_read_input_tokens", 0) * 4e-7
                    + kwargs.get("cache_creation_input_tokens", 0) * 1e-6
                ),
                0.0,
            )

        monkeypatch.setattr(
            "headroom.pricing.counterfactual._litellm",
            lambda: SimpleNamespace(
                model_cost={
                    "m": {
                        "input_cost_per_token": 1e-6,
                        "input_cost_per_token_above_200k_tokens": 2e-6,
                    }
                },
                cost_per_token=canonical,
            ),
        )
        factor = 2 if long_context else 1
        assert CostTracker()._get_cache_prices("m", long_context=long_context) == (
            4e-7 * factor,
            1e-6 * factor,
            1e-6 * factor,
            1e-6 * factor,
        )
        assert len(calls) == 2
        assert all(
            call["model"] == "m" and call["prompt_tokens"] == (200_001 if long_context else 1)
            for call in calls
        )

    def test_inferred_writes_are_not_billed_twice_with_canonical_rates(self, monkeypatch):
        _patch_litellm(
            monkeypatch,
            {
                "m": {
                    "input_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": 5e-7,
                    "cache_creation_input_token_cost": 1e-6,
                }
            },
        )
        tracker = CostTracker()
        tracker.record_tokens(
            model="m",
            tokens_saved=0,
            tokens_sent=20_000,
            cache_read_tokens=10_000,
            cache_write_tokens=10_000,
            cache_write_5m_tokens=10_000,
            uncached_tokens=10_000,
            cache_inferred=True,
        )
        assert tracker.totals()[1] == pytest.approx(0.015)
        assert tracker.stats()["total_input_cost_usd"] == pytest.approx(0.015)
        assert tracker._api_cache_write_by_model["m"] == 0
        assert tracker._api_cache_write_5m_by_model["m"] == 0

    def test_explicit_zero_cache_tier_and_one_hour_prices_are_authoritative(self, monkeypatch):
        _patch_litellm(
            monkeypatch,
            {
                "m": {
                    "input_cost_per_token": 1e-6,
                    "input_cost_per_token_above_200k_tokens": 2e-6,
                    "cache_read_input_token_cost": 1e-7,
                    "cache_read_input_token_cost_above_200k_tokens": 0.0,
                    "cache_creation_input_token_cost": 1.25e-6,
                    "cache_creation_input_token_cost_above_200k_tokens": 0.0,
                    "cache_creation_input_token_cost_above_1hr": 0.0,
                }
            },
        )
        assert CostTracker()._get_cache_prices("m")[2] == 0.0
        assert CostTracker()._get_cache_prices("m", long_context=True) == (
            0.0,
            0.0,
            0.0,
            2e-6,
        )

    def test_explicit_list_rate_cache_prices_do_not_inherit_provider_discounts(self, monkeypatch):
        _patch_litellm(
            monkeypatch,
            {
                "m": {
                    "input_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": 1e-6,
                    "cache_creation_input_token_cost": 1e-6,
                }
            },
        )
        rates = resolve_rates("m", provider="anthropic")
        assert rates.read == rates.write_5m == 1e-6
        assert rates.read_is_catalog and rates.write_is_catalog


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_cache_rate_failure_warns_once_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising rate resolver used to read as 'unpriced' ($0) with no trace."""
    from headroom.proxy import cost

    monkeypatch.setattr(cost, "_pricing_warnings", WarnOnce(1024, "unpriceable models"))
    _patch_litellm(
        monkeypatch,
        {"m": {"input_cost_per_token": 1e-6, "litellm_provider": "anthropic"}},
    )
    capture = _Capture()
    cost.logger.addHandler(capture)
    try:
        assert CostTracker()._get_cache_prices("m") is not None
        assert not capture.records  # a resolvable model is silent

        def broken(*_args, **_kwargs):
            raise RuntimeError("pricing table corrupt")

        monkeypatch.setattr("headroom.pricing.counterfactual.resolve_rates", broken)
        assert CostTracker()._get_cache_prices("m") is None
        assert CostTracker()._get_cache_prices("m") is None
    finally:
        cost.logger.removeHandler(capture)

    warnings = [r.getMessage() for r in capture.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "for model 'm'" in warnings[0] and "RuntimeError" in warnings[0]
    assert "pricing table corrupt" not in warnings[0]


_CREDENTIAL_CANARY = "Authorization: Bearer sk-canary-7f3a9c"


def test_pricing_error_never_logs_its_text_at_any_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pricing backend error quoting a credential must not reach any log record.

    Formats every WARNING and DEBUG record with a real Formatter, which renders
    exc_info too, so a traceback carrying the text would be caught.
    """
    from headroom.proxy import cost

    class _LeakyLiteLLM:
        model_cost: dict = {}

        @staticmethod
        def cost_per_token(**_kwargs):
            raise RuntimeError(f"upstream rejected request ({_CREDENTIAL_CANARY})")

    monkeypatch.setattr(cost, "_pricing_warnings", WarnOnce(1024, "unpriceable models"))
    monkeypatch.setattr(cost, "_get_litellm_module", lambda: _LeakyLiteLLM())
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    capture = _Capture()
    previous_level = cost.logger.level
    cost.logger.setLevel(logging.DEBUG)
    cost.logger.addHandler(capture)
    try:
        assert CostTracker().estimate_cost("gpt-4o", 100, 10) is None
    finally:
        cost.logger.removeHandler(capture)
        cost.logger.setLevel(previous_level)

    formatter = logging.Formatter()
    rendered = [formatter.format(r) for r in capture.records if r.levelno >= logging.DEBUG]
    assert any("Failed to get pricing for model 'gpt-4o'" in line for line in rendered)
    assert all("sk-canary" not in line for line in rendered), rendered

"""xAI inferred writes and cache estimates honor billing provenance."""

import pytest


@pytest.mark.parametrize("inferred, expected", [(True, 0.0058), (False, 0.0108)])
@pytest.mark.parametrize("ttl_split", [False, True])
def test_xai_inferred_writes_are_not_billed_twice(monkeypatch, inferred, expected, ttl_split):
    import litellm

    from headroom.proxy.cost import CostTracker

    model = "xai/grok-review-inferred"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            # Keep the double-charge regression about writes, with known rates.
            "cache_read_input_token_cost": 0.16e-6,
            "cache_creation_input_token_cost": 1e-6,
            "litellm_provider": "xai",
            "mode": "chat",
        },
    )
    tracker = CostTracker()
    tracker.record_tokens(
        model,
        tokens_saved=0,
        tokens_sent=10_000,
        cache_read_tokens=5_000,
        cache_write_tokens=5_000,
        cache_write_5m_tokens=2_000 if ttl_split else 0,
        cache_write_1h_tokens=3_000 if ttl_split else 0,
        uncached_tokens=5_000,
        cache_inferred=inferred,
    )
    assert tracker.totals()[1] == pytest.approx(expected)
    assert tracker.stats()["total_input_cost_usd"] == pytest.approx(expected)


@pytest.mark.parametrize("catalog_read", [None, 0.05e-6])
def test_xai_cache_savings_and_billing_use_their_respective_price_sources(
    monkeypatch, catalog_read
):
    import litellm

    from headroom.pricing.counterfactual import (
        BASIS_CATALOG,
        BASIS_PROVIDER_RATIO,
        CacheMix,
        Region,
        price_savings,
        resolve_rates,
    )
    from headroom.proxy.cost import CostTracker

    # Provider dispatch uses LiteLLM's registered model set independently of
    # model_cost. A synthetic catalog key can therefore be rejected by older
    # supported LiteLLM versions before the canonical billing probe runs.
    model = next(
        model
        for model in sorted(litellm.xai_models)
        if model.startswith("xai/grok") and model in litellm.model_cost
    )
    row = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "litellm_provider": "xai",
        "mode": "chat",
    }
    if catalog_read is not None:
        row["cache_read_input_token_cost"] = catalog_read
    monkeypatch.setitem(litellm.model_cost, model, row)
    resolve_rates.cache_clear()
    try:
        expected_read = catalog_read if catalog_read is not None else 0.16e-6
        mix = CacheMix.from_usage(cache_read_tokens=10_000)
        saving = price_savings(10_000, model=model, mix=mix, provider="xai", region=Region.PREFIX)
        assert saving.usd == pytest.approx(10_000 * expected_read)
        assert saving.basis == (BASIS_PROVIDER_RATIO if catalog_read is None else BASIS_CATALOG)

        # A provider-wide counterfactual estimate is not a billed cache rate.
        # Use the real calculator for the same reported usage, as billing does.
        if catalog_read is None:
            canonical_input, _ = litellm.cost_per_token(
                model=model,
                prompt_tokens=10_000,
                completion_tokens=0,
                cache_read_input_tokens=10_000,
            )
        else:
            canonical_input = 10_000 * catalog_read
        tracker = CostTracker()
        tracker.record_tokens(model, tokens_saved=0, tokens_sent=10_000, cache_read_tokens=10_000)
        assert tracker.totals()[1] == pytest.approx(canonical_input)
        assert tracker.stats()["total_input_cost_usd"] == pytest.approx(canonical_input)
    finally:
        resolve_rates.cache_clear()


@pytest.mark.parametrize("user_agent", ["grok/1.2.3", "grok-shell/0.2.112"])
def test_grok_api_keys_keep_payg_policy_across_client_versions(user_agent):
    from headroom.proxy.auth_policy import (
        AuthMode,
        AuthSignals,
        classify_auth_signals,
        classify_client_signals,
    )

    signals = AuthSignals(user_agent=user_agent, authorization="Bearer xai-test-key")
    assert classify_client_signals(signals) == "grok_build"
    assert classify_auth_signals(signals) is AuthMode.PAYG


@pytest.mark.parametrize("user_agent", ["grok/1.2.3", "grok-shell/0.2.112"])
def test_grok_oauth_tokens_keep_oauth_policy_across_client_versions(user_agent):
    from headroom.proxy.auth_policy import AuthMode, AuthSignals, classify_auth_signals

    signals = AuthSignals(user_agent=user_agent, authorization="Bearer header.payload.signature")
    assert classify_auth_signals(signals) is AuthMode.OAUTH

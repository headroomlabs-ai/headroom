"""Generated output changes card when compression avoids a long-prompt tier."""

from __future__ import annotations

import pytest

from headroom.pricing.counterfactual import long_context_threshold, resolve_rates
from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.proxy.savings_tracker import SavingsTracker, estimate_request_savings_usd


@pytest.fixture
def haiku_card(monkeypatch):
    import litellm

    monkeypatch.setitem(
        litellm.model_cost,
        "claude-haiku-5-5",
        {
            "litellm_provider": "anthropic",
            "input_cost_per_token": 1e-7,
            "output_cost_per_token": 5e-7,
            "input_cost_per_token_above_100k_tokens": 5e-7,
            "output_cost_per_token_above_100k_tokens": 2.5e-6,
        },
    )
    resolve_rates.cache_clear()
    long_context_threshold.cache_clear()
    yield
    resolve_rates.cache_clear()
    long_context_threshold.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("through_metrics", [False, True], ids=["persistent", "metrics"])
@pytest.mark.parametrize(
    ("forwarded", "removed", "output", "expected"),
    [
        (90_000, 20_000, 1_000, 0.048),
        (110_000, 10_000, 1_000, 0.005),
        (90_000, 20_000, 0, 0.046),
        (90_000, 10_000, 1_000, 0.001),
        (100_000, 10_000, 1_000, 0.047),
        (90_000, 0, 1_000, 0.0),
    ],
    ids=[
        "crosses",
        "stays-above",
        "no-output",
        "original-at-threshold",
        "forwarded-at-threshold",
        "no-compression",
    ],
)
async def test_forwarded_output_premium_only_counts_when_compression_crosses_the_tier(
    tmp_path, haiku_card, through_metrics, forwarded, removed, output, expected
):
    tracker = SavingsTracker(path=str(tmp_path / "savings.json"), stateless=True)
    request = {
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "input_tokens": forwarded,
        "uncached_input_tokens": forwarded,
        "tokens_saved": removed,
        "output_tokens": output,
    }
    if through_metrics:
        metrics = PrometheusMetrics(savings_tracker=tracker, stateless=True)
        await metrics.record_request(**request, latency_ms=1)
    else:
        tracker.record_lifetime_request(**request, stack=None)
    lifetime = tracker.lifetime_response()
    assert lifetime["cost"]["compression_savings_usd"] == pytest.approx(expected)
    assert lifetime["tokens"]["output"] == output
    if through_metrics:
        assert tracker.snapshot()["lifetime"]["output_savings_usd"] == 0


@pytest.mark.asyncio
async def test_avoided_output_premium_is_separate_from_output_shaping(tmp_path, haiku_card):
    tracker = SavingsTracker(path=str(tmp_path / "savings.json"), stateless=True)
    metrics = PrometheusMetrics(savings_tracker=tracker, stateless=True)
    await metrics.record_request(
        provider="anthropic",
        model="claude-haiku-5-5",
        input_tokens=90_000,
        uncached_input_tokens=90_000,
        tokens_saved=20_000,
        output_tokens=1_000,
        output_tokens_saved=100,
        latency_ms=1,
    )
    lifetime = tracker.snapshot()["lifetime"]
    assert lifetime["compression_savings_usd"] == pytest.approx(0.048)
    assert lifetime["output_savings_usd"] == pytest.approx(0.00005)


@pytest.mark.parametrize(
    ("base", "long", "premium"),
    [
        (0.0, 2.5e-6, 0.0025),
        (5e-7, 0.0, 0.0),
        (None, 2.5e-6, 0.0),
        (5e-7, None, 0.0),
        (5e-7, "invalid", 0.0),
    ],
    ids=["zero-base", "zero-long", "missing-base", "missing-long", "invalid-long"],
)
def test_optional_output_rate_metadata_is_fail_soft(haiku_card, base, long, premium):
    import litellm

    row = litellm.model_cost["claude-haiku-5-5"]
    row["output_cost_per_token"] = base
    row["output_cost_per_token_above_100k_tokens"] = long
    priced = estimate_request_savings_usd(
        "claude-haiku-5-5",
        compression_tokens_saved=20_000,
        uncached_input_tokens=90_000,
        local_input_tokens=90_000,
        output_tokens=1_000,
    )
    assert priced["compression"] == pytest.approx(0.046 + premium)
    assert priced["compression_list"] == pytest.approx(0.01)


def test_billing_preserves_explicit_zero_cache_rates_at_the_model_tier(haiku_card):
    import litellm

    row = litellm.model_cost["claude-haiku-5-5"]
    for field in (
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
    ):
        row[f"{field}_above_100k_tokens"] = 0.0
    rates = resolve_rates("claude-haiku-5-5", long_context=True, for_billing=True)
    assert rates is not None
    assert rates.uncached == 5e-7
    assert (rates.read, rates.write_5m, rates.write_1h) == (0.0, 0.0, 0.0)
    assert rates.read_is_catalog and rates.write_is_catalog


def test_missing_billed_cache_rates_probe_the_model_specific_threshold(haiku_card, monkeypatch):
    import litellm

    prompts = []

    def canonical_cost(**kwargs):
        prompts.append(kwargs["prompt_tokens"])
        return kwargs["prompt_tokens"] * 3e-8, 0.0

    monkeypatch.setattr(litellm, "cost_per_token", canonical_cost)
    rates = resolve_rates("claude-haiku-5-5", long_context=True, for_billing=True)
    assert rates is not None
    assert rates.read == pytest.approx(3e-8)
    assert rates.write_5m == pytest.approx(3e-8)
    assert prompts == [100_001, 100_001]
    assert not rates.read_is_catalog and not rates.write_is_catalog

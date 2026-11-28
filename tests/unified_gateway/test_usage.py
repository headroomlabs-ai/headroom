"""Independent accounting fixtures; synthetic bounds are not live qualification."""

from decimal import Decimal
from importlib import import_module, util

import pytest

from headroom.proxy.gateway.config import GatewayConfigSnapshot, ModelBounds, PricingConfig
from tests.unified_gateway.test_admission import EXAMPLE


def usage_api():
    assert util.find_spec("headroom.proxy.gateway.usage") is not None, "usage contracts missing"
    return import_module("headroom.proxy.gateway.usage")


def tariff():
    return PricingConfig(
        input_usd_per_million="2",
        output_usd_per_million="4",
        cache_read_usd_per_million="1",
        cache_create_usd_per_million="5",
        revision="fake-v1",
    )


def priced_route():
    return (
        GatewayConfigSnapshot.load(EXAMPLE)
        .routes[0]
        .model_copy(
            update={
                "pricing": tariff(),
                "model_bounds": ModelBounds(
                    max_input_tokens=100,
                    max_output_tokens=20,
                    default_max_output_tokens=10,
                    provider_contract="fake-model-v1",
                ),
            }
        )
    )


def test_unknown_cost_and_allowance_units_are_not_zero_usd():
    api = usage_api()
    observation = api.UsageObservation(allowance_units=Decimal("3.5"), allowance_unit="credits")
    cost = api.evaluate_cost(observation, None)
    assert cost.known_micro_usd is None
    assert cost.basis == "unknown"
    assert observation.input_tokens is None
    assert observation.allowance_units == Decimal("3.5")
    with pytest.raises((AttributeError, TypeError)):
        observation.input_tokens = 0


@pytest.mark.parametrize(
    "protocol,payload,expected",
    [
        (
            "openai-chat",
            {
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 2},
                }
            },
            (10, 3, 2, 0),
        ),
        (
            "openai-responses",
            {
                "response": {
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "input_tokens_details": {"cached_tokens": 2},
                    }
                }
            },
            (10, 3, 2, 0),
        ),
        (
            "anthropic-messages",
            {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 2,
                    "cache_creation_input_tokens": 4,
                }
            },
            (10, 3, 2, 4),
        ),
        (
            "gemini-generate",
            {
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 3,
                    "cachedContentTokenCount": 2,
                }
            },
            (10, 3, 2, 0),
        ),
        (
            "vertex-generate",
            {
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 3,
                    "cachedContentTokenCount": 2,
                }
            },
            (10, 3, 2, 0),
        ),
        (
            "bedrock-invoke",
            {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 3,
                    "cacheReadInputTokens": 2,
                    "cacheWriteInputTokens": 4,
                }
            },
            (10, 3, 2, 4),
        ),
    ],
)
def test_usage_normalizers_preserve_partial_and_duplicate_terminal_counts(
    protocol, payload, expected
):
    api = usage_api()
    first = api.normalize_usage(protocol, payload)
    repeated = api.normalize_usage(protocol, payload, previous=first)
    assert (
        repeated.input_tokens,
        repeated.output_tokens,
        repeated.cache_read_tokens,
        repeated.cache_create_tokens,
    ) == expected
    assert repeated.availability == "complete"
    assert api.normalize_usage(protocol, {}, previous=first) == first


def test_missing_malformed_and_partial_usage_stays_unknown():
    api = usage_api()
    partial = api.normalize_usage("openai-chat", {"usage": {"prompt_tokens": 10}})
    assert partial.output_tokens is None
    assert partial.availability == "partial"
    malformed = api.normalize_usage(
        "openai-chat", {"usage": {"prompt_tokens": True, "completion_tokens": -1}}
    )
    assert malformed.input_tokens is None
    assert malformed.output_tokens is None
    assert api.evaluate_cost(partial, tariff()).known_micro_usd == 20
    assert api.evaluate_cost(partial, tariff()).complete is False


def test_strict_cost_bound_covers_rates_and_rejects_unbounded_shapes():
    api = usage_api()
    route = priced_route()
    qualified = frozenset({"fake-model-v1"})
    bound = api.conservative_cost_bound(
        route, {"max_output_tokens": 20}, qualified_contracts=qualified
    )
    # 100 input tokens at highest applicable $5/M + 20 output tokens at $4/M.
    assert bound.reserved_upper_micro_usd == 580
    assert bound.tariff_revision == "fake-v1"
    assert (
        api.conservative_cost_bound(
            route, {}, qualified_contracts=qualified
        ).reserved_upper_micro_usd
        == 540
    )
    assert api.conservative_cost_bound(route, {}).reserved_upper_micro_usd is None
    for body in (
        {"max_output_tokens": 21},
        {"tools": [{"type": "web_search"}]},
        {"input": [{"type": "input_image", "image_url": "example"}]},
    ):
        assert (
            api.conservative_cost_bound(
                route, body, qualified_contracts=qualified
            ).reserved_upper_micro_usd
            is None
        )
    no_default = route.model_copy(
        update={
            "model_bounds": route.model_bounds.model_copy(
                update={"default_max_output_tokens": None}
            )
        }
    )
    assert (
        api.conservative_cost_bound(
            no_default, {}, qualified_contracts=qualified
        ).reserved_upper_micro_usd
        is None
    )
    incomplete = route.model_copy(
        update={"pricing": route.pricing.model_copy(update={"cache_create_usd_per_million": None})}
    )
    assert (
        api.conservative_cost_bound(
            incomplete, {}, qualified_contracts=qualified
        ).reserved_upper_micro_usd
        is None
    )


def test_decimal_rounding_and_observed_over_bound_cost_are_truthful():
    api = usage_api()
    assert api.usd_to_micro(Decimal("0.0000001"), reservation=True) == 1
    assert api.usd_to_micro(Decimal("0.0000009"), reservation=False) == 0
    observed = api.UsageObservation(
        currency_charge=Decimal("0.000581"),
        currency="USD",
        availability="complete",
        provenance="provider_reported",
    )
    cost = api.evaluate_cost(observed, tariff(), reserved_upper_micro_usd=580)
    assert cost.known_micro_usd == 581
    assert cost.bound_violated is True
    mismatch = api.UsageObservation(input_tokens=10, output_tokens=1, tariff_revision="other")
    assert api.evaluate_cost(mismatch, tariff()).known_micro_usd is None


def test_stale_usage_snapshots_do_not_erase_known_cumulative_counts():
    api = usage_api()
    terminal = api.normalize_usage(
        "anthropic-messages",
        {"usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 2}},
    )
    stale = api.normalize_usage(
        "anthropic-messages", {"usage": {"input_tokens": 10, "output_tokens": 1}}, previous=terminal
    )
    assert stale.output_tokens == 5
    assert stale.cache_read_tokens == 2


def test_gemini_reasoning_counts_and_multi_candidate_bounds():
    api = usage_api()
    usage = api.normalize_usage(
        "gemini-generate",
        {
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 7,
            }
        },
    )
    assert usage.output_tokens == 10
    cost = api.conservative_cost_bound(
        priced_route(),
        {"generationConfig": {"candidateCount": 2}},
        qualified_contracts=frozenset({"fake-model-v1"}),
    )
    assert cost.reserved_upper_micro_usd is None


def test_provider_reported_currency_is_retained_even_when_tariff_revision_disagrees():
    api = usage_api()
    usage = api.UsageObservation(
        currency_charge=Decimal("0.000020"),
        currency="USD",
        availability="complete",
        tariff_revision="other",
    )
    assert api.evaluate_cost(usage, tariff()).known_micro_usd == 20

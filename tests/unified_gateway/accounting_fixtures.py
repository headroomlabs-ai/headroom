"""Explicit synthetic bounds for ledger tests, never production qualification."""

from dataclasses import replace
from pathlib import Path

from headroom.proxy.gateway.admission import AdmissionRequest
from headroom.proxy.gateway.config import GatewayConfigSnapshot, ModelBounds, PricingConfig
from headroom.proxy.gateway.usage import conservative_cost_bound


def qualified_request(request: AdmissionRequest) -> AdmissionRequest:
    upper = request.reserved_upper_micro_usd
    if upper is None:
        return request
    pricing = PricingConfig(
        input_usd_per_million="0",
        output_usd_per_million="1" if upper else "0",
        cache_read_usd_per_million="0",
        cache_create_usd_per_million="0",
        revision="synthetic-ledger-v1",
    )
    model = ModelBounds(
        max_input_tokens=1,
        max_output_tokens=100000000,
        default_max_output_tokens=1,
        provider_contract="synthetic-ledger-v1",
    )
    path = (
        Path(__file__).parents[2]
        / "docs/proposals/unified-api-gateway/examples/gateway.api-keys.json"
    )
    route = (
        GatewayConfigSnapshot.load(path)
        .routes[0]
        .model_copy(update={"pricing": pricing, "model_bounds": model})
    )
    bound = conservative_cost_bound(
        route,
        {"max_output_tokens": max(1, upper)},
        qualified_contracts=frozenset({"synthetic-ledger-v1"}),
    )
    return replace(
        request, estimated_cost=None, pricing=pricing, model_bounds=model, cost_bound=bound
    )

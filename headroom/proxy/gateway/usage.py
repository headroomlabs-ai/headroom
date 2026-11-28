"""Immutable usage facts and conservative, explicitly qualified USD bounds.

No tokenizer, price discovery, or credential access belongs in this module.
Input counts are uncached tokens; cached dimensions are priced separately.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from typing import Any, Literal

from headroom.proxy.gateway.config import PricingConfig, RouteConfig


def usd_to_micro(value: Decimal, *, reservation: bool = True) -> int:
    if not value.is_finite() or value < 0:
        raise ValueError("invalid monetary amount")
    with localcontext() as context:
        context.prec = max(28, len(value.as_tuple().digits) + 12)
        return int(
            (value * 1_000_000).to_integral_value(
                rounding=ROUND_CEILING if reservation else ROUND_FLOOR
            )
        )


@dataclass(frozen=True, slots=True)
class UsageObservation:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_create_tokens: int | None = None
    currency_charge: Decimal | None = None
    currency: str | None = None
    allowance_units: Decimal | None = None
    allowance_unit: str | None = None
    availability: Literal["unknown", "partial", "complete"] = "unknown"
    provenance: str = "unknown"
    charge_free: bool = False
    tariff_revision: str | None = None
    # Retain the authoritative inclusive counter. Uncached input can decrease
    # when a later snapshot refines cache detail, so it is not monotonic itself.
    input_total_tokens: int | None = None

    def __post_init__(self) -> None:
        for count in (
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.cache_create_tokens,
            self.input_total_tokens,
        ):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError("invalid usage count")
        for amount in (self.currency_charge, self.allowance_units):
            if amount is not None and (not amount.is_finite() or amount < 0):
                raise ValueError("invalid usage amount")
        if self.availability not in {"unknown", "partial", "complete"}:
            raise ValueError("invalid usage availability")


@dataclass(frozen=True, slots=True)
class CostEvaluation:
    known_micro_usd: int | None = None
    basis: Literal["provider_reported", "configured_tariff", "unknown"] = "unknown"
    tariff_revision: str | None = None
    reserved_upper_micro_usd: int | None = None
    complete: bool = False
    bound_violated: bool = False
    provider_contract: str | None = None
    qualified_bound: bool = False


def evaluate_cost(
    usage: UsageObservation,
    pricing: PricingConfig | None,
    *,
    reserved_upper_micro_usd: int | None = None,
) -> CostEvaluation:
    revision = pricing.revision if pricing else None
    base = CostEvaluation(
        tariff_revision=revision, reserved_upper_micro_usd=reserved_upper_micro_usd
    )
    if (
        usage.tariff_revision is not None
        and usage.tariff_revision != revision
        and not usage.charge_free
        and not (usage.currency_charge is not None and usage.currency == "USD")
    ):
        return base
    known: int | None = None
    complete = False
    basis: Literal["provider_reported", "configured_tariff", "unknown"] = "unknown"
    if usage.charge_free:
        known, complete, basis = 0, True, "provider_reported"
    elif usage.currency_charge is not None and usage.currency == "USD":
        known = usd_to_micro(usage.currency_charge)
        complete, basis = usage.availability == "complete", "provider_reported"
    elif pricing is not None:
        dimensions = (
            (usage.input_tokens, pricing.input_usd_per_million),
            (usage.output_tokens, pricing.output_usd_per_million),
            (usage.cache_read_tokens, pricing.cache_read_usd_per_million),
            (usage.cache_create_tokens, pricing.cache_create_usd_per_million),
        )
        priced = [
            (count, rate) for count, rate in dimensions if count is not None and rate is not None
        ]
        if priced:
            with localcontext() as context:
                context.prec = 80
                # USD/M-token multiplied by tokens is exactly micro-USD.
                known = int(
                    sum(
                        (Decimal(count) * Decimal(rate) for count, rate in priced), Decimal(0)
                    ).to_integral_value(rounding=ROUND_CEILING)
                )
            basis = "configured_tariff"
        complete = usage.availability == "complete" and all(
            count is not None and (count == 0 or rate is not None) for count, rate in dimensions
        )
    return replace(
        base,
        known_micro_usd=known,
        basis=basis,
        complete=complete,
        bound_violated=known is not None
        and reserved_upper_micro_usd is not None
        and known > reserved_upper_micro_usd,
    )


def conservative_cost_bound(
    route: RouteConfig,
    body: Mapping[str, Any],
    *,
    qualified_contracts: frozenset[str] = frozenset(),
) -> CostEvaluation:
    pricing, bounds = route.pricing, route.model_bounds
    unknown = CostEvaluation(
        tariff_revision=pricing.revision if pricing else None,
        provider_contract=bounds.provider_contract if bounds else None,
    )
    if pricing is None or bounds is None or bounds.provider_contract not in qualified_contracts:
        return unknown
    rates = (
        pricing.input_usd_per_million,
        pricing.cache_read_usd_per_million,
        pricing.cache_create_usd_per_million,
    )
    if any(rate is None for rate in rates) or _unbounded_shape(body):
        return unknown
    limits = [
        body[key]
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens")
        if key in body
    ]
    generation = body.get("generationConfig")
    if isinstance(generation, dict) and "maxOutputTokens" in generation:
        limits.append(generation["maxOutputTokens"])
    if len(limits) > 1:
        return unknown
    output = limits[0] if limits else bounds.default_max_output_tokens
    if type(output) is not int or output < 1 or output > bounds.max_output_tokens:
        return unknown
    with localcontext() as context:
        context.prec = 80
        cost = bounds.max_input_tokens * max(Decimal(rate) for rate in rates if rate is not None)
        cost += output * Decimal(pricing.output_usd_per_million)
        maximum = int(cost.to_integral_value(rounding=ROUND_CEILING))
    return replace(
        unknown, basis="configured_tariff", reserved_upper_micro_usd=maximum, qualified_bound=True
    )


def _unbounded_shape(body: Mapping[str, Any]) -> bool:
    # Explicitly narrow to text/function tools; no hosted tools, media, or n-way output.
    if body.get("n", 1) != 1 or body.get("candidateCount", 1) != 1:
        return True
    for key, value in body.items():
        if key in {"audio", "image", "images", "video", "inlineData", "fileData", "modalities"}:
            return True
        if key == "tools" and (
            not isinstance(value, list)
            or any(not isinstance(tool, dict) or tool.get("type") != "function" for tool in value)
        ):
            return True
        if (
            key == "type"
            and isinstance(value, str)
            and any(word in value for word in ("image", "audio", "video", "file"))
        ):
            return True
        if isinstance(value, dict) and _unbounded_shape(value):
            return True
        if isinstance(value, list) and any(
            isinstance(item, dict) and _unbounded_shape(item) for item in value
        ):
            return True
    return False


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def normalize_usage(
    protocol: str,
    payload: Mapping[str, Any],
    *,
    previous: UsageObservation | None = None,
) -> UsageObservation:
    root = payload.get("response", payload.get("message", payload))
    if not isinstance(root, dict):
        return previous or UsageObservation()
    raw = root.get(
        "usageMetadata" if protocol in {"gemini-generate", "vertex-generate"} else "usage"
    )
    if not isinstance(raw, dict):
        return previous or UsageObservation()
    cache_included = False
    cache_reported = created_reported = False
    created: int | None
    if protocol in {"openai-chat", "openai-responses"}:
        input_key, output_key = (
            ("prompt_tokens", "completion_tokens")
            if protocol == "openai-chat"
            else ("input_tokens", "output_tokens")
        )
        details = raw.get(input_key.replace("tokens", "tokens_details"), {})
        cache_reported = isinstance(details, dict) and "cached_tokens" in details
        cache = _count(details.get("cached_tokens", 0)) if isinstance(details, dict) else None
        created, cache_included = 0, True
    elif protocol == "anthropic-messages":
        input_key, output_key = "input_tokens", "output_tokens"
        cache = _count(raw.get("cache_read_input_tokens", 0))
        created = _count(raw.get("cache_creation_input_tokens", 0))
        cache_reported = "cache_read_input_tokens" in raw
        created_reported = "cache_creation_input_tokens" in raw
    elif protocol in {"gemini-generate", "vertex-generate"}:
        input_key, output_key = "promptTokenCount", "candidatesTokenCount"
        cache, created, cache_included = _count(raw.get("cachedContentTokenCount", 0)), 0, True
        cache_reported = "cachedContentTokenCount" in raw
    elif protocol == "bedrock-invoke":
        input_key, output_key = "inputTokens", "outputTokens"
        cache, created = (
            _count(raw.get("cacheReadInputTokens", 0)),
            _count(raw.get("cacheWriteInputTokens", 0)),
        )
        cache_reported = "cacheReadInputTokens" in raw
        created_reported = "cacheWriteInputTokens" in raw
    else:
        return previous or UsageObservation()
    input_count, output_count = _count(raw.get(input_key)), _count(raw.get(output_key))
    input_total = input_count if cache_included else None
    if protocol in {"gemini-generate", "vertex-generate"} and output_count is not None:
        thoughts = _count(raw.get("thoughtsTokenCount", 0))
        output_count = output_count + thoughts if thoughts is not None else None
    if cache_included and input_count is not None:
        input_count = input_count - cache if cache is not None and cache <= input_count else None
    # Preserve explicit cache-only refinements; absent input cannot assert
    # omitted cache dimensions are zero or validate a contradictory partition.
    if input_count is None:
        cache = cache if cache_reported and input_total is None else None
        created = created if created_reported and input_total is None else None
    result = UsageObservation(
        input_count,
        output_count,
        cache,
        created,
        provenance=protocol,
        input_total_tokens=input_total,
    )
    if previous is not None:
        updates: dict[str, Any] = {}
        for field in fields(result):
            if field.name.endswith("_tokens"):
                known = [
                    value
                    for value in (getattr(previous, field.name), getattr(result, field.name))
                    if value is not None
                ]
                updates[field.name] = max(known) if known else None
        result = replace(result, **updates)
    if cache_included:
        total, cached = result.input_total_tokens, result.cache_read_tokens
        result = replace(
            result,
            input_tokens=total - cached
            if total is not None and cached is not None and cached <= total
            else None,
        )
    counts = (
        result.input_tokens,
        result.output_tokens,
        result.cache_read_tokens,
        result.cache_create_tokens,
    )
    return replace(
        result,
        availability="complete"
        if all(c is not None for c in counts)
        else "partial"
        if any(c is not None for c in counts)
        else "unknown",
    )

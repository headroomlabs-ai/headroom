"""Pricing registry for LLM model cost estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any


@dataclass(frozen=True)
class ModelPricing:
    """Immutable pricing information for a specific model.

    All prices are in USD per 1 million tokens.
    """

    model: str
    input_per_1m: float
    output_per_1m: float
    provider: str = ""
    cached_input_per_1m: float | None = None
    batch_input_per_1m: float | None = None
    batch_output_per_1m: float | None = None
    context_window: int | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ModelPricing:
        """Create a ModelPricing instance from a dictionary."""
        return cls(
            model=str(data.get("model", "")),
            input_per_1m=float(data.get("input_per_1m", 0.0)),
            output_per_1m=float(data.get("output_per_1m", 0.0)),
            provider=str(data.get("provider", "")),
            cached_input_per_1m=float(data["cached_input_per_1m"])
            if data.get("cached_input_per_1m") is not None
            else None,
            batch_input_per_1m=float(data["batch_input_per_1m"])
            if data.get("batch_input_per_1m") is not None
            else None,
            batch_output_per_1m=float(data["batch_output_per_1m"])
            if data.get("batch_output_per_1m") is not None
            else None,
            context_window=int(data["context_window"])
            if data.get("context_window") is not None
            else None,
            notes=str(data["notes"]) if data.get("notes") is not None else None,
        )


@dataclass
class CostEstimate:
    """Result of a cost estimation calculation."""

    cost_usd: float
    breakdown: dict = field(default_factory=dict)
    pricing_date: date | None = None
    is_stale: bool = False
    warning: str | None = None


class PricingRegistry:
    """Registry of model pricing information with cost estimation capabilities."""

    # Pricing is considered stale after this many days
    STALENESS_THRESHOLD_DAYS = 30

    def __init__(
        self,
        last_updated: date | None = None,
        source_url: str | None = None,
        prices: dict[str, ModelPricing] | None = None,
    ):
        """Initialize the pricing registry.

        Args:
            last_updated: Date when pricing information was last verified (defaults to today).
            source_url: URL to the official pricing page.
            prices: Dictionary mapping model names to ModelPricing objects.
        """
        self.last_updated = last_updated or date.today()
        self.source_url = source_url
        self.prices: dict[str, ModelPricing] = prices or {}

    def register_price(self, pricing: ModelPricing) -> None:
        """Register or override pricing for a model."""
        self.prices[pricing.model] = pricing

    def get_price(self, model: str) -> ModelPricing | None:
        """Get pricing for a specific model.

        Supports direct lookup, case-insensitive lookup, and provider prefix stripping
        (e.g., 'azure/gpt-4o' -> 'gpt-4o').

        Args:
            model: The model name/identifier.

        Returns:
            ModelPricing if found, None otherwise.
        """
        if not model:
            return None

        # 1. Exact match
        if model in self.prices:
            return self.prices[model]

        # 2. Case-insensitive match
        model_lower = model.lower()
        for k, v in self.prices.items():
            if k.lower() == model_lower:
                return v

        # 3. Strip provider prefix (e.g. "azure/gpt-4" -> "gpt-4", "bedrock/us.anthropic..." -> "claude...")
        if "/" in model:
            stripped = model.split("/", 1)[1]
            price = self.get_price(stripped)
            if price:
                return price

        return None

    @classmethod
    def from_dict(cls, custom_pricing_dict: dict[str, Any]) -> PricingRegistry:
        """Create a PricingRegistry from a dictionary of model pricing configs."""
        registry = cls()
        for key, val in custom_pricing_dict.items():
            if isinstance(val, ModelPricing):
                registry.register_price(val)
            elif isinstance(val, dict):
                data = dict(val)
                if "model" not in data:
                    data["model"] = key
                registry.register_price(ModelPricing.from_dict(data))
        return registry

    def is_stale(self) -> bool:
        """Check if pricing information is potentially outdated.

        Returns:
            True if pricing data is older than STALENESS_THRESHOLD_DAYS.
        """
        age = date.today() - self.last_updated
        return age > timedelta(days=self.STALENESS_THRESHOLD_DAYS)

    def staleness_warning(self) -> str | None:
        """Get a warning message if pricing is stale.

        Returns:
            Warning message if stale, None otherwise.
        """
        if not self.is_stale():
            return None

        age_days = (date.today() - self.last_updated).days
        msg = f"Pricing data is {age_days} days old (last updated: {self.last_updated})."
        if self.source_url:
            msg += f" Please verify at: {self.source_url}"
        return msg

    def estimate_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        batch_input_tokens: int = 0,
        batch_output_tokens: int = 0,
    ) -> CostEstimate:
        """Estimate the cost for a given token usage.

        Args:
            model: The model name/identifier.
            input_tokens: Number of regular input tokens.
            output_tokens: Number of regular output tokens.
            cached_input_tokens: Number of cached input tokens.
            batch_input_tokens: Number of batch API input tokens.
            batch_output_tokens: Number of batch API output tokens.

        Returns:
            CostEstimate with calculated cost and breakdown.

        Raises:
            ValueError: If model is not found in registry.
        """
        pricing = self.get_price(model)
        if pricing is None:
            raise ValueError(f"Model '{model}' not found in registry")

        breakdown = {}
        total_cost = 0.0

        # Regular input tokens
        if input_tokens > 0:
            input_cost = (input_tokens / 1_000_000) * pricing.input_per_1m
            breakdown["input"] = {
                "tokens": input_tokens,
                "rate_per_1m": pricing.input_per_1m,
                "cost_usd": input_cost,
            }
            total_cost += input_cost

        # Regular output tokens
        if output_tokens > 0:
            output_cost = (output_tokens / 1_000_000) * pricing.output_per_1m
            breakdown["output"] = {
                "tokens": output_tokens,
                "rate_per_1m": pricing.output_per_1m,
                "cost_usd": output_cost,
            }
            total_cost += output_cost

        # Cached input tokens
        if cached_input_tokens > 0:
            if pricing.cached_input_per_1m is None:
                raise ValueError(f"Model '{model}' does not have cached input pricing")
            cached_cost = (cached_input_tokens / 1_000_000) * pricing.cached_input_per_1m
            breakdown["cached_input"] = {
                "tokens": cached_input_tokens,
                "rate_per_1m": pricing.cached_input_per_1m,
                "cost_usd": cached_cost,
            }
            total_cost += cached_cost

        # Batch input tokens
        if batch_input_tokens > 0:
            if pricing.batch_input_per_1m is None:
                raise ValueError(f"Model '{model}' does not have batch input pricing")
            batch_input_cost = (batch_input_tokens / 1_000_000) * pricing.batch_input_per_1m
            breakdown["batch_input"] = {
                "tokens": batch_input_tokens,
                "rate_per_1m": pricing.batch_input_per_1m,
                "cost_usd": batch_input_cost,
            }
            total_cost += batch_input_cost

        # Batch output tokens
        if batch_output_tokens > 0:
            if pricing.batch_output_per_1m is None:
                raise ValueError(f"Model '{model}' does not have batch output pricing")
            batch_output_cost = (batch_output_tokens / 1_000_000) * pricing.batch_output_per_1m
            breakdown["batch_output"] = {
                "tokens": batch_output_tokens,
                "rate_per_1m": pricing.batch_output_per_1m,
                "cost_usd": batch_output_cost,
            }
            total_cost += batch_output_cost

        return CostEstimate(
            cost_usd=total_cost,
            breakdown=breakdown,
            pricing_date=self.last_updated,
            is_stale=self.is_stale(),
            warning=self.staleness_warning(),
        )

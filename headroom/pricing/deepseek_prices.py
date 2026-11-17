"""DeepSeek model pricing information.

Every row here is the vendor's OFF-PEAK tier: these flat consumers have no
request instant, so they take the cheaper published tier, and per-request
costing applies the Beijing peak window through
:func:`headroom.pricing.deepseek_tiers.rates_for`.
"""

from .deepseek_tiers import (
    LAST_UPDATED as LAST_UPDATED,
)
from .deepseek_tiers import OFF_PEAK_RATES_PER_1M
from .deepseek_tiers import (
    SOURCE_URL as SOURCE_URL,
)
from .registry import ModelPricing, PricingRegistry

# ``LAST_UPDATED`` and ``SOURCE_URL`` live with the rates they describe, in
# ``deepseek_tiers``: this table is derived from that one, so the provenance date
# must move with the numbers rather than tracking a copy. They are re-exported
# here (the ``X as X`` form) so the existing
# ``from headroom.pricing.deepseek_prices import LAST_UPDATED`` path keeps working.

# All prices are in USD per 1 million tokens, off-peak.
_OFF_PEAK_NOTE = "off-peak; peak is 2x (Beijing 09:00-12:00, 14:00-18:00)"


def _flat_row(model_id: str, canonical: str, model_version: str) -> ModelPricing:
    """Build one off-peak row for ``model_id`` from the tier table."""
    cache_hit, miss, out = OFF_PEAK_RATES_PER_1M[canonical]
    return ModelPricing(
        model=model_id,
        provider="deepseek",
        input_per_1m=miss,
        output_per_1m=out,
        cached_input_per_1m=cache_hit,
        context_window=1_000_000,
        notes=f"{model_version} - {_OFF_PEAK_NOTE}",
    )


DEEPSEEK_PRICES: dict[str, ModelPricing] = {
    "deepseek-flash": _flat_row("deepseek-flash", "deepseek-flash", "DeepSeek V4.1-Flash"),
    "deepseek-v4-flash": _flat_row(
        "deepseek-v4-flash", "deepseek-flash", "Retired id, served as DeepSeek V4.1-Flash"
    ),
    "deepseek-v4-flash-vision-exp": _flat_row(
        "deepseek-v4-flash-vision-exp",
        "deepseek-flash",
        "Retired id, served as DeepSeek V4.1-Flash",
    ),
    "deepseek-v4-pro": _flat_row("deepseek-v4-pro", "deepseek-v4-pro", "DeepSeek V4-Pro-0813"),
}


def get_deepseek_registry() -> PricingRegistry:
    """Create and return a DeepSeek pricing registry.

    Returns:
        PricingRegistry configured with DeepSeek model prices.
    """
    return PricingRegistry(
        last_updated=LAST_UPDATED,
        source_url=SOURCE_URL,
        prices=DEEPSEEK_PRICES.copy(),
    )

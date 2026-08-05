"""LiteLLM-based pricing for model cost estimation.

Uses LiteLLM's community-maintained model cost database instead of
hardcoded values. This provides up-to-date pricing for 100+ models.

See: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from headroom.pricing.litellm_model_resolution import (
    pricing_lookup_candidates,
    resolve_litellm_model_name,
)

# litellm calls `dotenv.load_dotenv()` during its own import, which loads
# the project `.env` into `os.environ`. We don't want that side effect —
# importing a pricing helper should not silently leak API keys into the
# process. Snapshot `os.environ` around the import and undo any keys
# litellm added. The module itself is fully imported and cached in
# `sys.modules`; subsequent `import litellm` calls hit the cache and
# don't re-run the dotenv side effect.
try:
    import os as _os

    _env_snapshot = set(_os.environ)
    import litellm

    for _leaked_key in set(_os.environ) - _env_snapshot:
        del _os.environ[_leaked_key]
    del _env_snapshot, _os

    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None  # type: ignore[assignment]
    LITELLM_AVAILABLE = False

_resolved_model_cache: dict[str, str] = {}

logger = logging.getLogger("headroom.pricing")

# --- Gateway model-name resolution ---------------------------------------
# When Headroom sits behind a gateway (Kong, LiteLLM, ...) that aliases model
# names, the raw client name it sees (e.g. "claude-opus") is not a priced key
# in litellm.model_cost, so dollar savings read $0. HEADROOM_MODEL_ALIAS_MAP is
# an optional, gateway-agnostic, fail-soft static JSON map {client_name: target}
# that reduces that name to a priced model_cost key (trying the target as-is and
# with a bedrock/ or vertex_ai/ provider prefix stripped). Unset -> behavior is
# identical to today's bare-prefix resolution; pricing never breaks.
_GATEWAY_PROVIDER_PREFIXES = ("bedrock/", "vertex_ai/")


def _static_alias_map() -> dict[str, str]:
    raw = os.environ.get("HEADROOM_MODEL_ALIAS_MAP", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        logger.debug("invalid HEADROOM_MODEL_ALIAS_MAP JSON", exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if k and v}


def _reduce_to_priced_key(target: str) -> str | None:
    """Reduce a gateway target to a priced litellm.model_cost key, or None."""
    if not LITELLM_AVAILABLE or litellm is None:
        return None
    candidates = [target]
    for prefix in _GATEWAY_PROVIDER_PREFIXES:
        if target.startswith(prefix):
            candidates.append(target[len(prefix) :])
    for candidate in candidates:
        info = litellm.model_cost.get(candidate)
        if info and info.get("input_cost_per_token") is not None:
            return candidate
    return None


def resolve_litellm_model(model: str) -> str:
    """Resolve model name to one LiteLLM recognizes, adding provider prefix if needed.
    Results are cached per model name to avoid blocking the event loop
    with repeated synchronous litellm lookups.

    When HEADROOM_MODEL_ALIAS_MAP is configured, a raw client name / group alias
    is first reduced to a priced model_cost key; otherwise this falls through to
    the bare-prefix rules. Shared by the live (cost.py) and persisted
    (savings_tracker) pricing paths so both figures price identically.
    """
    if model in _resolved_model_cache:
        return _resolved_model_cache[model]
    priced: str | None = None
    alias = _static_alias_map()
    if alias:
        priced = _reduce_to_priced_key(alias.get(model, model))
    resolved = priced if priced is not None else _resolve_litellm_model_uncached(model)
    _resolved_model_cache[model] = resolved
    return resolved


def _resolve_litellm_model_uncached(model: str) -> str:
    """Uncached resolution — called once per unique model name."""
    if not LITELLM_AVAILABLE:
        return model

    def is_known_model(candidate: str) -> bool:
        try:
            litellm.cost_per_token(model=candidate, prompt_tokens=1, completion_tokens=0)
            return True
        except Exception:
            return False

    return resolve_litellm_model_name(model, is_known_model)


def _register_minimax_pricing() -> None:
    """Pre-register MiniMax-M3 in litellm.model_cost from `minimax/MiniMax-M3`.

    The proxy receives the bare model name `MiniMax-M3` from Claude Code.
    LiteLLM's community pricing database only stores it under the
    `minimax/MiniMax-M3` key. The resolver's `minimax-` prefix rule
    handles the lookup; this pre-registration is a safety net so
    `estimate_cost()` succeeds even if (a) the resolver cache is cold,
    or (b) LiteLLM drops the prefixed entry in a future release.
    Pricing mirrors the upstream DB (input $0.60/M, output $2.40/M,
    cache read $0.12/M as of 2026-06). Re-check after LiteLLM updates.
    """
    if not LITELLM_AVAILABLE:
        return
    source_key = "minimax/MiniMax-M3"
    if source_key not in litellm.model_cost:
        return
    if "MiniMax-M3" not in litellm.model_cost:
        litellm.model_cost["MiniMax-M3"] = dict(litellm.model_cost[source_key])


_register_minimax_pricing()


@dataclass
class LiteLLMModelPricing:
    """Pricing information from LiteLLM's database.

    All costs are in USD per 1 million tokens.
    """

    model: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_vision: bool = False
    supports_function_calling: bool = False


def get_litellm_model_cost() -> dict[str, Any]:
    """Get LiteLLM's full model cost dictionary.

    Returns:
        Dictionary mapping model names to their pricing/capability info.
        Empty dict if litellm is not installed.
    """
    if not LITELLM_AVAILABLE:
        return {}
    return litellm.model_cost  # type: ignore[no-any-return]


def get_model_pricing(model: str) -> LiteLLMModelPricing | None:
    """Get pricing for a model from LiteLLM's database.

    Args:
        model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022').

    Returns:
        LiteLLMModelPricing if found, None if not found or litellm not installed.
    """
    if not LITELLM_AVAILABLE:
        return None
    cost_data = litellm.model_cost

    info = None
    for candidate in pricing_lookup_candidates(model):
        info = cost_data.get(candidate)
        if info is not None:
            break

    if info is None:
        return None

    # LiteLLM stores cost per token, convert to per 1M
    input_per_token = info.get("input_cost_per_token", 0) or 0
    output_per_token = info.get("output_cost_per_token", 0) or 0

    return LiteLLMModelPricing(
        model=model,
        input_cost_per_1m=input_per_token * 1_000_000,
        output_cost_per_1m=output_per_token * 1_000_000,
        max_tokens=info.get("max_tokens"),
        max_input_tokens=info.get("max_input_tokens"),
        max_output_tokens=info.get("max_output_tokens"),
        supports_vision=info.get("supports_vision", False),
        supports_function_calling=info.get("supports_function_calling", False),
    )


def pricing_per_1m(model: str) -> tuple[float, float] | None:
    """``(input, output)`` USD per 1M tokens from LiteLLM, or ``None``.

    The tuple shape providers already use for their own tables, so a provider can
    prefer this over a hand-maintained copy with a single call. ``None`` means
    "LiteLLM can't answer" — either it isn't installed (the dependency is gated
    ``python_version < '3.14'``) or it doesn't know the model — which is the
    provider's cue to fall back.

    A found-but-zero price is returned as ``0.0`` rather than treated as missing:
    some models genuinely are free, and ``savings_ledger`` already made this call
    (see its note on "a legitimate 0.0 for genuinely free (0-priced) models").
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None
    # LiteLLM stores cost per token, so the x1e6 conversion leaves float noise
    # ($0.4/1M arrives as 0.39999999999999997). Round at this boundary: 6 places
    # is finer than any published rate and keeps the value printable.
    return (round(pricing.input_cost_per_1m, 6), round(pricing.output_cost_per_1m, 6))


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float | None:
    """Estimate cost for a model using LiteLLM's pricing.

    Args:
        model: Model name.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD, or None if model not found.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m
    return input_cost + output_cost


def list_available_models() -> list[str]:
    """List all models with pricing info in LiteLLM's database.

    Returns:
        List of model names. Empty list if litellm not installed.
    """
    if not LITELLM_AVAILABLE:
        return []
    return list(litellm.model_cost.keys())


# ============================================================
# DeepSeek V4 pricing injection
# ============================================================
# Vendored LiteLLM JSON predates DeepSeek V4 models. Inject pricing at
# import time so the primary cost-per-token path resolves them. Once
# upstream litellm adds these entries, injection becomes a no-op.
# ============================================================

_DEEPSEEK_V4_PRICING: dict[str, dict[str, float | str | int]] = {
    "deepseek-v4-flash": {
        "input_cost_per_token": 0.14 / 1_000_000,
        "output_cost_per_token": 0.28 / 1_000_000,
        "cache_read_input_token_cost": 0.0028 / 1_000_000,
        "input_cost_per_token_cache_hit": 0.0028 / 1_000_000,
        "litellm_provider": "deepseek",
        "max_tokens": 384_000,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 384_000,
    },
    "deepseek-v4-pro": {
        "input_cost_per_token": 0.435 / 1_000_000,
        "output_cost_per_token": 0.87 / 1_000_000,
        "cache_read_input_token_cost": 0.003625 / 1_000_000,
        "input_cost_per_token_cache_hit": 0.003625 / 1_000_000,
        "litellm_provider": "deepseek",
        "max_tokens": 384_000,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 384_000,
    },
}


def _inject_deepseek_pricing() -> None:
    """Inject DeepSeek V4 pricing into litellm's model_cost dict.

    Only injects entries not already present, so upstream litellm additions
    (once available) take precedence. Both bare and provider-prefixed keys
    are added so resolve_litellm_model() catches them via its deepseek/
    prefix loop.
    """
    if not LITELLM_AVAILABLE:
        return
    for model_name, pricing in _DEEPSEEK_V4_PRICING.items():
        if model_name not in litellm.model_cost:
            litellm.model_cost[model_name] = pricing
        prefixed = f"deepseek/{model_name}"
        if prefixed not in litellm.model_cost:
            litellm.model_cost[prefixed] = pricing


_inject_deepseek_pricing()

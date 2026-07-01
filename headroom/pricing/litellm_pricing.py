"""LiteLLM-based pricing for model cost estimation.

Uses LiteLLM's community-maintained model cost database instead of
hardcoded values. This provides up-to-date pricing for 100+ models.

See: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

# Aliases for models removed from LiteLLM's cost database (retired/renamed).
# Maps old model name -> current LiteLLM key that has equivalent pricing.
_MODEL_ALIASES: dict[str, str] = {
    # Claude 3.5 Sonnet retired Feb 2026, pricing same as claude-sonnet-4-20250514
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4-20250514",
    # Claude 3 Sonnet retired
    "claude-3-sonnet-20240229": "claude-3-haiku-20240307",
}

_resolved_model_cache: dict[str, str] = {}


def resolve_litellm_model(model: str) -> str:
    """Resolve model name to one LiteLLM recognizes, adding provider prefix if needed.
    Results are cached per model name to avoid blocking the event loop
    with repeated synchronous litellm lookups.
    """
    if model in _resolved_model_cache:
        return _resolved_model_cache[model]
    resolved = _resolve_litellm_model_uncached(model)
    _resolved_model_cache[model] = resolved
    return resolved


def _resolve_litellm_model_uncached(model: str) -> str:
    """Uncached resolution — called once per unique model name."""
    if not LITELLM_AVAILABLE:
        return model
    # Try as-is first
    try:
        litellm.cost_per_token(model=model, prompt_tokens=1, completion_tokens=0)
        return model
    except Exception:
        pass
    # Try with provider prefix
    prefixes = {
        "claude-": "anthropic/",
        "gpt-": "openai/",
        "o1-": "openai/",
        "o3-": "openai/",
        "o4-": "openai/",
        "gemini-": "google/",
        "minimax-": "minimax/",
        "deepseek-": "deepseek/",
    }
    # Case-insensitive prefix match: MiniMax uses mixed-case model
    # names like "MiniMax-M3" (capital M's); we shouldn't require the
    # user to lower-case their config to match.
    model_lower = model.lower()
    for pattern, prefix in prefixes.items():
        if model_lower.startswith(pattern):
            prefixed = f"{prefix}{model}"
            try:
                litellm.cost_per_token(model=prefixed, prompt_tokens=1, completion_tokens=0)
                return prefixed
            except Exception:
                break
    return model


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

    # Try exact match first
    info = cost_data.get(model)

    # Try common provider prefixes if not found
    if info is None:
        for prefix in ["openai/", "anthropic/", "google/", "mistral/", "deepseek/"]:
            if f"{prefix}{model}" in cost_data:
                info = cost_data[f"{prefix}{model}"]
                break

    # Try retired/renamed model aliases (LiteLLM removes old model keys over time)
    if info is None:
        alias = _MODEL_ALIASES.get(model)
        if alias:
            info = cost_data.get(alias)

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

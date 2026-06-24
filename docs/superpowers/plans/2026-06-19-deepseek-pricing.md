# DeepSeek V4 Pricing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `deepseek-v4-flash` and `deepseek-v4-pro` pricing to Headroom so cost estimation works when routing through `--anthropic-api-url https://api.deepseek.com/anthropic`.

**Architecture:** Four independent layers — (1) a new pricing data module following the `anthropic_prices.py` pattern, (2) runtime injection into `litellm.model_cost` so the primary cost-per-token path resolves DeepSeek V4, (3) a fallback in the Anthropic provider's `_get_pricing()` when LiteLLM is unavailable, and (4) vendored JSON entries for Rust-side context window lookups.

**Tech Stack:** Python 3.12+, LiteLLM, Rust (vendored JSON), pytest

## Global Constraints

- All prices in USD per 1 million tokens (`ModelPricing` dataclass convention)
- LiteLLM injection must not overwrite upstream entries if litellm already has DeepSeek V4
- Provider field must be `"deepseek"` for all DeepSeek models
- Follow exact patterns from `anthropic_prices.py` / `openai_prices.py` / `test_anthropic.py`
- Vendored JSON entries at `crates/headroom-proxy/data/model_prices_and_context_window.json`

---

### Task 1: DeepSeek Pricing Data Module

**Files:**
- Create: `headroom/pricing/deepseek_prices.py`
- Modify: `headroom/pricing/__init__.py`
- Test: `tests/test_providers/test_deepseek.py` (first test class)

**Interfaces:**
- Consumes: `ModelPricing`, `PricingRegistry` from `headroom.pricing.registry`
- Produces: `DEEPSEEK_PRICES: dict[str, ModelPricing]`, `get_deepseek_registry() -> PricingRegistry`, exported via `headroom.pricing`

- [ ] **Step 1: Write the failing tests for the pricing data module**

Create `tests/test_providers/test_deepseek.py`:

```python
"""Tests for DeepSeek model pricing and cost estimation."""

from headroom.pricing.deepseek_prices import (
    DEEPSEEK_PRICES,
    get_deepseek_registry,
)
from headroom.pricing.registry import PricingRegistry


class TestDeepSeekPricingModule:
    """Tests for the DeepSeek pricing data module."""

    def test_deepseek_pricing_contains_v4_models(self):
        assert "deepseek-v4-flash" in DEEPSEEK_PRICES
        assert "deepseek-v4-pro" in DEEPSEEK_PRICES
        assert len(DEEPSEEK_PRICES) == 2

    def test_deepseek_v4_flash_pricing(self):
        pricing = DEEPSEEK_PRICES["deepseek-v4-flash"]
        assert pricing.input_per_1m == 0.14
        assert pricing.output_per_1m == 0.28
        assert pricing.cached_input_per_1m == 0.0028
        assert pricing.context_window == 1_000_000
        assert pricing.provider == "deepseek"
        assert pricing.notes is not None

    def test_deepseek_v4_pro_pricing(self):
        pricing = DEEPSEEK_PRICES["deepseek-v4-pro"]
        assert pricing.input_per_1m == 0.435
        assert pricing.output_per_1m == 0.87
        assert pricing.cached_input_per_1m == 0.003625
        assert pricing.context_window == 1_000_000
        assert pricing.provider == "deepseek"
        assert pricing.notes is not None

    def test_get_deepseek_registry(self):
        registry = get_deepseek_registry()
        assert isinstance(registry, PricingRegistry)
        assert registry.get_price("deepseek-v4-flash") is not None
        assert registry.get_price("deepseek-v4-pro") is not None
        assert registry.get_price("nonexistent") is None

    def test_registry_staleness_and_source_url(self):
        registry = get_deepseek_registry()
        assert registry.source_url == "https://api-docs.deepseek.com/quick_start/pricing"
        assert not registry.is_stale()

    def test_deepseek_registry_estimate_cost(self):
        registry = get_deepseek_registry()
        cost = registry.estimate_cost("deepseek-v4-flash", input_tokens=1_000_000)
        assert cost.cost_usd == 0.14
        assert "input" in cost.breakdown
        assert cost.pricing_date is not None

    def test_deepseek_registry_estimate_cost_with_cached(self):
        registry = get_deepseek_registry()
        cost = registry.estimate_cost(
            "deepseek-v4-flash",
            input_tokens=1_000_000,
            cached_input_tokens=1_000_000,
        )
        assert cost.cost_usd == 0.14 + 0.0028
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd D:/headroom && python -m pytest tests/test_providers/test_deepseek.py::TestDeepSeekPricingModule -v
```

Expected: ModuleNotFoundError or ImportError — `deepseek_prices` doesn't exist yet.

- [ ] **Step 3: Create the pricing data module**

Create `headroom/pricing/deepseek_prices.py`:

```python
"""DeepSeek model pricing information."""

from datetime import date

from .registry import ModelPricing, PricingRegistry

# Last verified date for pricing information
LAST_UPDATED = date(2026, 6, 19)

# Official pricing page
SOURCE_URL = "https://api-docs.deepseek.com/quick_start/pricing"

# All prices are in USD per 1 million tokens
DEEPSEEK_PRICES: dict[str, ModelPricing] = {
    "deepseek-v4-flash": ModelPricing(
        model="deepseek-v4-flash",
        provider="deepseek",
        input_per_1m=0.14,
        output_per_1m=0.28,
        cached_input_per_1m=0.0028,
        context_window=1_000_000,
        notes="DeepSeek V4 Flash - 13B active params; non-thinking + thinking modes",
    ),
    "deepseek-v4-pro": ModelPricing(
        model="deepseek-v4-pro",
        provider="deepseek",
        input_per_1m=0.435,
        output_per_1m=0.87,
        cached_input_per_1m=0.003625,
        context_window=1_000_000,
        notes="DeepSeek V4 Pro - 49B active params",
    ),
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
```

- [ ] **Step 4: Wire into `__init__.py`**

Edit `headroom/pricing/__init__.py` — add these imports after the existing OpenAI imports:

```python
from .deepseek_prices import (
    DEEPSEEK_PRICES,
    get_deepseek_registry,
)
from .deepseek_prices import (
    LAST_UPDATED as DEEPSEEK_LAST_UPDATED,
)
```

And add to `__all__`:

```python
    # DeepSeek
    "DEEPSEEK_LAST_UPDATED",
    "DEEPSEEK_PRICES",
    "get_deepseek_registry",
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd D:/headroom && python -m pytest tests/test_providers/test_deepseek.py::TestDeepSeekPricingModule -v
```

Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add headroom/pricing/deepseek_prices.py headroom/pricing/__init__.py tests/test_providers/test_deepseek.py
git commit -m "feat(pricing): add DeepSeek V4 pricing data module

Add deepseek-v4-flash and deepseek-v4-pro ModelPricing entries and
registry factory, following the pattern of anthropic_prices.py.
Wire into pricing/__init__.py exports.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: LiteLLM Runtime Injection

**Files:**
- Modify: `headroom/pricing/litellm_pricing.py` (add injection after `LITELLM_AVAILABLE` block)
- Test: `tests/test_providers/test_deepseek.py` (add `TestDeepSeekLiteLLMInjection` class)

**Interfaces:**
- Consumes: `litellm.model_cost` dict (available after import)
- Produces: `"deepseek-v4-flash"`, `"deepseek/deepseek-v4-flash"`, `"deepseek-v4-pro"`, `"deepseek/deepseek-v4-pro"` keys in `litellm.model_cost`
- Depends on: DEEPSEEK_V4_PRICING constant defined within litellm_pricing.py

- [ ] **Step 1: Write failing injection tests**

Append to `tests/test_providers/test_deepseek.py`:

```python
import pytest


class TestDeepSeekLiteLLMInjection:
    """Tests for DeepSeek V4 pricing injection into litellm."""

    def test_deepseek_v4_models_in_litellm_model_cost(self):
        from headroom.pricing.litellm_pricing import litellm, LITELLM_AVAILABLE
        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        assert "deepseek-v4-flash" in litellm.model_cost
        assert "deepseek-v4-pro" in litellm.model_cost

    def test_deepseek_v4_prefixed_models_in_litellm_model_cost(self):
        from headroom.pricing.litellm_pricing import litellm, LITELLM_AVAILABLE
        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        assert "deepseek/deepseek-v4-flash" in litellm.model_cost
        assert "deepseek/deepseek-v4-pro" in litellm.model_cost

    def test_deepseek_v4_flash_litellm_pricing(self):
        from headroom.pricing.litellm_pricing import litellm, LITELLM_AVAILABLE
        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        flash = litellm.model_cost["deepseek-v4-flash"]
        assert flash["input_cost_per_token"] == 0.14 / 1_000_000
        assert flash["output_cost_per_token"] == 0.28 / 1_000_000
        assert flash["cache_read_input_token_cost"] == 0.0028 / 1_000_000
        assert flash["litellm_provider"] == "deepseek"

    def test_deepseek_v4_pro_litellm_pricing(self):
        from headroom.pricing.litellm_pricing import litellm, LITELLM_AVAILABLE
        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        pro = litellm.model_cost["deepseek-v4-pro"]
        assert pro["input_cost_per_token"] == 0.435 / 1_000_000
        assert pro["output_cost_per_token"] == 0.87 / 1_000_000
        assert pro["cache_read_input_token_cost"] == 0.003625 / 1_000_000
        assert pro["litellm_provider"] == "deepseek"

    def test_cost_per_token_resolves_deepseek_v4_flash(self):
        from headroom.pricing.litellm_pricing import litellm, LITELLM_AVAILABLE
        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        input_cost, output_cost = litellm.cost_per_token(
            model="deepseek-v4-flash",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert input_cost == pytest.approx(0.14, rel=0.01)
        assert output_cost == pytest.approx(0.28, rel=0.01)

    def test_injection_does_not_overwrite_existing_upstream_entries(self):
        """If litellm upstream already has these, our injection is a no-op."""
        from headroom.pricing.litellm_pricing import litellm, LITELLM_AVAILABLE
        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        # Force-inject with wrong value, then verify the injection guard
        litellm.model_cost["deepseek-v4-flash"] = {"input_cost_per_token": 999}
        # Reimport to trigger _inject_deepseek_pricing — but it should NOT overwrite
        import importlib
        import headroom.pricing.litellm_pricing as lp
        importlib.reload(lp)
        assert litellm.model_cost["deepseek-v4-flash"]["input_cost_per_token"] == 999
        # Reset to correct value
        litellm.model_cost["deepseek-v4-flash"] = {
            "input_cost_per_token": 0.14 / 1_000_000,
            "output_cost_per_token": 0.28 / 1_000_000,
            "cache_read_input_token_cost": 0.0028 / 1_000_000,
            "litellm_provider": "deepseek",
            "max_tokens": 384_000,
            "max_input_tokens": 1_000_000,
        }
```

- [ ] **Step 2: Run the injection tests to verify they fail**

```bash
cd D:/headroom && python -m pytest tests/test_providers/test_deepseek.py::TestDeepSeekLiteLLMInjection -v
```

Expected: Tests fail because `litellm.model_cost` doesn't have DeepSeek V4 entries yet.

- [ ] **Step 3: Add the runtime injection to `litellm_pricing.py`**

At the end of `headroom/pricing/litellm_pricing.py`, before the `_resolved_model_cache` and function definitions, add:

```python
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
        "litellm_provider": "deepseek",
        "max_tokens": 384_000,
        "max_input_tokens": 1_000_000,
    },
    "deepseek-v4-pro": {
        "input_cost_per_token": 0.435 / 1_000_000,
        "output_cost_per_token": 0.87 / 1_000_000,
        "cache_read_input_token_cost": 0.003625 / 1_000_000,
        "litellm_provider": "deepseek",
        "max_tokens": 384_000,
        "max_input_tokens": 1_000_000,
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd D:/headroom && python -m pytest tests/test_providers/test_deepseek.py::TestDeepSeekLiteLLMInjection -v
```

Expected: All 6 tests pass (or some skip if litellm is unavailable — that's acceptable).

- [ ] **Step 5: Commit**

```bash
git add headroom/pricing/litellm_pricing.py tests/test_providers/test_deepseek.py
git commit -m "feat(pricing): inject DeepSeek V4 pricing into litellm model_cost

Add runtime injection so litellm.cost_per_token() resolves
deepseek-v4-flash and deepseek-v4-pro pricing. Includes both bare
and provider-prefixed keys. No-ops if entries already exist.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Anthropic Provider DeepSeek Fallback

**Files:**
- Modify: `headroom/providers/anthropic.py` (add `_get_deepseek_pricing()` helper and call in `_get_pricing()`)
- Test: `tests/test_providers/test_deepseek.py` (add `TestDeepSeekAnthropicProviderFallback` class)

**Interfaces:**
- Consumes: `_get_pricing(self, model: str) -> dict | None` (inside `AnthropicProvider`)
- Produces: DeepSeek pricing dicts from `_get_pricing("deepseek-*")` and `estimate_cost("deepseek-*")`

- [ ] **Step 1: Write failing fallback tests**

Append to `tests/test_providers/test_deepseek.py`:

```python
class TestDeepSeekAnthropicProviderFallback:
    """Tests that Anthropic provider's _get_pricing handles DeepSeek models."""

    def test_deepseek_v4_flash_fallback(self):
        from headroom.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider()
        pricing = provider._get_pricing("deepseek-v4-flash")
        assert pricing is not None
        assert pricing["input"] == 0.14
        assert pricing["output"] == 0.28
        assert pricing["cached_input"] == 0.0028

    def test_deepseek_v4_pro_fallback(self):
        from headroom.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider()
        pricing = provider._get_pricing("deepseek-v4-pro")
        assert pricing is not None
        assert pricing["input"] == 0.435
        assert pricing["output"] == 0.87
        assert pricing["cached_input"] == 0.003625

    def test_deepseek_unknown_model_returns_none(self):
        from headroom.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider()
        pricing = provider._get_pricing("deepseek-unknown-model")
        assert pricing is None

    def test_deepseek_partial_match_v4_flash_alias(self):
        from headroom.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider()
        # Should match via partial match (flash in v4-flash)
        pricing = provider._get_pricing("deepseek-v4-flash-v1")
        assert pricing is not None

    def test_estimate_cost_deepseek_v4_flash(self):
        from headroom.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider()
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            model="deepseek-v4-flash",
        )
        assert cost is not None
        assert cost == 0.14

    def test_estimate_cost_deepseek_v4_flash_with_cache(self):
        from headroom.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider()
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            model="deepseek-v4-flash",
            cached_tokens=1_000_000,
        )
        assert cost is not None
        assert cost == 0.14 + 0.0028
```

- [ ] **Step 2: Run fallback tests to verify they fail**

```bash
cd D:/headroom && python -m pytest tests/test_providers/test_deepseek.py::TestDeepSeekAnthropicProviderFallback -v
```

Expected: Fail because `_get_pricing()` returns `None` for `deepseek-*` models.

- [ ] **Step 3: Add DeepSeek fallback to Anthropic provider**

In `headroom/providers/anthropic.py`, add a new helper function after the `_UNKNOWN_CLAUDE_DEFAULT` dict (around line 151) and modify `_get_pricing()`:

Add the helper (after `_UNKNOWN_CLAUDE_DEFAULT` at line 151):

```python
# DeepSeek fallback pricing for --anthropic-api-url deepseek routing
_DEEPSEEK_FALLBACK_PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28, "cached_input": 0.0028},
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87, "cached_input": 0.003625},
}


def _get_deepseek_pricing(model: str) -> dict[str, float] | None:
    """Get fallback pricing for a DeepSeek model.

    Used when the Anthropic provider encounters a deepseek-* model name
    (via --anthropic-api-url pointing at DeepSeek's Anthropic-compatible
    endpoint) and LiteLLM is unavailable.

    Args:
        model: The model name to look up.

    Returns:
        Pricing dict with input/output/cached_input keys, or None.
    """
    # Direct match
    if model in _DEEPSEEK_FALLBACK_PRICING:
        return cast(dict[str, float], _DEEPSEEK_FALLBACK_PRICING[model])
    # Partial match
    for known_model, prices in _DEEPSEEK_FALLBACK_PRICING.items():
        if model in known_model or known_model in model:
            return cast(dict[str, float], prices)
    return None
```

Modify `_get_pricing()` in the `AnthropicProvider` class. Add after the Claude default check (line 698):

```python
        # Default for unknown Claude models
        if model.startswith("claude"):
            return cast(dict[str, float], _UNKNOWN_CLAUDE_DEFAULT["pricing"])

        # DeepSeek model fallback (via --anthropic-api-url)
        if model.startswith("deepseek"):
            return _get_deepseek_pricing(model)

        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd D:/headroom && python -m pytest tests/test_providers/test_deepseek.py::TestDeepSeekAnthropicProviderFallback -v
```

Expected: All 6 tests pass.

- [ ] **Step 5: Run full test suite for pricing-related tests**

```bash
cd D:/headroom && python -m pytest tests/test_providers/test_anthropic.py tests/test_providers/test_deepseek.py tests/test_pricing.py tests/test_pricing_litellm.py tests/test_litellm_optional.py -v
```

Expected: All existing tests still pass, plus new tests pass.

- [ ] **Step 6: Commit**

```bash
git add headroom/providers/anthropic.py tests/test_providers/test_deepseek.py
git commit -m "feat(providers): add DeepSeek pricing fallback to Anthropic provider

When routing through --anthropic-api-url with a DeepSeek endpoint,
_get_pricing() now handles deepseek-* model names. Falls back to
hardcoded V4 pricing when LiteLLM is unavailable.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Vendored Rust JSON Entries

**Files:**
- Modify: `crates/headroom-proxy/data/model_prices_and_context_window.json`

**Interfaces:**
- Consumes: The existing DeepSeek JSON entry structure around line 12516
- Produces: `"deepseek-v4-flash"` and `"deepseek-v4-pro"` entries in the JSON

- [ ] **Step 1: Add DeepSeek V4 entries to the vendored JSON**

Open `crates/headroom-proxy/data/model_prices_and_context_window.json` and insert the following after the `"deepseek/deepseek-v3.2"` entry (after line 12516):

```json
    "deepseek-v4-flash": {
        "input_cost_per_token": 1.4e-07,
        "output_cost_per_token": 2.8e-07,
        "cache_read_input_token_cost": 2.8e-09,
        "litellm_provider": "deepseek",
        "max_input_tokens": 1000000,
        "max_output_tokens": 384000,
        "max_tokens": 384000,
        "mode": "chat",
        "supports_function_calling": true,
        "supports_native_streaming": true,
        "supports_prompt_caching": true,
        "source": "https://api-docs.deepseek.com/quick_start/pricing"
    },
    "deepseek-v4-pro": {
        "input_cost_per_token": 4.35e-07,
        "output_cost_per_token": 8.7e-07,
        "cache_read_input_token_cost": 3.625e-09,
        "litellm_provider": "deepseek",
        "max_input_tokens": 1000000,
        "max_output_tokens": 384000,
        "max_tokens": 384000,
        "mode": "chat",
        "supports_function_calling": true,
        "supports_native_streaming": true,
        "supports_prompt_caching": true,
        "source": "https://api-docs.deepseek.com/quick_start/pricing"
    },
```

Also add provider-prefixed variants after the bare entries (following the pattern where `deepseek/deepseek-v3.2` exists alongside `deepseek-v3-2-251201`):

```json
    "deepseek/deepseek-v4-flash": {
        "input_cost_per_token": 1.4e-07,
        "output_cost_per_token": 2.8e-07,
        "cache_read_input_token_cost": 2.8e-09,
        "input_cost_per_token_cache_hit": 2.8e-09,
        "litellm_provider": "deepseek",
        "max_input_tokens": 1000000,
        "max_output_tokens": 384000,
        "max_tokens": 384000,
        "mode": "chat",
        "supports_function_calling": true,
        "supports_native_streaming": true,
        "supports_prompt_caching": true,
        "supports_assistant_prefill": true,
        "source": "https://api-docs.deepseek.com/quick_start/pricing"
    },
    "deepseek/deepseek-v4-pro": {
        "input_cost_per_token": 4.35e-07,
        "output_cost_per_token": 8.7e-07,
        "cache_read_input_token_cost": 3.625e-09,
        "input_cost_per_token_cache_hit": 3.625e-09,
        "litellm_provider": "deepseek",
        "max_input_tokens": 1000000,
        "max_output_tokens": 384000,
        "max_tokens": 384000,
        "mode": "chat",
        "supports_function_calling": true,
        "supports_native_streaming": true,
        "supports_prompt_caching": true,
        "supports_assistant_prefill": true,
        "source": "https://api-docs.deepseek.com/quick_start/pricing"
    },
```

Note: The prefixed variants use `input_cost_per_token_cache_hit` (LiteLLM modern field) alongside `cache_read_input_token_cost` (legacy field) following the convention of existing entries like `deepseek/deepseek-v3.2`.

- [ ] **Step 2: Validate the JSON is still well-formed**

```bash
cd D:/headroom && python -m json.tool crates/headroom-proxy/data/model_prices_and_context_window.json > /dev/null && echo "Valid JSON"
```

Expected: "Valid JSON"

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
cd D:/headroom && python -m pytest tests/test_pricing.py tests/test_pricing_litellm.py tests/test_providers/test_deepseek.py -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/headroom-proxy/data/model_prices_and_context_window.json
git commit -m "feat(proxy): add DeepSeek V4 context window entries to vendored JSON

Add bare and provider-prefixed JSON entries for deepseek-v4-flash
and deepseek-v4-pro so Rust-side context window lookups work.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Run All Tests and Final Verification

- [ ] **Step 1: Run all pricing-related tests**

```bash
cd D:/headroom && python -m pytest tests/test_pricing.py tests/test_pricing_litellm.py tests/test_litellm_optional.py tests/test_providers/test_anthropic.py tests/test_providers/test_openai.py tests/test_providers/test_deepseek.py tests/test_providers/test_universal.py tests/test_cost_tracker_counterfactual.py tests/test_proxy_savings_history.py tests/test_provider_model_fallback.py -v
```

Expected: All tests pass. Note any skips (some may require litellm installed).

- [ ] **Step 2: Verify imports work cleanly**

```bash
cd D:/headroom && python -c "from headroom.pricing import DEEPSEEK_PRICES, get_deepseek_registry; print('Direct import OK:', list(DEEPSEEK_PRICES.keys()))"
cd D:/headroom && python -c "from headroom.pricing.litellm_pricing import get_model_pricing; p = get_model_pricing('deepseek-v4-flash'); print('LiteLLM pricing OK:', p.input_cost_per_1m if p else 'None')"
cd D:/headroom && python -c "from headroom.providers.anthropic import AnthropicProvider; p = AnthropicProvider(); print('Anthropic fallback OK:', p._get_pricing('deepseek-v4-flash'))"
```

Expected: All three commands print correct pricing values without errors.

- [ ] **Step 3: Verify the full proxy cost tracker end-to-end**

```bash
cd D:/headroom && python -c "
from headroom.proxy.cost import CostTracker
t = CostTracker()
cost = t.estimate_cost('deepseek-v4-flash', input_tokens=1000000, output_tokens=1000000)
print(f'CostTracker estimate: \${cost:.4f}' if cost else 'CostTracker returned None')
"
```

Expected: Prints `CostTracker estimate: $0.4200` (0.14 input + 0.28 output, may vary if litellm is available).

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git log --oneline -5
```

Expected: Shows the 4 commit history for this feature.

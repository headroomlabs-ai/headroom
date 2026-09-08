"""Pricing comes from LiteLLM; the built-in table is only a fallback.

`_PRICING` used to be authoritative, with no LiteLLM lookup in front of it. That
is how it went ~18 months stale and priced `gpt-4.1-nano` 300x over. The table is
still needed -- the `litellm` dependency is gated `python_version < '3.14'`, and
LiteLLM does not know every model -- but it must not outrank the live source.

Resolution order (mirroring `get_context_limit`, so limits and prices agree):

1. explicit user config (`HEADROOM_MODEL_LIMITS` / `models.json`)
2. LiteLLM
3. built-in table -> family pattern -> unknown default
"""

from __future__ import annotations

import pytest

from headroom.pricing.litellm_model_resolution import unwrapped_model_forms
from headroom.providers.openai import OpenAIProvider

litellm = pytest.importorskip("litellm")


def test_unwrapped_model_forms_drops_leading_segments() -> None:
    """Pure function: no gateway-prefix list to maintain."""
    assert unwrapped_model_forms("bedrock/anthropic.claude-x") == ("anthropic.claude-x",)
    assert unwrapped_model_forms("accounts/fireworks/models/kimi-k2") == (
        "fireworks/models/kimi-k2",
        "models/kimi-k2",
        "kimi-k2",
    )
    assert unwrapped_model_forms("gpt-4o") == ()


@pytest.mark.parametrize(
    ("model", "want_in", "want_out"),
    [
        # Gateway-routed names. litellm.model_cost keys the UNWRAPPED form, so
        # these all returned None (-> $2.50/$10.00 unknown default) before.
        ("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", 3.00, 15.00),
        ("bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", 3.00, 15.00),
        ("vertex_ai/claude-sonnet-4-5", 3.00, 15.00),
        ("groq/llama-3.3-70b-versatile", 0.59, 0.79),
        # Non-OpenAI models reachable through the OpenAI-compatible passthrough.
        ("gemini-2.5-flash", 0.30, 2.50),
        ("deepseek-chat", 0.28, 0.42),
    ],
)
def test_provider_prices_models_its_table_never_covered(
    model: str, want_in: float, want_out: float
) -> None:
    got_in, got_out = OpenAIProvider()._get_pricing(model)

    assert (round(got_in, 2), round(got_out, 2)) == (want_in, want_out)


def test_explicit_config_outranks_litellm() -> None:
    """A configured price is a decision, not a guess."""
    provider = OpenAIProvider()
    provider._pricing_overrides["gpt-4o"] = (99.0, 111.0)

    assert provider._get_pricing("gpt-4o") == (99.0, 111.0)


def test_falls_back_to_the_builtin_table_without_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline / Python >= 3.14 installs must still get sane numbers.

    Pinned because the fallback is exactly where the table's correctness still
    matters -- it is the only thing those installs see.
    """
    import headroom.pricing.litellm_pricing as lp

    monkeypatch.setattr(lp, "LITELLM_AVAILABLE", False)

    provider = OpenAIProvider()
    assert provider._get_pricing("gpt-4.1-nano") == (0.10, 0.40)
    assert provider._get_pricing("gpt-4") == (30.00, 60.00)
    assert provider._get_pricing("o3") == (2.00, 8.00)


def test_unknown_model_still_returns_a_usable_default() -> None:
    """Never raise, never return None, for a model nobody knows."""
    got = OpenAIProvider()._get_pricing("totally-made-up-model-xyz")

    assert got is not None
    assert got[0] > 0

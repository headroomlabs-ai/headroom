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
from tests._dotenv import importorskip_no_env_leak

# NOTE: no module-level importorskip("litellm") here on purpose. Only the
# catalog-price test below needs the live LiteLLM database; every other test
# in this module either exercises the built-in-table/unknown-default fallback
# (which must ALSO run on LiteLLM-free installs, including Python 3.14) or a
# pure function. The litellm-dependent test skips itself at function level
# via importorskip_no_env_leak, which quarantines litellm's dotenv import
# side effect (see tests/_dotenv.py).


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
        # Any gateway-prefixed name here must be one litellm still prices:
        # when it prunes a model the unwrap finds nothing and _get_pricing
        # silently returns the $2.50/$10.00 GPT-4o default, which is what
        # this test exists to catch. litellm dropped
        # groq/llama-3.3-70b-versatile on 2026-09-23 (since restored) and
        # groq/llama-guard-3-8b (2026-10-07); see issue #3732.
        ("groq/llama-3.3-70b-versatile", 0.59, 0.79),
        # Non-OpenAI models reachable through the OpenAI-compatible passthrough.
        ("gemini-2.5-flash", 0.30, 2.50),
        ("deepseek-chat", 0.28, 0.42),
    ],
)
def test_provider_prices_models_its_table_never_covered(
    model: str, want_in: float, want_out: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The catalog interface needs LiteLLM; other tests also run without it.
    litellm = importorskip_no_env_leak("litellm")
    # Pin the catalog input, not a mutable live-catalog version. This exercises
    # real provider lookup for models outside its built-in table.
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            model: {
                "input_cost_per_token": want_in / 1_000_000,
                "output_cost_per_token": want_out / 1_000_000,
            }
        },
    )

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


def test_unknown_model_warns_once_and_is_flagged_estimated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The unknown-model default must be loud and distinguishable (issue #3732).

    Fails before the fix: no warning was emitted at all (only a debug log via
    the generic fallback path) and callers had no way to tell the $2.50/$10.00
    tuple apart from a real price lookup.
    """
    import logging

    import headroom.pricing.litellm_pricing as lp
    import headroom.providers.openai as openai_mod

    monkeypatch.setattr(lp, "LITELLM_AVAILABLE", False)
    # Isolation: the unknown-model registry is module-global.
    monkeypatch.setattr(openai_mod, "_UNKNOWN_PRICING_MODELS", set())

    provider = OpenAIProvider()
    with caplog.at_level(logging.WARNING, logger="headroom.providers.openai"):
        got = provider._get_pricing("no-such-model-warns-once")
        provider._get_pricing("no-such-model-warns-once")

    # Still a usable number -- a number is genuinely required here.
    assert got == (2.50, 10.00)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "no-such-model-warns-once" in warnings[0].getMessage()
    assert "GUESS" in warnings[0].getMessage()

    # ...but now it is representable as a guess.
    assert provider.pricing_is_estimated("no-such-model-warns-once") is True


def test_resolved_prices_are_not_flagged_estimated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Table hits and explicit config are decisions, not guesses (issue #3732)."""
    import logging

    import headroom.pricing.litellm_pricing as lp
    import headroom.providers.openai as openai_mod

    monkeypatch.setattr(lp, "LITELLM_AVAILABLE", False)
    monkeypatch.setattr(openai_mod, "_UNKNOWN_PRICING_MODELS", set())

    provider = OpenAIProvider()
    provider._pricing_overrides["my-custom-model"] = (1.0, 2.0)

    with caplog.at_level(logging.WARNING, logger="headroom.providers.openai"):
        assert provider._get_pricing("gpt-4.1-nano") == (0.10, 0.40)
        assert provider._get_pricing("my-custom-model") == (1.0, 2.0)
        assert provider.pricing_is_estimated("gpt-4.1-nano") is False
        assert provider.pricing_is_estimated("my-custom-model") is False

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_pricing_provenance_is_per_lookup_not_sticky(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A later explicit override clears the estimated flag (review follow-up).

    Provenance is derived from the lookup that just ran, never from the
    module-global warning-dedup set: once a model resolves (explicit config,
    catalog update), the next lookup reports it as a real price. The old
    sticky-set implementation kept reporting True forever after the first
    warning.
    """
    import logging

    import headroom.pricing.litellm_pricing as lp
    import headroom.providers.openai as openai_mod

    monkeypatch.setattr(lp, "LITELLM_AVAILABLE", False)
    monkeypatch.setattr(openai_mod, "_UNKNOWN_PRICING_MODELS", set())

    provider = OpenAIProvider()
    model = "mystery-model-per-lookup"

    with caplog.at_level(logging.WARNING, logger="headroom.providers.openai"):
        assert provider.pricing_is_estimated(model) is True

    # The user adds the model to models.json (or HEADROOM_MODEL_LIMITS); the
    # next lookup sees a decision, not a guess -- no process restart needed
    # for the flag to clear, because nothing was cached.
    provider._pricing_overrides[model] = (1.0, 2.0)
    with caplog.at_level(logging.WARNING, logger="headroom.providers.openai"):
        assert provider.pricing_is_estimated(model) is False
        assert provider._get_pricing(model) == (1.0, 2.0)

    # And the warning still fired exactly once for the unknown phase.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert model in warnings[0].getMessage()

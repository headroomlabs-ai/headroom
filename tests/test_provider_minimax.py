"""Tests for the MiniMax provider (token counts, context limits, costs, vision)."""
from __future__ import annotations

import pytest

from headroom.providers.minimax import (
    MODEL_CONTEXT_LIMITS,
    MODEL_INPUT_COST,
    MODEL_OUTPUT_COST,
    MiniMaxProvider,
)


class TestMiniMaxModelMetadata:
    """Model registry has correct limits and pricing for known M3 / M2.7 family."""

    def test_known_models_have_context_limit(self) -> None:
        provider = MiniMaxProvider()
        for model in ("MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"):
            assert provider.get_context_limit(model) == MODEL_CONTEXT_LIMITS[model], model

    def test_m3_has_million_token_context(self) -> None:
        provider = MiniMaxProvider()
        assert provider.get_context_limit("MiniMax-M3") == 1_000_000

    def test_m27_family_has_200k_context(self) -> None:
        provider = MiniMaxProvider()
        assert provider.get_context_limit("MiniMax-M2.7") == 204_800
        assert provider.get_context_limit("MiniMax-M2.7-highspeed") == 204_800

    def test_unknown_model_returns_safe_default(self) -> None:
        provider = MiniMaxProvider()
        # Unknown model falls back to 128K (conservative)
        assert provider.get_context_limit("MiniMax-M99-future") == 128_000

    def test_model_prefix_is_stripped(self) -> None:
        provider = MiniMaxProvider()
        # Clients may send "minimax/MiniMax-M3" — must resolve to the bare model.
        assert provider.get_context_limit("minimax/MiniMax-M3") == 1_000_000
        assert provider.get_max_output_tokens("minimax/MiniMax-M2.7-highspeed") == 65_536

    def test_max_output_tokens(self) -> None:
        provider = MiniMaxProvider()
        assert provider.get_max_output_tokens("MiniMax-M3") == 131_072
        assert provider.get_max_output_tokens("MiniMax-M2.7") == 65_536

    def test_pricing_tables_are_populated(self) -> None:
        # Sanity: every model in the context table has an entry in both cost tables
        for model in MODEL_CONTEXT_LIMITS:
            assert model in MODEL_INPUT_COST, f"missing input cost for {model}"
            assert model in MODEL_OUTPUT_COST, f"missing output cost for {model}"
            assert MODEL_INPUT_COST[model] > 0
            assert MODEL_OUTPUT_COST[model] > 0

    def test_input_cheaper_than_output(self) -> None:
        # Industry standard: input is cheaper than output
        for model in MODEL_CONTEXT_LIMITS:
            assert MODEL_INPUT_COST[model] < MODEL_OUTPUT_COST[model], model


class TestMiniMaxCapabilities:
    """Capability flags: tool use, vision, streaming."""

    def test_supports_tools_for_every_model(self) -> None:
        provider = MiniMaxProvider()
        for model in ("MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"):
            assert provider.supports_tools(model) is True

    def test_vision_only_for_m3(self) -> None:
        provider = MiniMaxProvider()
        assert provider.supports_vision("MiniMax-M3") is True
        assert provider.supports_vision("MiniMax-M2.7") is False
        assert provider.supports_vision("MiniMax-M2.7-highspeed") is False

    def test_supports_streaming(self) -> None:
        provider = MiniMaxProvider()
        assert provider.supports_streaming("MiniMax-M3") is True


class TestMiniMaxTokenCounter:
    """Token counter delegates to AnthropicProvider's tiktoken (cl100k_base)."""

    def test_token_counter_reports_positive_count(self) -> None:
        provider = MiniMaxProvider()
        counter = provider.get_token_counter("MiniMax-M3")
        # Roughly: 1 token per ~4 chars for English text
        count = counter.count_text("Hello, world! " * 10)
        assert count > 0
        assert count < 100  # 130 chars / 4 ≈ 30 tokens

    def test_token_counter_reusable_across_models(self) -> None:
        provider = MiniMaxProvider()
        c1 = provider.get_token_counter("MiniMax-M3")
        c2 = provider.get_token_counter("MiniMax-M2.7")
        # Same encoding family — counts should be identical for the same input.
        text = "The quick brown fox jumps over the lazy dog."
        assert c1.count_text(text) == c2.count_text(text)


class TestMiniMaxProviderName:
    def test_provider_name(self) -> None:
        assert MiniMaxProvider.name == "minimax"


@pytest.mark.parametrize(
    "model,expected_limit",
    [
        ("MiniMax-M3", 1_000_000),
        ("MiniMax-M2.7", 204_800),
        ("MiniMax-M2.7-highspeed", 204_800),
        ("MiniMax-M2.5-highspeed", 204_800),
        ("MiniMax-M2.1", 204_800),
        ("MiniMax-M2", 204_800),
    ],
)
def test_model_context_limit_parametrized(model: str, expected_limit: int) -> None:
    provider = MiniMaxProvider()
    assert provider.get_context_limit(model) == expected_limit

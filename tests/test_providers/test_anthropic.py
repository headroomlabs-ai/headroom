"""Tests for Anthropic provider."""

import pytest


class TestAnthropicModelSanitization:
    def test_sanitize_model_id_removes_ansi_escape_sequences(self):
        from headroom.providers.anthropic import sanitize_anthropic_model_id

        assert sanitize_anthropic_model_id("claude-opus-4-8\x1b[1m") == "claude-opus-4-8"

    def test_sanitize_model_id_removes_displayed_style_suffix(self):
        from headroom.providers.anthropic import sanitize_anthropic_model_id

        assert sanitize_anthropic_model_id("claude-opus-4-8[1m]") == "claude-opus-4-8"

    def test_sanitize_model_metadata_cleans_nested_model_ids(self):
        from headroom.providers.anthropic import sanitize_anthropic_model_metadata

        payload = {
            "data": [
                {"id": "claude-opus-4-8\x1b[1m", "display_name": "Claude Opus 4.8"},
                {"id": "claude-sonnet-4-5[1m]"},
            ],
            "model": "claude-opus-4-8[1m]",
        }

        assert sanitize_anthropic_model_metadata(payload) == {
            "data": [
                {"id": "claude-opus-4-8", "display_name": "Claude Opus 4.8"},
                {"id": "claude-sonnet-4-5"},
            ],
            "model": "claude-opus-4-8",
        }


class TestAnthropicTokenCounting:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_count_text_fallback(self, anthropic_provider):
        # Without API client, should use tiktoken fallback
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        count = counter.count_text("Hello world")
        assert count > 0

    def test_count_messages_basic(self, anthropic_provider):
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        messages = [{"role": "user", "content": "Hello"}]
        count = counter.count_messages(messages)
        assert count > 0

    def test_count_text_allows_literal_special_tokens(self, anthropic_provider):
        counter = anthropic_provider.get_token_counter("claude-3-5-sonnet-20241022")
        count = counter.count_text("prefix <|fim_suffix|> suffix")
        assert count > 0


class TestAnthropicModelLimits:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_get_context_limit_claude_sonnet(self, anthropic_provider):
        limit = anthropic_provider.get_context_limit("claude-3-5-sonnet-20241022")
        assert limit == 200000

    def test_get_context_limit_claude_opus(self, anthropic_provider):
        limit = anthropic_provider.get_context_limit("claude-3-opus-20240229")
        assert limit == 200000

    def test_get_context_limit_strips_ansi_model_suffix(self, anthropic_provider):
        assert anthropic_provider.get_context_limit("claude-opus-4-7[1m]") == 1000000

    def test_get_context_limit_claude_opus_4_8(self, anthropic_provider):
        assert anthropic_provider.get_context_limit("claude-opus-4-8") == 1000000

    def test_get_context_limit_claude_sonnet_4_6(self, anthropic_provider):
        # Sonnet 4.6 ships with the 1M context window (long-context pricing tier).
        assert anthropic_provider.get_context_limit("claude-sonnet-4-6") == 1000000

    def test_supports_model_known(self, anthropic_provider):
        assert anthropic_provider.supports_model("claude-3-5-sonnet-20241022")

    def test_supports_model_prefix(self, anthropic_provider):
        assert anthropic_provider.supports_model("claude-3-5-sonnet-latest")

    def test_token_counter_cache_uses_sanitized_model_id(self, anthropic_provider):
        plain = anthropic_provider.get_token_counter("claude-opus-4-7")
        styled = anthropic_provider.get_token_counter("claude-opus-4-7\x1b[1m")

        assert styled is plain


class TestAnthropicCostEstimation:
    @pytest.fixture
    def anthropic_provider(self):
        from headroom.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    def test_estimate_cost_basic(self, anthropic_provider):
        cost = anthropic_provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=0,
            model="claude-3-5-sonnet-20241022",
        )
        # $3.00 per 1M input
        assert cost == pytest.approx(3.00, rel=0.1)

    def test_pricing_lookup_strips_ansi_model_suffix(self, anthropic_provider):
        assert anthropic_provider._get_pricing("claude-opus-4-7[1m]") == (
            anthropic_provider._get_pricing("claude-opus-4-7")
        )

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-4-5-20251101", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8"],
    )
    def test_current_opus_tier_pricing(self, anthropic_provider, model):
        # Opus 4.5–4.8 share the same current-tier rates per anthropic.com/pricing
        # (verified 2026-06-27): $5/MTok input, $25/MTok output, cache read = 0.1x input ($0.50).
        assert anthropic_provider._get_pricing(model) == {
            "input": 5.00,
            "output": 25.00,
            "cached_input": 0.50,
        }

    @pytest.mark.parametrize(
        "model", ["claude-sonnet-4-5", "claude-sonnet-4-6", "claude-sonnet-4-20250514"]
    )
    def test_current_sonnet_tier_pricing(self, anthropic_provider, model):
        # Sonnet 4 / 4.5 / 4.6 all sit at the current Sonnet tier per anthropic.com/pricing
        # (verified 2026-06-27): $3/MTok input, $15/MTok output, cache read = 0.1x input ($0.30).
        assert anthropic_provider._get_pricing(model) == {
            "input": 3.00,
            "output": 15.00,
            "cached_input": 0.30,
        }

    def test_current_haiku_tier_pricing(self, anthropic_provider):
        # Haiku 4.5 (current Haiku tier, verified 2026-06-27): $1/MTok input,
        # $5/MTok output, cache read = 0.1x input ($0.10). Was previously stored
        # with Haiku 3.5 rates ($0.80/$4/$0.08).
        assert anthropic_provider._get_pricing("claude-haiku-4-5-20251001") == {
            "input": 1.00,
            "output": 5.00,
            "cached_input": 0.10,
        }

"""Tests for OpenAI provider."""

import pytest

from headroom.providers.openai import (
    _get_encoding_name_for_model,
)


class TestOpenAITokenCounting:
    def test_count_text_empty(self, openai_tokenizer):
        assert openai_tokenizer.count_text("") == 0

    def test_count_text_simple(self, openai_tokenizer):
        count = openai_tokenizer.count_text("Hello world")
        assert count > 0
        assert count < 10  # Should be ~2 tokens

    def test_count_text_with_special_chars(self, openai_tokenizer):
        text = "Hello 🌍! Special chars: @#$%"
        count = openai_tokenizer.count_text(text)
        assert count > 0

    def test_count_text_allows_literal_special_tokens(self, openai_tokenizer):
        """count_text must not raise on literal tiktoken special-token strings.

        Regression: a /v1/responses request whose context contained the literal
        "<|endoftext|>" made tiktoken raise ValueError (default
        disallowed_special="all"), which the proxy turned into an HTTP 413
        compression_refused. Markers must be counted as ordinary text instead.
        """
        text = "before <|endoftext|> after"
        count = openai_tokenizer.count_text(text)
        assert count > openai_tokenizer.count_text("before  after")

    def test_count_messages_single(self, openai_tokenizer):
        messages = [{"role": "user", "content": "Hello"}]
        count = openai_tokenizer.count_messages(messages)
        assert count > 0

    def test_count_messages_with_tools(self, openai_tokenizer):
        messages = [
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
            },
        ]
        count = openai_tokenizer.count_messages(messages)
        assert count > 10  # Tool calls add overhead

    def test_count_message_overhead(self, openai_tokenizer):
        # Each message has ~4 tokens overhead
        msg = {"role": "user", "content": ""}
        count = openai_tokenizer.count_message(msg)
        assert count >= 4


class TestOpenAIModelLimits:
    def test_get_context_limit_gpt4o(self, openai_provider):
        assert openai_provider.get_context_limit("gpt-4o") == 128000

    def test_get_context_limit_o1(self, openai_provider):
        assert openai_provider.get_context_limit("o1") == 200000

    def test_get_context_limit_unknown_model(self, openai_provider):
        # Unknown models now get a fallback value instead of raising
        limit = openai_provider.get_context_limit("unknown-model")
        assert limit == 128000  # Default fallback

    def test_supports_model_known(self, openai_provider):
        assert openai_provider.supports_model("gpt-4o") is True
        assert openai_provider.supports_model("gpt-4o-mini") is True

    def test_supports_model_unknown(self, openai_provider):
        assert openai_provider.supports_model("claude-3") is False


class TestOpenAICostEstimation:
    def test_estimate_cost_input_only(self, openai_provider):
        cost = openai_provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=0,
            model="gpt-4o",
        )
        assert cost == pytest.approx(2.50, rel=0.01)

    def test_estimate_cost_with_output(self, openai_provider):
        cost = openai_provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=1000000,
            model="gpt-4o",
        )
        # $2.50 input + $10.00 output = $12.50
        assert cost == pytest.approx(12.50, rel=0.01)

    def test_estimate_cost_with_cached(self, openai_provider):
        cost = openai_provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=0,
            model="gpt-4o",
            cached_tokens=500000,
        )
        # 500K regular @ $2.50/M = $1.25, 500K cached @ $1.25/M = $0.625
        assert cost == pytest.approx(1.875, rel=0.01)

    def test_estimate_cost_unknown_model(self, openai_provider):
        # Unknown models now get fallback pricing (gpt-4o tier)
        cost = openai_provider.estimate_cost(
            input_tokens=1000,
            output_tokens=1000,
            model="unknown-model",
        )
        # Fallback uses gpt-4o pricing: $2.50/M input + $10/M output
        # = (1000/1M * 2.50) + (1000/1M * 10.00) = 0.0025 + 0.01 = 0.0125
        assert cost == pytest.approx(0.0125, rel=0.01)


class TestEncodingSelection:
    def test_gpt4o_uses_o200k(self):
        assert _get_encoding_name_for_model("gpt-4o") == "o200k_base"

    def test_gpt4_uses_cl100k(self):
        assert _get_encoding_name_for_model("gpt-4") == "cl100k_base"

    def test_versioned_model_prefix_match(self):
        assert _get_encoding_name_for_model("gpt-4o-2024-11-20") == "o200k_base"

    def test_unknown_model_uses_fallback(self):
        # Unknown models now get a fallback encoding instead of raising
        encoding = _get_encoding_name_for_model("completely-unknown")
        assert encoding == "o200k_base"  # Default fallback

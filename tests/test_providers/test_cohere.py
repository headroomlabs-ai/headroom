"""Tests for Cohere provider."""

from __future__ import annotations

import pytest

from headroom.providers import CohereProvider


class TestCohereProvider:
    """Tests for CohereProvider."""

    @pytest.fixture
    def provider(self):
        """Create Cohere provider without client (estimation mode)."""
        return CohereProvider()

    def test_name(self, provider):
        """Test provider name."""
        assert provider.name == "cohere"

    def test_supports_command_models(self, provider):
        """Test support for Command models."""
        assert provider.supports_model("command-r-plus") is True
        assert provider.supports_model("command-r") is True
        assert provider.supports_model("command-a") is True
        assert provider.supports_model("command") is True

    def test_not_supports_other_models(self, provider):
        """Test non-support for other models."""
        assert provider.supports_model("gpt-4o") is False
        assert provider.supports_model("claude-3") is False
        assert provider.supports_model("gemini-2.0") is False

    def test_get_token_counter(self, provider):
        """Test getting token counter."""
        counter = provider.get_token_counter("command-r-plus")
        assert counter is not None
        count = counter.count_text("Hello, world!")
        assert count > 0

    def test_get_context_limit_command_a(self, provider):
        """Test context limit for Command A (256K)."""
        limit = provider.get_context_limit("command-a")
        assert limit == 256000

    def test_get_context_limit_command_r_plus(self, provider):
        """Test context limit for Command R+."""
        limit = provider.get_context_limit("command-r-plus")
        assert limit == 128000

    def test_get_context_limit_command_r(self, provider):
        """Test context limit for Command R."""
        limit = provider.get_context_limit("command-r")
        assert limit == 128000

    def test_get_context_limit_legacy_command(self, provider):
        """Test context limit for legacy Command."""
        limit = provider.get_context_limit("command")
        assert limit == 4096

    def test_estimate_cost_command_r_plus(self, provider):
        """Test cost estimation for Command R+."""
        cost = provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=500000,
            model="command-r-plus",
        )
        assert cost is not None
        # 1M input * $2.50/1M + 0.5M output * $10.00/1M = $2.50 + $5.00 = $7.50
        assert abs(cost - 7.50) < 0.01

    def test_estimate_cost_command_r(self, provider):
        """Test cost estimation for Command R."""
        cost = provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=500000,
            model="command-r",
        )
        assert cost is not None
        # 1M input * $0.15/1M + 0.5M output * $0.60/1M = $0.15 + $0.30 = $0.45
        assert abs(cost - 0.45) < 0.01

    def test_estimate_cost_unknown_model(self, provider):
        """Test cost estimation returns None for unknown model."""
        cost = provider.estimate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="unknown-model",
        )
        assert cost is None


class TestCohereTokenCounter:
    """Tests for CohereTokenCounter."""

    @pytest.fixture
    def counter(self):
        """Create token counter without client."""
        provider = CohereProvider()
        return provider.get_token_counter("command-r-plus")

    def test_count_text_empty(self, counter):
        """Test counting empty text."""
        assert counter.count_text("") == 0

    def test_count_text_simple(self, counter):
        """Test counting simple text."""
        count = counter.count_text("Hello, world!")
        assert count > 0
        assert count < 20  # Should be a few tokens

    def test_count_messages(self, counter):
        """Test counting messages."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = counter.count_messages(messages)
        assert count > 0

    def test_count_messages_empty(self, counter):
        """Test counting empty messages."""
        assert counter.count_messages([]) == 0

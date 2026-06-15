"""Tests for Deepseek provider."""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from headroom.providers.deepseek import (
    DeepseekProvider,
    DeepseekTokenCounter,
    _CONTEXT_LIMITS,
    _PRICING,
    _UNKNOWN_MODEL_WARNINGS,
    _check_pricing_staleness,
)
from headroom.providers.openai_compatible import (
    OpenAICompatibleProvider,
    create_deepseek_provider,
)


class TestDeepseekProvider:
    """Tests for DeepseekProvider."""

    def test_name(self):
        provider = DeepseekProvider()
        assert provider.name == "deepseek"

    def test_supports_known_models(self):
        provider = DeepseekProvider()
        for model in [
            "deepseek-chat",
            "deepseek-v3",
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "deepseek-coder",
            "deepseek-reasoner",
        ]:
            assert provider.supports_model(model)

    def test_supports_unknown_deepseek_model(self):
        provider = DeepseekProvider()
        assert provider.supports_model("deepseek-unknown-model")

    def test_does_not_support_other_providers(self):
        provider = DeepseekProvider()
        assert not provider.supports_model("gpt-4o")
        assert not provider.supports_model("claude-3-opus")

    def test_context_limits_known_models(self):
        provider = DeepseekProvider()
        assert provider.get_context_limit("deepseek-chat") == 131072
        assert provider.get_context_limit("deepseek-v4-flash") == 1000000
        assert provider.get_context_limit("deepseek-v4-pro") == 1000000
        # deepseek-coder may be overridden by LiteLLM to 128000; verify >= 16384
        assert provider.get_context_limit("deepseek-coder") >= 16384

    def test_context_limit_unknown_defaults_128k(self):
        provider = DeepseekProvider()
        assert provider.get_context_limit("deepseek-future-model") == 128000

    def test_estimate_cost_known_model(self):
        provider = DeepseekProvider()
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="deepseek-chat",
        )
        assert cost is not None
        assert cost > 0

    def test_estimate_cost_unknown_model_returns_none(self):
        provider = DeepseekProvider()
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="nonexistent-model",
        )
        assert cost is None

    def test_get_token_counter_returns_counter(self):
        provider = DeepseekProvider()
        counter = provider.get_token_counter("deepseek-chat")
        assert isinstance(counter, DeepseekTokenCounter)

    def test_get_token_counter_caches(self):
        provider = DeepseekProvider()
        counter1 = provider.get_token_counter("deepseek-chat")
        counter2 = provider.get_token_counter("deepseek-chat")
        assert counter1 is counter2

    def test_output_buffer_default(self):
        provider = DeepseekProvider()
        assert provider.get_output_buffer("deepseek-chat") == 8_192

    def test_output_buffer_v4(self):
        provider = DeepseekProvider()
        assert provider.get_output_buffer("deepseek-v4-flash") == 384_000

    def test_warns_for_unknown_model(self, caplog):
        import logging

        _UNKNOWN_MODEL_WARNINGS.clear()
        provider = DeepseekProvider()
        with caplog.at_level(logging.WARNING):
            limit = provider.get_context_limit("deepseek-future-model")
        assert limit == 128_000
        assert "Unknown Deepseek model" in caplog.text

    def test_uses_litellm_for_context_limit(self, monkeypatch):
        mock_litellm = MagicMock()
        mock_litellm.get_model_info.return_value = {"max_input_tokens": 256000}
        monkeypatch.setattr("headroom.providers.deepseek.litellm", mock_litellm)
        monkeypatch.setattr("headroom.providers.deepseek.LITELLM_AVAILABLE", True)
        provider = DeepseekProvider()
        limit = provider.get_context_limit("deepseek-chat")
        assert limit == 256000

    def test_supports_model_uses_instance_limits(self):
        provider = DeepseekProvider()
        provider._context_limits["custom-model"] = 32000
        assert provider.supports_model("custom-model")


class TestDeepseekCustomConfig:
    """Tests for custom model configuration via env vars."""

    def test_custom_context_limit_from_env(self, monkeypatch):
        import json

        monkeypatch.setenv(
            "HEADROOM_MODEL_LIMITS",
            json.dumps({"deepseek": {"context_limits": {"my-fine-tuned": 64000}}}),
        )
        from headroom.providers.deepseek import _load_custom_model_config

        config = _load_custom_model_config()
        assert config["context_limits"]["my-fine-tuned"] == 64000

    def test_custom_pricing_from_env(self, monkeypatch):
        import json

        monkeypatch.setenv(
            "HEADROOM_MODEL_LIMITS",
            json.dumps({"deepseek": {"pricing": {"my-model": [1.0, 2.0]}}}),
        )
        from headroom.providers.deepseek import _load_custom_model_config

        config = _load_custom_model_config()
        assert config["pricing"]["my-model"] == [1.0, 2.0]


class TestDeepseekTokenCounter:
    """Tests for DeepseekTokenCounter."""

    def _make_counter(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.count_text = MagicMock(side_effect=lambda t: len(t.split()))
        with patch(
            "headroom.providers.deepseek.get_tokenizer", return_value=mock_tokenizer
        ):
            return DeepseekTokenCounter("deepseek-chat")

    def test_count_text(self):
        counter = self._make_counter()
        tokens = counter.count_text("Hello, world!")
        assert tokens > 0

    def test_count_message(self):
        counter = self._make_counter()
        message = {"role": "user", "content": "Hello, world!"}
        tokens = counter.count_message(message)
        assert tokens > 0

    def test_count_messages(self):
        counter = self._make_counter()
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        tokens = counter.count_messages(messages)
        assert tokens > 0

    def test_count_message_with_tool_calls(self):
        counter = self._make_counter()
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                }
            ],
        }
        tokens = counter.count_message(message)
        assert tokens > 0


class TestContextLimitsCoverage:
    """Tests for data consistency."""

    def test_all_pricing_have_context_limits(self):
        for model in _PRICING:
            assert model in _CONTEXT_LIMITS, (
                f"Model '{model}' is in _PRICING but not in _CONTEXT_LIMITS"
            )


class TestCreateDeepseekProvider:
    """Tests for the create_deepseek_provider factory function."""

    def test_returns_openai_compatible_provider(self):
        provider = create_deepseek_provider()
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_provider_name(self):
        provider = create_deepseek_provider()
        assert provider.name == "deepseek"

    def test_base_url(self):
        provider = create_deepseek_provider()
        assert provider.base_url == "https://api.deepseek.com/v1"

    def test_api_key_passthrough(self):
        provider = create_deepseek_provider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_supports_registered_models(self):
        provider = create_deepseek_provider()
        for model in [
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "deepseek-v3",
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-v2",
            "deepseek-v2-chat",
            "deepseek-coder",
            "deepseek-coder-v2",
        ]:
            assert provider.supports_model(model)

    def test_registered_model_context_windows(self):
        provider = create_deepseek_provider()
        assert provider.get_context_limit("deepseek-v4-flash") == 1_000_000
        assert provider.get_context_limit("deepseek-v4-pro") == 1_000_000
        assert provider.get_context_limit("deepseek-chat") == 131_072
        assert provider.get_context_limit("deepseek-coder") == 16_384

    def test_registered_model_cost_estimation(self):
        provider = create_deepseek_provider()
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="deepseek-chat",
        )
        assert cost is not None
        assert cost > 0


class TestPricingStaleness:
    """Tests for pricing staleness warning."""

    def test_pricing_staleness_warning(self, monkeypatch):
        import headroom.providers.deepseek as ds_mod

        monkeypatch.setattr(ds_mod, "_PRICING_WARNING_SHOWN", False)
        monkeypatch.setattr(ds_mod, "_PRICING_STALE_DAYS", 0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _check_pricing_staleness()
            assert result is not None
            assert "days old" in result

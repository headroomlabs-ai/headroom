"""Tests for provider model fallback and configuration."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from headroom.providers.anthropic import (
    AnthropicProvider,
    _infer_model_tier,
)
from headroom.providers.anthropic import (
    _load_custom_model_config as anthropic_load_config,
)
from headroom.providers.openai import (
    OpenAIProvider,
    _infer_model_family,
)
from headroom.providers.openai import (
    _load_custom_model_config as openai_load_config,
)


class TestAnthropicModelFallback:
    """Tests for Anthropic provider model fallback."""

    def test_known_claude_4_models(self):
        """Test that Claude 4/4.5 models are recognized."""
        provider = AnthropicProvider()

        # Claude Opus 4.5
        assert provider.get_context_limit("claude-opus-4-5-20251101") == 200000
        assert provider.supports_model("claude-opus-4-5-20251101")

        # Claude Sonnet 4
        assert provider.get_context_limit("claude-sonnet-4-20250514") == 200000
        assert provider.supports_model("claude-sonnet-4-20250514")

        # Claude Haiku 4
        assert provider.get_context_limit("claude-haiku-4-5-20251001") == 200000
        assert provider.supports_model("claude-haiku-4-5-20251001")

    def test_pattern_based_inference_opus(self):
        """Test pattern-based inference for opus models."""
        provider = AnthropicProvider()

        # Future opus model should infer 200K and opus pricing
        limit = provider.get_context_limit("claude-opus-5-20260101")
        assert limit == 200000

        pricing = provider._get_pricing("claude-opus-5-20260101")
        assert pricing["input"] == 15.00
        assert pricing["output"] == 75.00

    def test_pattern_based_inference_sonnet(self):
        """Test pattern-based inference for sonnet models."""
        provider = AnthropicProvider()

        limit = provider.get_context_limit("claude-sonnet-5-20260101")
        assert limit == 200000

        pricing = provider._get_pricing("claude-sonnet-5-20260101")
        assert pricing["input"] == 3.00
        assert pricing["output"] == 15.00

    def test_pattern_based_inference_haiku(self):
        """Test pattern-based inference for haiku models."""
        provider = AnthropicProvider()

        limit = provider.get_context_limit("claude-haiku-5-20260101")
        assert limit == 200000

        pricing = provider._get_pricing("claude-haiku-5-20260101")
        assert pricing["input"] == 0.80
        assert pricing["output"] == 4.00

    def test_unknown_claude_model_fallback(self):
        """Test fallback for unknown Claude models."""
        provider = AnthropicProvider()

        # Unknown Claude model should get 200K default
        limit = provider.get_context_limit("claude-unknown-model")
        assert limit == 200000

        # Should still support it
        assert provider.supports_model("claude-unknown-model")

    def test_no_exception_for_unknown_model(self):
        """Test that unknown models don't raise exceptions."""
        provider = AnthropicProvider()

        # Should not raise
        limit = provider.get_context_limit("claude-future-model-xyz")
        assert limit > 0

    def test_infer_model_tier(self):
        """Test model tier inference."""
        assert _infer_model_tier("claude-opus-4-5-20251101") == "opus"
        assert _infer_model_tier("claude-sonnet-4-20250514") == "sonnet"
        assert _infer_model_tier("claude-haiku-4-5-20251001") == "haiku"
        assert _infer_model_tier("claude-3-5-sonnet-latest") == "sonnet"
        assert _infer_model_tier("CLAUDE-OPUS-FUTURE") == "opus"  # Case insensitive
        assert _infer_model_tier("some-other-model") is None

    def test_explicit_context_limits_override(self):
        """Test that explicit context_limits override defaults."""
        provider = AnthropicProvider(context_limits={"custom-model": 500000})

        assert provider.get_context_limit("custom-model") == 500000

    def test_pricing_for_known_models(self):
        """Test pricing retrieval for known models."""
        provider = AnthropicProvider()

        # Claude Opus 4.5
        pricing = provider._get_pricing("claude-opus-4-5-20251101")
        assert pricing["input"] == 15.00
        assert pricing["output"] == 75.00
        assert pricing["cached_input"] == 1.50

    def test_cost_estimation_for_new_models(self):
        """Test cost estimation works for new models."""
        provider = AnthropicProvider()

        cost = provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=100000,
            model="claude-opus-4-5-20251101",
            cached_tokens=0,
        )

        # $15/1M input + $75/1M * 0.1M output = $15 + $7.5 = $22.5
        assert cost == pytest.approx(22.5, rel=0.01)


class TestAnthropicConfigLoading:
    """Tests for Anthropic config file/env var loading."""

    def test_load_from_env_var_json(self):
        """Test loading config from JSON env var."""
        config = {"context_limits": {"test-model": 300000}}

        with patch.dict(os.environ, {"HEADROOM_MODEL_LIMITS": json.dumps(config)}):
            loaded = anthropic_load_config()
            assert loaded["context_limits"]["test-model"] == 300000

    def test_load_from_env_var_file(self):
        """Test loading config from file path in env var."""
        config = {"context_limits": {"file-model": 400000}}

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_limits.json"
            config_path.write_text(json.dumps(config))

            with patch.dict(os.environ, {"HEADROOM_MODEL_LIMITS": str(config_path)}):
                loaded = anthropic_load_config()
                assert loaded["context_limits"]["file-model"] == 400000

    def test_load_from_config_file(self):
        """Test loading from ~/.headroom/models.json."""
        config = {
            "anthropic": {
                "context_limits": {"config-model": 250000},
                "pricing": {"config-model": {"input": 5.0, "output": 25.0}},
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".headroom"
            config_dir.mkdir()
            config_file = config_dir / "models.json"
            config_file.write_text(json.dumps(config))

            with patch.object(Path, "home", return_value=Path(tmpdir)):
                loaded = anthropic_load_config()
                assert loaded["context_limits"]["config-model"] == 250000

    def test_env_var_overrides_config_file(self):
        """Test that env var takes precedence over config file."""
        env_config = {"context_limits": {"test-model": 100000}}
        file_config = {"anthropic": {"context_limits": {"test-model": 200000}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".headroom"
            config_dir.mkdir()
            config_file = config_dir / "models.json"
            config_file.write_text(json.dumps(file_config))

            with patch.object(Path, "home", return_value=Path(tmpdir)):
                with patch.dict(os.environ, {"HEADROOM_MODEL_LIMITS": json.dumps(env_config)}):
                    loaded = anthropic_load_config()
                    # Env var should win
                    assert loaded["context_limits"]["test-model"] == 100000


class TestOpenAIModelFallback:
    """Tests for OpenAI provider model fallback."""

    def test_known_models(self):
        """Test that known models work."""
        provider = OpenAIProvider()

        assert provider.get_context_limit("gpt-4o") == 128000
        assert provider.get_context_limit("gpt-4o-mini") == 128000
        assert provider.get_context_limit("o1") == 200000
        assert provider.get_context_limit("o3-mini") == 200000

    def test_pattern_based_inference_gpt4o(self):
        """Test pattern-based inference for gpt-4o models."""
        provider = OpenAIProvider()

        # Future gpt-4o model
        limit = provider.get_context_limit("gpt-4o-2025-01-01")
        assert limit == 128000

    def test_pattern_based_inference_o1(self):
        """Test pattern-based inference for o1 models."""
        provider = OpenAIProvider()

        limit = provider.get_context_limit("o1-super-2025")
        assert limit == 200000

    def test_pattern_based_inference_o3(self):
        """Test pattern-based inference for o3 models."""
        provider = OpenAIProvider()

        limit = provider.get_context_limit("o3-large-2025")
        assert limit == 200000

    def test_unknown_model_fallback(self):
        """Test fallback for unknown models."""
        provider = OpenAIProvider()

        # Unknown model should get 128K default
        limit = provider.get_context_limit("gpt-5-future")
        assert limit == 128000

    def test_no_exception_for_unknown_model(self):
        """Test that unknown models don't raise exceptions."""
        provider = OpenAIProvider()

        # Should not raise
        limit = provider.get_context_limit("gpt-future-xyz")
        assert limit > 0

    def test_infer_model_family(self):
        """Test model family inference."""
        assert _infer_model_family("gpt-4o-2024-11-20") == "gpt-4o"
        assert _infer_model_family("gpt-4-turbo-preview") == "gpt-4-turbo"
        assert _infer_model_family("gpt-4") == "gpt-4"
        assert _infer_model_family("gpt-3.5-turbo") == "gpt-3.5"
        assert _infer_model_family("o1-preview") == "o1"
        assert _infer_model_family("o3-mini") == "o3"
        assert _infer_model_family("unknown") is None

    def test_explicit_context_limits_override(self):
        """Test that explicit context_limits override defaults."""
        provider = OpenAIProvider(context_limits={"custom-model": 500000})

        assert provider.get_context_limit("custom-model") == 500000

    def test_supports_model_expanded(self):
        """Test that supports_model works for new patterns."""
        provider = OpenAIProvider()

        # Should support any gpt-* or o1/o3
        assert provider.supports_model("gpt-4o")
        assert provider.supports_model("gpt-4o-future")
        assert provider.supports_model("gpt-5-future")
        assert provider.supports_model("o1-mega")
        assert provider.supports_model("o3-ultra")


class TestOpenAIConfigLoading:
    """Tests for OpenAI config file/env var loading."""

    def test_load_from_env_var_json(self):
        """Test loading config from JSON env var."""
        config = {"openai": {"context_limits": {"test-model": 300000}}}

        with patch.dict(os.environ, {"HEADROOM_MODEL_LIMITS": json.dumps(config)}):
            loaded = openai_load_config()
            assert loaded["context_limits"]["test-model"] == 300000

    def test_load_pricing_from_config(self):
        """Test loading pricing from config."""
        config = {"openai": {"pricing": {"test-model": [5.0, 15.0]}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_limits.json"
            config_path.write_text(json.dumps(config))

            with patch.dict(os.environ, {"HEADROOM_MODEL_LIMITS": str(config_path)}):
                loaded = openai_load_config()
                assert loaded["pricing"]["test-model"] == [5.0, 15.0]


class TestCrossProviderConsistency:
    """Tests for consistency across providers."""

    def test_both_providers_use_same_env_var(self):
        """Test that both providers use HEADROOM_MODEL_LIMITS."""
        config = {
            "anthropic": {"context_limits": {"anthropic-model": 100000}},
            "openai": {"context_limits": {"openai-model": 200000}},
        }

        with patch.dict(os.environ, {"HEADROOM_MODEL_LIMITS": json.dumps(config)}):
            anthropic = anthropic_load_config()
            openai = openai_load_config()

            assert anthropic["context_limits"]["anthropic-model"] == 100000
            assert openai["context_limits"]["openai-model"] == 200000

    def test_both_providers_never_raise_for_unknown_models(self):
        """Test that neither provider raises for unknown models."""
        anthropic = AnthropicProvider()
        openai = OpenAIProvider()

        # Neither should raise
        anthropic.get_context_limit("claude-future-model-xyz")
        openai.get_context_limit("gpt-future-model-xyz")

    def test_both_providers_warn_for_unknown_models(self):
        """Test that both providers warn for unknown models."""
        # Clear warning caches
        from headroom.providers import anthropic as anthropic_module
        from headroom.providers import openai as openai_module

        anthropic_module._UNKNOWN_MODEL_WARNINGS.clear()
        openai_module._UNKNOWN_MODEL_WARNINGS.clear()

        with (
            patch.object(anthropic_module.logger, "warning") as anthropic_warning,
            patch.object(openai_module.logger, "warning") as openai_warning,
        ):
            anthropic = AnthropicProvider()
            anthropic.get_context_limit("claude-test-unknown-model")

            openai = OpenAIProvider()
            openai.get_context_limit("gpt-test-unknown-model")

        anthropic_warning.assert_called_once()
        openai_warning.assert_called_once()
        assert "claude-test-unknown-model" in anthropic_warning.call_args.args[0]
        assert "gpt-test-unknown-model" in openai_warning.call_args.args[0]

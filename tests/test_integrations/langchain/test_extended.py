"""Tests for extended LangChain integration modules.

Tests cover:
1. langchain_providers - Provider auto-detection
2. langchain_agents - HeadroomToolWrapper end-to-end compression
3. langchain_langsmith - LangSmith integration
4. HeadroomChatModel provider auto-detection
"""

import json
from unittest.mock import MagicMock

import pytest

# Check if LangChain is available
try:
    import langchain_core  # noqa: F401

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Skip all tests if LangChain not installed
pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


class TestProviderDetection:
    """Tests for langchain_providers module."""

    def test_detect_openai_provider(self):
        """Detect OpenAI from ChatOpenAI class."""
        from headroom.integrations.langchain.providers import detect_provider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai.chat_models"

        provider = detect_provider(mock_model)
        assert provider == "openai"

    def test_detect_anthropic_provider(self):
        """Detect Anthropic from ChatAnthropic class."""
        from headroom.integrations.langchain.providers import detect_provider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatAnthropic"
        mock_model.__class__.__module__ = "langchain_anthropic.chat_models"

        provider = detect_provider(mock_model)
        assert provider == "anthropic"

    def test_detect_google_provider(self):
        """Detect Google from ChatGoogleGenerativeAI class."""
        from headroom.integrations.langchain.providers import detect_provider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_model.__class__.__module__ = "langchain_google_genai"

        provider = detect_provider(mock_model)
        assert provider == "google"

    def test_detect_fallback_to_openai(self):
        """Fall back to OpenAI for unknown models."""
        from headroom.integrations.langchain.providers import detect_provider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "CustomChatModel"
        mock_model.__class__.__module__ = "my_custom_module"

        provider = detect_provider(mock_model)
        assert provider == "openai"

    def test_detect_from_model_name_claude(self):
        """Detect Anthropic from model name containing 'claude'."""
        from headroom.integrations.langchain.providers import detect_provider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "CustomModel"
        mock_model.__class__.__module__ = "custom"
        mock_model.model_name = "claude-3-5-sonnet-20241022"

        provider = detect_provider(mock_model)
        assert provider == "anthropic"

    def test_get_headroom_provider_openai(self):
        """Get OpenAIProvider for OpenAI model."""
        from headroom.integrations.langchain.providers import get_headroom_provider
        from headroom.providers import OpenAIProvider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai"

        provider = get_headroom_provider(mock_model)
        assert isinstance(provider, OpenAIProvider)

    def test_get_headroom_provider_anthropic(self):
        """Get AnthropicProvider for Anthropic model."""
        from headroom.integrations.langchain.providers import get_headroom_provider
        from headroom.providers import AnthropicProvider

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatAnthropic"
        mock_model.__class__.__module__ = "langchain_anthropic"

        provider = get_headroom_provider(mock_model)
        assert isinstance(provider, AnthropicProvider)

    def test_get_model_name_from_langchain(self):
        """Extract model name from LangChain model."""
        from headroom.integrations.langchain.providers import get_model_name_from_langchain

        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o"

        name = get_model_name_from_langchain(mock_model)
        assert name == "gpt-4o"

    def test_get_model_name_fallback(self):
        """Fall back when model name not available."""
        from headroom.integrations.langchain.providers import get_model_name_from_langchain

        mock_model = MagicMock(spec=[])
        mock_model.__class__.__name__ = "ChatOpenAI"

        name = get_model_name_from_langchain(mock_model)
        assert name == "gpt-4o"  # Default for OpenAI


class TestHeadroomToolWrapper:
    """End-to-end HeadroomToolWrapper compression (compress_tool_result not mocked)."""

    def test_call_compresses_large_json(self):
        """Large JSON outputs get compressed."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "search"

        # Large JSON output
        large_output = json.dumps([{"id": i, "data": "x" * 100} for i in range(50)])
        mock_tool.invoke.return_value = large_output

        wrapper = HeadroomToolWrapper(mock_tool, min_chars_to_compress=100)
        result = wrapper("query")

        # Should be smaller after compression
        assert len(result) <= len(large_output)


class TestHeadroomLangSmithCallbackHandler:
    """Tests for LangSmith integration."""

    def test_init(self):
        """Initialize handler."""
        from headroom.integrations.langchain.langsmith import (
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler(auto_update_runs=False)

        assert handler._auto_update is False
        assert handler._pending_metrics == {}

    def test_set_headroom_metrics(self):
        """Set metrics for a run."""
        from headroom.integrations.langchain.langsmith import (
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler(auto_update_runs=False)

        handler.set_headroom_metrics(
            run_id="test-run-123",
            tokens_before=1000,
            tokens_after=800,
            transforms_applied=["smart_crusher"],
        )

        assert "test-run-123" in handler._pending_metrics
        metrics = handler._pending_metrics["test-run-123"]
        assert metrics.tokens_before == 1000
        assert metrics.tokens_after == 800
        assert metrics.tokens_saved == 200
        assert metrics.savings_percent == 20.0

    def test_get_run_metrics(self):
        """Get metrics for a specific run."""
        from headroom.integrations.langchain.langsmith import (
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler(auto_update_runs=False)
        handler._run_metrics["run-1"] = {"headroom.tokens_saved": 100}

        metrics = handler.get_run_metrics("run-1")
        assert metrics["headroom.tokens_saved"] == 100

    def test_get_summary(self):
        """Get summary statistics."""
        from headroom.integrations.langchain.langsmith import (
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler(auto_update_runs=False)
        handler._run_metrics = {
            "run-1": {"headroom.tokens_saved": 100, "headroom.savings_percent": 20},
            "run-2": {"headroom.tokens_saved": 200, "headroom.savings_percent": 30},
        }

        summary = handler.get_summary()
        assert summary["total_runs"] == 2
        assert summary["total_tokens_saved"] == 300
        assert summary["average_savings_percent"] == 25.0

    def test_reset(self):
        """Reset clears all metrics."""
        from headroom.integrations.langchain.langsmith import (
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler(auto_update_runs=False)
        handler._run_metrics = {"run-1": {}}
        handler._pending_metrics = {"run-2": MagicMock()}

        handler.reset()

        assert handler._run_metrics == {}
        assert handler._pending_metrics == {}


class TestAutoDetectProviderInChatModel:
    """Tests for auto_detect_provider in HeadroomChatModel."""

    def test_auto_detect_enabled_by_default(self):
        """auto_detect_provider is True by default."""
        from headroom.integrations import HeadroomChatModel

        mock_model = MagicMock()
        mock_model._llm_type = "test"
        mock_model._identifying_params = {}
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai"

        model = HeadroomChatModel(mock_model)
        assert model.auto_detect_provider is True

    def test_auto_detect_can_be_disabled(self):
        """auto_detect_provider can be set to False."""
        from headroom.integrations import HeadroomChatModel

        mock_model = MagicMock()
        mock_model._llm_type = "test"
        mock_model._identifying_params = {}

        model = HeadroomChatModel(mock_model, auto_detect_provider=False)
        assert model.auto_detect_provider is False

    def test_pipeline_uses_detected_provider(self):
        """Pipeline uses auto-detected provider."""
        from headroom.integrations import HeadroomChatModel
        from headroom.providers import AnthropicProvider

        mock_model = MagicMock()
        mock_model._llm_type = "test"
        mock_model._identifying_params = {}
        mock_model.__class__.__name__ = "ChatAnthropic"
        mock_model.__class__.__module__ = "langchain_anthropic"

        model = HeadroomChatModel(mock_model)
        _ = model.pipeline  # Force lazy init

        assert isinstance(model._provider, AnthropicProvider)

"""Tests for universal provider support.

Tests OpenAICompatibleProvider, GoogleProvider, and LiteLLMProvider.
"""

from __future__ import annotations

import pytest

from headroom.providers import (
    GoogleProvider,
    LiteLLMProvider,
    ModelCapabilities,
    OpenAICompatibleProvider,
    create_anyscale_provider,
    create_fireworks_provider,
    create_groq_provider,
    create_litellm_provider,
    create_lmstudio_provider,
    create_ollama_provider,
    create_together_provider,
    create_vllm_provider,
    is_litellm_available,
)


def _transformers_available() -> bool:
    """Check if transformers is available."""
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


class TestOpenAICompatibleProvider:
    """Tests for OpenAICompatibleProvider."""

    def test_init_default(self):
        """Test initialization with defaults."""
        provider = OpenAICompatibleProvider()
        assert provider.name == "openai_compatible"
        assert provider.base_url is None

    def test_init_with_config(self):
        """Test initialization with configuration."""
        provider = OpenAICompatibleProvider(
            name="custom",
            base_url="http://localhost:8080/v1",
            api_key="test-key",
        )
        assert provider.name == "custom"
        assert provider.base_url == "http://localhost:8080/v1"
        assert provider.api_key == "test-key"

    def test_supports_any_model(self):
        """Test that provider supports any model."""
        provider = OpenAICompatibleProvider()
        assert provider.supports_model("any-model") is True
        assert provider.supports_model("llama-3") is True
        assert provider.supports_model("custom-finetuned") is True

    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers not installed - needed for HuggingFace tokenizer",
    )
    def test_get_token_counter(self):
        """Test getting token counter."""
        provider = OpenAICompatibleProvider()
        counter = provider.get_token_counter("llama-3-8b")
        assert counter is not None
        # Should be able to count tokens
        count = counter.count_text("Hello, world!")
        assert count > 0

    def test_get_context_limit_known_model(self):
        """Test context limit for known models."""
        provider = OpenAICompatibleProvider()
        # Llama 3.1 has 128K context
        limit = provider.get_context_limit("llama-3.1-8b")
        assert limit == 128000

    def test_get_context_limit_deepseek_v3_is_1m(self):
        """DeepSeek V3/V4 support 1M context, not 128K (#1038)."""
        provider = OpenAICompatibleProvider()
        assert provider.get_context_limit("deepseek-v3") == 1048576
        assert provider.get_context_limit("deepseek-v4") == 1048576
        assert provider.get_context_limit("deepseek") == 1048576
        assert provider.get_context_limit("deepseek-v2") == 128000

    def test_get_context_limit_unknown_model(self):
        """Test context limit for unknown models (defaults to 128K)."""
        provider = OpenAICompatibleProvider()
        limit = provider.get_context_limit("unknown-model")
        assert limit == 128000

    def test_register_model(self):
        """Test registering a custom model."""
        provider = OpenAICompatibleProvider()
        provider.register_model(
            "my-model",
            context_window=64000,
            max_output_tokens=8192,
            input_cost_per_1m=1.0,
            output_cost_per_1m=2.0,
        )
        assert provider.get_context_limit("my-model") == 64000

    def test_estimate_cost_registered_model(self):
        """Test cost estimation for registered model."""
        provider = OpenAICompatibleProvider()
        provider.register_model(
            "priced-model",
            input_cost_per_1m=1.0,
            output_cost_per_1m=2.0,
        )
        cost = provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=500000,
            model="priced-model",
        )
        assert cost == 2.0  # 1.0 + 1.0

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation returns None for unknown model."""
        provider = OpenAICompatibleProvider()
        cost = provider.estimate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="unknown-model",
        )
        assert cost is None

    def test_register_model_accepts_capabilities_object(self):
        provider = OpenAICompatibleProvider()
        caps = ModelCapabilities(model="caps-model", context_window=16000, tokenizer_backend="test")

        provider.register_model("caps-model", capabilities=caps)

        assert provider.get_context_limit("caps-model") == 16000

    def test_get_token_counter_uses_registered_tokenizer_backend(self, monkeypatch):
        recorded: list[tuple[str, str | None]] = []

        class DummyTokenizer:
            def count_text(self, text: str) -> int:
                return len(text.split())

        monkeypatch.setattr(
            "headroom.providers.openai_compatible.get_tokenizer",
            lambda model, backend=None: recorded.append((model, backend)) or DummyTokenizer(),
        )
        provider = OpenAICompatibleProvider(
            models={
                "custom-model": ModelCapabilities(
                    model="custom-model",
                    tokenizer_backend="custom-backend",
                )
            }
        )

        counter = provider.get_token_counter("custom-model")

        assert counter.count_text("one two three") == 3
        assert recorded == [("custom-model", "custom-backend")]

    def test_openai_compatible_token_counter_counts_message_parts(self, monkeypatch):
        class DummyTokenizer:
            def count_text(self, text: str) -> int:
                return len(text)

        monkeypatch.setattr(
            "headroom.providers.openai_compatible.get_tokenizer",
            lambda model, backend=None: DummyTokenizer(),
        )
        counter = OpenAICompatibleProvider().get_token_counter("demo-model")

        tokens = counter.count_message(
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}, "there"],
                "name": "tester",
                "tool_calls": [{"function": {"name": "lookup", "arguments": '{"x":1}'}}],
                "tool_call_id": "call_123",
            }
        )
        total = counter.count_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": ["world"]},
            ]
        )

        assert tokens == 55
        assert total == 34

    def test_openai_compatible_token_counter_ignores_unhandled_content_shapes(self, monkeypatch):
        class DummyTokenizer:
            def count_text(self, text: str) -> int:
                return len(text)

        monkeypatch.setattr(
            "headroom.providers.openai_compatible.get_tokenizer",
            lambda model, backend=None: DummyTokenizer(),
        )
        counter = OpenAICompatibleProvider().get_token_counter("demo-model")

        assert counter.count_message({"role": "user", "content": {}}) == 8
        assert counter.count_message({"role": "user", "content": [{"type": "image"}, 123]}) == 8

    def test_get_context_limit_prefix_output_buffer_and_partial_pricing(self):
        provider = OpenAICompatibleProvider(
            models={
                "buffered": ModelCapabilities(
                    model="buffered",
                    max_output_tokens=1200,
                    input_cost_per_1m=1.0,
                )
            }
        )

        assert provider.get_context_limit("mistral-custom") == 32768
        assert provider.get_output_buffer("buffered", default=4000) == 1200
        assert provider.get_output_buffer("unknown", default=2222) == 2222
        assert provider.estimate_cost(1000, 1000, "buffered") is None


class TestModelCapabilities:
    """Tests for ModelCapabilities dataclass."""

    def test_default_values(self):
        """Test default capability values."""
        caps = ModelCapabilities(model="test-model")
        assert caps.context_window == 128000
        assert caps.max_output_tokens == 4096
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.supports_streaming is True

    def test_custom_values(self):
        """Test custom capability values."""
        caps = ModelCapabilities(
            model="custom-model",
            context_window=32000,
            max_output_tokens=16384,
            supports_tools=False,
            supports_vision=True,
            input_cost_per_1m=0.5,
            output_cost_per_1m=1.5,
        )
        assert caps.context_window == 32000
        assert caps.max_output_tokens == 16384
        assert caps.supports_tools is False
        assert caps.supports_vision is True
        assert caps.input_cost_per_1m == 0.5
        assert caps.output_cost_per_1m == 1.5


class TestGoogleProvider:
    """Tests for GoogleProvider."""

    @pytest.fixture
    def provider(self):
        """Create Google provider."""
        return GoogleProvider()

    def test_name(self, provider):
        """Test provider name."""
        assert provider.name == "google"

    def test_supports_gemini_models(self, provider):
        """Test support for Gemini models."""
        assert provider.supports_model("gemini-2.0-flash") is True
        assert provider.supports_model("gemini-1.5-pro") is True
        assert provider.supports_model("gemini-1.5-flash") is True

    def test_not_supports_other_models(self, provider):
        """Test non-support for other models."""
        assert provider.supports_model("gpt-4o") is False
        assert provider.supports_model("claude-3") is False

    def test_get_token_counter(self, provider):
        """Test getting token counter."""
        counter = provider.get_token_counter("gemini-2.0-flash")
        assert counter is not None
        count = counter.count_text("Hello, world!")
        assert count > 0

    def test_get_context_limit_gemini_2(self, provider):
        """Test context limit for Gemini 2.0."""
        limit = provider.get_context_limit("gemini-2.0-flash")
        # LiteLLM returns 1048576 (2^20), fallback returns 1000000
        assert limit in (1000000, 1048576)  # ~1M tokens

    def test_get_context_limit_gemini_1_5_pro(self, provider):
        """Test context limit for Gemini 1.5 Pro (2M!)."""
        limit = provider.get_context_limit("gemini-1.5-pro")
        # LiteLLM returns 2097152 (2^21), fallback returns 2000000
        assert limit in (2000000, 2097152)  # ~2M tokens!

    def test_estimate_cost(self, provider):
        """Test cost estimation."""
        cost = provider.estimate_cost(
            input_tokens=1000000,
            output_tokens=500000,
            model="gemini-2.0-flash",
        )
        assert cost is not None
        # 1M input * $0.10 + 0.5M output * $0.40 = $0.10 + $0.20 = $0.30
        assert abs(cost - 0.30) < 0.01

    def test_openai_compatible_url(self):
        """Test OpenAI-compatible URL."""
        url = GoogleProvider.get_openai_compatible_url("test-key")
        assert "generativelanguage.googleapis.com" in url


class TestProviderFactoryFunctions:
    """Tests for provider factory functions."""

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        provider = create_ollama_provider()
        assert provider.name == "ollama"
        assert provider.base_url == "http://localhost:11434/v1"

    def test_create_ollama_provider_custom_url(self):
        """Test creating Ollama provider with custom URL."""
        provider = create_ollama_provider("http://192.168.1.100:11434/v1")
        assert provider.base_url == "http://192.168.1.100:11434/v1"

    def test_create_together_provider(self):
        """Test creating Together provider."""
        provider = create_together_provider()
        assert provider.name == "together"
        assert "together.xyz" in provider.base_url

    def test_create_groq_provider(self):
        """Test creating Groq provider."""
        provider = create_groq_provider()
        assert provider.name == "groq"
        assert "groq.com" in provider.base_url

    def test_create_vllm_provider(self):
        """Test creating vLLM provider."""
        provider = create_vllm_provider("http://localhost:8000/v1")
        assert provider.name == "vllm"
        assert provider.base_url == "http://localhost:8000/v1"

    def test_create_lmstudio_provider(self):
        """Test creating LM Studio provider."""
        provider = create_lmstudio_provider()
        assert provider.name == "lmstudio"
        assert provider.base_url == "http://localhost:1234/v1"

    def test_create_fireworks_and_anyscale_providers(self):
        fireworks = create_fireworks_provider(api_key="fireworks-key")
        anyscale = create_anyscale_provider(api_key="anyscale-key")

        assert fireworks.name == "fireworks"
        assert fireworks.base_url == "https://api.fireworks.ai/inference/v1"
        assert fireworks.api_key == "fireworks-key"
        assert anyscale.name == "anyscale"
        assert anyscale.base_url == "https://api.endpoints.anyscale.com/v1"
        assert anyscale.api_key == "anyscale-key"


class TestLiteLLMProvider:
    """Tests for LiteLLM provider."""

    def test_is_litellm_available(self):
        """Test checking LiteLLM availability."""
        result = is_litellm_available()
        assert isinstance(result, bool)

    def test_unavailable_litellm_paths(self, monkeypatch):
        import headroom.providers.litellm as litellm_module

        monkeypatch.setattr(litellm_module, "LITELLM_AVAILABLE", False)

        assert litellm_module.is_litellm_available() is False
        assert litellm_module.LiteLLMProvider.list_supported_providers() == []
        with pytest.raises(RuntimeError, match="LiteLLM is required"):
            litellm_module.LiteLLMTokenCounter("gpt-4o")
        with pytest.raises(RuntimeError, match="LiteLLM is required"):
            litellm_module.LiteLLMProvider()

    def test_litellm_token_counter_fallback_paths(self, monkeypatch):
        import headroom.providers.litellm as litellm_module

        class DummyFallback:
            def count_text(self, text: str) -> int:
                return len(text.split())

        monkeypatch.setattr(litellm_module, "LITELLM_AVAILABLE", True)
        monkeypatch.setattr(
            litellm_module,
            "litellm_token_counter",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(litellm_module, "EstimatingTokenCounter", DummyFallback)

        counter = litellm_module.LiteLLMTokenCounter("gpt-4o")

        assert counter.count_text("") == 0
        assert counter.count_text("one two three") == 3
        assert counter.count_message({"content": "one two"}) == 6
        assert counter.count_messages([]) == 0
        assert counter.count_messages([{"content": "one two"}, {"content": "three"}]) == 14

    def test_litellm_provider_info_and_cost_fallbacks(self, monkeypatch):
        import headroom.providers.litellm as litellm_module

        monkeypatch.setattr(litellm_module, "LITELLM_AVAILABLE", True)
        monkeypatch.setattr(
            litellm_module,
            "litellm_get_model_info",
            lambda model: {
                "ctx-model": {"max_input_tokens": 64000},
                "max-model": {"max_tokens": 32000},
                "none-model": {"max_input_tokens": None, "max_output_tokens": None},
                "output-model": {"max_output_tokens": 6000},
            }[model],
        )
        monkeypatch.setattr(
            litellm_module,
            "litellm",
            type(
                "LiteLLM",
                (),
                {
                    "completion_cost": staticmethod(
                        lambda **kwargs: (
                            1.23
                            if kwargs["model"] == "priced-model"
                            else (_ for _ in ()).throw(RuntimeError("missing price"))
                        )
                    )
                },
            )(),
        )

        provider = litellm_module.LiteLLMProvider()

        assert provider.get_context_limit("ctx-model") == 64000
        assert provider.get_context_limit("max-model") == 32000
        assert provider.get_context_limit("none-model") == 128000
        assert provider.get_output_buffer("output-model", default=4000) == 4000
        assert provider.get_output_buffer("none-model", default=2222) == 2222
        assert provider.estimate_cost(1000, 1000, "priced-model") == 1.23
        assert provider.estimate_cost(1000, 1000, "missing-price") is None

    def test_litellm_provider_handles_info_exceptions_and_factory(self, monkeypatch):
        import headroom.providers.litellm as litellm_module

        monkeypatch.setattr(litellm_module, "LITELLM_AVAILABLE", True)
        monkeypatch.setattr(
            litellm_module,
            "litellm_get_model_info",
            lambda model: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        provider = create_litellm_provider()

        assert isinstance(provider, LiteLLMProvider)
        assert provider.get_context_limit("gpt-4o") == 128000
        assert provider.get_output_buffer("gpt-4o", default=3333) == 3333

    @pytest.mark.skipif(
        not is_litellm_available(),
        reason="LiteLLM not installed",
    )
    def test_create_litellm_provider(self):
        """Test creating LiteLLM provider."""
        from headroom.providers import create_litellm_provider

        provider = create_litellm_provider()
        assert provider.name == "litellm"

    @pytest.mark.skipif(
        not is_litellm_available(),
        reason="LiteLLM not installed",
    )
    def test_litellm_supports_any_model(self):
        """Test LiteLLM supports any model."""
        from headroom.providers import create_litellm_provider

        provider = create_litellm_provider()
        assert provider.supports_model("gpt-4o") is True
        assert provider.supports_model("claude-3-sonnet") is True
        assert provider.supports_model("any-model") is True

    @pytest.mark.skipif(
        not is_litellm_available(),
        reason="LiteLLM not installed",
    )
    def test_litellm_list_providers(self):
        """Test listing LiteLLM providers."""
        from headroom.providers import LiteLLMProvider

        providers = LiteLLMProvider.list_supported_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers

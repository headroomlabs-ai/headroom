"""OpenAI-compatible provider for universal LLM support.

This provider supports any LLM service that implements the OpenAI API format:
- Ollama (local)
- vLLM (local/cloud)
- Together AI
- Groq
- Fireworks AI
- Anyscale
- LM Studio
- LocalAI
- Hugging Face Inference Endpoints
- Azure OpenAI
- And many more...

The key insight: 70%+ of LLM providers use OpenAI-compatible APIs,
so supporting this format gives near-universal coverage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from headroom.tokenizers import get_tokenizer

from .base import Provider

logger = logging.getLogger(__name__)


@dataclass
class ModelCapabilities:
    """Model capability metadata.

    Stores information about a model's capabilities and constraints
    that the provider needs for token counting and cost estimation.
    """

    model: str
    context_window: int = 128000  # Default to 128K
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    tokenizer_backend: str | None = None  # Force specific tokenizer
    input_cost_per_1m: float | None = None  # Cost per 1M input tokens
    output_cost_per_1m: float | None = None  # Cost per 1M output tokens


# Default context limits for common open models
# These are reasonable defaults; users can override
_DEFAULT_CONTEXT_LIMITS: dict[str, int] = {
    # Llama 3 family
    "llama-3": 8192,
    "llama-3-8b": 8192,
    "llama-3-70b": 8192,
    "llama-3.1": 128000,
    "llama-3.1-8b": 128000,
    "llama-3.1-70b": 128000,
    "llama-3.1-405b": 128000,
    "llama-3.2": 128000,
    "llama-3.3": 128000,
    # Llama 2 family
    "llama-2": 4096,
    "llama-2-7b": 4096,
    "llama-2-13b": 4096,
    "llama-2-70b": 4096,
    "codellama": 16384,
    # Mistral family
    "mistral": 32768,
    "mistral-7b": 32768,
    "mistral-nemo": 128000,
    "mistral-small": 32768,
    "mistral-large": 128000,
    "mixtral": 32768,
    "mixtral-8x7b": 32768,
    "mixtral-8x22b": 65536,
    # Qwen family
    "qwen": 32768,
    "qwen2": 32768,
    "qwen2-7b": 32768,
    "qwen2-72b": 32768,
    "qwen2.5": 131072,
    # DeepSeek
    "deepseek": 32768,
    "deepseek-coder": 16384,
    "deepseek-v2": 128000,
    "deepseek-v3": 128000,
    # Yi
    "yi": 32768,
    "yi-34b": 32768,
    # Phi
    "phi-2": 2048,
    "phi-3": 4096,
    "phi-3-mini": 4096,
    "phi-3-medium": 4096,
    # Others
    "falcon": 2048,
    "falcon-40b": 2048,
    "falcon-180b": 2048,
    "gemma": 8192,
    "gemma-2": 8192,
    "starcoder": 8192,
    "starcoder2": 16384,
}


class OpenAICompatibleTokenCounter:
    """Token counter for OpenAI-compatible providers.

    Uses the TokenizerRegistry to get the appropriate tokenizer
    for the model, falling back to estimation if needed.
    """

    def __init__(
        self,
        model: str,
        tokenizer_backend: str | None = None,
    ):
        """Initialize token counter.

        Args:
            model: Model name.
            tokenizer_backend: Force specific tokenizer backend.
        """
        self.model = model
        self._tokenizer = get_tokenizer(model, backend=tokenizer_backend)

    def count_text(self, text: str) -> int:
        """Count tokens in text."""
        return self._tokenizer.count_text(text)

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a single message."""
        # Use OpenAI-style message overhead
        tokens = 4  # Base overhead

        role = message.get("role", "")
        tokens += self.count_text(role)

        content = message.get("content")
        if content:
            if isinstance(content, str):
                tokens += self.count_text(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            tokens += self.count_text(part.get("text", ""))
                    elif isinstance(part, str):
                        tokens += self.count_text(part)

        name = message.get("name")
        if name:
            tokens += self.count_text(name) + 1

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                tokens += self.count_text(func.get("name", ""))
                tokens += self.count_text(func.get("arguments", ""))
                tokens += 10

        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            tokens += self.count_text(tool_call_id) + 2

        return tokens

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        total = sum(self.count_message(msg) for msg in messages)
        total += 3  # Priming tokens
        return total


class OpenAICompatibleProvider(Provider):
    """Provider for OpenAI-compatible LLM services.

    Works with any service implementing the OpenAI chat completions API:
    - Ollama (local)
    - vLLM (local/cloud)
    - Together AI
    - Groq
    - Fireworks AI
    - LM Studio
    - LocalAI
    - And many more...

    Example:
        # For Ollama
        provider = OpenAICompatibleProvider(
            name="ollama",
            base_url="http://localhost:11434/v1",
            default_model="llama3.1",
        )

        # For Together AI
        provider = OpenAICompatibleProvider(
            name="together",
            base_url="https://api.together.xyz/v1",
        )

        # Get token counter for a specific model
        counter = provider.get_token_counter("llama-3.1-8b")
    """

    def __init__(
        self,
        name: str = "openai_compatible",
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        models: dict[str, ModelCapabilities] | None = None,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            name: Provider name for identification.
            base_url: API base URL (e.g., 'http://localhost:11434/v1').
            api_key: API key (if required).
            default_model: Default model for operations.
            models: Custom model configurations.
        """
        self._name = name
        self.base_url = base_url
        self.api_key = api_key
        self.default_model = default_model
        self._models: dict[str, ModelCapabilities] = models or {}

    @property
    def name(self) -> str:
        return self._name

    def register_model(
        self,
        model: str,
        capabilities: ModelCapabilities | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a model with its capabilities.

        Args:
            model: Model name.
            capabilities: Model capabilities object.
            **kwargs: Alternative way to specify capabilities.
        """
        if capabilities is not None:
            self._models[model] = capabilities
        else:
            self._models[model] = ModelCapabilities(model=model, **kwargs)

    def supports_model(self, model: str) -> bool:
        """Check if model is supported.

        OpenAI-compatible providers support any model by default,
        using estimation for token counting.
        """
        return True  # Always return True - we can estimate

    def get_token_counter(self, model: str) -> OpenAICompatibleTokenCounter:
        """Get token counter for a model.

        Uses the TokenizerRegistry to find the best tokenizer,
        with fallback to estimation.
        """
        tokenizer_backend = None

        # Check for registered model with specific tokenizer
        if model in self._models:
            tokenizer_backend = self._models[model].tokenizer_backend

        return OpenAICompatibleTokenCounter(model, tokenizer_backend)

    def get_context_limit(self, model: str) -> int:
        """Get context limit for a model.

        Priority:
        1. Registered model capabilities
        2. Default limits for known models
        3. Prefix matching
        4. Default 128K
        """
        # Check registered models
        if model in self._models:
            return self._models[model].context_window

        model_lower = model.lower()

        # Check default limits
        if model_lower in _DEFAULT_CONTEXT_LIMITS:
            return _DEFAULT_CONTEXT_LIMITS[model_lower]

        # Prefix match
        for prefix, limit in _DEFAULT_CONTEXT_LIMITS.items():
            if model_lower.startswith(prefix):
                return limit

        # Default to 128K for modern models
        return 128000

    def get_output_buffer(self, model: str, default: int = 4000) -> int:
        """Get recommended output buffer."""
        if model in self._models:
            return min(self._models[model].max_output_tokens, default)
        return default

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate cost if pricing is configured.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.
            cached_tokens: Number of cached tokens.

        Returns:
            Estimated cost in USD, or None if pricing unknown.
        """
        if model not in self._models:
            return None

        caps = self._models[model]
        if caps.input_cost_per_1m is None or caps.output_cost_per_1m is None:
            return None

        input_cost = (input_tokens / 1_000_000) * caps.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * caps.output_cost_per_1m

        return input_cost + output_cost


# Pre-configured provider factories for common services


def create_ollama_provider(
    base_url: str = "http://localhost:11434/v1",
) -> OpenAICompatibleProvider:
    """Create provider for Ollama.

    Ollama is a popular local LLM runner that supports many open models.

    Args:
        base_url: Ollama API URL (default: http://localhost:11434/v1).

    Returns:
        Configured provider.
    """
    return OpenAICompatibleProvider(
        name="ollama",
        base_url=base_url,
    )


def create_together_provider(
    api_key: str | None = None,
) -> OpenAICompatibleProvider:
    """Create provider for Together AI.

    Together AI offers high-performance inference for open models.

    Args:
        api_key: Together AI API key.

    Returns:
        Configured provider with Together AI pricing.
    """
    provider = OpenAICompatibleProvider(
        name="together",
        base_url="https://api.together.xyz/v1",
        api_key=api_key,
    )

    # Register common Together models with pricing
    # Pricing as of Jan 2025 (verify current rates)
    provider.register_model(
        "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        context_window=128000,
        input_cost_per_1m=0.18,
        output_cost_per_1m=0.18,
    )
    provider.register_model(
        "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        context_window=128000,
        input_cost_per_1m=0.88,
        output_cost_per_1m=0.88,
    )
    provider.register_model(
        "meta-llama/Llama-3.1-405B-Instruct-Turbo",
        context_window=128000,
        input_cost_per_1m=3.50,
        output_cost_per_1m=3.50,
    )

    return provider


def create_groq_provider(
    api_key: str | None = None,
) -> OpenAICompatibleProvider:
    """Create provider for Groq.

    Groq offers ultra-fast inference on custom hardware.

    Args:
        api_key: Groq API key.

    Returns:
        Configured provider with Groq pricing.
    """
    provider = OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )

    # Register common Groq models with pricing
    # Pricing as of Jan 2025 (verify current rates)
    provider.register_model(
        "llama-3.1-8b-instant",
        context_window=128000,
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.08,
    )
    provider.register_model(
        "llama-3.1-70b-versatile",
        context_window=128000,
        input_cost_per_1m=0.59,
        output_cost_per_1m=0.79,
    )
    provider.register_model(
        "mixtral-8x7b-32768",
        context_window=32768,
        input_cost_per_1m=0.24,
        output_cost_per_1m=0.24,
    )

    return provider


def create_fireworks_provider(
    api_key: str | None = None,
) -> OpenAICompatibleProvider:
    """Create provider for Fireworks AI.

    Args:
        api_key: Fireworks API key.

    Returns:
        Configured provider.
    """
    return OpenAICompatibleProvider(
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
    )


def create_anyscale_provider(
    api_key: str | None = None,
) -> OpenAICompatibleProvider:
    """Create provider for Anyscale Endpoints.

    Args:
        api_key: Anyscale API key.

    Returns:
        Configured provider.
    """
    return OpenAICompatibleProvider(
        name="anyscale",
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key=api_key,
    )


def create_vllm_provider(
    base_url: str,
) -> OpenAICompatibleProvider:
    """Create provider for vLLM server.

    vLLM is a high-performance inference engine.

    Args:
        base_url: vLLM server URL (e.g., 'http://localhost:8000/v1').

    Returns:
        Configured provider.
    """
    return OpenAICompatibleProvider(
        name="vllm",
        base_url=base_url,
    )


def create_lmstudio_provider(
    base_url: str = "http://localhost:1234/v1",
) -> OpenAICompatibleProvider:
    """Create provider for LM Studio.

    LM Studio is a desktop app for running local LLMs.

    Args:
        base_url: LM Studio API URL.

    Returns:
        Configured provider.
    """
    return OpenAICompatibleProvider(
        name="lmstudio",
        base_url=base_url,
    )


def create_deepseek_provider(
    api_key: str | None = None,
) -> OpenAICompatibleProvider:
    """Create provider for Deepseek.

    Deepseek offers high-performance LLMs at very competitive pricing,
    including V4, V3, Coder, and Reasoner models.

    Args:
        api_key: Deepseek API key.

    Returns:
        Configured provider with Deepseek pricing.
    """
    provider = OpenAICompatibleProvider(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key=api_key,
    )

    # V4 models
    provider.register_model(
        "deepseek-v4-flash",
        context_window=1_000_000,
        max_output_tokens=384_000,
        input_cost_per_1m=0.27,
        output_cost_per_1m=1.10,
    )
    provider.register_model(
        "deepseek-v4-pro",
        context_window=1_000_000,
        max_output_tokens=384_000,
        input_cost_per_1m=0.54,
        output_cost_per_1m=2.19,
    )

    # V3 models
    provider.register_model(
        "deepseek-v3",
        context_window=128_000,
        input_cost_per_1m=0.27,
        output_cost_per_1m=1.10,
    )
    provider.register_model(
        "deepseek-chat",
        context_window=131_072,
        input_cost_per_1m=0.27,
        output_cost_per_1m=1.10,
    )
    provider.register_model(
        "deepseek-reasoner",
        context_window=131_072,
        max_output_tokens=16_384,
        input_cost_per_1m=0.55,
        output_cost_per_1m=2.19,
    )

    # V2 models
    provider.register_model(
        "deepseek-v2",
        context_window=128_000,
        input_cost_per_1m=0.27,
        output_cost_per_1m=1.10,
    )
    provider.register_model(
        "deepseek-v2-chat",
        context_window=128_000,
        input_cost_per_1m=0.27,
        output_cost_per_1m=1.10,
    )

    # Coder models
    provider.register_model(
        "deepseek-coder",
        context_window=16_384,
        input_cost_per_1m=0.14,
        output_cost_per_1m=0.28,
    )
    provider.register_model(
        "deepseek-coder-v2",
        context_window=128_000,
        input_cost_per_1m=0.27,
        output_cost_per_1m=1.10,
    )

    return provider

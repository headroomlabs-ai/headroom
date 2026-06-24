"""LiteLLM provider for universal LLM support.

LiteLLM provides a unified interface to 100+ LLM providers:
- OpenAI, Azure OpenAI
- Anthropic
- Google (Vertex AI, AI Studio)
- AWS Bedrock
- Cohere
- Replicate
- Hugging Face
- Ollama
- Together AI
- Groq
- And many more...

This integration allows Headroom to work with any LiteLLM-supported
model without needing provider-specific implementations.

Requires: pip install litellm
"""

from __future__ import annotations

import logging
import os
from typing import Any

from headroom.tokenizers import EstimatingTokenCounter

from .base import Provider, TokenCounter

logger = logging.getLogger(__name__)

# Check if litellm is available
try:
    # LiteLLM can print its provider-list banner during import, before the
    # module-level suppression flags below can be set.
    os.environ.setdefault("LITELLM_SUPPRESS_DEBUG_INFO", "True")

    import litellm

    # Suppress litellm's startup banner ("Provider List: https://...") and
    # verbose debug output that spams stdout on every worker import.
    litellm.suppress_debug_info = True
    litellm.set_verbose = False

    from litellm import get_model_info as litellm_get_model_info
    from litellm import model_cost as litellm_model_cost
    from litellm import token_counter as litellm_token_counter

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None  # type: ignore[assignment]
    litellm_token_counter = None  # type: ignore[assignment]
    litellm_model_cost = None  # type: ignore[assignment]
    litellm_get_model_info = None  # type: ignore[assignment]


def is_litellm_available() -> bool:
    """Check if LiteLLM is installed.

    Returns:
        True if litellm is available.
    """
    return LITELLM_AVAILABLE


class LiteLLMTokenCounter:
    """Token counter using LiteLLM's token counting.

    LiteLLM provides accurate token counting for most providers
    by using the appropriate tokenizer for each model.
    """

    def __init__(self, model: str):
        """Initialize LiteLLM token counter.

        Args:
            model: Model name in LiteLLM format (e.g., 'gpt-4o', 'claude-3-sonnet').
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError(
                "LiteLLM is required for LiteLLMProvider. Install with: pip install litellm"
            )
        self.model = model
        # Fallback estimator for when litellm counting fails
        self._fallback = EstimatingTokenCounter()

    def count_text(self, text: str) -> int:
        """Count tokens in text using LiteLLM."""
        if not text:
            return 0
        try:
            # LiteLLM's token_counter expects messages format
            # We wrap text in a simple message
            return litellm_token_counter(
                model=self.model,
                messages=[{"role": "user", "content": text}],
            )
        except Exception as e:
            logger.debug(f"LiteLLM token count failed for {self.model}: {e}")
            return self._fallback.count_text(text)

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a single message."""
        try:
            return litellm_token_counter(
                model=self.model,
                messages=[message],
            )
        except Exception as e:
            logger.debug(f"LiteLLM message count failed for {self.model}: {e}")
            # Fallback to estimation
            tokens = 4  # Base overhead
            content = message.get("content", "")
            if isinstance(content, str):
                tokens += self._fallback.count_text(content)
            return tokens

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in messages using LiteLLM."""
        if not messages:
            return 0
        try:
            return litellm_token_counter(
                model=self.model,
                messages=messages,
            )
        except Exception as e:
            logger.debug(f"LiteLLM messages count failed for {self.model}: {e}")
            # Fallback to estimation
            total = sum(self.count_message(msg) for msg in messages)
            total += 3  # Priming
            return total


class LiteLLMProvider(Provider):
    """Provider using LiteLLM for universal model support.

    LiteLLM supports 100+ LLM providers with a unified interface.
    This provider leverages LiteLLM's:
    - Token counting (accurate for most providers)
    - Model info (context limits, capabilities)
    - Cost estimation (from LiteLLM's model database)

    Example:
        from headroom.providers import LiteLLMProvider

        provider = LiteLLMProvider()

        # Works with any LiteLLM-supported model
        counter = provider.get_token_counter("gpt-4o")
        counter = provider.get_token_counter("claude-3-5-sonnet-20241022")
        counter = provider.get_token_counter("gemini/gemini-1.5-pro")
        counter = provider.get_token_counter("bedrock/anthropic.claude-v2")
        counter = provider.get_token_counter("ollama/llama3")

    Model Format:
        LiteLLM uses a provider/model format for some providers:
        - OpenAI: "gpt-4o" or "openai/gpt-4o"
        - Anthropic: "claude-3-sonnet" or "anthropic/claude-3-sonnet"
        - Google: "gemini/gemini-1.5-pro"
        - Azure: "azure/gpt-4"
        - Bedrock: "bedrock/anthropic.claude-v2"
        - Ollama: "ollama/llama3"

    See LiteLLM docs for full model list:
    https://docs.litellm.ai/docs/providers
    """

    def __init__(self):
        """Initialize LiteLLM provider."""
        if not LITELLM_AVAILABLE:
            raise RuntimeError(
                "LiteLLM is required for LiteLLMProvider. Install with: pip install litellm"
            )

    @property
    def name(self) -> str:
        return "litellm"

    def supports_model(self, model: str) -> bool:
        """Check if LiteLLM supports this model.

        LiteLLM supports most models, so this returns True
        for any model. Actual support depends on credentials.
        """
        return True  # LiteLLM handles validation

    def get_token_counter(self, model: str) -> TokenCounter:
        """Get token counter for a model."""
        return LiteLLMTokenCounter(model)

    def get_context_limit(self, model: str) -> int:
        """Get context limit using LiteLLM's model info."""
        try:
            if litellm_get_model_info is not None:
                info = litellm_get_model_info(model)
                if info and "max_input_tokens" in info:
                    result = info["max_input_tokens"]
                    return result if result is not None else 128000
                if info and "max_tokens" in info:
                    result = info["max_tokens"]
                    return result if result is not None else 128000
        except Exception as e:
            logger.debug(f"LiteLLM get_model_info failed for {model}: {e}")

        # Fallback to reasonable default
        return 128000

    def get_output_buffer(self, model: str, default: int = 4000) -> int:
        """Get recommended output buffer."""
        try:
            if litellm_get_model_info is not None:
                info = litellm_get_model_info(model)
                if info and "max_output_tokens" in info:
                    max_output = info["max_output_tokens"]
                    if max_output is not None:
                        return min(max_output, default)
        except Exception:
            pass
        return default

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate cost using LiteLLM's cost database.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.
            cached_tokens: Cached tokens (may not be supported by all providers).

        Returns:
            Estimated cost in USD, or None if pricing unknown.
        """
        try:
            # LiteLLM's cost calculation
            cost = litellm.completion_cost(
                model=model,
                prompt="",  # We're using token counts directly
                completion="",
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            return cost
        except Exception as e:
            logger.debug(f"LiteLLM cost estimation failed for {model}: {e}")
            return None

    @classmethod
    def list_supported_providers(cls) -> list[str]:
        """List providers supported by LiteLLM.

        Returns:
            List of provider names.
        """
        if not LITELLM_AVAILABLE:
            return []

        # Major providers supported by LiteLLM
        return [
            "openai",
            "anthropic",
            "azure",
            "google",
            "vertex_ai",
            "bedrock",
            "cohere",
            "replicate",
            "huggingface",
            "ollama",
            "together_ai",
            "groq",
            "fireworks_ai",
            "anyscale",
            "deepinfra",
            "perplexity",
            "mistral",
            "cloudflare",
            "ai21",
            "nlp_cloud",
            "aleph_alpha",
            "petals",
            "baseten",
            "openrouter",
            "vllm",
            "xinference",
            "text-generation-inference",
        ]


def create_litellm_provider() -> LiteLLMProvider:
    """Create a LiteLLM provider.

    Returns:
        Configured LiteLLMProvider.

    Raises:
        RuntimeError: If LiteLLM is not installed.
    """
    return LiteLLMProvider()

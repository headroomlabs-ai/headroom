"""Google Gemini provider for Headroom SDK.

Supports Google's Gemini models through two interfaces:
1. OpenAI-compatible endpoint (recommended for Headroom)
2. Native Google AI SDK (for advanced features)

Token counting uses Google's official countTokens API when a client
is provided. This gives accurate counts for all content types.

Usage:
    import google.generativeai as genai
    from headroom import GoogleProvider

    genai.configure(api_key="your-api-key")
    provider = GoogleProvider(client=genai)  # Accurate counting via API

    # Or without client (uses estimation - less accurate)
    provider = GoogleProvider()  # Warning: approximate counting
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from typing import Any

from headroom.tokenizers import EstimatingTokenCounter

from .base import Provider, TokenCounter

# Check if litellm is available for pricing/context limit lookups
try:
    import litellm
    from litellm import get_model_info as litellm_get_model_info

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None  # type: ignore[assignment]
    litellm_get_model_info = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Warning flags
_FALLBACK_WARNING_SHOWN = False

# Pricing metadata
_PRICING_LAST_UPDATED = date(2025, 1, 6)

# Google model context limits
_CONTEXT_LIMITS: dict[str, int] = {
    # Gemini 2.0
    "gemini-2.0-flash": 1000000,
    "gemini-2.0-flash-exp": 1000000,
    "gemini-2.0-flash-thinking": 1000000,
    # Gemini 1.5
    "gemini-1.5-pro": 2000000,
    "gemini-1.5-pro-latest": 2000000,
    "gemini-1.5-flash": 1000000,
    "gemini-1.5-flash-latest": 1000000,
    "gemini-1.5-flash-8b": 1000000,
    # Gemini 1.0
    "gemini-1.0-pro": 32768,
    "gemini-pro": 32768,
}

# Fallback pricing - LiteLLM is preferred source
# Pricing per 1M tokens (input, output)
# Note: Google has different pricing tiers based on context length
_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-exp": (0.10, 0.40),  # Experimental, may change
    "gemini-1.5-pro": (1.25, 5.00),  # Up to 128K context
    "gemini-1.5-flash": (0.075, 0.30),  # Up to 128K context
    "gemini-1.5-flash-8b": (0.0375, 0.15),
    "gemini-1.0-pro": (0.50, 1.50),
}


class GeminiTokenCounter:
    """Token counter for Gemini models.

    When a google.generativeai client is provided, uses the official
    countTokens API for accurate counting. Falls back to estimation
    when no client is available.

    Usage:
        import google.generativeai as genai
        genai.configure(api_key="...")

        # With API (accurate)
        counter = GeminiTokenCounter("gemini-2.0-flash", client=genai)

        # Without API (estimation)
        counter = GeminiTokenCounter("gemini-2.0-flash")
    """

    def __init__(self, model: str, client: Any = None):
        """Initialize Gemini token counter.

        Args:
            model: Gemini model name.
            client: Optional google.generativeai module for API-based counting.
        """
        global _FALLBACK_WARNING_SHOWN

        self.model = model
        self._client = client
        self._use_api = client is not None
        self._genai_model = None

        # Gemini uses ~4 chars per token (similar to GPT models)
        self._estimator = EstimatingTokenCounter(chars_per_token=4.0)

        if not self._use_api and not _FALLBACK_WARNING_SHOWN:
            warnings.warn(
                "GoogleProvider: No client provided, using estimation. "
                "For accurate counting, pass google.generativeai: "
                "GoogleProvider(client=genai)",
                UserWarning,
                stacklevel=4,
            )
            _FALLBACK_WARNING_SHOWN = True

    def _get_model(self):
        """Lazy-load the GenerativeModel for API calls."""
        if self._genai_model is None and self._client is not None:
            self._genai_model = self._client.GenerativeModel(self.model)
        return self._genai_model

    def count_text(self, text: str) -> int:
        """Count tokens in text.

        Uses countTokens API if client available, otherwise estimates.
        """
        if not text:
            return 0

        if self._use_api:
            try:
                model = self._get_model()
                response = model.count_tokens(text)
                return response.total_tokens
            except Exception as e:
                logger.debug(f"Google countTokens API failed: {e}, using estimation")

        return self._estimator.count_text(text)

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a message."""
        # For API-based counting, convert message to content and count
        if self._use_api:
            try:
                content = self._message_to_content(message)
                model = self._get_model()
                response = model.count_tokens(content)
                return response.total_tokens
            except Exception as e:
                logger.debug(f"Google countTokens API failed: {e}, using estimation")

        # Fallback to estimation
        return self._estimate_message(message)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in messages.

        Uses countTokens API with full conversation if available.
        """
        if not messages:
            return 0

        if self._use_api:
            try:
                # Convert to Gemini content format
                contents = [self._message_to_content(msg) for msg in messages]
                model = self._get_model()
                response = model.count_tokens(contents)
                return response.total_tokens
            except Exception as e:
                logger.debug(f"Google countTokens API failed: {e}, using estimation")

        # Fallback to estimation
        total = sum(self._estimate_message(msg) for msg in messages)
        total += 3  # Priming tokens
        return total

    def _message_to_content(self, message: dict[str, Any]) -> str:
        """Convert OpenAI-format message to text content for counting."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "\n".join(parts)
        return str(content)

    def _estimate_message(self, message: dict[str, Any]) -> int:
        """Estimate tokens in a message without API."""
        tokens = 4  # Message overhead

        role = message.get("role", "")
        tokens += self._estimator.count_text(role)

        content = message.get("content")
        if content:
            if isinstance(content, str):
                tokens += self._estimator.count_text(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            tokens += self._estimator.count_text(part.get("text", ""))
                    elif isinstance(part, str):
                        tokens += self._estimator.count_text(part)

        return tokens


class GoogleProvider(Provider):
    """Provider for Google Gemini models.

    Supports Gemini 1.5 and 2.0 model families through:
    - OpenAI-compatible endpoint (generativelanguage.googleapis.com)
    - Native Google AI SDK (for accurate token counting)

    Example:
        import google.generativeai as genai
        genai.configure(api_key="...")

        # With client (accurate token counting via API)
        provider = GoogleProvider(client=genai)

        # Without client (estimation-based counting)
        provider = GoogleProvider()

        # Token counting
        counter = provider.get_token_counter("gemini-2.0-flash")
        tokens = counter.count_text("Hello, world!")

        # Context limits
        limit = provider.get_context_limit("gemini-1.5-pro")  # 2M tokens!

        # Cost estimation
        cost = provider.estimate_cost(
            input_tokens=100000,
            output_tokens=10000,
            model="gemini-1.5-pro",
        )
    """

    # OpenAI-compatible endpoint for Gemini
    OPENAI_COMPATIBLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

    def __init__(self, client: Any = None):
        """Initialize Google provider.

        Args:
            client: Optional google.generativeai module for API-based token counting.
                    If provided, uses countTokens API for accurate counts.
        """
        self._client = client

    @property
    def name(self) -> str:
        return "google"

    def supports_model(self, model: str) -> bool:
        """Check if model is a known Gemini model."""
        model_lower = model.lower()
        if model_lower in _CONTEXT_LIMITS:
            return True
        # Check prefix match
        for prefix in ["gemini-2", "gemini-1.5", "gemini-1.0", "gemini-pro"]:
            if model_lower.startswith(prefix):
                return True
        return False

    def get_token_counter(self, model: str) -> TokenCounter:
        """Get token counter for a Gemini model.

        Uses countTokens API if client was provided, otherwise estimates.
        """
        if not self.supports_model(model):
            raise ValueError(
                f"Model '{model}' is not recognized as a Google model. "
                f"Supported models: {list(_CONTEXT_LIMITS.keys())}"
            )
        return GeminiTokenCounter(model, client=self._client)

    def get_context_limit(self, model: str) -> int:
        """Get context limit for a Gemini model.

        Tries LiteLLM first for up-to-date limits, falls back to hardcoded values.
        Note: Gemini 1.5 Pro has 2M token context!
        """
        model_lower = model.lower()

        # Try LiteLLM first for up-to-date context limits
        if LITELLM_AVAILABLE and litellm_get_model_info is not None:
            # Try different model name formats that LiteLLM might recognize
            model_variants = [
                f"gemini/{model_lower}",  # gemini/gemini-1.5-pro
                model_lower,  # gemini-1.5-pro
            ]
            for variant in model_variants:
                try:
                    info = litellm_get_model_info(variant)
                    if info:
                        if "max_input_tokens" in info and info["max_input_tokens"]:
                            return info["max_input_tokens"]
                        if "max_tokens" in info and info["max_tokens"]:
                            return info["max_tokens"]
                except Exception:
                    continue

        # Fallback to hardcoded limits
        # Direct match
        if model_lower in _CONTEXT_LIMITS:
            return _CONTEXT_LIMITS[model_lower]

        # Prefix match
        for prefix, limit in [
            ("gemini-2.0", 1000000),
            ("gemini-1.5-pro", 2000000),
            ("gemini-1.5-flash", 1000000),
            ("gemini-1.0", 32768),
            ("gemini-pro", 32768),
        ]:
            if model_lower.startswith(prefix):
                return limit

        raise ValueError(
            f"Unknown context limit for model '{model}'. "
            f"Known models: {list(_CONTEXT_LIMITS.keys())}"
        )

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate cost for Gemini API call.

        Tries LiteLLM first for up-to-date pricing, falls back to hardcoded values.

        Note: Google has tiered pricing based on context length.
        This uses the standard pricing (up to 128K context).
        For >128K context, actual costs may be higher.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.
            cached_tokens: Number of cached tokens (not used by Google).

        Returns:
            Estimated cost in USD, or None if pricing unknown.
        """
        model_lower = model.lower()

        # Try LiteLLM first for up-to-date pricing
        if LITELLM_AVAILABLE and litellm is not None:
            # Try different model name formats that LiteLLM might recognize
            model_variants = [
                f"gemini/{model_lower}",  # gemini/gemini-1.5-pro
                model_lower,  # gemini-1.5-pro
            ]
            for variant in model_variants:
                try:
                    cost = litellm.completion_cost(
                        model=variant,
                        prompt="",
                        completion="",
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                    )
                    if cost is not None:
                        return cost
                except Exception:
                    continue

        # Fallback to hardcoded pricing
        input_price, output_price = None, None
        for model_prefix, (inp, outp) in _PRICING.items():
            if model_lower.startswith(model_prefix):
                input_price, output_price = inp, outp
                break

        if input_price is None:
            return None

        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * (output_price or 0)

        return input_cost + output_cost

    def get_output_buffer(self, model: str, default: int = 4000) -> int:
        """Get recommended output buffer."""
        # Gemini models can output up to 8K tokens
        return min(8192, default)

    @classmethod
    def get_openai_compatible_url(cls, api_key: str) -> str:
        """Get OpenAI-compatible endpoint URL.

        Use this with the OpenAI client:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url=GoogleProvider.get_openai_compatible_url(api_key),
            )

        Args:
            api_key: Google AI API key.

        Returns:
            Base URL for OpenAI-compatible requests.
        """
        return cls.OPENAI_COMPATIBLE_BASE_URL

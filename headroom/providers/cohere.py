"""Cohere provider for Headroom SDK.

Token counting uses Cohere's official tokenize API when a client
is provided. This gives accurate counts for all content types.

Usage:
    import cohere
    from headroom import CohereProvider

    client = cohere.ClientV2()  # Uses CO_API_KEY env var
    provider = CohereProvider(client=client)  # Accurate counting via API

    # Or without client (uses estimation - less accurate)
    provider = CohereProvider()  # Warning: approximate counting
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from typing import Any

from headroom.tokenizers import EstimatingTokenCounter

from .base import Provider, TokenCounter

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Warning flags
_FALLBACK_WARNING_SHOWN = False

# Pricing metadata
_PRICING_LAST_UPDATED = date(2025, 1, 6)

# Cohere model context limits
_CONTEXT_LIMITS: dict[str, int] = {
    # Command A (latest, 2025)
    "command-a-03-2025": 256000,
    "command-a": 256000,
    # Command R+ (2024)
    "command-r-plus-08-2024": 128000,
    "command-r-plus": 128000,
    # Command R (2024)
    "command-r-08-2024": 128000,
    "command-r": 128000,
    # Command (legacy)
    "command": 4096,
    "command-light": 4096,
    "command-nightly": 128000,
    # Embed models
    "embed-english-v3.0": 512,
    "embed-multilingual-v3.0": 512,
    "embed-english-light-v3.0": 512,
    "embed-multilingual-light-v3.0": 512,
}

# Fallback pricing - LiteLLM is preferred source
# Pricing per 1M tokens (input, output)
_PRICING: dict[str, tuple[float, float]] = {
    "command-a-03-2025": (2.50, 10.00),
    "command-a": (2.50, 10.00),
    "command-r-plus-08-2024": (2.50, 10.00),
    "command-r-plus": (2.50, 10.00),
    "command-r-08-2024": (0.15, 0.60),
    "command-r": (0.15, 0.60),
    "command": (1.00, 2.00),
    "command-light": (0.30, 0.60),
}


class CohereTokenCounter:
    """Token counter for Cohere models.

    When a Cohere client is provided, uses the official tokenize API
    for accurate counting. Falls back to estimation when no client
    is available.

    Usage:
        import cohere
        client = cohere.ClientV2()

        # With API (accurate)
        counter = CohereTokenCounter("command-r-plus", client=client)

        # Without API (estimation)
        counter = CohereTokenCounter("command-r-plus")
    """

    def __init__(self, model: str, client: Any = None):
        """Initialize Cohere token counter.

        Args:
            model: Cohere model name.
            client: Optional cohere.ClientV2 for API-based counting.
        """
        global _FALLBACK_WARNING_SHOWN

        self.model = model
        self._client = client
        self._use_api = client is not None

        # Cohere uses ~4 chars per token
        self._estimator = EstimatingTokenCounter(chars_per_token=4.0)

        if not self._use_api and not _FALLBACK_WARNING_SHOWN:
            warnings.warn(
                "CohereProvider: No client provided, using estimation. "
                "For accurate counting, pass a Cohere client: "
                "CohereProvider(client=cohere.ClientV2())",
                UserWarning,
                stacklevel=4,
            )
            _FALLBACK_WARNING_SHOWN = True

    def count_text(self, text: str) -> int:
        """Count tokens in text.

        Uses tokenize API if client available, otherwise estimates.
        """
        if not text:
            return 0

        if self._use_api:
            try:
                response = self._client.tokenize(
                    text=text,
                    model=self.model,
                )
                return len(response.tokens)
            except Exception as e:
                logger.debug(f"Cohere tokenize API failed: {e}, using estimation")

        return self._estimator.count_text(text)

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a message."""
        content = self._extract_content(message)
        tokens = self.count_text(content)
        tokens += 4  # Message overhead (role tokens, etc.)
        return tokens

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in messages."""
        if not messages:
            return 0

        # For API-based counting, concatenate all content
        if self._use_api:
            try:
                all_content = []
                for msg in messages:
                    content = self._extract_content(msg)
                    role = msg.get("role", "user")
                    all_content.append(f"{role}: {content}")

                full_text = "\n".join(all_content)
                response = self._client.tokenize(
                    text=full_text,
                    model=self.model,
                )
                return len(response.tokens)
            except Exception as e:
                logger.debug(f"Cohere tokenize API failed: {e}, using estimation")

        # Fallback to estimation
        total = sum(self.count_message(msg) for msg in messages)
        total += 3  # Priming tokens
        return total

    def _extract_content(self, message: dict[str, Any]) -> str:
        """Extract text content from message."""
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


class CohereProvider(Provider):
    """Provider for Cohere Command models.

    Supports Command R, Command R+, and Command A model families.

    Example:
        import cohere
        client = cohere.ClientV2()

        # With client (accurate token counting via API)
        provider = CohereProvider(client=client)

        # Without client (estimation-based counting)
        provider = CohereProvider()

        # Token counting
        counter = provider.get_token_counter("command-r-plus")
        tokens = counter.count_text("Hello, world!")

        # Context limits
        limit = provider.get_context_limit("command-a")  # 256K tokens

        # Cost estimation
        cost = provider.estimate_cost(
            input_tokens=100000,
            output_tokens=10000,
            model="command-r-plus",
        )
    """

    def __init__(self, client: Any = None):
        """Initialize Cohere provider.

        Args:
            client: Optional cohere.ClientV2 for API-based token counting.
                    If provided, uses tokenize API for accurate counts.
        """
        self._client = client

    @property
    def name(self) -> str:
        return "cohere"

    def supports_model(self, model: str) -> bool:
        """Check if model is a known Cohere model."""
        model_lower = model.lower()
        if model_lower in _CONTEXT_LIMITS:
            return True
        # Check prefix match
        for prefix in ["command-a", "command-r", "command", "embed-"]:
            if model_lower.startswith(prefix):
                return True
        return False

    def get_token_counter(self, model: str) -> TokenCounter:
        """Get token counter for a Cohere model.

        Uses tokenize API if client was provided, otherwise estimates.
        """
        if not self.supports_model(model):
            raise ValueError(
                f"Model '{model}' is not recognized as a Cohere model. "
                f"Supported models: {list(_CONTEXT_LIMITS.keys())}"
            )
        return CohereTokenCounter(model, client=self._client)

    def get_context_limit(self, model: str) -> int:
        """Get context limit for a Cohere model.

        Tries LiteLLM first (with and without 'cohere/' prefix),
        then falls back to built-in limits.
        """
        # Try LiteLLM first
        if LITELLM_AVAILABLE:
            for model_variant in [f"cohere/{model}", model]:
                try:
                    info = litellm.get_model_info(model_variant)
                    if info and "max_input_tokens" in info:
                        result = info["max_input_tokens"]
                        if result is not None:
                            return int(result)
                    if info and "max_tokens" in info:
                        result = info["max_tokens"]
                        if result is not None:
                            return int(result)
                except Exception:
                    pass

        # Fallback to built-in limits
        model_lower = model.lower()

        # Direct match
        if model_lower in _CONTEXT_LIMITS:
            return _CONTEXT_LIMITS[model_lower]

        # Prefix match
        for prefix, limit in [
            ("command-a", 256000),
            ("command-r-plus", 128000),
            ("command-r", 128000),
            ("command", 4096),
            ("embed-", 512),
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
        """Estimate cost for Cohere API call.

        Tries LiteLLM first (with and without 'cohere/' prefix),
        then falls back to built-in pricing.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.
            cached_tokens: Not used by Cohere.

        Returns:
            Estimated cost in USD, or None if pricing unknown.
        """
        # Try LiteLLM first
        if LITELLM_AVAILABLE:
            for model_variant in [f"cohere/{model}", model]:
                try:
                    cost = litellm.completion_cost(
                        model=model_variant,
                        prompt="",
                        completion="",
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                    )
                    if cost is not None:
                        return float(cost)
                except Exception:
                    pass

        # Fallback to built-in pricing
        model_lower = model.lower()

        # Find pricing
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
        return default

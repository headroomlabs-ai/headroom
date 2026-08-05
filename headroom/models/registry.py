"""Model registry with capabilities database.

Centralized database of LLM models with their capabilities, context limits,
and provider information. Supports dynamic registration of custom models
and automatic provider detection.

Pricing is fetched dynamically from LiteLLM's community-maintained database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from headroom.pricing.litellm_pricing import estimate_cost as litellm_estimate_cost
from headroom.pricing.litellm_pricing import get_model_pricing


@dataclass(frozen=True)
class ModelInfo:
    """Information about an LLM model.

    Attributes:
        name: Model identifier.
        provider: Provider name (openai, anthropic, etc.).
        context_window: Maximum context window in tokens.
        max_output_tokens: Maximum output tokens.
        supports_tools: Whether model supports tool/function calling.
        supports_vision: Whether model supports image inputs.
        supports_streaming: Whether model supports streaming responses.
        supports_json_mode: Whether model supports JSON output mode.
        tokenizer_backend: Tokenizer backend to use.
        aliases: Alternative names for the model.
        notes: Additional notes about the model.

    Note:
        Pricing is fetched dynamically from LiteLLM's database.
        Use ModelRegistry.estimate_cost() to get current pricing.
    """

    name: str
    provider: str
    context_window: int = 128000
    max_output_tokens: int = 4096
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = True
    tokenizer_backend: str | None = None
    aliases: tuple[str, ...] = ()
    notes: str = ""


# Built-in model database
# Pricing as of January 2025 - verify current rates
_MODELS: dict[str, ModelInfo] = {}


def _register_builtin_models() -> None:
    """Register built-in models.

    Note: Pricing is fetched dynamically from LiteLLM's database.
    """

    # ============================================================
    # OpenAI Models
    # ============================================================

    # GPT-4o family
    _MODELS["gpt-4o"] = ModelInfo(
        name="gpt-4o",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="tiktoken",
        aliases=("gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13"),
        notes="Latest GPT-4o with vision and tools",
    )

    _MODELS["gpt-4o-mini"] = ModelInfo(
        name="gpt-4o-mini",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="tiktoken",
        aliases=("gpt-4o-mini-2024-07-18",),
        notes="Cost-effective GPT-4o variant",
    )

    # o1 reasoning models
    _MODELS["o1"] = ModelInfo(
        name="o1",
        provider="openai",
        context_window=200000,
        max_output_tokens=100000,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="tiktoken",
        notes="Full reasoning model with extended thinking",
    )

    _MODELS["o1-mini"] = ModelInfo(
        name="o1-mini",
        provider="openai",
        context_window=128000,
        max_output_tokens=65536,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="tiktoken",
        notes="Fast reasoning model",
    )

    _MODELS["o3-mini"] = ModelInfo(
        name="o3-mini",
        provider="openai",
        context_window=200000,
        max_output_tokens=100000,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="tiktoken",
        notes="Latest reasoning model",
    )

    # GPT-4 Turbo
    _MODELS["gpt-4-turbo"] = ModelInfo(
        name="gpt-4-turbo",
        provider="openai",
        context_window=128000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="tiktoken",
        aliases=("gpt-4-turbo-preview", "gpt-4-turbo-2024-04-09"),
        notes="GPT-4 Turbo with vision",
    )

    # GPT-4
    _MODELS["gpt-4"] = ModelInfo(
        name="gpt-4",
        provider="openai",
        context_window=8192,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="tiktoken",
        aliases=("gpt-4-0613",),
        notes="Original GPT-4",
    )

    _MODELS["gpt-4-32k"] = ModelInfo(
        name="gpt-4-32k",
        provider="openai",
        context_window=32768,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="tiktoken",
        notes="Extended context GPT-4",
    )

    # GPT-3.5
    _MODELS["gpt-3.5-turbo"] = ModelInfo(
        name="gpt-3.5-turbo",
        provider="openai",
        context_window=16385,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="tiktoken",
        aliases=("gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"),
        notes="Fast and cost-effective",
    )

    # ============================================================
    # Anthropic Models
    # ============================================================

    _MODELS["claude-3-5-sonnet-20241022"] = ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="anthropic",
        aliases=("claude-3-5-sonnet-latest", "claude-sonnet-4-20250514"),
        notes="Claude 3.5 Sonnet - Best balance of speed and capability",
    )

    _MODELS["claude-3-5-haiku-20241022"] = ModelInfo(
        name="claude-3-5-haiku-20241022",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="anthropic",
        aliases=("claude-3-5-haiku-latest",),
        notes="Claude 3.5 Haiku - Fast and cost-effective",
    )

    _MODELS["claude-3-opus-20240229"] = ModelInfo(
        name="claude-3-opus-20240229",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="anthropic",
        aliases=("claude-3-opus-latest",),
        notes="Claude 3 Opus - Most capable",
    )

    _MODELS["claude-3-haiku-20240307"] = ModelInfo(
        name="claude-3-haiku-20240307",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="anthropic",
        notes="Claude 3 Haiku - Legacy fast model",
    )

    # ============================================================
    # Google Models
    # ============================================================

    _MODELS["gemini-2.0-flash"] = ModelInfo(
        name="gemini-2.0-flash",
        provider="google",
        context_window=1000000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="google",
        aliases=("gemini-2.0-flash-exp",),
        notes="Gemini 2.0 Flash - Fast multimodal",
    )

    _MODELS["gemini-1.5-pro"] = ModelInfo(
        name="gemini-1.5-pro",
        provider="google",
        context_window=2000000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="google",
        aliases=("gemini-1.5-pro-latest",),
        notes="Gemini 1.5 Pro - 2M context window",
    )

    _MODELS["gemini-1.5-flash"] = ModelInfo(
        name="gemini-1.5-flash",
        provider="google",
        context_window=1000000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        tokenizer_backend="google",
        aliases=("gemini-1.5-flash-latest",),
        notes="Gemini 1.5 Flash - Cost-effective",
    )

    # ============================================================
    # Meta Llama Models (open source)
    # ============================================================

    _MODELS["llama-3.3-70b"] = ModelInfo(
        name="llama-3.3-70b",
        provider="meta",
        context_window=128000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        aliases=("llama-3.3-70b-instruct", "meta-llama/Llama-3.3-70B-Instruct"),
        notes="Llama 3.3 70B - Open source",
    )

    _MODELS["llama-3.1-405b"] = ModelInfo(
        name="llama-3.1-405b",
        provider="meta",
        context_window=128000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        aliases=("llama-3.1-405b-instruct", "meta-llama/Llama-3.1-405B-Instruct"),
        notes="Llama 3.1 405B - Largest open source",
    )

    _MODELS["llama-3.1-70b"] = ModelInfo(
        name="llama-3.1-70b",
        provider="meta",
        context_window=128000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        aliases=("llama-3.1-70b-instruct", "meta-llama/Llama-3.1-70B-Instruct"),
        notes="Llama 3.1 70B",
    )

    _MODELS["llama-3.1-8b"] = ModelInfo(
        name="llama-3.1-8b",
        provider="meta",
        context_window=128000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        aliases=("llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        notes="Llama 3.1 8B - Fast and efficient",
    )

    # ============================================================
    # Mistral Models
    # ============================================================

    _MODELS["mistral-large"] = ModelInfo(
        name="mistral-large",
        provider="mistral",
        context_window=128000,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        aliases=("mistral-large-latest",),
        notes="Mistral Large - Best capability",
    )

    _MODELS["mistral-small"] = ModelInfo(
        name="mistral-small",
        provider="mistral",
        context_window=32768,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        aliases=("mistral-small-latest",),
        notes="Mistral Small - Cost-effective",
    )

    _MODELS["mixtral-8x7b"] = ModelInfo(
        name="mixtral-8x7b",
        provider="mistral",
        context_window=32768,
        max_output_tokens=4096,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        aliases=("mixtral-8x7b-instruct",),
        notes="Mixtral 8x7B - MoE architecture",
    )

    _MODELS["mistral-7b"] = ModelInfo(
        name="mistral-7b",
        provider="mistral",
        context_window=32768,
        max_output_tokens=4096,
        supports_tools=False,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        aliases=("mistral-7b-instruct",),
        notes="Mistral 7B - Open source",
    )

    # ============================================================
    # DeepSeek Models
    # ============================================================

    _MODELS["deepseek-v3"] = ModelInfo(
        name="deepseek-v3",
        provider="deepseek",
        context_window=128000,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        notes="DeepSeek V3 - High performance, low cost",
    )

    _MODELS["deepseek-coder"] = ModelInfo(
        name="deepseek-coder",
        provider="deepseek",
        context_window=16384,
        max_output_tokens=4096,
        supports_tools=False,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        notes="DeepSeek Coder - Specialized for code",
    )

    _MODELS["deepseek-v4-flash"] = ModelInfo(
        name="deepseek-v4-flash",
        provider="deepseek",
        context_window=1_000_000,
        max_output_tokens=384_000,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        notes="DeepSeek V4 Flash - 13B active params; non-thinking + thinking modes",
    )

    _MODELS["deepseek-v4-pro"] = ModelInfo(
        name="deepseek-v4-pro",
        provider="deepseek",
        context_window=1_000_000,
        max_output_tokens=384_000,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        notes="DeepSeek V4 Pro - 49B active params",
    )

    # ============================================================
    # Qwen Models
    # ============================================================

    _MODELS["qwen2.5-72b"] = ModelInfo(
        name="qwen2.5-72b",
        provider="alibaba",
        context_window=131072,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        aliases=("qwen2.5-72b-instruct",),
        notes="Qwen 2.5 72B - Strong multilingual",
    )

    _MODELS["qwen2.5-7b"] = ModelInfo(
        name="qwen2.5-7b",
        provider="alibaba",
        context_window=131072,
        max_output_tokens=8192,
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        tokenizer_backend="huggingface",
        aliases=("qwen2.5-7b-instruct",),
        notes="Qwen 2.5 7B - Efficient",
    )


# Initialize built-in models
_register_builtin_models()

# Build alias lookup
_ALIASES: dict[str, str] = {}
for model_name, info in _MODELS.items():
    for alias in info.aliases:
        _ALIASES[alias.lower()] = model_name


_PROVIDER_TOKENIZER_BACKENDS = {
    "anthropic": "anthropic",
    "google": "google",
    "openai": "tiktoken",
}

_PROVIDER_FAMILY_DEFAULTS: dict[str, tuple[tuple[str, dict[str, Any]], ...]] = {
    "google": (
        (
            "gemini-1.0",
            {
                "context_window": 32768,
                "max_output_tokens": 4096,
                "supports_vision": False,
                "tokenizer_backend": "google",
                "notes": "Google Gemini 1.0 family fallback",
            },
        ),
        (
            "gemini-pro",
            {
                "context_window": 32768,
                "max_output_tokens": 4096,
                "supports_vision": False,
                "tokenizer_backend": "google",
                "notes": "Legacy Google Gemini Pro fallback",
            },
        ),
        (
            "gemini-1.5-pro",
            {
                "context_window": 2000000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "tokenizer_backend": "google",
                "notes": "Google Gemini 1.5 Pro family fallback",
            },
        ),
        (
            "gemini-1.5-flash",
            {
                "context_window": 1000000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "tokenizer_backend": "google",
                "notes": "Google Gemini 1.5 Flash family fallback",
            },
        ),
        (
            "gemini-2",
            {
                "context_window": 1000000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "tokenizer_backend": "google",
                "notes": "Google Gemini 2 family fallback",
            },
        ),
        (
            "gemini-",
            {
                "context_window": 1000000,
                "max_output_tokens": 8192,
                "supports_vision": True,
                "tokenizer_backend": "google",
                "notes": "Future Google Gemini family fallback",
            },
        ),
    ),
}


def _infer_provider(model: str) -> str | None:
    """Infer a provider from common model id prefixes."""
    model_lower = model.lower()
    if model_lower.startswith(("gemini-", "gemini/gemini-", "google/gemini-")):
        return "google"
    if model_lower.startswith(("claude", "anthropic/claude")):
        return "anthropic"
    if model_lower.startswith(("gpt-", "o1", "o3", "o4", "openai/gpt-")):
        return "openai"
    return None


def _unprefixed_model_id(model: str) -> str:
    """Drop a common LiteLLM provider prefix before family matching."""
    model_lower = model.lower()
    for prefix in ("anthropic/", "gemini/", "google/", "openai/"):
        if model_lower.startswith(prefix):
            return model_lower[len(prefix) :]
    return model_lower


def _family_fallback(model: str, provider: str) -> ModelInfo | None:
    """Return a provider-scoped family fallback for plausible future models."""
    model_lower = _unprefixed_model_id(model)
    for prefix, defaults in _PROVIDER_FAMILY_DEFAULTS.get(provider, ()):
        if model_lower.startswith(prefix):
            return ModelInfo(name=model, provider=provider, **defaults)
    return None


class ModelRegistry:
    """Registry of LLM models and their capabilities.

    Singleton registry providing access to model information.
    Supports built-in models and custom registration.

    Example:
        # Get model info
        info = ModelRegistry.get("gpt-4o")
        print(f"Context: {info.context_window}")

        # Register custom model
        ModelRegistry.register(
            "my-model",
            provider="custom",
            context_window=32000,
        )

        # List models by provider
        openai_models = ModelRegistry.list_models(provider="openai")
    """

    @classmethod
    def get(cls, model: str) -> ModelInfo | None:
        """Get model information.

        Args:
            model: Model name or alias.

        Returns:
            ModelInfo if found, None otherwise.
        """
        model_lower = model.lower()

        # Direct lookup
        if model_lower in _MODELS:
            return _MODELS[model_lower]

        # Alias lookup
        if model_lower in _ALIASES:
            return _MODELS[_ALIASES[model_lower]]

        # Prefix matching. Two rules keep this from silently mis-resolving
        # newer or larger variants:
        #   1. The registered name must end at a version boundary in the query
        #      (next char is a separator like ``-``/``/``/``:``/``@``/``_``), so
        #      ``gpt-4.1`` — a distinct model, not a variant of ``gpt-4`` — does
        #      not inherit gpt-4's 8192-token window (the ``.`` is not a
        #      boundary, so it no longer matches).
        #   2. When several names qualify, the LONGEST wins, so
        #      ``gpt-4-32k-0613`` resolves to ``gpt-4-32k`` (32768) rather than
        #      the shorter ``gpt-4`` (8192) that happens to be registered first.
        best: ModelInfo | None = None
        best_len = -1
        for name, info in _MODELS.items():
            if not model_lower.startswith(name):
                continue
            rest = model_lower[len(name) :]
            if rest and rest[0] not in ("-", "/", ":", "@", "_"):
                continue
            if len(name) > best_len:
                best = info
                best_len = len(name)

        return best

    @classmethod
    def resolve(
        cls,
        model: str,
        provider: str | None = None,
        default_context_window: int = 128000,
    ) -> ModelInfo | None:
        """Resolve model capabilities for a runtime provider path.

        This is the tolerant runtime counterpart to :meth:`get`: it first
        checks the built-in registry, then LiteLLM metadata, then
        provider-scoped family fallbacks. Unknown models for unrelated
        providers still return ``None`` instead of being claimed globally.

        Args:
            model: Model identifier from a request/provider.
            provider: Optional provider hint (for example ``"google"``).
            default_context_window: Conservative context window when a
                plausible family fallback has no exact catalog hit.

        Returns:
            Resolved ModelInfo when the model belongs to the hinted or
            inferred provider, otherwise None.
        """
        provider_hint = provider.lower() if provider else None

        info = cls.get(model)
        if info is not None and (provider_hint is None or info.provider == provider_hint):
            return info

        inferred_provider = _infer_provider(model)
        if (
            provider_hint is not None
            and inferred_provider is not None
            and inferred_provider != provider_hint
        ):
            return None

        resolved_provider = provider_hint or inferred_provider
        if resolved_provider is None:
            return None

        fallback = _family_fallback(model, resolved_provider)
        if provider_hint is not None and inferred_provider is None and fallback is None:
            return None

        pricing = get_model_pricing(model)
        if pricing is not None:
            context_window = (
                pricing.max_input_tokens or pricing.max_tokens or default_context_window
            )
            max_output_tokens = pricing.max_output_tokens or 4096
            return ModelInfo(
                name=model,
                provider=resolved_provider,
                context_window=int(context_window),
                max_output_tokens=int(max_output_tokens),
                supports_vision=pricing.supports_vision,
                tokenizer_backend=_PROVIDER_TOKENIZER_BACKENDS.get(resolved_provider),
                notes="Resolved from LiteLLM pricing metadata",
            )

        if fallback is not None:
            return fallback

        return None

    @classmethod
    def register(
        cls,
        model: str,
        provider: str,
        context_window: int = 128000,
        **kwargs: Any,
    ) -> ModelInfo:
        """Register a custom model.

        Args:
            model: Model name.
            provider: Provider name.
            context_window: Maximum context window.
            **kwargs: Additional ModelInfo fields.

        Returns:
            Registered ModelInfo.
        """
        info = ModelInfo(
            name=model,
            provider=provider,
            context_window=context_window,
            **kwargs,
        )
        _MODELS[model.lower()] = info

        # Register aliases
        for alias in info.aliases:
            _ALIASES[alias.lower()] = model.lower()

        return info

    @classmethod
    def list_models(
        cls,
        provider: str | None = None,
        supports_tools: bool | None = None,
        supports_vision: bool | None = None,
        min_context: int | None = None,
    ) -> list[ModelInfo]:
        """List models matching criteria.

        Args:
            provider: Filter by provider.
            supports_tools: Filter by tool support.
            supports_vision: Filter by vision support.
            min_context: Minimum context window.

        Returns:
            List of matching ModelInfo.
        """
        results = []
        for info in _MODELS.values():
            if provider and info.provider != provider:
                continue
            if supports_tools is not None and info.supports_tools != supports_tools:
                continue
            if supports_vision is not None and info.supports_vision != supports_vision:
                continue
            if min_context and info.context_window < min_context:
                continue
            results.append(info)
        return results

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all known providers.

        Returns:
            List of provider names.
        """
        return list({info.provider for info in _MODELS.values()})

    @classmethod
    def get_context_limit(cls, model: str, default: int = 128000) -> int:
        """Get context limit for a model.

        Args:
            model: Model name.
            default: Default if model not found.

        Returns:
            Context window size.
        """
        info = cls.get(model)
        return info.context_window if info else default

    @classmethod
    def estimate_cost(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate API cost for a model using LiteLLM's pricing database.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cached_tokens: Number of cached input tokens (not currently used).

        Returns:
            Estimated cost in USD, or None if pricing unknown.
        """
        # Use LiteLLM's pricing database
        return litellm_estimate_cost(model, input_tokens, output_tokens)

    @classmethod
    def get_pricing(cls, model: str) -> tuple[float, float] | None:
        """Get pricing for a model from LiteLLM's database.

        Args:
            model: Model name.

        Returns:
            Tuple of (input_cost_per_1m, output_cost_per_1m) or None if not found.
        """
        pricing = get_model_pricing(model)
        if pricing is None:
            return None
        return (pricing.input_cost_per_1m, pricing.output_cost_per_1m)


# Convenience functions
def get_model_info(model: str) -> ModelInfo | None:
    """Get information about a model.

    Args:
        model: Model name or alias.

    Returns:
        ModelInfo if found, None otherwise.
    """
    return ModelRegistry.get(model)


def list_models(
    provider: str | None = None,
    **kwargs: Any,
) -> list[ModelInfo]:
    """List models matching criteria.

    Args:
        provider: Filter by provider.
        **kwargs: Additional filter criteria.

    Returns:
        List of matching ModelInfo.
    """
    return ModelRegistry.list_models(provider=provider, **kwargs)


def register_model(
    model: str,
    provider: str,
    context_window: int = 128000,
    **kwargs: Any,
) -> ModelInfo:
    """Register a custom model.

    Args:
        model: Model name.
        provider: Provider name.
        context_window: Maximum context window.
        **kwargs: Additional ModelInfo fields.

    Returns:
        Registered ModelInfo.
    """
    return ModelRegistry.register(model, provider, context_window, **kwargs)

"""Provider-owned Agno model detection policies."""

from __future__ import annotations

import logging
from typing import Any

from headroom.providers import (
    AnthropicProvider,
    CohereProvider,
    GoogleProvider,
    OpenAIProvider,
)
from headroom.providers.base import Provider

logger = logging.getLogger(__name__)

_AGNO_MODEL_PROVIDERS: dict[str, type[Provider]] = {
    "OpenAIChat": OpenAIProvider,
    "OpenAILike": OpenAIProvider,
    "Claude": AnthropicProvider,
    "Anthropic": AnthropicProvider,
    "AwsBedrock": AnthropicProvider,
    "BedrockClaude": AnthropicProvider,
    "Gemini": GoogleProvider,
    "GoogleGenerativeAI": GoogleProvider,
    "VertexAI": GoogleProvider,
    "LiteLLM": OpenAIProvider,
    "LiteLLMChat": OpenAIProvider,
    "Groq": OpenAIProvider,
    "Mistral": OpenAIProvider,
    "MistralChat": OpenAIProvider,
    "Together": OpenAIProvider,
    "TogetherChat": OpenAIProvider,
    "Fireworks": OpenAIProvider,
    "FireworksChat": OpenAIProvider,
    "Ollama": OpenAIProvider,
    "OllamaChat": OpenAIProvider,
    "DeepSeek": OpenAIProvider,
    "DeepSeekChat": OpenAIProvider,
    "xAI": OpenAIProvider,
    "XAI": OpenAIProvider,
    "Grok": OpenAIProvider,
    "Cohere": CohereProvider,
    "CohereChat": CohereProvider,
    "Perplexity": OpenAIProvider,
    "Anyscale": OpenAIProvider,
    "OpenRouter": OpenAIProvider,
    "Replicate": OpenAIProvider,
    "HuggingFace": OpenAIProvider,
    "HuggingFaceChat": OpenAIProvider,
}

_MODULE_PROVIDER_HINTS: tuple[tuple[tuple[str, ...], type[Provider]], ...] = (
    (("anthropic",), AnthropicProvider),
    (("google", "gemini"), GoogleProvider),
    (("cohere",), CohereProvider),
    (("openai", "litellm"), OpenAIProvider),
)

_MODEL_ID_PROVIDER_HINTS: tuple[tuple[tuple[str, ...], type[Provider]], ...] = (
    (("claude",), AnthropicProvider),
    (("gemini",), GoogleProvider),
    (("gpt", "o1", "o3"), OpenAIProvider),
    (("command", "cohere"), CohereProvider),
)


def create_provider(agno_model: Any) -> Provider:
    """Create the appropriate Headroom provider for an Agno model."""
    class_name = agno_model.__class__.__name__
    if class_name in _AGNO_MODEL_PROVIDERS:
        provider_class = _AGNO_MODEL_PROVIDERS[class_name]
        logger.debug("Detected provider %s from class %s", provider_class.__name__, class_name)
        return provider_class()

    module_path = agno_model.__class__.__module__
    module_path_lower = module_path.lower()
    for hints, provider_class in _MODULE_PROVIDER_HINTS:
        if any(hint in module_path_lower for hint in hints):
            logger.debug(
                "Detected provider %s from module %s", provider_class.__name__, module_path
            )
            return provider_class()

    model_id = extract_model_id(agno_model)
    if model_id:
        model_id_lower = model_id.lower()
        for hints, provider_class in _MODEL_ID_PROVIDER_HINTS:
            if any(hint in model_id_lower for hint in hints):
                logger.debug(
                    "Detected provider %s from model ID %s",
                    provider_class.__name__,
                    model_id,
                )
                return provider_class()

    logger.warning(
        "Unknown Agno model class '%s', defaulting to OpenAIProvider. "
        "Token counting may be inaccurate.",
        class_name,
    )
    return OpenAIProvider()


def extract_model_id(agno_model: Any) -> str:
    """Extract model ID from an Agno model."""
    for attr in ["id", "model", "model_name", "model_id"]:
        value = getattr(agno_model, attr, None)
        if value and isinstance(value, str):
            return str(value)

    return ""


def default_model_name(agno_model: Any) -> str:
    """Return an Agno model name or provider-owned default."""
    model_id = extract_model_id(agno_model)
    if model_id:
        return model_id

    class_name = agno_model.__class__.__name__
    logger.warning(
        "Could not extract model name from %s (no 'id', 'model', "
        "'model_name', or 'model_id' attribute). Defaulting to 'gpt-4o'. "
        "Token counting may be inaccurate.",
        class_name,
    )
    return "gpt-4o"

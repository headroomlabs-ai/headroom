"""Provider detection for Agno models.

Automatically detects the correct Headroom provider based on the Agno model type.
"""

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

# Mapping from Agno model class names to Headroom providers
_AGNO_MODEL_PROVIDERS: dict[str, type[Provider]] = {
    # OpenAI models
    "OpenAIChat": OpenAIProvider,
    "OpenAILike": OpenAIProvider,
    # Anthropic models (direct and cloud variants)
    "Claude": AnthropicProvider,
    "Anthropic": AnthropicProvider,
    "AwsBedrock": AnthropicProvider,  # Bedrock Claude models
    "BedrockClaude": AnthropicProvider,
    # Google models
    "Gemini": GoogleProvider,
    "GoogleGenerativeAI": GoogleProvider,
    "VertexAI": GoogleProvider,
    # LiteLLM (uses OpenAI-compatible interface, provider detected from model ID)
    "LiteLLM": OpenAIProvider,
    "LiteLLMChat": OpenAIProvider,
    # Others default to OpenAI-compatible tokenization
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


def get_headroom_provider(agno_model: Any) -> Provider:
    """Get the appropriate Headroom provider for an Agno model.

    Detection strategy:
    1. Check model class name against known Agno model types
    2. Check for provider hints in model attributes
    3. Fall back to OpenAI provider (most compatible)

    Args:
        agno_model: An Agno model instance (OpenAIChat, Claude, etc.)

    Returns:
        Appropriate Headroom Provider instance.

    Example:
        from agno.models.openai import OpenAIChat
        from headroom.integrations.agno.providers import get_headroom_provider

        model = OpenAIChat(id="gpt-4o")
        provider = get_headroom_provider(model)  # Returns OpenAIProvider
    """
    # Strategy 1: Class name matching
    class_name = agno_model.__class__.__name__
    if class_name in _AGNO_MODEL_PROVIDERS:
        provider_class = _AGNO_MODEL_PROVIDERS[class_name]
        logger.debug(f"Detected provider {provider_class.__name__} from class {class_name}")
        return provider_class()

    # Strategy 2: Check module path
    module_path = agno_model.__class__.__module__
    if "anthropic" in module_path.lower():
        logger.debug(f"Detected AnthropicProvider from module {module_path}")
        return AnthropicProvider()
    elif "google" in module_path.lower() or "gemini" in module_path.lower():
        logger.debug(f"Detected GoogleProvider from module {module_path}")
        return GoogleProvider()
    elif "cohere" in module_path.lower():
        logger.debug(f"Detected CohereProvider from module {module_path}")
        return CohereProvider()
    elif "openai" in module_path.lower() or "litellm" in module_path.lower():
        logger.debug(f"Detected OpenAIProvider from module {module_path}")
        return OpenAIProvider()

    # Strategy 3: Check model ID/name for hints
    model_id = getattr(agno_model, "id", "") or getattr(agno_model, "model", "")
    if isinstance(model_id, str) and model_id:
        model_id_lower = model_id.lower()
        if "claude" in model_id_lower:
            logger.debug(f"Detected AnthropicProvider from model ID {model_id}")
            return AnthropicProvider()
        elif "gemini" in model_id_lower:
            logger.debug(f"Detected GoogleProvider from model ID {model_id}")
            return GoogleProvider()
        elif "gpt" in model_id_lower or "o1" in model_id_lower or "o3" in model_id_lower:
            logger.debug(f"Detected OpenAIProvider from model ID {model_id}")
            return OpenAIProvider()
        elif "command" in model_id_lower or "cohere" in model_id_lower:
            logger.debug(f"Detected CohereProvider from model ID {model_id}")
            return CohereProvider()

    # Strategy 4: Default fallback
    logger.warning(
        f"Unknown Agno model class '{class_name}', defaulting to OpenAIProvider. "
        "Token counting may be inaccurate."
    )
    return OpenAIProvider()


def get_model_name_from_agno(agno_model: Any) -> str:
    """Extract the model name/ID from an Agno model.

    Args:
        agno_model: An Agno model instance

    Returns:
        Model name string (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
    """
    # Try common attribute names
    for attr in ["id", "model", "model_name", "model_id"]:
        value = getattr(agno_model, attr, None)
        if value and isinstance(value, str):
            return str(value)

    # Fallback with warning
    class_name = agno_model.__class__.__name__
    logger.warning(
        f"Could not extract model name from {class_name} (no 'id', 'model', "
        f"'model_name', or 'model_id' attribute). Defaulting to 'gpt-4o'. "
        "Token counting may be inaccurate."
    )
    return "gpt-4o"

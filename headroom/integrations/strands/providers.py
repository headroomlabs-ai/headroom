"""Provider detection for Strands models.

Automatically detects the correct Headroom provider based on the Strands model type.
"""

from __future__ import annotations

import logging
from typing import Any

from headroom.providers.anthropic import AnthropicProvider
from headroom.providers.base import Provider
from headroom.providers.google import GoogleProvider
from headroom.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)

# Mapping from Strands model class names to Headroom providers
_STRANDS_MODEL_PROVIDERS: dict[str, type[Provider]] = {
    # Bedrock models (primarily Claude via Bedrock)
    "BedrockModel": AnthropicProvider,
    # Anthropic models (direct API)
    "AnthropicModel": AnthropicProvider,
    # OpenAI models
    "OpenAIModel": OpenAIProvider,
    # LiteLLM (uses OpenAI-compatible interface)
    "LiteLLMModel": OpenAIProvider,
    # Ollama (uses OpenAI-compatible interface)
    "OllamaModel": OpenAIProvider,
    # Google Gemini models
    "GeminiModel": GoogleProvider,
    # Writer models (uses OpenAI-compatible interface)
    "WriterModel": OpenAIProvider,
}


def get_headroom_provider(model: Any) -> Provider:
    """Get the appropriate Headroom provider for a Strands model.

    Detection strategy:
    1. Check model class name against known Strands model types
    2. Check for provider hints in model attributes
    3. Fall back to OpenAI provider (most compatible)

    Args:
        model: A Strands model instance (BedrockModel, AnthropicModel, etc.)

    Returns:
        Appropriate Headroom Provider instance.

    Example:
        from strands.models import BedrockModel
        from headroom.integrations.strands.providers import get_headroom_provider

        model = BedrockModel(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        provider = get_headroom_provider(model)  # Returns AnthropicProvider
    """
    # Strategy 1: Class name matching
    class_name = model.__class__.__name__
    if class_name in _STRANDS_MODEL_PROVIDERS:
        provider_class = _STRANDS_MODEL_PROVIDERS[class_name]
        logger.debug(f"Detected provider {provider_class.__name__} from class {class_name}")
        return provider_class()

    # Strategy 2: Check module path
    module_path = model.__class__.__module__
    if "anthropic" in module_path.lower():
        logger.debug(f"Detected AnthropicProvider from module {module_path}")
        return AnthropicProvider()
    elif "bedrock" in module_path.lower():
        logger.debug(f"Detected AnthropicProvider from module {module_path}")
        return AnthropicProvider()
    elif "google" in module_path.lower() or "gemini" in module_path.lower():
        logger.debug(f"Detected GoogleProvider from module {module_path}")
        return GoogleProvider()
    elif "openai" in module_path.lower() or "litellm" in module_path.lower():
        logger.debug(f"Detected OpenAIProvider from module {module_path}")
        return OpenAIProvider()

    # Strategy 3: Check model ID/name for hints
    model_id = _extract_model_id(model)
    if model_id:
        model_id_lower = model_id.lower()
        if "claude" in model_id_lower or "anthropic" in model_id_lower:
            logger.debug(f"Detected AnthropicProvider from model ID {model_id}")
            return AnthropicProvider()
        elif "gemini" in model_id_lower:
            logger.debug(f"Detected GoogleProvider from model ID {model_id}")
            return GoogleProvider()
        elif "gpt" in model_id_lower or "o1" in model_id_lower or "o3" in model_id_lower:
            logger.debug(f"Detected OpenAIProvider from model ID {model_id}")
            return OpenAIProvider()

    # Strategy 4: Default fallback
    logger.warning(
        f"Unknown Strands model class '{class_name}', defaulting to OpenAIProvider. "
        "Token counting may be inaccurate."
    )
    return OpenAIProvider()


def _extract_model_id(model: Any) -> str:
    """Extract model ID from a Strands model using various attribute names.

    Args:
        model: A Strands model instance

    Returns:
        Model ID string or empty string if not found
    """
    # Try common attribute names used by Strands models
    for attr in ["model_id", "model", "model_name", "id"]:
        value = getattr(model, attr, None)
        if value and isinstance(value, str):
            return str(value)

    # Try to get from config if available (config can be dict or object)
    config = getattr(model, "config", None)
    if config:
        for attr in ["model_id", "model", "model_name"]:
            # Handle dict-style config (Strands uses this)
            if isinstance(config, dict):
                value = config.get(attr)
            else:
                value = getattr(config, attr, None)
            if value and isinstance(value, str):
                return str(value)

    # Try get_config() method (Strands Model interface)
    if hasattr(model, "get_config"):
        try:
            config_dict = model.get_config()
            if isinstance(config_dict, dict):
                for attr in ["model_id", "model", "model_name"]:
                    value = config_dict.get(attr)
                    if value and isinstance(value, str):
                        return str(value)
        except Exception:
            pass

    return ""


def get_model_name_from_strands(model: Any) -> str:
    """Extract the model name/ID from a Strands model.

    Args:
        model: A Strands model instance

    Returns:
        Model name string (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
    """
    model_id = _extract_model_id(model)
    if model_id:
        return str(model_id)

    # Fallback with warning
    class_name = model.__class__.__name__
    logger.warning(
        f"Could not extract model name from {class_name} (no 'model_id', 'model', "
        f"'model_name', or 'id' attribute). Defaulting to 'gpt-4o'. "
        "Token counting may be inaccurate."
    )
    return "gpt-4o"

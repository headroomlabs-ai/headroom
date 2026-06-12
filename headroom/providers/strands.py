"""Provider-owned Strands model detection policies."""

from __future__ import annotations

import logging
from typing import Any

from headroom.providers.anthropic import AnthropicProvider
from headroom.providers.base import Provider
from headroom.providers.google import GoogleProvider
from headroom.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)

_STRANDS_MODEL_PROVIDERS: dict[str, type[Provider]] = {
    "BedrockModel": AnthropicProvider,
    "AnthropicModel": AnthropicProvider,
    "OpenAIModel": OpenAIProvider,
    "LiteLLMModel": OpenAIProvider,
    "OllamaModel": OpenAIProvider,
    "GeminiModel": GoogleProvider,
    "WriterModel": OpenAIProvider,
}

_MODULE_PROVIDER_HINTS: tuple[tuple[tuple[str, ...], type[Provider]], ...] = (
    (("anthropic", "bedrock"), AnthropicProvider),
    (("google", "gemini"), GoogleProvider),
    (("openai", "litellm"), OpenAIProvider),
)

_MODEL_ID_PROVIDER_HINTS: tuple[tuple[tuple[str, ...], type[Provider]], ...] = (
    (("claude", "anthropic"), AnthropicProvider),
    (("gemini",), GoogleProvider),
    (("gpt", "o1", "o3"), OpenAIProvider),
)


def create_provider(model: Any) -> Provider:
    """Create a Headroom provider for a Strands model."""
    class_name = model.__class__.__name__
    if class_name in _STRANDS_MODEL_PROVIDERS:
        provider_class = _STRANDS_MODEL_PROVIDERS[class_name]
        logger.debug("Detected provider %s from class %s", provider_class.__name__, class_name)
        return provider_class()

    module_path = model.__class__.__module__
    module_path_lower = module_path.lower()
    for hints, provider_class in _MODULE_PROVIDER_HINTS:
        if any(hint in module_path_lower for hint in hints):
            logger.debug(
                "Detected provider %s from module %s", provider_class.__name__, module_path
            )
            return provider_class()

    model_id = extract_model_id(model)
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
        "Unknown Strands model class '%s', defaulting to OpenAIProvider. "
        "Token counting may be inaccurate.",
        class_name,
    )
    return OpenAIProvider()


def extract_model_id(model: Any) -> str:
    """Extract model ID from a Strands model using provider-known attributes."""
    for attr in ["model_id", "model", "model_name", "id"]:
        value = getattr(model, attr, None)
        if value and isinstance(value, str):
            return str(value)

    config = getattr(model, "config", None)
    if config:
        for attr in ["model_id", "model", "model_name"]:
            if isinstance(config, dict):
                value = config.get(attr)
            else:
                value = getattr(config, attr, None)
            if value and isinstance(value, str):
                return str(value)

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


def default_model_name(model: Any) -> str:
    """Return a Strands model name or provider-owned default."""
    model_id = extract_model_id(model)
    if model_id:
        return str(model_id)

    class_name = model.__class__.__name__
    logger.warning(
        "Could not extract model name from %s (no 'model_id', 'model', "
        "'model_name', or 'id' attribute). Defaulting to 'gpt-4o'. "
        "Token counting may be inaccurate.",
        class_name,
    )
    return "gpt-4o"

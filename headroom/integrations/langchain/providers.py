"""Provider detection for LangChain models.

This module provides automatic provider detection from LangChain chat models
without requiring explicit provider imports. It uses duck-typing based on
class paths to identify the appropriate Headroom provider.

Example:
    from langchain_anthropic import ChatAnthropic
    from headroom.integrations.langchain import get_headroom_provider

    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    provider = get_headroom_provider(model)  # Returns AnthropicProvider
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from headroom.providers.base import Provider

logger = logging.getLogger(__name__)

# Provider detection patterns
# Maps provider name to list of class path patterns to match
PROVIDER_PATTERNS: dict[str, list[str]] = {
    "openai": [
        "langchain_openai.ChatOpenAI",
        "langchain_openai.chat_models.ChatOpenAI",
        "langchain_community.chat_models.ChatOpenAI",
        "langchain.chat_models.ChatOpenAI",
        "ChatOpenAI",
    ],
    "anthropic": [
        "langchain_anthropic.ChatAnthropic",
        "langchain_anthropic.chat_models.ChatAnthropic",
        "langchain_community.chat_models.ChatAnthropic",
        "langchain.chat_models.ChatAnthropic",
        "ChatAnthropic",
    ],
    "google": [
        "langchain_google_genai.ChatGoogleGenerativeAI",
        "langchain_google_genai.chat_models.ChatGoogleGenerativeAI",
        "langchain_community.chat_models.ChatGoogleGenerativeAI",
        "ChatGoogleGenerativeAI",
        # Also match Vertex AI
        "langchain_google_vertexai.ChatVertexAI",
        "ChatVertexAI",
    ],
    "cohere": [
        "langchain_cohere.ChatCohere",
        "langchain_community.chat_models.ChatCohere",
        "ChatCohere",
    ],
    "mistral": [
        "langchain_mistralai.ChatMistralAI",
        "langchain_community.chat_models.ChatMistralAI",
        "ChatMistralAI",
    ],
}

# Model name patterns for fallback detection
MODEL_NAME_PATTERNS: dict[str, list[str]] = {
    "anthropic": ["claude", "anthropic"],
    "openai": ["gpt", "o1", "o3", "davinci", "turbo"],
    "google": ["gemini", "palm", "bison"],
    "cohere": ["command", "cohere"],
    "mistral": ["mistral", "mixtral"],
}


def detect_provider(model: Any) -> str:
    """Detect provider name from a LangChain model using duck-typing.

    Detection strategy:
    1. Check class module and name against known patterns
    2. Check model_name attribute against known model patterns
    3. Fall back to "openai" as safe default

    Args:
        model: Any LangChain chat model instance

    Returns:
        Provider name string: "openai", "anthropic", "google", "cohere", "mistral"

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> detect_provider(model)
        'anthropic'
    """
    # Strategy 1: Check class path
    class_module = getattr(model.__class__, "__module__", "")
    class_name = model.__class__.__name__
    class_path = f"{class_module}.{class_name}"

    for provider_name, patterns in PROVIDER_PATTERNS.items():
        for pattern in patterns:
            if pattern in class_path or class_name == pattern.split(".")[-1]:
                logger.debug(f"Detected provider '{provider_name}' from class path: {class_path}")
                return provider_name

    # Strategy 2: Check model_name attribute
    model_name = _get_model_name(model)
    if model_name:
        model_name_lower = model_name.lower()
        for provider_name, name_patterns in MODEL_NAME_PATTERNS.items():
            for pattern in name_patterns:
                if pattern in model_name_lower:
                    logger.debug(
                        f"Detected provider '{provider_name}' from model name: {model_name}"
                    )
                    return provider_name

    # Strategy 3: Fall back to OpenAI (most common, safe default)
    logger.debug(f"Could not detect provider for {class_path}, falling back to 'openai'")
    return "openai"


def _get_model_name(model: Any) -> str | None:
    """Extract model name from a LangChain model.

    Tries common attribute names used by different LangChain models.
    """
    # Try common attribute names
    for attr in ["model_name", "model", "model_id", "_model_name"]:
        value = getattr(model, attr, None)
        if isinstance(value, str):
            return value

    return None


def get_headroom_provider(model: Any) -> Provider:
    """Get appropriate Headroom Provider instance for a LangChain model.

    This function automatically detects the provider from the model type
    and returns a configured Headroom provider for accurate token counting
    and context limit detection.

    Args:
        model: Any LangChain chat model instance

    Returns:
        Configured Headroom Provider instance

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> provider = get_headroom_provider(model)
        >>> provider.name
        'anthropic'
    """
    # Import providers lazily to avoid circular imports
    from headroom.providers import (
        AnthropicProvider,
        GoogleProvider,
        OpenAIProvider,
    )

    provider_name = detect_provider(model)

    if provider_name == "anthropic":
        return AnthropicProvider()
    elif provider_name == "google":
        return GoogleProvider()
    # Cohere and Mistral fall back to OpenAI-compatible for now
    # TODO: Add dedicated providers when needed

    # Default to OpenAI
    return OpenAIProvider()


def get_model_name_from_langchain(model: Any) -> str:
    """Extract the model name string from a LangChain model.

    Useful for getting the model identifier for token counting
    and context limit lookup.

    Args:
        model: Any LangChain chat model instance

    Returns:
        Model name string (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
    """
    name = _get_model_name(model)
    if name:
        return name

    # Try to infer from class name
    class_name = model.__class__.__name__
    if "GPT" in class_name or "OpenAI" in class_name:
        return "gpt-4o"  # Safe default for OpenAI
    elif "Anthropic" in class_name or "Claude" in class_name:
        return "claude-3-5-sonnet-20241022"  # Safe default for Anthropic
    elif "Google" in class_name or "Gemini" in class_name:
        return "gemini-1.5-pro"  # Safe default for Google

    return "gpt-4o"  # Ultimate fallback

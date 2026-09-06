"""Provider-owned LangChain model detection policies."""

from __future__ import annotations

import logging
from typing import Any

from headroom.providers.anthropic import AnthropicProvider
from headroom.providers.base import Provider
from headroom.providers.google import GoogleProvider
from headroom.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)

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

MODEL_NAME_PATTERNS: dict[str, list[str]] = {
    "anthropic": ["claude", "anthropic"],
    "openai": ["gpt", "o1", "o3", "davinci", "turbo"],
    "google": ["gemini", "palm", "bison"],
    "cohere": ["command", "cohere"],
    "mistral": ["mistral", "mixtral"],
}

DEFAULT_MODEL_BY_PROVIDER = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "google": "gemini-1.5-pro",
    "openai": "gpt-4o",
}


def detect_provider(model: Any) -> str:
    """Detect provider name from a LangChain model using provider-owned patterns."""
    class_module = getattr(model.__class__, "__module__", "")
    class_name = model.__class__.__name__
    class_path = f"{class_module}.{class_name}"

    for provider_name, patterns in PROVIDER_PATTERNS.items():
        for pattern in patterns:
            if pattern in class_path or class_name == pattern.split(".")[-1]:
                logger.debug(
                    "Detected provider '%s' from class path: %s", provider_name, class_path
                )
                return provider_name

    model_name = get_model_name(model)
    if model_name:
        model_name_lower = model_name.lower()
        for provider_name, name_patterns in MODEL_NAME_PATTERNS.items():
            for pattern in name_patterns:
                if pattern in model_name_lower:
                    logger.debug(
                        "Detected provider '%s' from model name: %s",
                        provider_name,
                        model_name,
                    )
                    return provider_name

    logger.debug("Could not detect provider for %s, falling back to 'openai'", class_path)
    return "openai"


def get_model_name(model: Any) -> str | None:
    """Extract model name from a LangChain model."""
    for attr in ["model_name", "model", "model_id", "_model_name"]:
        value = getattr(model, attr, None)
        if isinstance(value, str):
            return value

    return None


def create_provider(model: Any) -> Provider:
    """Create a Headroom provider for a LangChain model."""
    provider_name = detect_provider(model)
    if provider_name == "anthropic":
        return AnthropicProvider()
    if provider_name == "google":
        return GoogleProvider()

    return OpenAIProvider()


def default_model_name(model: Any) -> str:
    """Return a provider-appropriate default model for a LangChain model."""
    name = get_model_name(model)
    if name:
        return name

    class_name = model.__class__.__name__
    class_name_lower = class_name.lower()
    for provider_name, patterns in MODEL_NAME_PATTERNS.items():
        if any(pattern in class_name_lower for pattern in patterns):
            return DEFAULT_MODEL_BY_PROVIDER.get(provider_name, DEFAULT_MODEL_BY_PROVIDER["openai"])

    if "openai" in class_name_lower:
        return DEFAULT_MODEL_BY_PROVIDER["openai"]

    return DEFAULT_MODEL_BY_PROVIDER["openai"]

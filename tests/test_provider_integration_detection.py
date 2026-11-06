"""Provider-owned integration detection tests."""

from __future__ import annotations

from headroom.providers.anthropic import AnthropicProvider
from headroom.providers.google import GoogleProvider
from headroom.providers.langchain import (
    create_provider as create_langchain_provider,
)
from headroom.providers.langchain import (
    default_model_name as langchain_default_model_name,
)
from headroom.providers.langchain import (
    detect_provider,
)
from headroom.providers.openai import OpenAIProvider
from headroom.providers.strands import (
    create_provider as create_strands_provider,
)
from headroom.providers.strands import (
    default_model_name as strands_default_model_name,
)


class ChatAnthropic:
    __module__ = "langchain_anthropic"
    model = "claude-3-5-sonnet-20241022"


class ChatOpenAI:
    __module__ = "langchain_openai"
    model_name = "gpt-4o"


class ChatGoogleGenerativeAI:
    __module__ = "langchain_google_genai"


class UnknownLangChainModel:
    __module__ = "example"


class BedrockModel:
    __module__ = "strands.models.bedrock"
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"


class GeminiModel:
    __module__ = "strands.models.google"
    config = {"model_id": "gemini-1.5-pro"}


class UnknownStrandsModel:
    __module__ = "example"


def test_langchain_provider_detection_from_class_path() -> None:
    assert detect_provider(ChatAnthropic()) == "anthropic"
    assert detect_provider(ChatOpenAI()) == "openai"
    assert detect_provider(ChatGoogleGenerativeAI()) == "google"


def test_langchain_provider_instances() -> None:
    assert isinstance(create_langchain_provider(ChatAnthropic()), AnthropicProvider)
    assert isinstance(create_langchain_provider(ChatGoogleGenerativeAI()), GoogleProvider)
    assert isinstance(create_langchain_provider(UnknownLangChainModel()), OpenAIProvider)


def test_langchain_default_model_name() -> None:
    assert langchain_default_model_name(ChatOpenAI()) == "gpt-4o"
    assert langchain_default_model_name(UnknownLangChainModel()) == "gpt-4o"


def test_strands_provider_instances() -> None:
    assert isinstance(create_strands_provider(BedrockModel()), AnthropicProvider)
    assert isinstance(create_strands_provider(GeminiModel()), GoogleProvider)
    assert isinstance(create_strands_provider(UnknownStrandsModel()), OpenAIProvider)


def test_strands_default_model_name() -> None:
    assert strands_default_model_name(BedrockModel()) == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert strands_default_model_name(UnknownStrandsModel()) == "gpt-4o"

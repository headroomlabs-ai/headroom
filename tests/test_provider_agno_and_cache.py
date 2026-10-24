"""Provider-owned Agno and cache policy tests."""

from __future__ import annotations

from headroom.providers.agno import create_provider, default_model_name
from headroom.providers.anthropic import AnthropicProvider
from headroom.providers.cache_economics import cache_economics_for_provider
from headroom.providers.defaults import DEFAULT_TOKEN_MODEL, create_default_provider
from headroom.providers.google import GoogleProvider
from headroom.providers.openai import OpenAIProvider


class OpenAIChat:
    __module__ = "agno.models.openai"
    id = "gpt-4o"


class Claude:
    __module__ = "agno.models.anthropic"
    id = "claude-sonnet-4-20250514"


class Gemini:
    __module__ = "agno.models.google"
    id = "gemini-1.5-pro"


class UnknownAgnoModel:
    __module__ = "example"


def test_agno_provider_detection_from_class_name() -> None:
    assert isinstance(create_provider(OpenAIChat()), OpenAIProvider)
    assert isinstance(create_provider(Claude()), AnthropicProvider)
    assert isinstance(create_provider(Gemini()), GoogleProvider)


def test_agno_provider_detection_falls_back_to_openai() -> None:
    assert isinstance(create_provider(UnknownAgnoModel()), OpenAIProvider)


def test_agno_model_name_default() -> None:
    assert default_model_name(OpenAIChat()) == "gpt-4o"
    assert default_model_name(UnknownAgnoModel()) == "gpt-4o"


def test_cache_economics_are_provider_owned() -> None:
    anthropic = cache_economics_for_provider("anthropic")
    openai = cache_economics_for_provider("openai")
    unknown = cache_economics_for_provider("unknown")

    assert anthropic.read_discount == 0.9
    assert anthropic.write_penalty == 0.25
    assert openai.read_discount == 0.5
    assert openai.write_penalty == 0.0
    assert unknown == anthropic


def test_default_provider_factory_is_provider_owned() -> None:
    provider = create_default_provider()

    assert isinstance(provider, OpenAIProvider)
    assert DEFAULT_TOKEN_MODEL == "gpt-4o"

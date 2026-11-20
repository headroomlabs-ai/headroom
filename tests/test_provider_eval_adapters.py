"""Provider-owned eval adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

from headroom.providers.anthropic_subscription import (
    ANTHROPIC_OAUTH_USAGE_BETA_HEADER,
    ANTHROPIC_OAUTH_USAGE_URL,
)
from headroom.providers.evals import call_eval_llm, default_eval_model
from headroom.providers.litellm import completion_api_base_for_model


def test_default_eval_models_are_provider_owned() -> None:
    assert default_eval_model("anthropic") == "claude-sonnet-4-20250514"
    assert default_eval_model("openai") == "gpt-4o"
    assert default_eval_model("litellm") == "gpt-4o"


def test_openai_eval_completion_shape() -> None:
    class Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            assert kwargs["model"] == "gpt-4o"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="openai text"))]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )

    assert (
        call_eval_llm(
            "openai",
            client,
            "gpt-4o",
            [{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0.0,
        )
        == "openai text"
    )


def test_anthropic_eval_completion_shape() -> None:
    class Messages:
        @staticmethod
        def create(**kwargs: object) -> object:
            assert kwargs["model"] == "claude-sonnet-4-20250514"
            return SimpleNamespace(content=[SimpleNamespace(text="anthropic text")])

    client = SimpleNamespace(messages=Messages())

    assert (
        call_eval_llm(
            "anthropic",
            client,
            "claude-sonnet-4-20250514",
            [{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0.0,
        )
        == "anthropic text"
    )


def test_ollama_eval_completion_shape() -> None:
    class OllamaClient:
        @staticmethod
        def chat(**kwargs: object) -> dict[str, object]:
            assert kwargs["model"] == "llama3"
            return {"message": {"content": "ollama text"}}

    assert (
        call_eval_llm(
            "ollama",
            OllamaClient(),
            "llama3",
            [{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0.0,
        )
        == "ollama text"
    )


def test_litellm_eval_completion_shape() -> None:
    class LiteLLMModule:
        @staticmethod
        def completion(**kwargs: object) -> object:
            assert kwargs["model"] == "gpt-4o"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="litellm text"))]
            )

    assert (
        call_eval_llm(
            "litellm",
            LiteLLMModule(),
            "gpt-4o",
            [{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0.0,
        )
        == "litellm text"
    )


def test_litellm_api_base_overrides_are_provider_owned() -> None:
    assert completion_api_base_for_model("claude-sonnet-4-20250514") == "https://api.anthropic.com"
    assert completion_api_base_for_model("gpt-4o") is None


def test_anthropic_subscription_constants_are_provider_owned() -> None:
    assert ANTHROPIC_OAUTH_USAGE_URL == "https://api.anthropic.com/api/oauth/usage"
    assert ANTHROPIC_OAUTH_USAGE_BETA_HEADER == "oauth-2025-04-20"

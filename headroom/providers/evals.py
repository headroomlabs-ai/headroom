"""Provider-owned LLM helpers for evaluation tooling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalProviderAdapter:
    """Evaluation-time provider SDK behavior."""

    provider: str
    default_model: str
    create_client: Callable[[], Any]
    complete: Callable[
        [Any, str, list[dict[str, Any]], int, float],
        str,
    ]


def _openai_client() -> Any:
    try:
        import openai
    except ImportError as exc:
        raise ImportError("openai package required. Install with: pip install openai") from exc

    return openai.OpenAI()


def _anthropic_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package required. Install with: pip install anthropic"
        ) from exc

    return anthropic.Anthropic()


def _ollama_client() -> Any:
    try:
        import ollama
    except ImportError as exc:
        raise ImportError("ollama package required. Install with: pip install ollama") from exc

    return ollama.Client()


def _litellm_client() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise ImportError("LiteLLM package required. Install with: pip install litellm") from exc

    return litellm


def _openai_complete(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    content = response.choices[0].message.content
    return str(content) if content else ""


def _anthropic_complete(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    return str(response.content[0].text)


def _ollama_complete(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    del max_tokens
    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return str(response["message"]["content"])


def _litellm_complete(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


_EVAL_PROVIDER_ADAPTERS: dict[str, EvalProviderAdapter] = {
    "anthropic": EvalProviderAdapter(
        provider="anthropic",
        default_model="claude-sonnet-4-20250514",
        create_client=_anthropic_client,
        complete=_anthropic_complete,
    ),
    "openai": EvalProviderAdapter(
        provider="openai",
        default_model="gpt-4o",
        create_client=_openai_client,
        complete=_openai_complete,
    ),
    "ollama": EvalProviderAdapter(
        provider="ollama",
        default_model="llama3",
        create_client=_ollama_client,
        complete=_ollama_complete,
    ),
    "litellm": EvalProviderAdapter(
        provider="litellm",
        default_model="gpt-4o",
        create_client=_litellm_client,
        complete=_litellm_complete,
    ),
}

EVAL_JUDGE_PROVIDER_CHOICES = ("openai", "anthropic", "litellm", "simple")


def create_eval_judge(provider: str, model: str) -> Callable[[str, str, str], tuple[float, str]]:
    """Create an eval judge function from provider-owned selection policy."""
    from headroom.evals.memory import (
        create_anthropic_judge,
        create_litellm_judge,
        create_openai_judge,
        simple_judge,
    )

    judge_factories: dict[str, Callable[[str], Callable[[str, str, str], tuple[float, str]]]] = {
        "anthropic": create_anthropic_judge,
        "litellm": create_litellm_judge,
        "openai": create_openai_judge,
        "simple": lambda _model: simple_judge,
    }

    try:
        return judge_factories[provider](model)
    except KeyError as exc:
        raise ValueError(f"Unknown judge provider: {provider}") from exc


def describe_eval_judge(provider: str, model: str) -> str:
    """Return display text for an eval judge provider/model pair."""
    if provider == "simple":
        return "ENABLED (rule-based F1)"
    return f"ENABLED ({provider}: {model})"


def get_eval_provider_adapter(provider: str) -> EvalProviderAdapter:
    """Return evaluation provider behavior for a provider name."""
    try:
        return _EVAL_PROVIDER_ADAPTERS[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider: {provider}") from exc


def default_eval_model(provider: str) -> str:
    """Return provider-owned default eval model."""
    return get_eval_provider_adapter(provider).default_model


def create_eval_client(provider: str) -> Any:
    """Create the SDK client/module for an eval provider."""
    return get_eval_provider_adapter(provider).create_client()


def call_eval_llm(
    provider: str,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call a provider LLM and return text content."""
    return get_eval_provider_adapter(provider).complete(
        client,
        model,
        messages,
        max_tokens,
        temperature,
    )

"""Request/response handling shared by the evals that call Anthropic directly."""

from __future__ import annotations

from typing import Any


def is_sonnet_5_5(model: str) -> bool:
    """Whether ``model`` is Claude Sonnet 5.5, with or without a LiteLLM prefix."""
    return model.removeprefix("anthropic/").startswith("claude-sonnet-5-5")


def anthropic_thinking_params(model: str) -> dict[str, Any]:
    """Native Messages API fields for ``model``, as used in batch ``params``.

    Claude Sonnet 5.5 thinks by default, and thinking counts against
    ``max_tokens``. The evals ask for short answers, so turn up-front thinking
    off (``between_tools``; ``disabled`` is rejected by this model).
    """
    if is_sonnet_5_5(model):
        return {"thinking": {"type": "between_tools"}}
    return {}


def anthropic_thinking_kwargs(model: str) -> dict[str, Any]:
    """The same fields as arguments for an SDK ``create`` call.

    They go through ``extra_body`` because the oldest ``anthropic`` release the
    ``evals`` extra accepts has no ``thinking`` argument.
    """
    params = anthropic_thinking_params(model)
    return {"extra_body": params} if params else {}


def litellm_model_kwargs(model: str) -> dict[str, Any]:
    """``litellm.completion`` model and request settings for a deterministic call.

    Claude Sonnet 5.5 rejects sampling parameters. It also gets an explicit
    provider prefix, so routing does not depend on LiteLLM's model table.
    """
    if not is_sonnet_5_5(model):
        return {"model": model, "temperature": 0.0}
    return {
        "model": f"anthropic/{model.removeprefix('anthropic/')}",
        **anthropic_thinking_params(model),
    }


def anthropic_response_text(response: Any) -> str:
    """Join the text blocks of a response, skipping thinking blocks."""
    return "".join(block.text for block in response.content if block.type == "text")

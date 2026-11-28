"""Request/response handling shared by the evals that call Anthropic directly."""

from __future__ import annotations

from typing import Any


def anthropic_thinking_kwargs(model: str) -> dict[str, Any]:
    """Extra ``messages.create`` arguments for ``model``.

    Claude Sonnet 5.5 thinks by default, and thinking counts against
    ``max_tokens``. The evals ask for short answers, so turn up-front thinking
    off (``between_tools``; ``disabled`` is rejected by this model).
    """
    if model.startswith("claude-sonnet-5-5"):
        return {"thinking": {"type": "between_tools"}}
    return {}


def anthropic_response_text(response: Any) -> str:
    """Join the text blocks of a response, skipping thinking blocks."""
    return "".join(block.text for block in response.content if block.type == "text")

"""MiniMax provider for Headroom SDK.

MiniMax provides Anthropic-compatible APIs for M3, M2.7, M2.5, M2.1, and M2 models.
See: https://platform.minimaxi.com/docs/api-reference/text-anthropic-api

Token counting uses tiktoken (cl100k_base) approximation — the same as
AnthropicProvider since MiniMax's API format mirrors Anthropic's Messages API.

Auth: MiniMax accepts both ``x-api-key`` and ``Authorization: Bearer`` headers.
The recommended approach for the proxy is to inject ``x-api-key`` (from
MINIMAX_API_KEY env var) alongside the existing client Authorization header.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import Provider

if TYPE_CHECKING:
    from headroom.providers.base import TokenCounter

logger = logging.getLogger(__name__)

# Default base URL for MiniMax Anthropic-compatible API.
DEFAULT_API_URL = "https://api.minimaxi.com/anthropic"

# MiniMax model context limits (tokens).
# M3 supports 1M context; M2.x series supports 200K.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "MiniMax-M3": 1_000_000,
    "MiniMax-M2.7": 204_800,
    "MiniMax-M2.7-highspeed": 204_800,
    "MiniMax-M2.5": 204_800,
    "MiniMax-M2.5-highspeed": 204_800,
    "MiniMax-M2.1": 204_800,
    "MiniMax-M2.1-highspeed": 204_800,
    "MiniMax-M2": 204_800,
}

# MiniMax model max output tokens.
MODEL_MAX_OUTPUT: dict[str, int] = {
    "MiniMax-M3": 131_072,
    "MiniMax-M2.7": 65_536,
    "MiniMax-M2.7-highspeed": 65_536,
    "MiniMax-M2.5": 65_536,
    "MiniMax-M2.5-highspeed": 65_536,
    "MiniMax-M2.1": 65_536,
    "MiniMax-M2.1-highspeed": 65_536,
    "MiniMax-M2": 65_536,
}

# MiniMax pricing (USD per 1M tokens, approximate as of 2025).
# Input and output costs vary by model. These are conservative defaults.
MODEL_INPUT_COST: dict[str, float] = {
    "MiniMax-M3": 1.0,  # ~$1/M input
    "MiniMax-M2.7": 0.8,
    "MiniMax-M2.7-highspeed": 0.5,
    "MiniMax-M2.5": 0.5,
    "MiniMax-M2.5-highspeed": 0.3,
    "MiniMax-M2.1": 0.3,
    "MiniMax-M2.1-highspeed": 0.2,
    "MiniMax-M2": 0.2,
}

MODEL_OUTPUT_COST: dict[str, float] = {
    "MiniMax-M3": 5.0,  # ~$5/M output
    "MiniMax-M2.7": 4.0,
    "MiniMax-M2.7-highspeed": 2.5,
    "MiniMax-M2.5": 2.5,
    "MiniMax-M2.5-highspeed": 1.5,
    "MiniMax-M2.1": 1.5,
    "MiniMax-M2.1-highspeed": 1.0,
    "MiniMax-M2": 1.0,
}


class MiniMaxProvider(Provider):
    """Provider for MiniMax Anthropic-compatible API.

    Uses tiktoken (cl100k_base) for token counting approximation.
    Supports MiniMax-M3, M2.7, M2.5, M2.1, and M2 model families.

    Auth: The proxy injects ``x-api-key`` from MINIMAX_API_KEY env var.
    MiniMax also accepts ``Authorization: Bearer`` with the same key.

    Example env vars::

        MINIMAX_API_URL=https://api.minimaxi.com/anthropic
        MINIMAX_API_KEY=sk-cp-xxxxxxxxxxxxxxxxxxxxxxxx
    """

    name = "minimax"

    def get_token_counter(self, model: str) -> TokenCounter:
        """Return a tiktoken-based token counter for the model."""
        from headroom.providers.anthropic import AnthropicProvider

        # Reuse AnthropicProvider's tiktoken counter since MiniMax uses
        # the same cl100k_base encoding as OpenAI/Anthropic.
        anthropic = AnthropicProvider(warn=False)
        return anthropic.get_token_counter(model)

    def get_context_limit(self, model: str) -> int:
        """Return the context window limit for the model."""
        # Strip provider prefix if present (e.g. "minimax/MiniMax-M3" -> "MiniMax-M3").
        normalized = model.split("/")[-1]
        if normalized not in MODEL_CONTEXT_LIMITS:
            logger.warning(
                "Unknown MiniMax model %r; using conservative default (128K). "
                "Known models: %s",
                model,
                ", ".join(sorted(MODEL_CONTEXT_LIMITS)),
            )
            return 128_000
        return MODEL_CONTEXT_LIMITS[normalized]

    def get_max_output_tokens(self, model: str) -> int:
        """Return the max output tokens for the model."""
        normalized = model.split("/")[-1]
        return MODEL_MAX_OUTPUT.get(normalized, 65_536)

    def get_input_cost_per_1m(self, model: str) -> float | None:
        """Return the input cost per 1M tokens, or None if unknown."""
        normalized = model.split("/")[-1]
        return MODEL_INPUT_COST.get(normalized)

    def get_output_cost_per_1m(self, model: str) -> float | None:
        """Return the output cost per 1M tokens, or None if unknown."""
        normalized = model.split("/")[-1]
        return MODEL_OUTPUT_COST.get(normalized)

    def supports_tools(self, model: str) -> bool:
        """Return whether the model supports tool calling."""
        return True

    def supports_vision(self, model: str) -> bool:
        """Return whether the model supports image inputs."""
        normalized = model.split("/")[-1]
        # Only M3 supports vision natively.
        return normalized == "MiniMax-M3"

    def supports_streaming(self, model: str) -> bool:
        """Return whether the model supports streaming."""
        return True

    def supports_model(self, model: str) -> bool:
        """Return whether this provider recognises the given model name.

        Accepts both bare model names ("MiniMax-M3") and the prefixed
        form ("minimax/MiniMax-M3"). Returns True even for unknown
        models — the upstream API is permissive and will return its
        own error for genuinely invalid ones.
        """
        normalized = model.split("/")[-1]
        return bool(normalized)

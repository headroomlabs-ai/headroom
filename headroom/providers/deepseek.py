"""Deepseek provider for Headroom SDK.

Token counting uses HuggingFace tokenizers for accurate counts on all
Deepseek model variants.
"""

from __future__ import annotations

import logging
from typing import Any

from headroom.tokenizers import get_tokenizer

from .base import Provider, TokenCounter

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

logger = logging.getLogger(__name__)

_CONTEXT_LIMITS: dict[str, int] = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v3": 128_000,
    "deepseek-chat": 131_072,
    "deepseek-v2": 128_000,
    "deepseek-v2-chat": 128_000,
    "deepseek-coder": 16_384,
    "deepseek-coder-v2": 128_000,
    "deepseek-reasoner": 131_072,
}

_PRICING: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash": (0.27, 1.10),
    "deepseek-v4-pro": (0.54, 2.19),
    "deepseek-v3": (0.27, 1.10),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-v2": (0.27, 1.10),
    "deepseek-v2-chat": (0.27, 1.10),
    "deepseek-coder": (0.14, 0.28),
    "deepseek-coder-v2": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}

_MAX_OUTPUT: dict[str, int] = {
    "deepseek-v4-flash": 384_000,
    "deepseek-v4-pro": 384_000,
    "deepseek-v3": 8_192,
    "deepseek-chat": 8_192,
    "deepseek-v2": 8_192,
    "deepseek-v2-chat": 8_192,
    "deepseek-coder": 4_096,
    "deepseek-coder-v2": 16_384,
    "deepseek-reasoner": 16_384,
}


class DeepseekTokenCounter:
    """Token counter for Deepseek models using HuggingFace tokenizers."""

    def __init__(self, model: str):
        self.model = model
        self._tokenizer = get_tokenizer(model, backend="huggingface")

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return self._tokenizer.count_text(text)

    def count_message(self, message: dict[str, Any]) -> int:
        tokens = 4  # base overhead
        role = message.get("role", "")
        tokens += self.count_text(role)
        content = message.get("content")
        if content:
            if isinstance(content, str):
                tokens += self.count_text(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        tokens += self.count_text(part.get("text", ""))
                    elif isinstance(part, str):
                        tokens += self.count_text(part)
        name = message.get("name")
        if name:
            tokens += self.count_text(name) + 1
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                tokens += self.count_text(func.get("name", ""))
                tokens += self.count_text(func.get("arguments", ""))
                tokens += self.count_text(tc.get("id", ""))
                tokens += 10  # structural overhead
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            tokens += self.count_text(tool_call_id) + 2
        return tokens

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        total = sum(self.count_message(msg) for msg in messages)
        total += 3  # priming tokens
        return total


class DeepseekProvider(Provider):
    """Provider for Deepseek models."""

    def __init__(self):
        self._context_limits = dict(_CONTEXT_LIMITS)
        self._pricing = dict(_PRICING)
        self._max_output = dict(_MAX_OUTPUT)
        self._token_counters: dict[str, DeepseekTokenCounter] = {}

    @property
    def name(self) -> str:
        return "deepseek"

    def supports_model(self, model: str) -> bool:
        model_lower = model.lower()
        if model_lower in _CONTEXT_LIMITS:
            return True
        return model_lower.startswith("deepseek-") or model_lower.startswith("deepseek_")

    def get_token_counter(self, model: str) -> TokenCounter:
        if not self.supports_model(model):
            raise ValueError(
                f"Model '{model}' is not a recognized Deepseek model. "
                f"Supported models: {list(_CONTEXT_LIMITS.keys())}"
            )
        if model not in self._token_counters:
            self._token_counters[model] = DeepseekTokenCounter(model)
        return self._token_counters[model]

    def get_context_limit(self, model: str) -> int:
        model_lower = model.lower()
        if model_lower in _CONTEXT_LIMITS:
            return _CONTEXT_LIMITS[model_lower]
        # prefix match
        for prefix, limit in _CONTEXT_LIMITS.items():
            if model_lower.startswith(prefix):
                return limit
        return 128_000

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float | None:
        # Try LiteLLM first
        if LITELLM_AVAILABLE:
            for model_variant in [f"deepseek/{model}", model]:
                try:
                    cost = litellm.completion_cost(
                        model=model_variant,
                        prompt="",
                        completion="",
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                    )
                    if cost is not None:
                        return float(cost)
                except Exception:
                    pass
        # Fallback to hardcoded pricing
        model_lower = model.lower()
        for model_prefix, (inp, outp) in _PRICING.items():
            if model_lower.startswith(model_prefix):
                input_cost = (input_tokens / 1_000_000) * inp
                output_cost = (output_tokens / 1_000_000) * outp
                return input_cost + output_cost
        return None

    def get_output_buffer(self, model: str, default: int = 4000) -> int:
        model_lower = model.lower()
        if model_lower in _MAX_OUTPUT:
            return _MAX_OUTPUT[model_lower]
        return default

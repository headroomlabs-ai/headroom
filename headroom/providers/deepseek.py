"""Deepseek provider for Headroom SDK.

Token counting uses HuggingFace tokenizers for accurate counts on all
Deepseek model variants. Cost estimates use LiteLLM's pricing database
when available, with hardcoded fallbacks.

Usage:
    from headroom import DeepseekProvider

    provider = DeepseekProvider()
    counter = provider.get_token_counter("deepseek-chat")
    tokens = counter.count_text("Hello, world!")

    # Cost estimation
    cost = provider.estimate_cost(
        input_tokens=100000,
        output_tokens=10000,
        model="deepseek-chat",
    )
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import date
from typing import Any

from headroom import paths as _paths
from headroom.tokenizers import get_tokenizer

from .base import Provider, TokenCounter

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

logger = logging.getLogger(__name__)

_UNKNOWN_MODEL_WARNINGS: set[str] = set()

_PRICING_LAST_UPDATED = date(2025, 6, 1)
_PRICING_STALE_DAYS = 60
_PRICING_WARNING_SHOWN = False


def _check_pricing_staleness() -> str | None:
    """Check if pricing data is stale and return warning message if so."""
    global _PRICING_WARNING_SHOWN
    days_old = (date.today() - _PRICING_LAST_UPDATED).days
    if days_old > _PRICING_STALE_DAYS and not _PRICING_WARNING_SHOWN:
        _PRICING_WARNING_SHOWN = True
        return (
            f"Deepseek pricing data is {days_old} days old. "
            "Cost estimates may be inaccurate. Verify against actual billing."
        )
    return None

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


def _load_custom_model_config() -> dict[str, Any]:
    """Load custom model configuration from environment or config file.

    Checks (in order):
    1. HEADROOM_MODEL_LIMITS environment variable (JSON string or file path)
    2. ~/.headroom/models.json config file

    Returns:
        Dict with 'context_limits', 'pricing', and 'max_output' keys.
    """
    config: dict[str, Any] = {"context_limits": {}, "pricing": {}, "max_output": {}}

    env_config = os.environ.get("HEADROOM_MODEL_LIMITS", "")
    if env_config:
        try:
            if os.path.isfile(env_config):
                with open(env_config) as f:
                    loaded = json.load(f)
            else:
                loaded = json.loads(env_config)
            deepseek_config = loaded.get("deepseek", loaded)
            if "context_limits" in deepseek_config:
                config["context_limits"].update(deepseek_config["context_limits"])
            if "pricing" in deepseek_config:
                config["pricing"].update(deepseek_config["pricing"])
            if "max_output" in deepseek_config:
                config["max_output"].update(deepseek_config["max_output"])
            logger.debug("Loaded custom Deepseek model config from HEADROOM_MODEL_LIMITS")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load HEADROOM_MODEL_LIMITS: {e}")

    config_file = _paths.models_config_path()
    if not config_file.exists():
        legacy_models = _paths.workspace_dir() / "models.json"
        if legacy_models.exists():
            config_file = legacy_models
    if config_file.exists():
        try:
            with open(config_file) as f:
                loaded = json.load(f)
            deepseek_config = loaded.get("deepseek", {})
            if "context_limits" in deepseek_config:
                for model, limit in deepseek_config["context_limits"].items():
                    if model not in config["context_limits"]:
                        config["context_limits"][model] = limit
            if "pricing" in deepseek_config:
                for model, pricing in deepseek_config["pricing"].items():
                    if model not in config["pricing"]:
                        config["pricing"][model] = pricing
            if "max_output" in deepseek_config:
                for model, output in deepseek_config["max_output"].items():
                    if model not in config["max_output"]:
                        config["max_output"][model] = output
            logger.debug(f"Loaded custom Deepseek model config from {config_file}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {config_file}: {e}")

    return config


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

    def __init__(self) -> None:
        self._context_limits = {**_CONTEXT_LIMITS}
        self._pricing = {**_PRICING}
        self._max_output = {**_MAX_OUTPUT}

        custom_config = _load_custom_model_config()
        self._context_limits.update(custom_config["context_limits"])
        self._max_output.update(custom_config["max_output"])
        for model, pricing in custom_config["pricing"].items():
            if isinstance(pricing, list | tuple) and len(pricing) >= 2:
                self._pricing[model] = (float(pricing[0]), float(pricing[1]))

        self._token_counters: dict[str, DeepseekTokenCounter] = {}

    @property
    def name(self) -> str:
        return "deepseek"

    def supports_model(self, model: str) -> bool:
        model_lower = model.lower()
        if model_lower in self._context_limits:
            return True
        return model_lower.startswith("deepseek-") or model_lower.startswith("deepseek_")

    def get_token_counter(self, model: str) -> TokenCounter:
        if not self.supports_model(model):
            raise ValueError(
                f"Model '{model}' is not a recognized Deepseek model. "
                f"Supported models: {list(self._context_limits.keys())}"
            )
        if model not in self._token_counters:
            self._token_counters[model] = DeepseekTokenCounter(model)
        return self._token_counters[model]

    def get_context_limit(self, model: str) -> int:
        """Get context limit for a Deepseek model.

        Tries LiteLLM first (with and without 'deepseek/' prefix),
        then falls back to built-in limits.
        """
        if LITELLM_AVAILABLE:
            for model_variant in [f"deepseek/{model}", model]:
                try:
                    info = litellm.get_model_info(model_variant)
                    if info and "max_input_tokens" in info:
                        result = info["max_input_tokens"]
                        if result is not None:
                            return int(result)
                    if info and "max_tokens" in info:
                        result = info["max_tokens"]
                        if result is not None:
                            return int(result)
                except Exception:
                    pass

        model_lower = model.lower()
        if model_lower in self._context_limits:
            return self._context_limits[model_lower]
        for prefix, limit in self._context_limits.items():
            if model_lower.startswith(prefix):
                return limit
        self._warn_unknown_model(model, 128_000, "using default limit")
        return 128_000

    def _warn_unknown_model(self, model: str, limit: int, reason: str) -> None:
        """Warn about unknown model (once per model)."""
        if model not in _UNKNOWN_MODEL_WARNINGS:
            _UNKNOWN_MODEL_WARNINGS.add(model)
            logger.warning(
                f"Unknown Deepseek model '{model}': {reason} ({limit:,} tokens). "
                f"To configure explicitly, set HEADROOM_MODEL_LIMITS env var or "
                f"add to ~/.headroom/models.json"
            )

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
        staleness_warning = _check_pricing_staleness()
        if staleness_warning:
            warnings.warn(staleness_warning, UserWarning, stacklevel=2)
        for model_prefix, (inp, outp) in self._pricing.items():
            if model_lower.startswith(model_prefix):
                input_cost = (input_tokens / 1_000_000) * inp
                output_cost = (output_tokens / 1_000_000) * outp
                return input_cost + output_cost
        return None

    def get_output_buffer(self, model: str, default: int = 4000) -> int:
        model_lower = model.lower()
        if model_lower in self._max_output:
            return self._max_output[model_lower]
        return default

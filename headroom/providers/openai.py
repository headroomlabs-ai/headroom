"""OpenAI provider implementation for Headroom SDK.

Token counting is accurate (uses tiktoken).
Cost estimates are APPROXIMATE - always verify against your actual billing.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import warnings
from datetime import date
from functools import lru_cache
from typing import Any, cast

from headroom import paths as _paths

from .base import Provider, TokenCounter

logger = logging.getLogger(__name__)

# Pricing metadata for transparency
_PRICING_LAST_UPDATED = date(2025, 1, 14)
_PRICING_STALE_DAYS = 60  # Warn if pricing data is older than this

# Warning tracking
_PRICING_WARNING_SHOWN = False
_UNKNOWN_MODEL_WARNINGS: set[str] = set()

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

LITELLM_AVAILABLE = importlib.util.find_spec("litellm") is not None


def _get_litellm_module() -> Any | None:
    """Import LiteLLM only when pricing/context metadata is needed."""
    if not LITELLM_AVAILABLE:
        return None

    try:
        import litellm
    except ImportError:
        return None

    return litellm


# OpenAI model to tiktoken encoding mappings
_MODEL_ENCODINGS: dict[str, str] = {
    # GPT-4o and newer use o200k_base
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-2024": "o200k_base",
    "o1": "o200k_base",
    "o1-preview": "o200k_base",
    "o1-mini": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    # GPT-4 and GPT-3.5 use cl100k_base
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
}

# OpenAI context window limits
_CONTEXT_LIMITS: dict[str, int] = {
    # GPT-4o series
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-2024-11-20": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-05-13": 128000,
    # GPT-4 Turbo
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-1106-preview": 128000,
    # GPT-4
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    # GPT-3.5
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    # o1/o3 reasoning models
    "o1": 200000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3": 200000,
    "o3-mini": 200000,
    # DeepSeek (often accessed via OpenAI-compatible API). Values verified
    # against api-docs.deepseek.com (V4) and LiteLLM model_cost (deprecated
    # aliases). LiteLLM lookup is still attempted first in get_context_limit;
    # these are the manual fallback when LiteLLM doesn't know the model.
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-chat": 131_072,
    "deepseek-reasoner": 131_072,
    "deepseek-coder": 16384,
}

# Fallback pricing - LiteLLM is preferred source
# OpenAI pricing per 1M tokens (input, output)
# NOTE: These are ESTIMATES. Always verify against actual OpenAI billing.
# Last updated: 2025-01-14
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-preview": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3": (10.00, 40.00),
    "o3-mini": (1.10, 4.40),
}

# Pattern-based defaults for unknown models
_PATTERN_DEFAULTS = {
    "gpt-4o": {"context": 128000, "encoding": "o200k_base", "pricing": (2.50, 10.00)},
    "gpt-4-turbo": {"context": 128000, "encoding": "cl100k_base", "pricing": (10.00, 30.00)},
    "gpt-4": {"context": 8192, "encoding": "cl100k_base", "pricing": (30.00, 60.00)},
    "gpt-3.5": {"context": 16385, "encoding": "cl100k_base", "pricing": (0.50, 1.50)},
    "o1": {"context": 200000, "encoding": "o200k_base", "pricing": (15.00, 60.00)},
    "o3": {"context": 200000, "encoding": "o200k_base", "pricing": (10.00, 40.00)},
}

# Default for completely unknown OpenAI models
_UNKNOWN_OPENAI_DEFAULT = {
    "context": 128000,
    "encoding": "o200k_base",
    "pricing": (2.50, 10.00),  # GPT-4o tier as reasonable default
}


def _load_custom_model_config() -> dict[str, Any]:
    """Load custom model configuration from environment or config file.

    Checks (in order):
    1. HEADROOM_MODEL_LIMITS environment variable (JSON string or file path)
    2. ~/.headroom/models.json config file

    Returns:
        Dict with 'context_limits' and 'pricing' keys.
    """
    config: dict[str, Any] = {"context_limits": {}, "pricing": {}, "encodings": {}}

    # Check environment variable
    env_config = os.environ.get("HEADROOM_MODEL_LIMITS", "")
    if env_config:
        try:
            # Check if it's a file path
            if os.path.isfile(env_config):
                with open(env_config, encoding="utf-8") as f:
                    loaded = json.load(f)
            else:
                # Try to parse as JSON string
                loaded = json.loads(env_config)

            openai_config = loaded.get("openai", loaded)
            if "context_limits" in openai_config:
                config["context_limits"].update(openai_config["context_limits"])
            if "pricing" in openai_config:
                config["pricing"].update(openai_config["pricing"])
            if "encodings" in openai_config:
                config["encodings"].update(openai_config["encodings"])

            logger.debug("Loaded custom OpenAI model config from HEADROOM_MODEL_LIMITS")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load HEADROOM_MODEL_LIMITS: {e}")

    # Check config file. Prefer the canonical config-dir location, then fall
    # back to the legacy workspace-root location for backward compatibility.
    config_file = _paths.models_config_path()
    if not config_file.exists():
        legacy_models = _paths.workspace_dir() / "models.json"
        if legacy_models.exists():
            config_file = legacy_models
    if config_file.exists():
        try:
            with open(config_file, encoding="utf-8") as f:
                loaded = json.load(f)

            openai_config = loaded.get("openai", {})
            if "context_limits" in openai_config:
                for model, limit in openai_config["context_limits"].items():
                    if model not in config["context_limits"]:
                        config["context_limits"][model] = limit
            if "pricing" in openai_config:
                for model, pricing in openai_config["pricing"].items():
                    if model not in config["pricing"]:
                        config["pricing"][model] = pricing
            if "encodings" in openai_config:
                for model, encoding in openai_config["encodings"].items():
                    if model not in config["encodings"]:
                        config["encodings"][model] = encoding

            logger.debug(f"Loaded custom OpenAI model config from {config_file}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {config_file}: {e}")

    return config


def _infer_model_family(model: str) -> str | None:
    """Infer the model family from model name for pattern-based defaults."""
    model_lower = model.lower()

    # Check in order of specificity
    if model_lower.startswith("gpt-4o"):
        return "gpt-4o"
    elif model_lower.startswith("gpt-4-turbo"):
        return "gpt-4-turbo"
    elif model_lower.startswith("gpt-4"):
        return "gpt-4"
    elif model_lower.startswith("gpt-3.5"):
        return "gpt-3.5"
    elif model_lower.startswith("o1"):
        return "o1"
    elif model_lower.startswith("o3"):
        return "o3"

    return None


def _check_pricing_staleness() -> str | None:
    """Check if pricing data is stale and return warning message if so."""
    global _PRICING_WARNING_SHOWN
    days_old = (date.today() - _PRICING_LAST_UPDATED).days
    if days_old > _PRICING_STALE_DAYS and not _PRICING_WARNING_SHOWN:
        _PRICING_WARNING_SHOWN = True
        return (
            f"OpenAI pricing data is {days_old} days old. "
            "Cost estimates may be inaccurate. Verify against actual billing."
        )
    return None


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str) -> Any:
    """Get tiktoken encoding, cached."""
    if not TIKTOKEN_AVAILABLE:
        raise RuntimeError(
            "tiktoken is required for OpenAI provider. Install with: pip install tiktoken"
        )
    return tiktoken.get_encoding(encoding_name)


def _get_encoding_name_for_model(model: str, custom_encodings: dict[str, str] | None = None) -> str:
    """Get the encoding name for a model with fallback support."""
    # Check custom encodings first
    if custom_encodings and model in custom_encodings:
        return custom_encodings[model]

    # Direct match
    if model in _MODEL_ENCODINGS:
        return _MODEL_ENCODINGS[model]

    # Prefix match for versioned models
    for prefix, encoding in _MODEL_ENCODINGS.items():
        if model.startswith(prefix):
            return encoding

    # Pattern-based inference
    family = _infer_model_family(model)
    if family and family in _PATTERN_DEFAULTS:
        return cast(str, _PATTERN_DEFAULTS[family]["encoding"])

    # Default for unknown models
    return cast(str, _UNKNOWN_OPENAI_DEFAULT["encoding"])


class OpenAITokenCounter:
    """Token counter using tiktoken for OpenAI models."""

    def __init__(self, model: str, custom_encodings: dict[str, str] | None = None):
        """
        Initialize token counter for a model.

        Args:
            model: OpenAI model name.
            custom_encodings: Optional custom model -> encoding mappings.

        Raises:
            RuntimeError: If tiktoken is not installed.
        """
        self.model = model
        encoding_name = _get_encoding_name_for_model(model, custom_encodings)
        self._encoding = _get_encoding(encoding_name)

    def count_text(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        try:
            return len(self._encoding.encode(text))
        except ValueError:
            # Passthrough content can legitimately contain strings that look
            # like tiktoken special tokens (e.g. "<|endoftext|>"). Treat them
            # as ordinary text instead of raising. Matches
            # AnthropicTokenCounter.count_text.
            return len(self._encoding.encode(text, disallowed_special=()))

    def count_message(self, message: dict[str, Any]) -> int:
        """
        Count tokens in a single message.

        Accounts for ChatML format overhead.
        """
        # Base overhead per message (role + delimiters)
        tokens = 4

        role = message.get("role", "")
        tokens += self.count_text(role)

        content = message.get("content")
        if content:
            if isinstance(content, str):
                tokens += self.count_text(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            tokens += self.count_text(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            tokens += 85  # Low detail image estimate
                    elif isinstance(part, str):
                        tokens += self.count_text(part)

        # Name field
        name = message.get("name")
        if name:
            tokens += self.count_text(name) + 1

        # Tool calls in assistant messages
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                tokens += self.count_text(func.get("name", ""))
                tokens += self.count_text(func.get("arguments", ""))
                tokens += self.count_text(tc.get("id", ""))
                tokens += 10  # Structural overhead

        # Tool call ID for tool responses
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            tokens += self.count_text(tool_call_id) + 2

        return tokens

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of messages."""
        total = sum(self.count_message(msg) for msg in messages)
        # Add priming tokens for assistant response
        total += 3
        return total


class OpenAIProvider(Provider):
    """Provider implementation for OpenAI models.

    Custom Model Configuration:
        You can configure custom models via environment variable or config file:

        1. Environment variable (JSON string):
           export HEADROOM_MODEL_LIMITS='{"openai": {"context_limits": {"my-model": 128000}}}'

        2. Environment variable (file path):
           export HEADROOM_MODEL_LIMITS=/path/to/models.json

        3. Config file (~/.headroom/models.json):
           {
             "openai": {
               "context_limits": {"my-model": 128000},
               "pricing": {"my-model": [2.50, 10.00]}
             }
           }
    """

    def __init__(self, context_limits: dict[str, int] | None = None):
        """Initialize OpenAI provider.

        Args:
            context_limits: Optional override for model context limits.
        """
        # Build limits: defaults -> config file -> env var -> explicit
        self._context_limits = {**_CONTEXT_LIMITS}
        self._pricing = {**_PRICING}
        self._encodings: dict[str, str] = {**_MODEL_ENCODINGS}

        # Load from config file and env var
        custom_config = _load_custom_model_config()
        self._context_limits.update(custom_config["context_limits"])
        self._encodings.update(custom_config["encodings"])

        # Handle pricing (can be tuple or list from JSON)
        for model, pricing in custom_config["pricing"].items():
            if isinstance(pricing, list | tuple) and len(pricing) >= 2:
                self._pricing[model] = (float(pricing[0]), float(pricing[1]))

        # Explicit overrides take precedence
        if context_limits:
            self._context_limits.update(context_limits)

        self._token_counters: dict[str, OpenAITokenCounter] = {}

    @property
    def name(self) -> str:
        return "openai"

    def supports_model(self, model: str) -> bool:
        """Check if model is a known OpenAI model."""
        if model in self._context_limits:
            return True
        # Check prefix match
        for prefix in self._context_limits:
            if model.startswith(prefix):
                return True
        # Support any gpt-* or o1/o3 model
        model_lower = model.lower()
        return (
            model_lower.startswith("gpt-")
            or model_lower.startswith("o1")
            or model_lower.startswith("o3")
        )

    def get_token_counter(self, model: str) -> TokenCounter:
        """Get token counter for an OpenAI model."""
        if model not in self._token_counters:
            self._token_counters[model] = OpenAITokenCounter(
                model=model, custom_encodings=self._encodings
            )
        return self._token_counters[model]

    def get_context_limit(self, model: str) -> int:
        """Get context limit for an OpenAI model.

        Resolution order:
        1. LiteLLM (if available, most up-to-date)
        2. Explicit context_limits passed to constructor
        3. HEADROOM_MODEL_LIMITS environment variable
        4. ~/.headroom/models.json config file
        5. Built-in _CONTEXT_LIMITS
        6. Pattern-based inference (gpt-4o, gpt-4, etc.)
        7. Default fallback (128K)

        Never raises an exception - uses sensible defaults for unknown models.
        """
        # Try LiteLLM first
        litellm = _get_litellm_module()
        if litellm is not None:
            try:
                info = litellm.get_model_info(model)
                if info and "max_input_tokens" in info:
                    max_tokens = info["max_input_tokens"]
                    if max_tokens is not None:
                        return int(max_tokens)
            except Exception:
                pass

        # Fall back to hardcoded
        return self._get_context_limit_manual(model)

    def _get_context_limit_manual(self, model: str) -> int:
        """Get context limit using hardcoded values (fallback)."""
        if model in self._context_limits:
            return self._context_limits[model]

        # Prefix match
        for prefix, limit in self._context_limits.items():
            if model.startswith(prefix):
                return limit

        # Pattern-based inference
        family = _infer_model_family(model)
        if family and family in _PATTERN_DEFAULTS:
            limit = cast(int, _PATTERN_DEFAULTS[family]["context"])
            self._warn_unknown_model(model, limit, f"inferred from '{family}' family")
            self._context_limits[model] = limit
            return limit

        # Default for unknown OpenAI models
        limit = cast(int, _UNKNOWN_OPENAI_DEFAULT["context"])
        self._warn_unknown_model(model, limit, "using default limit")
        self._context_limits[model] = limit
        return limit

    def _warn_unknown_model(self, model: str, limit: int, reason: str) -> None:
        """Warn about unknown model (once per model)."""
        global _UNKNOWN_MODEL_WARNINGS
        if model not in _UNKNOWN_MODEL_WARNINGS:
            _UNKNOWN_MODEL_WARNINGS.add(model)
            logger.warning(
                f"Unknown OpenAI model '{model}': {reason} ({limit:,} tokens). "
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
        """Estimate cost for OpenAI API call.

        ⚠️ IMPORTANT: This is an ESTIMATE only.
        - Pricing data may be outdated
        - Cached token discount assumed at 50% (actual may vary)
        - Always verify against your actual OpenAI billing

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.
            cached_tokens: Number of cached tokens (estimated 50% discount).

        Returns:
            Estimated cost in USD, or None if pricing unknown.
        """
        # Try LiteLLM first (most up-to-date pricing)
        litellm = _get_litellm_module()
        if litellm is not None:
            try:
                # LiteLLM uses per-token pricing, returns total cost
                cost = litellm.completion_cost(
                    model=model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                )
                if cost is not None and cost > 0:
                    return float(cost)
            except Exception:
                pass  # Fall through to manual pricing

        # Fall back to hardcoded pricing
        return self._estimate_cost_manual(input_tokens, output_tokens, model, cached_tokens)

    def _estimate_cost_manual(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate cost using hardcoded pricing (fallback)."""
        # Check for stale pricing and warn once
        staleness_warning = _check_pricing_staleness()
        if staleness_warning:
            warnings.warn(staleness_warning, UserWarning, stacklevel=2)

        pricing = self._get_pricing(model)
        if not pricing:
            return None

        input_price, output_price = pricing

        # Calculate cost (cached tokens get estimated 50% discount)
        # NOTE: Actual OpenAI cache discount may vary
        regular_input = input_tokens - cached_tokens
        cached_cost = (cached_tokens / 1_000_000) * input_price * 0.5
        regular_cost = (regular_input / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price

        return cached_cost + regular_cost + output_cost

    def _get_pricing(self, model: str) -> tuple[float, float] | None:
        """Get pricing for a model with fallback logic."""
        # Direct match
        if model in self._pricing:
            return self._pricing[model]

        # Prefix match
        for model_prefix, pricing in self._pricing.items():
            if model.startswith(model_prefix):
                return pricing

        # Pattern-based inference
        family = _infer_model_family(model)
        if family and family in _PATTERN_DEFAULTS:
            return cast(tuple[float, float], _PATTERN_DEFAULTS[family]["pricing"])

        # Default for unknown models
        return cast(tuple[float, float], _UNKNOWN_OPENAI_DEFAULT["pricing"])

    def get_output_buffer(self, model: str, default: int = 4000) -> int:
        """Get recommended output buffer."""
        # Reasoning models produce longer outputs
        if model.startswith("o1") or model.startswith("o3"):
            return 8000
        return default

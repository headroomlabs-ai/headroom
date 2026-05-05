"""Anthropic provider implementation for Headroom SDK.

Token counting uses Anthropic's official Token Count API when a client
is provided. This gives accurate counts for all content types including
JSON, non-English text, and tool definitions.

Usage:
    from anthropic import Anthropic
    from headroom import AnthropicProvider

    client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
    provider = AnthropicProvider(client=client)  # Accurate counting via API

    # Or without client (uses tiktoken approximation - less accurate)
    provider = AnthropicProvider()  # Warning: approximate counting
"""

import importlib.util
import json
import logging
import os
import warnings
from typing import Any, cast

from headroom import paths as _paths

from .base import Provider, TokenCounter

LITELLM_AVAILABLE = importlib.util.find_spec("litellm") is not None


def _get_litellm_clients() -> tuple[Any | None, Any | None]:
    """Import LiteLLM only when pricing/context metadata is needed."""
    if not LITELLM_AVAILABLE:
        return None, None

    try:
        import litellm
        from litellm import get_model_info as litellm_get_model_info
    except ImportError:
        return None, None

    return litellm, litellm_get_model_info


logger = logging.getLogger(__name__)

# Warning flags
_FALLBACK_WARNING_SHOWN = False
_UNKNOWN_MODEL_WARNINGS: set[str] = set()


# Anthropic model context limits
# All Claude 3+ models have 200K context
ANTHROPIC_CONTEXT_LIMITS: dict[str, int] = {
    # Claude 4.7 (Opus 4.7) - 1M context. Claude Code sends the model
    # name with a `[1m]` suffix to select the 1M tier; both forms are
    # registered explicitly because the lookup chain does not strip the
    # tier suffix and LiteLLM does not key on it either.
    "claude-opus-4-7": 1000000,
    "claude-opus-4-7[1m]": 1000000,
    # Claude 4.6 (Opus 4.6) - 1M context
    "claude-opus-4-6": 1000000,
    # Claude 4.5 (Opus 4.5)
    "claude-opus-4-5-20251101": 200000,
    # Claude 4 (Sonnet 4, Haiku 4)
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    # Claude 3.5
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet-latest": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-5-haiku-latest": 200000,
    # Claude 3
    "claude-3-opus-20240229": 200000,
    "claude-3-opus-latest": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    # Claude 2
    "claude-2.1": 200000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000,
}

# Fallback pricing - LiteLLM is preferred source
# NOTE: These are ESTIMATES. Always verify against actual Anthropic billing.
# Last updated: 2025-01-14
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    # Claude 4.7 (Opus tier pricing) — registered for both the bare
    # name and the `[1m]` tier-suffixed form Claude Code sends.
    "claude-opus-4-7": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-opus-4-7[1m]": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    # Claude 4.6 (Opus tier pricing)
    "claude-opus-4-6": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    # Claude 4.5 (Opus tier pricing)
    "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    # Claude 4 (Sonnet/Haiku tier pricing)
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    # Claude 3.5
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    # Claude 3
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-3-opus-latest": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25, "cached_input": 0.03},
}

# Default limits for pattern-based inference
# Used when a model isn't in the explicit list but matches a known pattern
_PATTERN_DEFAULTS = {
    "opus": {"context": 200000, "pricing": {"input": 15.00, "output": 75.00, "cached_input": 1.50}},
    "sonnet": {
        "context": 200000,
        "pricing": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    },
    "haiku": {"context": 200000, "pricing": {"input": 0.80, "output": 4.00, "cached_input": 0.08}},
}

# Fallback for completely unknown Claude models
_UNKNOWN_CLAUDE_DEFAULT = {
    "context": 200000,  # Safe assumption for Claude 3+
    "pricing": {"input": 3.00, "output": 15.00, "cached_input": 0.30},  # Sonnet-tier pricing
}


def _load_custom_model_config() -> dict[str, Any]:
    """Load custom model configuration from environment or config file.

    Checks (in order):
    1. HEADROOM_MODEL_LIMITS environment variable (JSON string or file path)
    2. ~/.headroom/models.json config file

    Returns:
        Dict with 'context_limits' and 'pricing' keys.
    """
    config: dict[str, Any] = {"context_limits": {}, "pricing": {}}

    # Check environment variable
    env_config = os.environ.get("HEADROOM_MODEL_LIMITS", "")
    if env_config:
        try:
            # Check if it's a file path
            if os.path.isfile(env_config):
                with open(env_config) as f:
                    loaded = json.load(f)
            else:
                # Try to parse as JSON string
                loaded = json.loads(env_config)

            # Check for anthropic-specific config, fall back to root level
            anthropic_config = loaded.get("anthropic", loaded)
            if "context_limits" in anthropic_config:
                config["context_limits"].update(anthropic_config["context_limits"])
            if "pricing" in anthropic_config:
                config["pricing"].update(anthropic_config["pricing"])

            logger.debug(f"Loaded custom model config from HEADROOM_MODEL_LIMITS: {loaded}")
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
            with open(config_file) as f:
                loaded = json.load(f)

            # Only load anthropic-specific config
            anthropic_config = loaded.get("anthropic", loaded)
            if "context_limits" in anthropic_config:
                # Don't override env var settings
                for model, limit in anthropic_config["context_limits"].items():
                    if model not in config["context_limits"]:
                        config["context_limits"][model] = limit
            if "pricing" in anthropic_config:
                for model, pricing in anthropic_config["pricing"].items():
                    if model not in config["pricing"]:
                        config["pricing"][model] = pricing

            logger.debug(f"Loaded custom model config from {config_file}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {config_file}: {e}")

    return config


def _infer_model_tier(model: str) -> str | None:
    """Infer the model tier (opus/sonnet/haiku) from model name.

    Uses pattern matching to handle future model releases.
    """
    model_lower = model.lower()

    # Check for tier keywords in model name
    if "opus" in model_lower:
        return "opus"
    elif "sonnet" in model_lower:
        return "sonnet"
    elif "haiku" in model_lower:
        return "haiku"

    return None


class AnthropicTokenCounter(TokenCounter):
    """Token counter for Anthropic models.

    When an Anthropic client is provided, uses the official Token Count API
    (/v1/messages/count_tokens) for accurate counting. This handles:
    - JSON-heavy tool payloads
    - Non-English text
    - Tool definitions and structured content

    Falls back to tiktoken approximation only when no client is available.
    """

    def __init__(self, model: str, client: Any = None):
        """Initialize token counter.

        Args:
            model: Anthropic model name.
            client: Optional anthropic.Anthropic client for API-based counting.
                    If not provided, falls back to tiktoken approximation.
        """
        global _FALLBACK_WARNING_SHOWN

        self.model = model
        self._client = client
        self._encoding: Any = None
        self._use_api = client is not None

        if not self._use_api and not _FALLBACK_WARNING_SHOWN:
            warnings.warn(
                "AnthropicProvider: No client provided, using tiktoken approximation. "
                "For accurate counting, pass an Anthropic client: "
                "AnthropicProvider(client=Anthropic())",
                UserWarning,
                stacklevel=4,
            )
            _FALLBACK_WARNING_SHOWN = True

        # Load tiktoken as fallback
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            if not self._use_api:
                warnings.warn(
                    "tiktoken not installed - token counting will be very approximate. "
                    "Install tiktoken or provide an Anthropic client.",
                    UserWarning,
                    stacklevel=4,
                )

    def count_text(self, text: str) -> int:
        """Count tokens in text.

        Note: For single text strings, uses tiktoken approximation even when
        API is available (API only supports full message counting).
        """
        if not text:
            return 0

        if self._encoding:
            # tiktoken with ~1.1x multiplier for Claude
            try:
                base_count = len(self._encoding.encode(text))
            except ValueError:
                # Real tool output can legitimately contain strings that look like
                # tiktoken special tokens (for example FIM markers in code spans).
                # Treat them as ordinary text for estimation instead of failing.
                base_count = len(self._encoding.encode(text, disallowed_special=()))
            return int(base_count * 1.1)

        # Character-based fallback
        return max(1, len(text) // 3)

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a single message.

        Uses API if available, otherwise falls back to estimation.
        """
        if self._use_api:
            return self._count_message_via_api(message)
        return self._count_message_estimated(message)

    def _count_message_via_api(self, message: dict[str, Any]) -> int:
        """Count tokens using Anthropic Token Count API."""
        try:
            # Convert to Anthropic message format if needed
            messages = [self._normalize_message(message)]
            response = self._client.messages.count_tokens(
                model=self.model,
                messages=messages,
            )
            return int(response.input_tokens)
        except Exception:
            # Fall back to estimation on API error
            return self._count_message_estimated(message)

    def _count_message_estimated(self, message: dict[str, Any]) -> int:
        """Estimate token count without API."""
        tokens = 4  # Role overhead

        content = message.get("content")
        if isinstance(content, str):
            tokens += self.count_text(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        tokens += self.count_text(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tokens += self.count_text(block.get("name", ""))
                        tokens += self.count_text(str(block.get("input", {})))
                    elif block.get("type") == "tool_result":
                        tokens += self.count_text(str(block.get("content", "")))

        # OpenAI format tool calls
        if "tool_calls" in message:
            for tool_call in message.get("tool_calls", []):
                if isinstance(tool_call, dict):
                    func = tool_call.get("function", {})
                    tokens += self.count_text(func.get("name", ""))
                    tokens += self.count_text(func.get("arguments", ""))

        return tokens

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Normalize message to Anthropic format."""
        role = message.get("role", "user")

        # Map OpenAI roles to Anthropic
        if role == "system":
            # System messages need special handling - count as user for API
            return {"role": "user", "content": message.get("content", "")}
        elif role == "tool":
            # Tool results in OpenAI format
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": message.get("content", ""),
                    }
                ],
            }

        return {"role": role, "content": message.get("content", "")}

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of messages.

        Uses the Token Count API for accurate counting when available.
        """
        if self._use_api:
            return self._count_messages_via_api(messages)
        return self._count_messages_estimated(messages)

    def _count_messages_via_api(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens using Anthropic Token Count API."""
        try:
            # Separate system message (Anthropic handles it differently)
            system_content = None
            api_messages = []

            for msg in messages:
                if msg.get("role") == "system":
                    system_content = msg.get("content", "")
                else:
                    api_messages.append(self._normalize_message(msg))

            # Ensure we have at least one message
            if not api_messages:
                api_messages = [{"role": "user", "content": ""}]

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": api_messages,
            }
            if system_content:
                kwargs["system"] = system_content

            response = self._client.messages.count_tokens(**kwargs)
            return int(response.input_tokens)

        except Exception as e:
            # Fall back to estimation on API error
            warnings.warn(
                f"Token Count API failed ({e}), using estimation", UserWarning, stacklevel=3
            )
            return self._count_messages_estimated(messages)

    def _count_messages_estimated(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token count without API."""
        total = sum(self._count_message_estimated(msg) for msg in messages)
        return total + 3  # Base overhead


class AnthropicProvider(Provider):
    """Provider implementation for Anthropic Claude models.

    For accurate token counting, provide an Anthropic client:

        from anthropic import Anthropic
        provider = AnthropicProvider(client=Anthropic())

    This uses Anthropic's official Token Count API which accurately handles:
    - JSON-heavy tool payloads
    - Non-English text
    - Long system prompts
    - Tool definitions and structured content

    Without a client, falls back to tiktoken approximation (less accurate).

    Custom Model Configuration:
        You can configure custom models via environment variable or config file:

        1. Environment variable (JSON string):
           export HEADROOM_MODEL_LIMITS='{"context_limits": {"my-model": 200000}}'

        2. Environment variable (file path):
           export HEADROOM_MODEL_LIMITS=/path/to/models.json

        3. Config file (~/.headroom/models.json):
           {
             "anthropic": {
               "context_limits": {"my-model": 200000},
               "pricing": {"my-model": {"input": 3.0, "output": 15.0}}
             }
           }
    """

    def __init__(
        self,
        client: Any = None,
        context_limits: dict[str, int] | None = None,
    ):
        """Initialize Anthropic provider.

        Args:
            client: Optional anthropic.Anthropic client for accurate token counting.
                    If not provided, uses tiktoken approximation.
            context_limits: Optional override for model context limits.

        Example:
            from anthropic import Anthropic
            provider = AnthropicProvider(client=Anthropic())
        """
        self._client = client
        self._token_counters: dict[str, AnthropicTokenCounter] = {}

        # Build context limits: defaults -> config file -> env var -> explicit
        self._context_limits = {**ANTHROPIC_CONTEXT_LIMITS}
        self._pricing = {**ANTHROPIC_PRICING}

        # Load from config file and env var
        custom_config = _load_custom_model_config()
        self._context_limits.update(custom_config["context_limits"])
        self._pricing.update(custom_config["pricing"])

        # Explicit overrides take precedence
        if context_limits:
            self._context_limits.update(context_limits)

    @property
    def name(self) -> str:
        return "anthropic"

    def get_token_counter(self, model: str) -> TokenCounter:
        """Get token counter for a model.

        If a client was provided to the provider, uses the Token Count API.
        Otherwise falls back to tiktoken approximation.
        """
        if model not in self._token_counters:
            self._token_counters[model] = AnthropicTokenCounter(
                model=model,
                client=self._client,
            )
        return self._token_counters[model]

    def get_context_limit(self, model: str) -> int:
        """Get context window limit for a model.

        Resolution order:
        1. Explicit context_limits passed to constructor
        2. HEADROOM_MODEL_LIMITS environment variable
        3. ~/.headroom/models.json config file
        4. LiteLLM model info (if available)
        5. Built-in ANTHROPIC_CONTEXT_LIMITS
        6. Pattern-based inference (opus/sonnet/haiku)
        7. Default fallback (200K for any Claude model)

        Never raises an exception - uses sensible defaults for unknown models.
        """
        # Check explicit and loaded limits
        if model in self._context_limits:
            return self._context_limits[model]

        # Check for partial matches (e.g., "claude-3-5-sonnet" matches "claude-3-5-sonnet-20241022")
        for known_model, limit in self._context_limits.items():
            if model in known_model or known_model in model:
                return limit

        # Try LiteLLM for context limit
        _, litellm_get_model_info = _get_litellm_clients()
        if litellm_get_model_info is not None:
            try:
                info = litellm_get_model_info(model)
                if info:
                    if "max_input_tokens" in info and info["max_input_tokens"] is not None:
                        limit = int(info["max_input_tokens"])
                        self._context_limits[model] = limit
                        return limit
                    if "max_tokens" in info and info["max_tokens"] is not None:
                        limit = int(info["max_tokens"])
                        self._context_limits[model] = limit
                        return limit
            except Exception as e:
                logger.debug(f"LiteLLM get_model_info failed for {model}: {e}")

        # Pattern-based inference for new models
        tier = _infer_model_tier(model)
        if tier and tier in _PATTERN_DEFAULTS:
            limit = cast(int, _PATTERN_DEFAULTS[tier]["context"])
            self._warn_unknown_model(model, limit, f"inferred from '{tier}' tier")
            # Cache for future calls
            self._context_limits[model] = limit
            return limit

        # Fallback for unknown Claude models
        if model.startswith("claude"):
            limit = cast(int, _UNKNOWN_CLAUDE_DEFAULT["context"])
            self._warn_unknown_model(model, limit, "using default Claude limit")
            self._context_limits[model] = limit
            return limit

        # Non-Claude model - use conservative default
        limit = 128000
        self._warn_unknown_model(model, limit, "unknown provider, using conservative default")
        self._context_limits[model] = limit
        return limit

    def _warn_unknown_model(self, model: str, limit: int, reason: str) -> None:
        """Warn about unknown model (once per model)."""
        global _UNKNOWN_MODEL_WARNINGS
        if model not in _UNKNOWN_MODEL_WARNINGS:
            _UNKNOWN_MODEL_WARNINGS.add(model)
            logger.warning(
                f"Unknown Anthropic model '{model}': {reason} ({limit:,} tokens). "
                f"To configure explicitly, set HEADROOM_MODEL_LIMITS env var or "
                f"add to ~/.headroom/models.json"
            )

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model."""
        if model in self._context_limits:
            return True
        # Check prefix matches - support all Claude models
        return model.startswith("claude")

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cached_tokens: int = 0,
    ) -> float | None:
        """Estimate cost for a request.

        Tries LiteLLM first for up-to-date pricing, falls back to manual pricing.
        """
        # Try LiteLLM first for cost estimation
        litellm, litellm_get_model_info = _get_litellm_clients()
        if litellm is not None:
            try:
                cost = litellm.completion_cost(
                    model=model,
                    prompt="",
                    completion="",
                    prompt_tokens=input_tokens - cached_tokens,
                    completion_tokens=output_tokens,
                )
                # Add cached token cost if applicable
                if cached_tokens > 0:
                    try:
                        # Get cached input pricing from LiteLLM model info
                        info = (
                            litellm_get_model_info(model)
                            if litellm_get_model_info is not None
                            else None
                        )
                        if info and "input_cost_per_token" in info:
                            # LiteLLM typically applies 90% discount for cached tokens
                            cached_cost = cached_tokens * info["input_cost_per_token"] * 0.1
                            cost += cached_cost
                    except Exception:
                        # Fall back to manual cached pricing
                        pricing = self._get_pricing(model)
                        if pricing:
                            cached_cost = (cached_tokens / 1_000_000) * pricing.get(
                                "cached_input", pricing["input"]
                            )
                            cost += cached_cost
                return cost  # type: ignore[no-any-return]
            except Exception as e:
                logger.debug(f"LiteLLM cost estimation failed for {model}: {e}")

        # Fall back to manual pricing
        pricing = self._get_pricing(model)
        if not pricing:
            return None

        # Calculate cost
        non_cached_input = input_tokens - cached_tokens
        cost = (
            (non_cached_input / 1_000_000) * pricing["input"]
            + (cached_tokens / 1_000_000) * pricing.get("cached_input", pricing["input"])
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        return cost  # type: ignore[no-any-return]

    def _get_pricing(self, model: str) -> dict[str, float] | None:
        """Get pricing for a model with fallback logic."""
        # Direct match
        if model in self._pricing:
            return self._pricing[model]

        # Partial match
        for known_model, prices in self._pricing.items():
            if model in known_model or known_model in model:
                return prices

        # Pattern-based inference
        tier = _infer_model_tier(model)
        if tier and tier in _PATTERN_DEFAULTS:
            return cast(dict[str, float], _PATTERN_DEFAULTS[tier]["pricing"])

        # Default for unknown Claude models
        if model.startswith("claude"):
            return cast(dict[str, float], _UNKNOWN_CLAUDE_DEFAULT["pricing"])

        return None

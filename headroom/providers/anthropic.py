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
import re
import warnings
from typing import Any, cast

from headroom import paths as _paths
from headroom.pricing.litellm_pricing import estimate_cost_from_tokens
from headroom.tokenizers.base import (
    TokenCountCache,
    coerce_countable_text,
    count_content_blocks,
)

from .base import Provider, TokenCounter

LITELLM_AVAILABLE = importlib.util.find_spec("litellm") is not None


def _get_litellm_clients() -> tuple[Any | None, Any | None]:
    """Import LiteLLM only when pricing/context metadata is needed."""
    if not LITELLM_AVAILABLE:
        return None, None

    try:
        import litellm

        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        from litellm import get_model_info as litellm_get_model_info
    except ImportError:
        return None, None

    return litellm, litellm_get_model_info


logger = logging.getLogger(__name__)

# Warning flags
_FALLBACK_WARNING_SHOWN = False
_UNKNOWN_MODEL_WARNINGS: set[str] = set()
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_DANGLING_ANSI_STYLE_SUFFIX_RE = re.compile(r"(?:\[[0-9;]*m\])+$")


def sanitize_anthropic_model_id(model: str) -> str:
    """Return an Anthropic model id without terminal styling artifacts."""
    cleaned = _ANSI_ESCAPE_RE.sub("", str(model)).strip()
    return _DANGLING_ANSI_STYLE_SUFFIX_RE.sub("", cleaned)


# `[1m]` is not only an ANSI artifact: Claude Code appends it to a model id to
# request the 1M context tier, and only then sends the `context-1m` beta header
# (#1158). Upstream rejects the suffix, so `sanitize_anthropic_model_id()` must
# keep stripping it before forwarding (#2027) — but the tier it encodes has to
# be read off the id *before* that happens, or a 1M request gets budgeted as if
# it were the base model's window.
_CONTEXT_1M_SUFFIX_RE = re.compile(r"(?:\[1m\])+$")
CONTEXT_1M_TOKENS = 1_000_000


def has_context_1m_suffix(model: str) -> bool:
    """Return True if ``model`` carries Claude Code's ``[1m]`` 1M-tier marker."""
    return bool(_CONTEXT_1M_SUFFIX_RE.search(_ANSI_ESCAPE_RE.sub("", str(model)).strip()))


def sanitize_anthropic_model_metadata(value: Any) -> Any:
    """Strip model-id styling artifacts from Anthropic model metadata payloads."""
    if isinstance(value, list):
        return [sanitize_anthropic_model_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value

    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"id", "model"} and isinstance(item, str):
            cleaned[key] = sanitize_anthropic_model_id(item)
        else:
            cleaned[key] = sanitize_anthropic_model_metadata(item)
    return cleaned


# Anthropic model context limits
# All Claude 3+ models have 200K context
ANTHROPIC_CONTEXT_LIMITS: dict[str, int] = {
    # Claude Fable 5 - 1M context
    "claude-fable-5": 1000000,
    # Claude Opus 4.8 - 1M context
    "claude-opus-4-8": 1000000,
    # Claude 4.7 (Opus 4.7) - 1M context
    "claude-opus-4-7": 1000000,
    # Claude 4.6 (Opus 4.6) - 1M context
    "claude-opus-4-6": 1000000,
    # Claude 4.5 (Opus 4.5)
    "claude-opus-4-5-20251101": 200000,
    # Claude Sonnet 5 - 1M context
    "claude-sonnet-5": 1000000,
    # Claude Sonnet 4.6 - 1M context window
    "claude-sonnet-4-6": 1000000,
    # Claude Sonnet 4.5
    "claude-sonnet-4-5": 200000,
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
# Last updated: 2026-07-04
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    # Claude Fable 5 (anthropic.com/pricing): $10 in / $50 out, cache read $1.
    "claude-fable-5": {"input": 10.00, "output": 50.00, "cached_input": 1.00},
    # Claude Opus 4.8 — current Opus tier: $5 in / $25 out, cache read $0.50.
    "claude-opus-4-8": {"input": 5.00, "output": 25.00, "cached_input": 0.50},
    # Claude 4.7 (current Opus tier)
    "claude-opus-4-7": {"input": 5.00, "output": 25.00, "cached_input": 0.50},
    # Claude 4.6 (current Opus tier)
    "claude-opus-4-6": {"input": 5.00, "output": 25.00, "cached_input": 0.50},
    # Claude 4.5 (current Opus tier — same rates as 4.6–4.8)
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00, "cached_input": 0.50},
    # Claude Sonnet 5 / 4.6 / 4.5 (current Sonnet tier): $3 in / $15 out, cache read $0.30
    "claude-sonnet-5": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    # Claude 4 (Sonnet/Haiku tier pricing)
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "cached_input": 0.10},
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

# Anthropic's long-context premium. On models that reach 1M over a 200K base,
# a prompt above 200K re-prices the *entire* request -- input, output and cache
# alike -- rather than only the tokens past the threshold. Multipliers are
# derived from LiteLLM's `*_above_200k_tokens` fields ($3->$6 in, $15->$22.50
# out, $0.30->$0.60 cache read).
#
# Only the Sonnet 4 / 4.5 family is tiered: Opus, and Sonnet 4.6 onward, are
# flat-rated across their whole window. This is the same population that needs
# the `[1m]` suffix to reach 1M at all, so a session that fills the window this
# unlocks is billed at these rates.
_LONG_CONTEXT_THRESHOLD = 200_000
_LONG_CONTEXT_PREMIUM: dict[str, float] = {"input": 2.0, "output": 1.5, "cached_input": 2.0}
_LONG_CONTEXT_TIERED_MODELS = (
    "claude-sonnet-4-5",
    "claude-sonnet-4-20250514",
    "claude-4-sonnet-20250514",
)


def _apply_long_context_premium(
    model: str, pricing: dict[str, float], input_tokens: int
) -> dict[str, float]:
    """Return ``pricing`` scaled by the long-context premium where it applies.

    Used only on the manual fallback path; the LiteLLM path already applies the
    published above-threshold rates itself.
    """
    if input_tokens <= _LONG_CONTEXT_THRESHOLD:
        return pricing
    if not any(model.startswith(tiered) for tiered in _LONG_CONTEXT_TIERED_MODELS):
        return pricing
    return {key: rate * _LONG_CONTEXT_PREMIUM.get(key, 1.0) for key, rate in pricing.items()}


# Default limits for pattern-based inference
# Used when a model isn't in the explicit list but matches a known pattern
_PATTERN_DEFAULTS = {
    "opus": {"context": 200000, "pricing": {"input": 5.00, "output": 25.00, "cached_input": 0.50}},
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


# DeepSeek fallback pricing for --anthropic-api-url deepseek routing
_DEEPSEEK_FALLBACK_PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28, "cached_input": 0.0028},
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87, "cached_input": 0.003625},
}


def _get_deepseek_pricing(model: str) -> dict[str, float] | None:
    """Get fallback pricing for a DeepSeek model.

    Used when the Anthropic provider encounters a deepseek-* model name
    (via --anthropic-api-url pointing at DeepSeek's Anthropic-compatible
    endpoint) and LiteLLM is unavailable.

    Args:
        model: The model name to look up.

    Returns:
        Pricing dict with input/output/cached_input keys, or None.
    """
    # Direct match
    if model in _DEEPSEEK_FALLBACK_PRICING:
        return cast(dict[str, float], _DEEPSEEK_FALLBACK_PRICING[model])
    # Partial match
    for known_model, prices in _DEEPSEEK_FALLBACK_PRICING.items():
        if model in known_model or known_model in model:
            return cast(dict[str, float], prices)
    return None


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
                with open(env_config, encoding="utf-8") as f:
                    loaded = json.load(f)
            else:
                # Try to parse as JSON string
                loaded = json.loads(env_config)

            if not isinstance(loaded, dict):
                raise ValueError(
                    f"HEADROOM_MODEL_LIMITS must be a JSON object, got {type(loaded).__name__}"
                )

            # Check for anthropic-specific config, fall back to root level
            anthropic_config = loaded.get("anthropic", loaded)
            if "context_limits" in anthropic_config:
                config["context_limits"].update(anthropic_config["context_limits"])
            if "pricing" in anthropic_config:
                config["pricing"].update(anthropic_config["pricing"])
            if (
                "context_limits" not in anthropic_config
                and "pricing" not in anthropic_config
            ):
                # Valid JSON object, but none of the keys we consume. Previously
                # this was a SILENT no-op: the unknown-model warning tells the
                # operator to "set HEADROOM_MODEL_LIMITS", they set the obvious
                # flat shape {"my-model": 262144}, nothing happens, and there is
                # no diagnostic anywhere. Name the expected shape instead.
                logger.warning(
                    "HEADROOM_MODEL_LIMITS parsed but contained no 'context_limits' "
                    "or 'pricing' key, so it had NO EFFECT. Expected shape: "
                    '{"context_limits": {"<model>": <int>}, "pricing": {...}} '
                    '(optionally nested under an "anthropic" key). '
                    f"Got top-level keys: {sorted(map(str, anthropic_config))[:10]}"
                )

            logger.debug(f"Loaded custom model config from HEADROOM_MODEL_LIMITS: {loaded}")
        except (ValueError, OSError) as e:
            # ValueError covers json.JSONDecodeError (a subclass) and the
            # non-object guard above, so a malformed value warns and falls back
            # to defaults instead of crashing provider init.
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

            if not isinstance(loaded, dict):
                raise ValueError(f"{config_file} must contain a JSON object")

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
        except (ValueError, OSError) as e:
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

    def __init__(self, model: str, client: Any = None, warn: bool = True):
        """Initialize token counter.

        Args:
            model: Anthropic model name.
            client: Optional anthropic.Anthropic client for API-based counting.
                    If not provided, falls back to tiktoken approximation.
            warn: If False, suppresses the no-client UserWarning (useful for
                  internal proxy usage where approximation is intentional).
        """
        global _FALLBACK_WARNING_SHOWN

        self.model = model
        self._client = client
        self._encoding: Any = None
        self._count_cache = TokenCountCache()
        self._use_api = client is not None

        if not self._use_api and warn and not _FALLBACK_WARNING_SHOWN:
            warnings.warn(
                "AnthropicProvider: No client provided, using tiktoken approximation. "
                "For accurate counting, pass an Anthropic client: "
                "AnthropicProvider(client=Anthropic())",
                UserWarning,
                stacklevel=4,
            )
            _FALLBACK_WARNING_SHOWN = True

        # Load tiktoken as fallback — bounded, so a stalled vocab download can't
        # hang token counting inside a request (tiktoken's downloader has no
        # network timeout); on timeout we estimate by characters instead (GH #956).
        try:
            from headroom.tokenizers.tiktoken_counter import (
                TiktokenLoadError,
                load_encoding,
            )

            self._encoding = load_encoding("cl100k_base")
        except TiktokenLoadError:
            self._encoding = None  # count_text() falls back to a character estimate
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

        cached = self._count_cache.get(text)
        if cached is not None:
            return cached
        count = self._count_text_uncached(text)
        self._count_cache.put(text, count)
        return count

    def _count_text_uncached(self, text: str) -> int:
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
            # Delegate to the audited shared walker instead of a partial
            # per-provider one. Each provider counter had grown its own
            # shortened branch list, so every modern block priced at ~0:
            # measured on a 6,800-char block this returned 8 tokens for
            # tool_result, thinking, document, mcp_tool_result — and for
            # output_text / refusal, which are OpenAI's OWN Responses shapes.
            # The shared walker is also image-safe: a 200KB base64 image gets
            # a pixel-based 1600, not the ~50K phantom text tokens a naive
            # str(block) catch-all would produce.
            tokens += count_content_blocks(content, self.count_text)

        # OpenAI format tool calls. Guard the value, not just the key: an
        # OpenAI-format assistant message often carries `tool_calls: null` on a
        # no-tool turn, and `for ... in None` would raise TypeError.
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                func = tool_call.get("function") or {}
                tokens += self.count_text(coerce_countable_text(func.get("name")))
                tokens += self.count_text(coerce_countable_text(func.get("arguments")))

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
        warn: bool = True,
    ):
        """Initialize Anthropic provider.

        Args:
            client: Optional anthropic.Anthropic client for accurate token counting.
                    If not provided, uses tiktoken approximation.
            context_limits: Optional override for model context limits.
            warn: If False, suppresses the no-client UserWarning. Set to False
                  in contexts where tiktoken approximation is intentional (e.g.
                  the internal proxy pipeline provider).

        Example:
            from anthropic import Anthropic
            provider = AnthropicProvider(client=Anthropic())
        """
        self._client = client
        self._warn = warn
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
        model = sanitize_anthropic_model_id(model)
        if model not in self._token_counters:
            self._token_counters[model] = AnthropicTokenCounter(
                model=model,
                client=self._client,
                warn=self._warn,
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

        A ``[1m]`` suffix raises the result to at least 1M: the caller asked for
        the 1M tier and Claude Code sent the `context-1m` beta header, so the
        real upstream window is 1M even when the base model's default is 200K.

        Never raises an exception - uses sensible defaults for unknown models.
        """
        if has_context_1m_suffix(model):
            # Recursion terminates: the sanitized id has no `[1m]` left.
            base = self.get_context_limit(sanitize_anthropic_model_id(model))
            return max(base, CONTEXT_1M_TOKENS)
        model = sanitize_anthropic_model_id(model)
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
        model = sanitize_anthropic_model_id(model)
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
        Both paths apply Anthropic's long-context premium: on the Sonnet 4 / 4.5
        family a prompt over 200K re-prices the whole request (see
        ``_LONG_CONTEXT_PREMIUM``).
        """
        model = sanitize_anthropic_model_id(model)
        # LiteLLM knows per-model cache and long-context rates, so let it price
        # the whole request rather than rebuilding the rate card here.
        cost = estimate_cost_from_tokens(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )
        if cost is not None:
            return cost

        # Fall back to manual pricing
        pricing = self._get_pricing(model)
        if not pricing:
            return None

        rates = _apply_long_context_premium(model, pricing, input_tokens)

        # Calculate cost
        non_cached_input = input_tokens - cached_tokens
        cost = (
            (non_cached_input / 1_000_000) * rates["input"]
            + (cached_tokens / 1_000_000) * rates.get("cached_input", rates["input"])
            + (output_tokens / 1_000_000) * rates["output"]
        )

        return cost

    def _get_pricing(self, model: str) -> dict[str, float] | None:
        """Get pricing for a model with fallback logic."""
        model = sanitize_anthropic_model_id(model)
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

        # DeepSeek model fallback (via --anthropic-api-url)
        if model.startswith("deepseek"):
            return _get_deepseek_pricing(model)

        return None

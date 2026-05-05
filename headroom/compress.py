"""One-function compression API for Headroom.

The simplest way to use Headroom — no proxy, no config, just compress:

    from headroom import compress

    result = compress(messages, model="claude-sonnet-4-5-20250929")
    result.messages          # Compressed messages (same format, fewer tokens)
    result.tokens_saved      # Tokens saved
    result.compression_ratio # e.g., 0.35 means 65% saved

Works with any LLM client, any proxy, any framework. Just compress
the messages before sending them.

Examples:

    # With Anthropic SDK
    from anthropic import Anthropic
    from headroom import compress

    client = Anthropic()
    messages = [{"role": "user", "content": huge_tool_output}]
    compressed = compress(messages, model="claude-sonnet-4-5-20250929")
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        messages=compressed.messages,
    )

    # With OpenAI SDK
    from openai import OpenAI
    from headroom import compress

    client = OpenAI()
    messages = [{"role": "user", "content": "analyze this"}, {"role": "tool", "content": big_data}]
    compressed = compress(messages, model="gpt-4o")
    response = client.chat.completions.create(model="gpt-4o", messages=compressed.messages)

    # With LiteLLM
    import litellm
    from headroom import compress

    messages = [...]
    compressed = compress(messages, model="bedrock/claude-sonnet")
    response = litellm.completion(model="bedrock/claude-sonnet", messages=compressed.messages)

    # With any HTTP client
    import httpx
    from headroom import compress

    compressed = compress(messages, model="claude-sonnet-4-5-20250929")
    httpx.post("https://api.anthropic.com/v1/messages", json={
        "model": "claude-sonnet-4-5-20250929",
        "messages": compressed.messages,
    })
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from .observability import get_otel_metrics
from .pipeline import PipelineExtensionManager, PipelineStage, summarize_routing_markers
from .utils import extract_user_query as _extract_user_query

logger = logging.getLogger(__name__)


# Lazy-initialized singleton pipeline
_pipeline = None
_pipeline_lock = threading.Lock()


@dataclass
class CompressConfig:
    """User-facing compression options.

    Controls what gets compressed, how aggressively, and with which model.
    Pass to ``compress()`` or any integration that uses headroom.

    Examples::

        # Coding agent (default — skip user messages, protect recent)
        compress(messages, model="gpt-4o")

        # Financial document (compress everything, keep 50%)
        compress(messages, model="claude-opus-4-20250514",
            compress_user_messages=True,
            target_ratio=0.5,
            protect_recent=0,
        )

        # Aggressive (logs, search results)
        compress(messages, model="gpt-4o", target_ratio=0.2)
    """

    # What to compress
    compress_user_messages: bool = False
    """Compress user messages too (default: skip them for coding agents).
    Set True for document compression, RAG pipelines, or when user messages
    contain large tool outputs."""

    compress_system_messages: bool = True
    """Compress system messages (default: True).
    Set False to preserve system prompts exactly as-is. Useful for voice
    agents where tool definitions and instructions must not be altered."""

    protect_recent: int = 4
    """Don't compress the last N messages (they're the active conversation).
    Set 0 to compress everything."""

    protect_analysis_context: bool = True
    """Detect 'analyze'/'review' intent and protect code from compression."""

    # How aggressive
    target_ratio: float | None = None
    """Keep ratio for Kompress. None = model decides (~15% kept, aggressive).
    0.5 = keep 50% (safe for documents). 0.7 = keep 70% (conservative).
    Only affects Kompress (text compression). SmartCrusher (JSON) has its
    own logic based on array dedup."""

    min_tokens_to_compress: int = 250
    """Minimum token count for a message to be compressed.
    Messages shorter than this are left unchanged. Default 250.
    Set lower for voice agents where turns are short."""

    # Model variant
    kompress_model: str | None = None
    """Kompress model ID. None = default (chopratejas/kompress-base).
    Set to a HuggingFace model ID for domain-specific compression.
    Set to 'disabled' to skip ML compression entirely
    (only SmartCrusher + CacheAligner will run)."""


@dataclass
class CompressResult:
    """Result of compressing messages.

    Attributes:
        messages: The compressed messages (same format as input).
        tokens_before: Token count before compression.
        tokens_after: Token count after compression.
        tokens_saved: Tokens removed by compression.
        compression_ratio: Ratio of tokens saved (0.0 = no savings, 1.0 = 100% removed).
        transforms_applied: List of transforms that were applied.
    """

    messages: list[dict[str, Any]]
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 0.0
    transforms_applied: list[str] = field(default_factory=list)


def compress(
    messages: list[dict[str, Any]],
    model: str = "claude-sonnet-4-5-20250929",
    model_limit: int = 200000,
    optimize: bool = True,
    hooks: Any = None,
    config: CompressConfig | None = None,
    **kwargs: Any,
) -> CompressResult:
    """Compress messages using Headroom's full compression pipeline.

    This is the simplest way to use Headroom. No proxy, no config needed.
    Just pass messages and get compressed messages back.

    Args:
        messages: List of messages in Anthropic or OpenAI format.
        model: Model name (used for token counting and context limit).
        model_limit: Model's context window size in tokens.
        optimize: Whether to actually compress (False = passthrough for A/B testing).
        hooks: Optional CompressionHooks instance for custom behavior.
        config: Compression options (CompressConfig). Overrides defaults.
        **kwargs: Shorthand for CompressConfig fields. These override config:
            compress_user_messages, target_ratio, protect_recent,
            protect_analysis_context, kompress_model.

    Returns:
        CompressResult with compressed messages and metrics.

    Examples::

        # Default (coding agent)
        result = compress(messages, model="gpt-4o")

        # Financial document (keep 50%, compress everything)
        result = compress(messages, model="claude-opus-4-20250514",
            compress_user_messages=True,
            target_ratio=0.5,
            protect_recent=0,
        )
    """
    if not messages or not optimize:
        return CompressResult(messages=messages)

    # Build config from explicit config + kwargs
    cfg = config or CompressConfig()
    config_fields = {f.name for f in cfg.__dataclass_fields__.values()}
    for key, value in kwargs.items():
        if key in config_fields:
            setattr(cfg, key, value)

    pipeline = _get_pipeline()
    pipeline_extensions = PipelineExtensionManager(hooks=hooks, discover=False)

    try:
        # Compute biases from hooks if provided
        biases = None
        if hooks:
            from headroom.hooks import CompressContext

            ctx = CompressContext(model=model)
            messages = hooks.pre_compress(messages, ctx)
            biases = hooks.compute_biases(messages, ctx)

        received_event = pipeline_extensions.emit(
            PipelineStage.INPUT_RECEIVED,
            operation="compress",
            model=model,
            messages=messages,
        )
        if received_event.messages is not None:
            messages = received_event.messages

        # Extract user query from messages so transforms can score by
        # relevance.  Without this, SmartCrusher selects items by statistics
        # alone (position, anomaly) and may drop relevant content.
        context = _extract_user_query(messages)

        result = pipeline.apply(
            messages=messages,
            model=model,
            model_limit=model_limit,
            context=context,
            biases=biases,
            # Pass CompressConfig options through to transforms
            compress_user_messages=cfg.compress_user_messages,
            compress_system_messages=cfg.compress_system_messages,
            target_ratio=cfg.target_ratio,
            protect_recent=cfg.protect_recent,
            protect_analysis_context=cfg.protect_analysis_context,
            min_tokens_to_compress=cfg.min_tokens_to_compress,
            kompress_model=cfg.kompress_model,
        )

        tokens_before = result.tokens_before
        tokens_after = result.tokens_after
        compressed_messages = result.messages

        routing_markers = summarize_routing_markers(result.transforms_applied)
        if routing_markers:
            routed_event = pipeline_extensions.emit(
                PipelineStage.INPUT_ROUTED,
                operation="compress",
                model=model,
                messages=compressed_messages,
                metadata={
                    "routing_markers": routing_markers,
                    "transforms_applied": result.transforms_applied,
                },
            )
            if routed_event.messages is not None:
                compressed_messages = routed_event.messages

        compressed_event = pipeline_extensions.emit(
            PipelineStage.INPUT_COMPRESSED,
            operation="compress",
            model=model,
            messages=compressed_messages,
            metadata={
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "transforms_applied": result.transforms_applied,
            },
        )
        if compressed_event.messages is not None:
            compressed_messages = compressed_event.messages

        tokens_saved = tokens_before - tokens_after
        ratio = tokens_saved / tokens_before if tokens_before > 0 else 0.0

        # Post-compress hook
        if hooks and tokens_saved > 0:
            from headroom.hooks import CompressEvent

            hooks.post_compress(
                CompressEvent(
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    tokens_saved=tokens_saved,
                    compression_ratio=ratio,
                    transforms_applied=result.transforms_applied,
                    model=model,
                )
            )

        return CompressResult(
            messages=compressed_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            compression_ratio=ratio,
            transforms_applied=result.transforms_applied,
        )

    except Exception as e:
        get_otel_metrics().record_compression_failure(
            model=model,
            operation="compress",
            error_type=type(e).__name__,
        )
        logger.warning("Compression failed, returning original messages: %s", e)
        return CompressResult(
            messages=messages,
            tokens_before=0,
            tokens_after=0,
            tokens_saved=0,
            compression_ratio=0.0,
        )


def _get_pipeline() -> Any:
    """Get or create the singleton compression pipeline."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        from headroom.transforms import TransformPipeline

        # Default pipeline: CacheAligner → ContentRouter
        # CacheAligner: stabilizes prefix for provider KV cache hits
        # ContentRouter: routes to the right compressor per content type
        #   (SmartCrusher for JSON, CodeCompressor for code, Kompress for text)
        # Phase B PR-B1 retired the trailing context-management stage —
        # live-zone-only compression never drops messages.
        _pipeline = TransformPipeline()
        logger.debug("Headroom compression pipeline initialized")
        return _pipeline

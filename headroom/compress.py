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
from dataclasses import dataclass, field, replace
from typing import Any

from .agent_savings import apply_agent_savings_profile
from .observability import get_otel_metrics
from .pipeline import PipelineExtensionManager, PipelineStage, summarize_routing_markers
from .utils import extract_user_query as _extract_user_query

logger = logging.getLogger(__name__)


# Lazy-initialized singleton pipelines (default + agent regime)
_pipeline = None
_agent_pipeline = None
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
    """Kompress model ID. None = default (chopratejas/kompress-v2-base).
    Set to a HuggingFace model ID for domain-specific compression.
    Set to 'disabled' to skip ML compression entirely
    (only SmartCrusher + CacheAligner will run)."""

    savings_profile: str | None = None
    """Named high-savings profile, e.g. 'agent-90' for Codex/Claude/Cursor."""

    # Regime
    mode: str | None = None
    """Compression regime.

    - ``None`` (default): the original behavior — content-aware compression that
      may use CCR removal (replace-with-marker + retrieve later). Best for a
      *proxy* compressing conversation history the model has moved past.
    - ``"agent"``: live-agent mode for compressing tool results an agent will
      *read back*. Lossless densification only — no CCR/removal, no ML text
      compression — so it is deterministic and prompt-cache-safe (the same tool
      output densifies to identical bytes every turn, keeping the cached prefix
      stable). Forces :attr:`verify_lossless`, so output is never lossy: any
      message that cannot be proven to round-trip is reverted to its original.
      Use :func:`densify` as the shorthand."""

    verify_lossless: bool = False
    """Round-trip each compressed message and revert any that is not provably
    lossless (keeps the original for that message). Forced on when
    ``mode='agent'``. Sets :attr:`CompressResult.lossless`."""


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
    lossless: bool | None = None
    """Whether the output is provably lossless. ``True``/``False`` when
    verification ran (``mode='agent'`` or ``verify_lossless=True``); ``None``
    when not checked. ``False`` means a removal marker survived but the message
    could not be reverted (should not happen in ``mode='agent'``)."""
    reverted_messages: int = 0
    """Messages reverted to their original form because their compression could
    not be verified lossless. Only meaningful when verification ran."""


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
    if cfg.savings_profile:
        cfg = replace(cfg)
        apply_agent_savings_profile(cfg, cfg.savings_profile)

    agent_mode = cfg.mode == "agent"
    if agent_mode:
        # Live-agent regime: densify losslessly, never remove. Disable the ML
        # text compressor so only deterministic, query-independent lossless
        # transforms run (keeps the cached prefix stable across turns).
        cfg.kompress_model = "disabled"
        cfg.verify_lossless = True
        pipeline = _get_agent_pipeline()
    else:
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

        # Snapshot the pipeline input so lossless verification can compare each
        # message against its pre-compression form and revert any that did not
        # round-trip (mode='agent' / verify_lossless).
        original_messages = messages

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

        # Guard: if "optimization" inflated tokens, revert to originals.
        # Mirrors the inflation guards in the proxy handlers
        # (anthropic/openai/gemini/batch) — the library path had none.
        if tokens_after > tokens_before:
            logger.warning(
                "Optimization inflated tokens (%d -> %d); reverting to original messages",
                tokens_before,
                tokens_after,
            )
            return CompressResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                tokens_saved=0,
                compression_ratio=0.0,
                transforms_applied=["inflation_guard:reverted"],
                # Reverting to the originals is trivially lossless.
                lossless=True if cfg.verify_lossless else None,
            )

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

        # Value-factoring (agent mode): hoist repeated low-cardinality column
        # values out of densified rows (e.g. file paths in search results).
        # Lossless + reversible; applied before verification so the round-trip
        # check validates it too.
        if agent_mode:
            compressed_messages = _apply_value_factoring(compressed_messages)

        # Lossless verification: round-trip every compressed message and revert
        # any that is not provably lossless. Guarantees the output is never
        # lossy even if a transform slips a removal marker through.
        lossless: bool | None = None
        reverted = 0
        if cfg.verify_lossless:
            compressed_messages, reverted, lossless = _verify_and_revert(
                original_messages, compressed_messages
            )
            if reverted:
                tokens_after = _recount_tokens(pipeline, model, compressed_messages)

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
            lossless=lossless,
            reverted_messages=reverted,
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


def compress_spreadsheet(
    path: str,
    model: str = "claude-sonnet-4-5-20250929",
    model_limit: int = 200000,
    **kwargs: Any,
) -> CompressResult:
    """Compress a binary spreadsheet (``.xlsx`` / ``.xls``).

    Each sheet is rendered to CSV text and submitted as its own user message so
    the tabular compressor (CSV → SmartCrusher, lossless-first + lossy CCR
    fallback) is applied per sheet. Requires the ``spreadsheet`` extra
    (``pip install headroom-ai[spreadsheet]``).

    Args:
        path: Path to a ``.xlsx`` or ``.xls`` file.
        model: Model name (token counting / context limit).
        model_limit: Model context window size in tokens.
        **kwargs: Forwarded to :func:`compress` (e.g. ``target_ratio``).

    Returns:
        CompressResult over the per-sheet messages.
    """
    from headroom.transforms.spreadsheet_ingest import load_spreadsheet

    sheets = load_spreadsheet(path)
    messages = [{"role": "user", "content": text} for text in sheets.values()]
    if not messages:
        return CompressResult(messages=[])
    # User messages hold the table text, so they must be compressible here.
    kwargs.setdefault("compress_user_messages", True)
    return compress(messages, model=model, model_limit=model_limit, **kwargs)


def densify(
    messages: list[dict[str, Any]],
    model: str = "claude-sonnet-4-5-20250929",
    model_limit: int = 200000,
    **kwargs: Any,
) -> CompressResult:
    """Lossless, prompt-cache-safe compression for live agents.

    Shorthand for ``compress(messages, mode="agent")``: densifies tool outputs
    an agent will read back, with **no removal** (no CCR markers, no dropped
    rows, no ML text compression) and a **verified losslessness guarantee** —
    any message that cannot be proven to round-trip is reverted to its original.
    Because only deterministic, query-independent transforms run, the same tool
    output densifies to identical bytes every turn, so a provider prompt-cache
    prefix stays stable across the agent loop.

    Inspect :attr:`CompressResult.lossless` (always ``True`` here) and
    :attr:`CompressResult.reverted_messages` to see how many messages could not
    be densified losslessly.

    Args:
        messages: Messages in Anthropic or OpenAI format.
        model: Model name (token counting / context limit).
        model_limit: Model context window size in tokens.
        **kwargs: Forwarded to :func:`compress` (e.g. ``protect_recent``).

    Returns:
        CompressResult with losslessly densified messages.
    """
    kwargs["mode"] = "agent"
    return compress(messages, model=model, model_limit=model_limit, **kwargs)


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


def _get_agent_pipeline() -> Any:
    """Get or create the singleton live-agent pipeline (lossless densify only).

    A ContentRouter with CCR removal and the ML text compressor disabled, so it
    only runs deterministic, query-independent lossless transforms (SmartCrusher
    densification, diff compaction). No CacheAligner stage — agent callers pass
    the full message list each turn and rely on densification determinism for
    prompt-cache stability.
    """
    global _agent_pipeline

    if _agent_pipeline is not None:
        return _agent_pipeline

    with _pipeline_lock:
        if _agent_pipeline is not None:
            return _agent_pipeline

        from headroom.transforms import TransformPipeline
        from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

        router = ContentRouter(
            ContentRouterConfig(
                ccr_enabled=False,  # never remove content the agent will read
                ccr_inject_marker=False,  # → emit_opaque_markers off in the engine
                enable_kompress=False,  # lossless only: no ML text compression
                smart_crusher_with_compaction=True,  # lossless-first densification
            )
        )
        _agent_pipeline = TransformPipeline(transforms=[router])
        logger.debug("Headroom agent (densify-only) pipeline initialized")
        return _agent_pipeline


def _recount_tokens(pipeline: Any, model: str, messages: list[dict[str, Any]]) -> int:
    """Recount tokens over ``messages`` using the pipeline's tokenizer.

    Mirrors how the pipeline counts (``count_text(str(content))`` per message)
    so a post-verification revert produces a consistent ``tokens_after``.
    """
    try:
        tokenizer = pipeline._get_tokenizer(model)
        return sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
    except Exception:  # pragma: no cover - tokenizer failure is non-fatal
        return sum(len(str(m.get("content", ""))) // 4 for m in messages)


def _apply_value_factoring(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dictionary-encode low-cardinality columns in every densified string leaf.

    Walks message content and runs :func:`factor_values` on any densified block
    (no-op for non-densified or already-factored strings). Returns a new list;
    inputs are not mutated.
    """
    from .transforms.compaction_codec import factor_values, is_compacted

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            return factor_values(node) if is_compacted(node) else node
        if isinstance(node, list):
            return [walk(x) for x in node]
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        return node

    return [walk(m) for m in messages]


def _verify_and_revert(
    original: list[dict[str, Any]],
    compressed: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, bool]:
    """Revert any compressed message that is not provably lossless.

    Compares each compressed message to its original. A message is kept only if
    it is unchanged, or every string value it changed is a densified block that
    round-trips back to the original (and carries no removal marker). Anything
    unverifiable is reverted to the original — so the result is never lossy.

    Returns ``(messages, num_reverted, lossless)`` where ``lossless`` is True
    when the final message list contains no surviving removal markers.
    """
    from .transforms.compaction_codec import contains_removal_marker

    if len(original) != len(compressed):
        # Structure changed unexpectedly — fall back to the originals.
        return list(original), len(compressed), True

    out: list[dict[str, Any]] = []
    reverted = 0
    for orig, comp in zip(original, compressed):
        if orig == comp or _message_is_lossless(orig, comp):
            out.append(comp)
        else:
            out.append(orig)
            reverted += 1

    lossless = not any(contains_removal_marker(str(m.get("content", ""))) for m in out)
    return out, reverted, lossless


def _message_is_lossless(orig: dict[str, Any], comp: dict[str, Any]) -> bool:
    """True if ``comp`` is a lossless densification of ``orig``.

    Walks both messages in parallel collecting (original, compressed) string
    pairs at matching positions; each changed pair must be a densified block
    that expands back to the original JSON. Any structural divergence or
    unverifiable change returns False (caller reverts).
    """
    pairs: list[tuple[str, str]] = []
    if not _collect_string_pairs(orig, comp, pairs):
        return False
    return all(_string_pair_is_lossless(o, c) for o, c in pairs)


def _collect_string_pairs(orig: Any, comp: Any, pairs: list[tuple[str, str]]) -> bool:
    """Collect aligned string leaves from two parallel structures.

    Returns False if the structures diverge in shape (different types, lengths,
    or dict keys) — a shape we cannot verify and must conservatively revert.
    """
    if isinstance(orig, str) and isinstance(comp, str):
        if orig != comp:
            pairs.append((orig, comp))
        return True
    if isinstance(orig, dict) and isinstance(comp, dict):
        if orig.keys() != comp.keys():
            return False
        return all(_collect_string_pairs(orig[k], comp[k], pairs) for k in orig)
    if isinstance(orig, list) and isinstance(comp, list):
        if len(orig) != len(comp):
            return False
        return all(_collect_string_pairs(o, c, pairs) for o, c in zip(orig, comp))
    # Scalars (int/float/bool/None) must match exactly; transforms only touch
    # strings, so a changed non-string is unexpected.
    return bool(orig == comp)


def _string_pair_is_lossless(original: str, compressed: str) -> bool:
    """True if ``compressed`` densifies ``original`` reversibly.

    ``original`` is expected to be a JSON array string; ``compressed`` is the
    densified block. Lossless iff it carries no removal marker and expands back
    to the same parsed JSON.
    """
    import json as _json

    from .transforms.compaction_codec import (
        contains_removal_marker,
        expand_compacted,
        is_compacted,
    )

    if contains_removal_marker(compressed):
        return False
    if not is_compacted(compressed):
        return False
    decoded = expand_compacted(compressed)
    if decoded is None:
        return False
    try:
        return bool(_json.loads(original) == decoded)
    except (ValueError, TypeError):
        return False

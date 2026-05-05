"""Transform pipeline orchestration for Headroom SDK."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from ..config import (
    CacheAlignerConfig,
    DiffArtifact,
    HeadroomConfig,
    TransformDiff,
    TransformResult,
    WasteSignals,
)
from ..observability import get_headroom_tracer, get_otel_metrics
from ..tokenizer import Tokenizer
from ..utils import deep_copy_messages
from .base import Transform
from .cache_aligner import CacheAligner
from .content_router import ContentRouter
from .smart_crusher import SmartCrusher

if TYPE_CHECKING:
    from ..providers.base import Provider

logger = logging.getLogger(__name__)


class TransformPipeline:
    """
    Orchestrates multiple transforms in the correct order.

    Transform order:
    1. Cache Aligner - normalize prefix for cache hits
    2. Content Router - intelligent content-aware compression (routes to appropriate
       compressor: Kompress for text, SmartCrusher for JSON, CodeCompressor for code, etc.)
    3. SmartCrusher - fallback if ContentRouter disabled

    Phase B PR-B1 retired the IntelligentContextManager / RollingWindow
    "drop messages from history" stage. Live-zone-only compression is the
    sole strategy going forward — message-list mutation no longer happens
    in the pipeline.
    """

    def __init__(
        self,
        config: HeadroomConfig | None = None,
        transforms: list[Transform] | None = None,
        provider: Provider | None = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Headroom configuration.
            transforms: Optional custom transform list (overrides config).
            provider: Provider for model-specific behavior.
        """
        self.config = config or HeadroomConfig()
        self._provider = provider

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = self._build_default_transforms()

    def _build_default_transforms(self) -> list[Transform]:
        """Build default transform pipeline from config."""
        transforms: list[Transform] = []

        # Order matters!

        # 0. Tool-result interceptors (ast-grep Read outline, etc.) run first
        # so downstream compressors operate on the already-shrunk content.
        # OPT-IN: enable via HeadroomConfig.intercept_tool_results, or for
        # non-config callers (CLI / SDK / tests) the env var
        # HEADROOM_INTERCEPT_ENABLED=1. Off by default while this ships — lets
        # users try it and compare before we make it the default.
        import os as _os

        if getattr(self.config, "intercept_tool_results", False) or _os.environ.get(
            "HEADROOM_INTERCEPT_ENABLED"
        ):
            from headroom.proxy.interceptors import ToolResultInterceptorTransform

            transforms.append(ToolResultInterceptorTransform())

        # 1. Cache Aligner (prefix stabilization)
        if self.config.cache_aligner.enabled:
            transforms.append(CacheAligner(self.config.cache_aligner))

        # 2. Content-aware Compression
        # ContentRouter handles ALL content types intelligently:
        # - JSON arrays -> SmartCrusher
        # - Plain text -> Kompress (ML-based) or passthrough
        # - Code -> CodeCompressor (AST-aware)
        # - Logs -> LogCompressor
        # - Search results -> SearchCompressor
        # - HTML -> HTMLExtractor
        if self.config.content_router_enabled:
            transforms.append(ContentRouter())
            logger.info("Pipeline using ContentRouter for intelligent content-aware compression")
        elif self.config.smart_crusher.enabled:
            # Fallback: SmartCrusher only handles JSON arrays
            from .smart_crusher import SmartCrusherConfig as SCConfig

            smart_config = SCConfig(
                enabled=True,
                min_items_to_analyze=self.config.smart_crusher.min_items_to_analyze,
                min_tokens_to_crush=self.config.smart_crusher.min_tokens_to_crush,
                variance_threshold=self.config.smart_crusher.variance_threshold,
                uniqueness_threshold=self.config.smart_crusher.uniqueness_threshold,
                similarity_threshold=self.config.smart_crusher.similarity_threshold,
                max_items_after_crush=self.config.smart_crusher.max_items_after_crush,
                preserve_change_points=self.config.smart_crusher.preserve_change_points,
                factor_out_constants=self.config.smart_crusher.factor_out_constants,
                include_summaries=self.config.smart_crusher.include_summaries,
            )
            transforms.append(SmartCrusher(smart_config))

        return transforms

    def _get_tokenizer(self, model: str) -> Tokenizer:
        """Get tokenizer for model.

        Uses provider's tokenizer if available, otherwise falls back to
        the tokenizer registry which auto-detects the best backend per model:
        - OpenAI models: tiktoken (exact)
        - Anthropic models: calibrated estimation (~3.5 chars/token)
        - Open models: HuggingFace tokenizer (if installed)
        - Unknown models: character-based estimation
        """
        if self._provider is not None:
            token_counter = self._provider.get_token_counter(model)
            return Tokenizer(token_counter, model)

        # No provider — use the tokenizer registry (auto-detects per model)
        # TokenCounter from tokenizers and providers have the same interface
        # (count_text, count_messages) but are different Protocol types.
        from headroom.tokenizers import get_tokenizer

        return Tokenizer(get_tokenizer(model), model)  # type: ignore[arg-type]

    def _provider_name(self) -> str | None:
        if self._provider is None:
            return None

        name = getattr(self._provider, "provider_name", None)
        if isinstance(name, str) and name:
            return name

        return self._provider.__class__.__name__.removesuffix("Provider").lower()

    def apply(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> TransformResult:
        """
        Apply all transforms in sequence.

        Args:
            messages: List of messages to transform.
            model: Model name for token counting.
            **kwargs: Additional arguments passed to transforms.
                - model_limit: Context limit override.
                - output_buffer: Output buffer override.
                - tool_profiles: Per-tool compression profiles.
                - request_id: Optional request ID for diff artifact.

        Returns:
            Combined TransformResult.
        """
        record_metrics = kwargs.pop("record_metrics", True)
        tokenizer = self._get_tokenizer(model)
        provider_name = self._provider_name()

        # Get model limit from kwargs (should be set by client)
        model_limit = kwargs.get("model_limit")
        if model_limit is None:
            raise ValueError(
                "model_limit is required. Provide it via kwargs or "
                "configure model_context_limits in HeadroomClient."
            )

        # Start with original tokens
        t_count = time.perf_counter()
        tokens_before = tokenizer.count_messages(messages)
        count_ms = (time.perf_counter() - t_count) * 1000

        logger.debug(
            "Pipeline starting: %d messages, %d tokens, model=%s",
            len(messages),
            tokens_before,
            model,
        )

        tracer = get_headroom_tracer()
        span_attributes = {
            "headroom.model": model,
            "headroom.provider": provider_name or "unknown",
            "headroom.message_count": len(messages),
            "headroom.tokens.before": tokens_before,
        }
        pipeline_span_context = (
            tracer.start_as_current_span(
                "headroom.compression.pipeline",
                attributes=span_attributes,
            )
            if record_metrics
            else nullcontext()
        )

        with pipeline_span_context as pipeline_span:
            # Track all transforms applied
            all_transforms: list[str] = []
            all_markers: list[str] = []
            all_warnings: list[str] = []
            all_timing: dict[str, float] = {}  # transform_name → ms

            # Track transform diffs if enabled
            transform_diffs: list[TransformDiff] = []
            generate_diff = self.config.generate_diff_artifact

            t_copy = time.perf_counter()
            current_messages = deep_copy_messages(messages)
            copy_ms = (time.perf_counter() - t_copy) * 1000

            all_timing["_deep_copy"] = copy_ms
            all_timing["_initial_token_count"] = count_ms

            pipeline_start = time.perf_counter()

            request_id = kwargs.get("request_id", "")
            log_prefix = f"[{request_id}] " if request_id else ""

            frozen_count = kwargs.get("frozen_message_count", 0)
            if frozen_count > 0:
                logger.info(
                    "%sPipeline: freezing first %d/%d messages (prefix cached by provider)",
                    log_prefix,
                    frozen_count,
                    len(messages),
                )

            for transform in self.transforms:
                # Check if transform should run
                if not transform.should_apply(current_messages, tokenizer, **kwargs):
                    continue

                transform_span_context = (
                    tracer.start_as_current_span(
                        "headroom.compression.transform",
                        attributes={
                            "headroom.model": model,
                            "headroom.provider": provider_name or "unknown",
                            "headroom.transform": transform.name,
                        },
                    )
                    if record_metrics
                    else nullcontext()
                )

                with transform_span_context as transform_span:
                    # Time the transform
                    t0 = time.perf_counter()
                    result = transform.apply(current_messages, tokenizer, **kwargs)
                    duration_ms = (time.perf_counter() - t0) * 1000

                    # Update messages for next transform
                    current_messages = result.messages

                    # Use token counts reported by the transform itself — avoids
                    # redundant O(N) recount of the full message list after each step.
                    tokens_before_transform = result.tokens_before
                    tokens_after_transform = result.tokens_after

                    if transform_span is not None and transform_span.is_recording():
                        transform_span.set_attribute(
                            "headroom.tokens.before", tokens_before_transform
                        )
                        transform_span.set_attribute(
                            "headroom.tokens.after", tokens_after_transform
                        )
                        transform_span.set_attribute(
                            "headroom.tokens.saved",
                            tokens_before_transform - tokens_after_transform,
                        )
                        transform_span.set_attribute("headroom.duration_ms", duration_ms)
                        transform_span.set_attribute(
                            "headroom.transforms_applied",
                            len(result.transforms_applied),
                        )

                    # Accumulate results
                    all_transforms.extend(result.transforms_applied)
                    all_markers.extend(result.markers_inserted)
                    all_warnings.extend(result.warnings)
                    all_timing[transform.name] = duration_ms

                    # Merge sub-transform timing (e.g. ContentRouter's per-compressor breakdown)
                    if result.timing:
                        all_timing.update(result.timing)

                    # Log transform results
                    if result.transforms_applied:
                        logger.info(
                            "Transform %s: %d -> %d tokens (saved %d) [%.1fms]",
                            transform.name,
                            tokens_before_transform,
                            tokens_after_transform,
                            tokens_before_transform - tokens_after_transform,
                            duration_ms,
                        )
                    else:
                        logger.debug(
                            "Transform %s: no changes [%.1fms]", transform.name, duration_ms
                        )

                    # Record diff if enabled
                    if generate_diff:
                        transform_diffs.append(
                            TransformDiff(
                                transform_name=transform.name,
                                tokens_before=tokens_before_transform,
                                tokens_after=tokens_after_transform,
                                tokens_saved=tokens_before_transform - tokens_after_transform,
                                details=", ".join(result.transforms_applied)
                                if result.transforms_applied
                                else "",
                                duration_ms=duration_ms,
                            )
                        )

            # Single final token count — the only full recount in the pipeline.
            # Earlier per-transform counts come from each transform's own result.
            t_final_count = time.perf_counter()
            tokens_after = tokenizer.count_messages(current_messages)
            all_timing["_final_token_count"] = (time.perf_counter() - t_final_count) * 1000

            pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
            all_timing["pipeline_total"] = pipeline_ms

            # Log pipeline summary
            total_saved = tokens_before - tokens_after
            timing_parts = " ".join(f"{k}={v:.0f}ms" for k, v in all_timing.items())
            if total_saved > 0:
                logger.info(
                    "%sPipeline complete: %d -> %d tokens (saved %d, %.1f%% reduction) [%s]",
                    log_prefix,
                    tokens_before,
                    tokens_after,
                    total_saved,
                    (total_saved / tokens_before * 100) if tokens_before > 0 else 0,
                    timing_parts,
                )
            else:
                logger.debug("%sPipeline complete: no token savings [%s]", log_prefix, timing_parts)

            # Build diff artifact if enabled
            diff_artifact = None
            if generate_diff:
                diff_artifact = DiffArtifact(
                    request_id=kwargs.get("request_id", ""),
                    original_tokens=tokens_before,
                    optimized_tokens=tokens_after,
                    total_tokens_saved=tokens_before - tokens_after,
                    transforms=transform_diffs,
                )

            # Detect waste signals in original messages (only when significant compression)
            waste_signals: WasteSignals | None = None
            if tokens_before > tokens_after and (tokens_before - tokens_after) > 100:
                try:
                    from ..parser import parse_messages

                    _, _, waste_signals = parse_messages(messages, tokenizer)
                    if waste_signals.total() == 0:
                        waste_signals = None
                except Exception:
                    pass

            if pipeline_span is not None and pipeline_span.is_recording():
                pipeline_span.set_attribute("headroom.tokens.after", tokens_after)
                pipeline_span.set_attribute("headroom.tokens.saved", total_saved)
                pipeline_span.set_attribute("headroom.duration_ms", pipeline_ms)
                pipeline_span.set_attribute("headroom.transforms_applied", len(all_transforms))
                pipeline_span.set_attribute("headroom.warnings", len(all_warnings))

            if record_metrics:
                get_otel_metrics().record_pipeline_run(
                    model=model,
                    provider=provider_name,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    duration_ms=pipeline_ms,
                    timing=all_timing,
                    transforms_applied=all_transforms,
                    waste_signals=waste_signals.to_dict() if waste_signals is not None else None,
                )

        return TransformResult(
            messages=current_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=all_transforms,
            markers_inserted=all_markers,
            warnings=all_warnings,
            diff_artifact=diff_artifact,
            timing=all_timing,
            waste_signals=waste_signals,
        )

    def simulate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> TransformResult:
        """
        Simulate transforms without modifying messages.

        Same as apply() but returns what WOULD happen.

        Args:
            messages: List of messages.
            model: Model name.
            **kwargs: Additional arguments.

        Returns:
            TransformResult with simulated changes.
        """
        # apply() already works on a copy, so this is safe
        return self.apply(messages, model, record_metrics=False, **kwargs)


def create_pipeline(
    cache_aligner_config: CacheAlignerConfig | None = None,
) -> TransformPipeline:
    """
    Create a pipeline with specific configurations.

    Args:
        cache_aligner_config: Cache aligner configuration.

    Returns:
        Configured TransformPipeline.
    """
    config = HeadroomConfig()

    if cache_aligner_config is not None:
        config.cache_aligner = cache_aligner_config

    return TransformPipeline(config)

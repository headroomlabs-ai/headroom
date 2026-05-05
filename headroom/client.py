"""Main HeadroomClient implementation for Headroom SDK."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from .cache import (
    BaseCacheOptimizer,
    CacheConfig,
    CacheOptimizerRegistry,
    OptimizationContext,
    SemanticCacheLayer,
)
from .config import (
    HeadroomConfig,
    HeadroomMode,
    RequestMetrics,
    SimulationResult,
)
from .parser import parse_messages
from .pipeline import PipelineExtensionManager, PipelineStage, summarize_routing_markers
from .providers.base import Provider
from .providers.registry import call_client_transport
from .storage import create_storage
from .tokenizer import Tokenizer
from .transforms import CacheAligner, TransformPipeline
from .utils import (
    compute_messages_hash,
    compute_prefix_hash,
    estimate_cost,
    format_cost,
    generate_request_id,
)


class ChatCompletions:
    """Wrapper for chat.completions API (OpenAI-style)."""

    def __init__(self, client: HeadroomClient):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        # Headroom-specific parameters
        headroom_mode: str | None = None,
        headroom_cache_prefix_tokens: int | None = None,
        headroom_output_buffer_tokens: int | None = None,
        headroom_keep_turns: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        # Pass through all other kwargs
        **kwargs: Any,
    ) -> Any:
        """
        Create a chat completion with optional Headroom optimization.

        Args:
            model: Model name.
            messages: List of messages.
            stream: Whether to stream the response.
            headroom_mode: Override default mode ("audit" | "optimize").
            headroom_cache_prefix_tokens: Target cache-aligned prefix size.
            headroom_output_buffer_tokens: Reserve tokens for output.
            headroom_keep_turns: Never drop last N turns.
            headroom_tool_profiles: Per-tool compression config.
            **kwargs: Additional arguments passed to underlying client.

        Returns:
            Chat completion response (or stream iterator).
        """
        return self._client._create(
            model=model,
            messages=messages,
            stream=stream,
            headroom_mode=headroom_mode,
            headroom_cache_prefix_tokens=headroom_cache_prefix_tokens,
            headroom_output_buffer_tokens=headroom_output_buffer_tokens,
            headroom_keep_turns=headroom_keep_turns,
            headroom_tool_profiles=headroom_tool_profiles,
            api_style="openai",
            **kwargs,
        )

    def simulate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headroom_mode: str = "optimize",
        headroom_output_buffer_tokens: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Simulate optimization without calling the API.

        Args:
            model: Model name.
            messages: List of messages.
            headroom_mode: Mode to simulate.
            headroom_output_buffer_tokens: Output buffer to use.
            headroom_tool_profiles: Tool profiles to use.
            **kwargs: Additional arguments (ignored).

        Returns:
            SimulationResult with projected changes.
        """
        return self._client._simulate(
            model=model,
            messages=messages,
            headroom_mode=headroom_mode,
            headroom_output_buffer_tokens=headroom_output_buffer_tokens,
            headroom_tool_profiles=headroom_tool_profiles,
        )


class Messages:
    """Wrapper for messages API (Anthropic-style)."""

    def __init__(self, client: HeadroomClient):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        # Headroom-specific parameters
        headroom_mode: str | None = None,
        headroom_cache_prefix_tokens: int | None = None,
        headroom_output_buffer_tokens: int | None = None,
        headroom_keep_turns: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        # Pass through all other kwargs
        **kwargs: Any,
    ) -> Any:
        """
        Create a message with optional Headroom optimization.

        Args:
            model: Model name.
            messages: List of messages.
            max_tokens: Maximum tokens in response.
            headroom_mode: Override default mode ("audit" | "optimize").
            headroom_cache_prefix_tokens: Target cache-aligned prefix size.
            headroom_output_buffer_tokens: Reserve tokens for output.
            headroom_keep_turns: Never drop last N turns.
            headroom_tool_profiles: Per-tool compression config.
            **kwargs: Additional arguments passed to underlying client.

        Returns:
            Message response.
        """
        return self._client._create(
            model=model,
            messages=messages,
            stream=False,
            headroom_mode=headroom_mode,
            headroom_cache_prefix_tokens=headroom_cache_prefix_tokens,
            headroom_output_buffer_tokens=headroom_output_buffer_tokens,
            headroom_keep_turns=headroom_keep_turns,
            headroom_tool_profiles=headroom_tool_profiles,
            api_style="anthropic",
            max_tokens=max_tokens,
            **kwargs,
        )

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        # Headroom-specific parameters
        headroom_mode: str | None = None,
        headroom_cache_prefix_tokens: int | None = None,
        headroom_output_buffer_tokens: int | None = None,
        headroom_keep_turns: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        # Pass through all other kwargs
        **kwargs: Any,
    ) -> Any:
        """
        Stream a message with optional Headroom optimization.

        Args:
            model: Model name.
            messages: List of messages.
            max_tokens: Maximum tokens in response.
            headroom_mode: Override default mode ("audit" | "optimize").
            headroom_cache_prefix_tokens: Target cache-aligned prefix size.
            headroom_output_buffer_tokens: Reserve tokens for output.
            headroom_keep_turns: Never drop last N turns.
            headroom_tool_profiles: Per-tool compression config.
            **kwargs: Additional arguments passed to underlying client.

        Returns:
            Stream context manager.
        """
        return self._client._create(
            model=model,
            messages=messages,
            stream=True,
            headroom_mode=headroom_mode,
            headroom_cache_prefix_tokens=headroom_cache_prefix_tokens,
            headroom_output_buffer_tokens=headroom_output_buffer_tokens,
            headroom_keep_turns=headroom_keep_turns,
            headroom_tool_profiles=headroom_tool_profiles,
            api_style="anthropic",
            max_tokens=max_tokens,
            **kwargs,
        )

    def simulate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headroom_mode: str = "optimize",
        headroom_output_buffer_tokens: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Simulate optimization without calling the API.

        Args:
            model: Model name.
            messages: List of messages.
            headroom_mode: Mode to simulate.
            headroom_output_buffer_tokens: Output buffer to use.
            headroom_tool_profiles: Tool profiles to use.
            **kwargs: Additional arguments (ignored).

        Returns:
            SimulationResult with projected changes.
        """
        return self._client._simulate(
            model=model,
            messages=messages,
            headroom_mode=headroom_mode,
            headroom_output_buffer_tokens=headroom_output_buffer_tokens,
            headroom_tool_profiles=headroom_tool_profiles,
        )


class HeadroomClient:
    """
    Context Budget Controller wrapper for LLM API clients.

    Provides automatic context optimization, waste detection, and
    cache alignment while maintaining API compatibility.
    """

    def __init__(
        self,
        original_client: Any,
        provider: Provider,
        store_url: str | None = None,
        default_mode: str = "audit",
        model_context_limits: dict[str, int] | None = None,
        cache_optimizer: BaseCacheOptimizer | None = None,
        enable_cache_optimizer: bool = True,
        enable_semantic_cache: bool = False,
        config: HeadroomConfig | None = None,
    ):
        """
        Initialize HeadroomClient.

        Args:
            original_client: The underlying LLM client (OpenAI-compatible).
            provider: Provider instance for model-specific behavior.
            store_url: Storage URL (sqlite:// or jsonl://). Defaults to temp dir.
            default_mode: Default mode ("audit" | "optimize").
            model_context_limits: Override context limits for models.
            cache_optimizer: Optional custom cache optimizer. If None and
                enable_cache_optimizer=True, auto-detects from provider.
            enable_cache_optimizer: Enable provider-specific cache optimization.
            enable_semantic_cache: Enable query-level semantic caching.
            config: Optional HeadroomConfig for full control over all settings.
                When provided, takes precedence over individual settings like
                store_url, default_mode, etc.
        """
        self._original = original_client
        self._provider = provider

        # Set default store_url to temp directory for better DevEx
        if store_url is None:
            import os
            import tempfile

            db_path = os.path.join(tempfile.gettempdir(), "headroom.db")
            store_url = f"sqlite:///{db_path}"

        self._store_url = store_url
        self._default_mode = HeadroomMode(default_mode)

        # Use provided config or build from individual parameters
        if config is not None:
            self._config = config
            # Override store_url and mode if explicitly provided in config
            if config.store_url:
                self._store_url = config.store_url
            else:
                self._config.store_url = store_url
            self._default_mode = config.default_mode
        else:
            # Build config from individual parameters
            self._config = HeadroomConfig()
            self._config.store_url = store_url
            self._config.default_mode = self._default_mode
            self._config.cache_optimizer.enabled = enable_cache_optimizer
            self._config.cache_optimizer.enable_semantic_cache = enable_semantic_cache

        if model_context_limits:
            self._config.model_context_limits.update(model_context_limits)

        # Initialize storage
        self._storage = create_storage(store_url)

        # Initialize transform pipeline
        self._pipeline = TransformPipeline(self._config, provider=self._provider)
        self._pipeline_extensions = PipelineExtensionManager(
            extensions=self._config.pipeline_extensions,
            discover=self._config.discover_pipeline_extensions,
        )

        # Initialize cache optimizer
        self._cache_optimizer: BaseCacheOptimizer | None = None
        self._semantic_cache_layer: SemanticCacheLayer | None = None

        if enable_cache_optimizer:
            if cache_optimizer is not None:
                self._cache_optimizer = cache_optimizer
            else:
                # Auto-detect from provider
                provider_name = self._provider.name.lower()
                if CacheOptimizerRegistry.is_registered(provider_name):
                    cache_config = CacheConfig(
                        min_cacheable_tokens=self._config.cache_optimizer.min_cacheable_tokens,
                    )
                    self._cache_optimizer = CacheOptimizerRegistry.get(
                        provider_name,
                        config=cache_config,
                    )

            # Wrap with semantic cache if enabled
            if enable_semantic_cache and self._cache_optimizer is not None:
                self._semantic_cache_layer = SemanticCacheLayer(
                    self._cache_optimizer,
                    similarity_threshold=self._config.cache_optimizer.semantic_cache_similarity,
                    max_entries=self._config.cache_optimizer.semantic_cache_max_entries,
                    ttl_seconds=self._config.cache_optimizer.semantic_cache_ttl_seconds,
                )

        # Public API - OpenAI style
        self.chat = type("Chat", (), {"completions": ChatCompletions(self)})()
        # Public API - Anthropic style
        self.messages = Messages(self)
        self._pipeline_extensions.emit(
            PipelineStage.SETUP,
            operation="sdk.setup",
            provider=self._provider.name.lower(),
            metadata={
                "default_mode": self._default_mode.value,
                "cache_optimizer_enabled": enable_cache_optimizer,
                "semantic_cache_enabled": enable_semantic_cache,
            },
        )

    def _get_tokenizer(self, model: str) -> Tokenizer:
        """Get tokenizer for model using provider."""
        token_counter = self._provider.get_token_counter(model)
        return Tokenizer(token_counter, model)

    def _get_context_limit(self, model: str) -> int:
        """Get context limit from user config or provider."""
        # User override takes precedence
        limit = self._config.get_context_limit(model)
        if limit is not None:
            return limit
        # Fall back to provider
        return self._provider.get_context_limit(model)

    def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        headroom_mode: str | None = None,
        headroom_cache_prefix_tokens: int | None = None,
        headroom_output_buffer_tokens: int | None = None,
        headroom_keep_turns: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
        api_style: str = "openai",
        **kwargs: Any,
    ) -> Any:
        """Internal implementation of create."""
        request_id = generate_request_id()
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
        mode = HeadroomMode(headroom_mode) if headroom_mode else self._default_mode

        input_event = self._pipeline_extensions.emit(
            PipelineStage.INPUT_RECEIVED,
            operation="sdk.request",
            request_id=request_id,
            provider=self._provider.name.lower(),
            model=model,
            messages=messages,
            metadata={"api_style": api_style, "stream": stream, "mode": mode.value},
        )
        if input_event.messages is not None:
            messages = input_event.messages

        tokenizer = self._get_tokenizer(model)

        # Analyze original messages
        blocks, block_breakdown, waste_signals = parse_messages(messages, tokenizer)
        tokens_before = tokenizer.count_messages(messages)

        # Compute cache alignment score
        aligner = CacheAligner(self._config.cache_aligner)
        cache_alignment_score = aligner.get_alignment_score(messages)

        # Compute stable prefix hash
        stable_prefix_hash = compute_prefix_hash(messages)

        # Cache optimizer metrics (populated later if optimizer is used)
        cache_optimizer_used = None
        cache_optimizer_strategy = None
        cacheable_tokens = 0
        breakpoints_inserted = 0
        estimated_cache_hit = False
        estimated_savings_percent = 0.0
        semantic_cache_hit = False
        cached_response = None

        # Apply transforms if in optimize mode
        if mode == HeadroomMode.OPTIMIZE:
            output_buffer = headroom_output_buffer_tokens or self._config.output_buffer_tokens
            model_limit = self._get_context_limit(model)

            result = self._pipeline.apply(
                messages,
                model,
                model_limit=model_limit,
                output_buffer=output_buffer,
                tool_profiles=headroom_tool_profiles or {},
            )

            optimized_messages = result.messages
            tokens_after = result.tokens_after
            transforms_applied = result.transforms_applied

            routing_markers = summarize_routing_markers(transforms_applied)
            if routing_markers:
                routed_event = self._pipeline_extensions.emit(
                    PipelineStage.INPUT_ROUTED,
                    operation="sdk.request",
                    request_id=request_id,
                    provider=self._provider.name.lower(),
                    model=model,
                    messages=optimized_messages,
                    metadata={
                        "routing_markers": routing_markers,
                        "transforms_applied": transforms_applied,
                    },
                )
                if routed_event.messages is not None:
                    optimized_messages = routed_event.messages

            # Apply provider-specific cache optimization
            if self._cache_optimizer is not None or self._semantic_cache_layer is not None:
                cache_context = OptimizationContext(
                    provider=self._provider.name.lower(),
                    model=model,
                    query=self._extract_query(optimized_messages),
                )

                # Check semantic cache first (if enabled)
                if self._semantic_cache_layer is not None:
                    cache_result = self._semantic_cache_layer.process(
                        optimized_messages, cache_context
                    )
                    semantic_cache_hit = cache_result.semantic_cache_hit
                    if semantic_cache_hit:
                        cached_response = cache_result.cached_response

                    # Update metrics from cache result
                    cache_optimizer_used = getattr(
                        cache_result.metrics, "optimizer_name", None
                    ) or (self._cache_optimizer.name if self._cache_optimizer else "")
                    cache_optimizer_strategy = getattr(cache_result.metrics, "strategy", "")
                    cacheable_tokens = cache_result.metrics.cacheable_tokens
                    breakpoints_inserted = cache_result.metrics.breakpoints_inserted
                    estimated_cache_hit = cache_result.metrics.estimated_cache_hit
                    estimated_savings_percent = cache_result.metrics.estimated_savings_percent

                    # Apply optimized messages (with cache_control blocks for Anthropic)
                    if cache_result.messages:
                        optimized_messages = cache_result.messages

                elif self._cache_optimizer is not None:
                    # Direct cache optimizer (no semantic layer)
                    cache_result = self._cache_optimizer.optimize(optimized_messages, cache_context)
                    cache_optimizer_used = self._cache_optimizer.name
                    cache_optimizer_strategy = self._cache_optimizer.strategy.value
                    cacheable_tokens = cache_result.metrics.cacheable_tokens
                    breakpoints_inserted = cache_result.metrics.breakpoints_inserted
                    estimated_cache_hit = cache_result.metrics.estimated_cache_hit
                    estimated_savings_percent = cache_result.metrics.estimated_savings_percent

                    if cache_result.messages:
                        optimized_messages = cache_result.messages

                transforms_applied.extend(
                    f"cache_optimizer:{t}" for t in (cache_result.transforms_applied or [])
                )

            compressed_event = self._pipeline_extensions.emit(
                PipelineStage.INPUT_COMPRESSED,
                operation="sdk.request",
                request_id=request_id,
                provider=self._provider.name.lower(),
                model=model,
                messages=optimized_messages,
                metadata={
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "transforms_applied": transforms_applied,
                },
            )
            if compressed_event.messages is not None:
                optimized_messages = compressed_event.messages
                tokens_after = tokenizer.count_messages(optimized_messages)

            # Recalculate prefix hash after optimization
            stable_prefix_hash = compute_prefix_hash(optimized_messages)
        else:
            # Audit mode - no changes
            optimized_messages = messages
            tokens_after = tokens_before
            transforms_applied = []

        presend_event = self._pipeline_extensions.emit(
            PipelineStage.PRE_SEND,
            operation="sdk.request",
            request_id=request_id,
            provider=self._provider.name.lower(),
            model=model,
            messages=optimized_messages,
            metadata={"api_style": api_style, "stream": stream},
        )
        if presend_event.messages is not None:
            optimized_messages = presend_event.messages
            tokens_after = tokenizer.count_messages(optimized_messages)
            stable_prefix_hash = compute_prefix_hash(optimized_messages)

        # Create metrics
        metrics = RequestMetrics(
            request_id=request_id,
            timestamp=timestamp,
            model=model,
            stream=stream,
            mode=mode.value,
            tokens_input_before=tokens_before,
            tokens_input_after=tokens_after,
            block_breakdown=block_breakdown,
            waste_signals=waste_signals.to_dict(),
            stable_prefix_hash=stable_prefix_hash,
            cache_alignment_score=cache_alignment_score,
            transforms_applied=transforms_applied,
            messages_hash=compute_messages_hash(messages),
            # Cache optimizer metrics
            cache_optimizer_used=cache_optimizer_used,
            cache_optimizer_strategy=cache_optimizer_strategy,
            cacheable_tokens=cacheable_tokens,
            breakpoints_inserted=breakpoints_inserted,
            estimated_cache_hit=estimated_cache_hit,
            estimated_savings_percent=estimated_savings_percent,
            semantic_cache_hit=semantic_cache_hit,
        )

        # Update session stats
        self._update_session_stats(
            mode=mode,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            cache_hit=semantic_cache_hit,
        )

        # Return cached response if semantic cache hit
        if semantic_cache_hit and cached_response is not None:
            self._storage.save(metrics)
            return cached_response

        # Call underlying client based on API style
        try:
            response = call_client_transport(
                api_style,
                self,
                model=model,
                messages=optimized_messages,
                stream=stream,
                metrics=metrics,
                **kwargs,
            )

            self._pipeline_extensions.emit(
                PipelineStage.POST_SEND,
                operation="sdk.request",
                request_id=request_id,
                provider=self._provider.name.lower(),
                model=model,
                messages=optimized_messages,
                response=response,
                metadata={"api_style": api_style, "stream": stream},
            )
            self._pipeline_extensions.emit(
                PipelineStage.RESPONSE_RECEIVED,
                operation="sdk.request",
                request_id=request_id,
                provider=self._provider.name.lower(),
                model=model,
                response=response,
                metadata={"api_style": api_style, "stream": stream},
            )
            return response

        except Exception as e:
            metrics.error = str(e)
            self._storage.save(metrics)
            raise

    def _call_openai(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        metrics: RequestMetrics,
        **kwargs: Any,
    ) -> Any:
        """Call OpenAI-style API."""
        return call_client_transport(
            "openai",
            self,
            model=model,
            messages=messages,
            stream=stream,
            metrics=metrics,
            **kwargs,
        )

    def _call_anthropic(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        metrics: RequestMetrics,
        **kwargs: Any,
    ) -> Any:
        """Call Anthropic-style API."""
        return call_client_transport(
            "anthropic",
            self,
            model=model,
            messages=messages,
            stream=stream,
            metrics=metrics,
            **kwargs,
        )

    def _wrap_stream(
        self,
        stream: Iterator[Any],
        metrics: RequestMetrics,
    ) -> Iterator[Any]:
        """Wrap stream to pass through chunks and save metrics at end."""
        try:
            yield from stream
        finally:
            # Save metrics when stream completes
            # Note: output tokens unknown for streams
            self._storage.save(metrics)

    def _extract_query(self, messages: list[dict[str, Any]]) -> str:
        """Extract query from messages for semantic caching.

        Returns the last user message content as the query.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Content block format
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_val = block.get("text", "")
                            return str(text_val) if text_val else ""
                    return ""
        return ""

    def _store_response_in_semantic_cache(
        self,
        messages: list[dict[str, Any]],
        response: Any,
        model: str,
    ) -> None:
        """Store response in semantic cache for future hits."""
        if self._semantic_cache_layer is not None:
            cache_context = OptimizationContext(
                provider=self._provider.name.lower(),
                model=model,
                query=self._extract_query(messages),
            )
            # Extract response content for caching
            response_data = self._extract_response_content(response)
            if response_data:
                self._semantic_cache_layer.store_response(messages, response_data, cache_context)

    def _extract_response_content(self, response: Any) -> dict[str, Any] | None:
        """Extract cacheable content from API response."""
        try:
            # OpenAI format
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message"):
                    return {
                        "role": "assistant",
                        "content": choice.message.content,
                    }
            # Anthropic format
            elif hasattr(response, "content"):
                return {
                    "role": "assistant",
                    "content": response.content,
                }
        except Exception:
            pass
        return None

    def _simulate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headroom_mode: str = "optimize",
        headroom_output_buffer_tokens: int | None = None,
        headroom_tool_profiles: dict[str, dict[str, Any]] | None = None,
    ) -> SimulationResult:
        """Internal implementation of simulate."""
        tokenizer = self._get_tokenizer(model)

        # Analyze original
        blocks, block_breakdown, waste_signals = parse_messages(messages, tokenizer)
        tokens_before = tokenizer.count_messages(messages)

        # Compute original cache alignment
        aligner = CacheAligner(self._config.cache_aligner)
        cache_alignment_score = aligner.get_alignment_score(messages)
        compute_prefix_hash(messages)

        # Apply transforms
        output_buffer = headroom_output_buffer_tokens or self._config.output_buffer_tokens
        model_limit = self._get_context_limit(model)

        result = self._pipeline.simulate(
            messages,
            model,
            model_limit=model_limit,
            output_buffer=output_buffer,
            tool_profiles=headroom_tool_profiles or {},
        )

        tokens_saved = tokens_before - result.tokens_after

        # Estimate cost savings using provider (use output_buffer tokens)
        # Note: output_buffer reserves tokens for expected model output
        cost_before = estimate_cost(tokens_before, output_buffer, model, provider=self._provider)
        cost_after = estimate_cost(
            result.tokens_after, output_buffer, model, provider=self._provider
        )

        if cost_before is not None and cost_after is not None:
            savings = format_cost(cost_before - cost_after)
        else:
            savings = "N/A"

        # Recalculate prefix hash after optimization
        optimized_prefix_hash = compute_prefix_hash(result.messages)

        return SimulationResult(
            tokens_before=tokens_before,
            tokens_after=result.tokens_after,
            tokens_saved=tokens_saved,
            transforms=result.transforms_applied,
            estimated_savings=f"{savings} per request",
            messages_optimized=result.messages,
            block_breakdown=block_breakdown,
            waste_signals=waste_signals.to_dict(),
            stable_prefix_hash=optimized_prefix_hash,
            cache_alignment_score=cache_alignment_score,
        )

    def get_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        mode: str | None = None,
        limit: int = 100,
    ) -> list[RequestMetrics]:
        """
        Query stored metrics.

        Args:
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.
            model: Filter by model name.
            mode: Filter by mode.
            limit: Maximum results.

        Returns:
            List of RequestMetrics.
        """
        return self._storage.query(
            start_time=start_time,
            end_time=end_time,
            model=model,
            mode=mode,
            limit=limit,
        )

    def get_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get summary statistics.

        Args:
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.

        Returns:
            Summary statistics dict.
        """
        return self._storage.get_summary_stats(start_time, end_time)

    def close(self) -> None:
        """Close storage connection."""
        self._storage.close()

    def __enter__(self) -> HeadroomClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def validate_setup(self) -> dict[str, Any]:
        """Validate that Headroom is properly configured.

        This method checks:
        - Provider is valid and can count tokens
        - Storage is accessible and writable
        - Configuration is valid
        - Cache optimizer (if enabled) is working

        Returns:
            dict with validation results:
            {
                "valid": True/False,
                "provider": {"ok": bool, "name": str, "error": str | None},
                "storage": {"ok": bool, "url": str, "error": str | None},
                "config": {"ok": bool, "mode": str, "error": str | None},
                "cache_optimizer": {"ok": bool, "name": str | None, "error": str | None},
            }

        Raises:
            ValidationError: If validation fails and raise_on_error=True.

        Example:
            client = HeadroomClient(...)
            result = client.validate_setup()
            if not result["valid"]:
                print("Setup issues:", result)
        """
        result: dict[str, Any] = {
            "valid": True,
            "provider": {"ok": False, "name": None, "error": None},
            "storage": {"ok": False, "url": self._store_url, "error": None},
            "config": {"ok": False, "mode": self._default_mode.value, "error": None},
            "cache_optimizer": {"ok": True, "name": None, "error": None},
        }

        # Validate provider
        try:
            result["provider"]["name"] = self._provider.name
            # Test token counting
            test_messages = [{"role": "user", "content": "test"}]
            tokenizer = self._get_tokenizer("gpt-4")
            count = tokenizer.count_messages(test_messages)
            if count > 0:
                result["provider"]["ok"] = True
            else:
                result["provider"]["error"] = "Token count returned 0"
                result["valid"] = False
        except Exception as e:
            result["provider"]["error"] = str(e)
            result["valid"] = False

        # Validate storage
        try:
            # Try to get summary (tests read)
            self._storage.get_summary_stats()
            result["storage"]["ok"] = True
        except Exception as e:
            result["storage"]["error"] = str(e)
            result["valid"] = False

        # Validate config
        try:
            # Check mode is valid
            if self._default_mode in (HeadroomMode.AUDIT, HeadroomMode.OPTIMIZE):
                result["config"]["ok"] = True
            else:
                result["config"]["error"] = f"Invalid mode: {self._default_mode}"
                result["valid"] = False
        except Exception as e:
            result["config"]["error"] = str(e)
            result["valid"] = False

        # Validate cache optimizer (if enabled)
        if self._cache_optimizer is not None:
            try:
                result["cache_optimizer"]["name"] = self._cache_optimizer.name
                result["cache_optimizer"]["ok"] = True
            except Exception as e:
                result["cache_optimizer"]["error"] = str(e)
                # Don't fail validation for cache optimizer issues
        elif self._config.cache_optimizer.enabled:
            result["cache_optimizer"]["error"] = "Enabled but no optimizer loaded"
            # Don't fail validation, just warn

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get quick statistics without database query.

        This returns in-memory stats tracked during this session.
        For historical metrics, use get_metrics() or get_summary().

        Returns:
            dict with session statistics:
            {
                "session": {
                    "requests_total": int,
                    "requests_optimized": int,
                    "requests_audit": int,
                    "tokens_saved_total": int,
                    "cache_hits": int,
                },
                "config": {
                    "mode": str,
                    "provider": str,
                    "cache_optimizer": str | None,
                    "semantic_cache": bool,
                },
                "transforms": {
                    "smart_crusher_enabled": bool,
                    "cache_aligner_enabled": bool,
                },
            }

        Example:
            stats = client.get_stats()
            print(f"Saved {stats['session']['tokens_saved_total']} tokens this session")
        """
        # Initialize session stats if not present
        if not hasattr(self, "_session_stats"):
            self._session_stats = {
                "requests_total": 0,
                "requests_optimized": 0,
                "requests_audit": 0,
                "tokens_saved_total": 0,
                "cache_hits": 0,
            }

        return {
            "session": dict(self._session_stats),
            "config": {
                "mode": self._default_mode.value,
                "provider": self._provider.name,
                "cache_optimizer": (self._cache_optimizer.name if self._cache_optimizer else None),
                "semantic_cache": self._semantic_cache_layer is not None,
            },
            "transforms": {
                "smart_crusher_enabled": self._config.smart_crusher.enabled,
                "cache_aligner_enabled": self._config.cache_aligner.enabled,
            },
        }

    def _update_session_stats(
        self,
        mode: HeadroomMode,
        tokens_before: int,
        tokens_after: int,
        cache_hit: bool = False,
    ) -> None:
        """Update in-memory session statistics."""
        if not hasattr(self, "_session_stats"):
            self._session_stats = {
                "requests_total": 0,
                "requests_optimized": 0,
                "requests_audit": 0,
                "tokens_saved_total": 0,
                "cache_hits": 0,
            }

        self._session_stats["requests_total"] += 1

        if mode == HeadroomMode.OPTIMIZE:
            self._session_stats["requests_optimized"] += 1
            self._session_stats["tokens_saved_total"] += max(0, tokens_before - tokens_after)
        else:
            self._session_stats["requests_audit"] += 1

        if cache_hit:
            self._session_stats["cache_hits"] += 1

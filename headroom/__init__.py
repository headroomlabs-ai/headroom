"""
Headroom - The Context Optimization Layer for LLM Applications.

Cut your LLM costs by 50-90% without losing accuracy.

Headroom wraps LLM clients to provide:
- Smart compression of tool outputs (keeps errors, anomalies, relevant items)
- Cache-aligned prefix optimization for better provider cache hits
- Rolling window token management for long conversations
- Full streaming support with zero accuracy loss

Quick Start:

    from headroom import HeadroomClient, OpenAIProvider
    from openai import OpenAI

    # Wrap your existing client
    client = HeadroomClient(
        original_client=OpenAI(),
        provider=OpenAIProvider(),
        default_mode="optimize",
    )

    # Use exactly like the original client
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Hello!"},
        ],
    )

    # Check savings
    stats = client.get_stats()
    print(f"Tokens saved: {stats['session']['tokens_saved_total']}")

Verify It's Working:

    # Validate configuration
    result = client.validate_setup()
    if not result["valid"]:
        print("Issues:", result)

    # Enable logging to see what's happening
    import logging
    logging.basicConfig(level=logging.INFO)
    # INFO:headroom.transforms.pipeline:Pipeline complete: 45000 -> 4500 tokens

Simulate Before Sending:

    plan = client.chat.completions.simulate(
        model="gpt-4o",
        messages=large_messages,
    )
    print(f"Would save {plan.tokens_saved} tokens")
    print(f"Transforms: {plan.transforms}")

Error Handling:

    from headroom import HeadroomError, ConfigurationError, ProviderError

    try:
        response = client.chat.completions.create(...)
    except ConfigurationError as e:
        print(f"Config issue: {e.details}")
    except HeadroomError as e:
        print(f"Headroom error: {e}")

For more examples, see https://github.com/headroom-sdk/headroom/tree/main/examples
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._version import __version__  # noqa: F401
from .compress import CompressConfig, CompressResult, compress

# Keep a real callable bound for the one-function compression API so
# `from headroom import compress` is never shadowed by the submodule object.

__all__ = [
    # Main client
    "HeadroomClient",
    # Providers
    "Provider",
    "TokenCounter",
    "OpenAIProvider",
    "AnthropicProvider",
    # Exceptions
    "HeadroomError",
    "ConfigurationError",
    "ProviderError",
    "StorageError",
    "CompressionError",
    "TokenizationError",
    "CacheError",
    "ValidationError",
    "TransformError",
    # Config
    "HeadroomConfig",
    "HeadroomMode",
    "SmartCrusherConfig",
    "CacheAlignerConfig",
    "CacheOptimizerConfig",
    "RelevanceScorerConfig",
    # Data models
    "Block",
    "CachePrefixMetrics",
    "DiffArtifact",
    "RequestMetrics",
    "SimulationResult",
    "TransformDiff",
    "TransformResult",
    "WasteSignals",
    # Transforms
    "SmartCrusher",
    "CacheAligner",
    "TransformPipeline",
    # Cache optimizers
    "BaseCacheOptimizer",
    "CacheConfig",
    "CacheMetrics",
    "CacheResult",
    "CacheStrategy",
    "OptimizationContext",
    "CacheOptimizerRegistry",
    "AnthropicCacheOptimizer",
    "OpenAICacheOptimizer",
    "GoogleCacheOptimizer",
    "SemanticCache",
    "SemanticCacheLayer",
    # Relevance scoring - BM25 always available, embeddings require sentence-transformers
    "RelevanceScore",
    "RelevanceScorer",
    "BM25Scorer",
    "EmbeddingScorer",
    "HybridScorer",
    "create_scorer",
    "embedding_available",
    # Utilities
    "Tokenizer",
    "count_tokens_text",
    "count_tokens_messages",
    "generate_report",
    # Observability
    "HeadroomOtelMetrics",
    "HeadroomTracer",
    "LangfuseTracingConfig",
    "OTelMetricsConfig",
    "configure_otel_metrics",
    "configure_langfuse_tracing",
    "get_headroom_tracer",
    "get_langfuse_tracing_status",
    "get_otel_metrics",
    "get_otel_metrics_status",
    "reset_headroom_tracing",
    "reset_otel_metrics",
    # Memory - optional hierarchical memory system
    "with_memory",  # Main user-facing API
    "Memory",
    "ScopeLevel",
    "HierarchicalMemory",
    "MemoryConfig",
    "EmbedderBackend",
    # One-function compression API
    "compress",
    "CompressConfig",
    "CompressResult",
    # Hooks
    "CompressionHooks",
    "CompressContext",
    "CompressEvent",
    # Canonical pipeline
    "PipelineStage",
    "PipelineEvent",
    "PipelineExtensionManager",
    "CANONICAL_PIPELINE_STAGES",
    # Shared context for multi-agent workflows
    "SharedContext",
]

# Keep package-level imports lightweight so `import headroom` does not eagerly
# load provider SDKs, ML stacks, or optional proxy/runtime integrations.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Main client
    "HeadroomClient": ("headroom.client", "HeadroomClient"),
    # Providers
    "Provider": ("headroom.providers", "Provider"),
    "TokenCounter": ("headroom.providers", "TokenCounter"),
    "OpenAIProvider": ("headroom.providers", "OpenAIProvider"),
    "AnthropicProvider": ("headroom.providers", "AnthropicProvider"),
    # Exceptions
    "HeadroomError": ("headroom.exceptions", "HeadroomError"),
    "ConfigurationError": ("headroom.exceptions", "ConfigurationError"),
    "ProviderError": ("headroom.exceptions", "ProviderError"),
    "StorageError": ("headroom.exceptions", "StorageError"),
    "CompressionError": ("headroom.exceptions", "CompressionError"),
    "TokenizationError": ("headroom.exceptions", "TokenizationError"),
    "CacheError": ("headroom.exceptions", "CacheError"),
    "ValidationError": ("headroom.exceptions", "ValidationError"),
    "TransformError": ("headroom.exceptions", "TransformError"),
    # Config
    "HeadroomConfig": ("headroom.config", "HeadroomConfig"),
    "HeadroomMode": ("headroom.config", "HeadroomMode"),
    "SmartCrusherConfig": ("headroom.config", "SmartCrusherConfig"),
    "CacheAlignerConfig": ("headroom.config", "CacheAlignerConfig"),
    "CacheOptimizerConfig": ("headroom.config", "CacheOptimizerConfig"),
    "RelevanceScorerConfig": ("headroom.config", "RelevanceScorerConfig"),
    # Data models
    "Block": ("headroom.config", "Block"),
    "CachePrefixMetrics": ("headroom.config", "CachePrefixMetrics"),
    "DiffArtifact": ("headroom.config", "DiffArtifact"),
    "RequestMetrics": ("headroom.config", "RequestMetrics"),
    "SimulationResult": ("headroom.config", "SimulationResult"),
    "TransformDiff": ("headroom.config", "TransformDiff"),
    "TransformResult": ("headroom.config", "TransformResult"),
    "WasteSignals": ("headroom.config", "WasteSignals"),
    # Transforms
    "SmartCrusher": ("headroom.transforms", "SmartCrusher"),
    "CacheAligner": ("headroom.transforms", "CacheAligner"),
    "TransformPipeline": ("headroom.transforms", "TransformPipeline"),
    # Cache optimizers
    "BaseCacheOptimizer": ("headroom.cache", "BaseCacheOptimizer"),
    "CacheConfig": ("headroom.cache", "CacheConfig"),
    "CacheMetrics": ("headroom.cache", "CacheMetrics"),
    "CacheResult": ("headroom.cache", "CacheResult"),
    "CacheStrategy": ("headroom.cache", "CacheStrategy"),
    "OptimizationContext": ("headroom.cache", "OptimizationContext"),
    "CacheOptimizerRegistry": ("headroom.cache", "CacheOptimizerRegistry"),
    "AnthropicCacheOptimizer": ("headroom.cache", "AnthropicCacheOptimizer"),
    "OpenAICacheOptimizer": ("headroom.cache", "OpenAICacheOptimizer"),
    "GoogleCacheOptimizer": ("headroom.cache", "GoogleCacheOptimizer"),
    "SemanticCache": ("headroom.cache", "SemanticCache"),
    "SemanticCacheLayer": ("headroom.cache", "SemanticCacheLayer"),
    # Relevance scoring
    "RelevanceScore": ("headroom.relevance", "RelevanceScore"),
    "RelevanceScorer": ("headroom.relevance", "RelevanceScorer"),
    "BM25Scorer": ("headroom.relevance", "BM25Scorer"),
    "EmbeddingScorer": ("headroom.relevance", "EmbeddingScorer"),
    "HybridScorer": ("headroom.relevance", "HybridScorer"),
    "create_scorer": ("headroom.relevance", "create_scorer"),
    "embedding_available": ("headroom.relevance", "embedding_available"),
    # Utilities
    "Tokenizer": ("headroom.tokenizer", "Tokenizer"),
    "count_tokens_text": ("headroom.tokenizer", "count_tokens_text"),
    "count_tokens_messages": ("headroom.tokenizer", "count_tokens_messages"),
    "generate_report": ("headroom.reporting", "generate_report"),
    # Observability
    "HeadroomOtelMetrics": ("headroom.observability", "HeadroomOtelMetrics"),
    "HeadroomTracer": ("headroom.observability", "HeadroomTracer"),
    "LangfuseTracingConfig": ("headroom.observability", "LangfuseTracingConfig"),
    "OTelMetricsConfig": ("headroom.observability", "OTelMetricsConfig"),
    "configure_otel_metrics": ("headroom.observability", "configure_otel_metrics"),
    "configure_langfuse_tracing": ("headroom.observability", "configure_langfuse_tracing"),
    "get_headroom_tracer": ("headroom.observability", "get_headroom_tracer"),
    "get_langfuse_tracing_status": ("headroom.observability", "get_langfuse_tracing_status"),
    "get_otel_metrics": ("headroom.observability", "get_otel_metrics"),
    "get_otel_metrics_status": ("headroom.observability", "get_otel_metrics_status"),
    "reset_headroom_tracing": ("headroom.observability", "reset_headroom_tracing"),
    "reset_otel_metrics": ("headroom.observability", "reset_otel_metrics"),
    # One-function API
    "compress": ("headroom.compress", "compress"),
    # Hooks
    "CompressionHooks": ("headroom.hooks", "CompressionHooks"),
    "CompressContext": ("headroom.hooks", "CompressContext"),
    "CompressEvent": ("headroom.hooks", "CompressEvent"),
    # Canonical pipeline
    "PipelineStage": ("headroom.pipeline", "PipelineStage"),
    "PipelineEvent": ("headroom.pipeline", "PipelineEvent"),
    "PipelineExtensionManager": ("headroom.pipeline", "PipelineExtensionManager"),
    "CANONICAL_PIPELINE_STAGES": ("headroom.pipeline", "CANONICAL_PIPELINE_STAGES"),
    # Shared context
    "SharedContext": ("headroom.shared_context", "SharedContext"),
}

# Memory remains optional and preserves the long-standing behavior of exposing
# `None` when the extra dependencies are not installed.
_OPTIONAL_EXPORTS = {
    "with_memory": ("headroom.memory", "with_memory"),
    "Memory": ("headroom.memory", "Memory"),
    "ScopeLevel": ("headroom.memory", "ScopeLevel"),
    "HierarchicalMemory": ("headroom.memory", "HierarchicalMemory"),
    "MemoryConfig": ("headroom.memory", "MemoryConfig"),
    "EmbedderBackend": ("headroom.memory", "EmbedderBackend"),
}


def __getattr__(name: str) -> Any:
    """Resolve package exports lazily while preserving legacy import paths."""
    module_attr = _LAZY_EXPORTS.get(name)
    if module_attr is not None:
        module_name, attr_name = module_attr
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    optional_module_attr = _OPTIONAL_EXPORTS.get(name)
    if optional_module_attr is not None:
        module_name, attr_name = optional_module_attr
        try:
            value = getattr(import_module(module_name), attr_name)
        except ImportError:
            value = None
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

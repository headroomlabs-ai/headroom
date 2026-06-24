"""Compression hooks and pipeline lifecycle events.

Three hooks at well-defined pipeline stages:

1. pre_compress: modify messages before compression (dedup, filter, inject)
2. compute_biases: set per-message compression aggressiveness (position-aware, phase-aware)
3. post_compress: observe results after compression (learning, analytics, logging)

The canonical pipeline also emits lifecycle events through ``on_pipeline_event``.
That gives extensions one stable contract across SDK, ``compress()``, and proxy
request flow without replacing the existing compression hooks.

Default implementation is no-op — OSS behavior unchanged. Override these
in a subclass to customize (e.g., Headroom SaaS implements position-aware
compression and cross-turn deduplication via these hooks).

Usage:
    from headroom.hooks import CompressionHooks, CompressContext

    class MyHooks(CompressionHooks):
        def compute_biases(self, messages, ctx):
            # Position-aware: keep more in the middle (attention is weakest there)
            biases = {}
            for i in range(len(messages)):
                pos = i / max(len(messages) - 1, 1)
                biases[i] = 1.0 + 0.5 * (1.0 - abs(2 * pos - 1))
            return biases

    config = ProxyConfig(hooks=MyHooks())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .pipeline import PipelineEvent


@dataclass
class CompressContext:
    """Context passed to pre_compress and compute_biases hooks.

    Provides enough information for hooks to make decisions without
    needing to understand the proxy's internals.
    """

    model: str = ""
    user_query: str = ""
    turn_number: int = 0
    tool_calls: list[str] = field(default_factory=list)
    provider: str = ""  # "anthropic", "openai", "gemini"


@dataclass
class CompressEvent:
    """Data passed to post_compress hook after compression completes.

    Contains before/after state and full metrics for learning and analytics.
    """

    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 0.0
    transforms_applied: list[str] = field(default_factory=list)
    ccr_hashes: list[str] = field(default_factory=list)
    model: str = ""
    user_query: str = ""
    provider: str = ""


class CompressionHooks:
    """Base class for compression hooks. Override methods to customize.

    All methods have no-op defaults — OSS behavior is unchanged unless
    a subclass is provided via ProxyConfig(hooks=MyHooks()).
    """

    def pre_compress(
        self,
        messages: list[dict[str, Any]],
        ctx: CompressContext,
    ) -> list[dict[str, Any]]:
        """Called before the compression pipeline runs.

        Modify and return the messages list. Use for:
        - Cross-turn deduplication (compare against recent CCR entries)
        - Memory injection (add relevant context from external sources)
        - Pre-filtering (remove messages irrelevant to the user's query)
        - Phase detection (reorder/prioritize based on task phase)

        Args:
            messages: The full message list (will be compressed next).
            ctx: Compression context (model, query, turn, tool calls).

        Returns:
            Modified (or unmodified) messages list.
        """
        return messages

    def compute_biases(
        self,
        messages: list[dict[str, Any]],
        ctx: CompressContext,
    ) -> dict[int, float]:
        """Compute per-message compression bias.

        Return a dict mapping message index to compression bias:
        - 1.0 = default compression
        - >1.0 = keep more (compress less aggressively)
        - <1.0 = compress more aggressively
        - Missing indices get 1.0

        Use for:
        - Position-aware compression (middle messages get higher bias
          because LLM attention is weakest there)
        - Phase-aware budgets (old exploration messages get lower bias,
          recent execution messages get higher bias)
        - Per-tool learned biases (from TOIN analysis)

        Args:
            messages: The full message list.
            ctx: Compression context.

        Returns:
            Dict of {message_index: bias_float}. Empty dict = all default.
        """
        return {}

    def post_compress(self, event: CompressEvent) -> None:
        """Called after compression completes. Observational only.

        Use for:
        - Failure-driven learning (log events, analyze offline)
        - Per-org analytics and dashboards
        - A/B testing of compression strategies
        - Anomaly detection (alert on sudden ratio changes)

        Args:
            event: Full compression event with before/after metrics.
        """
        pass

    def on_pipeline_event(self, event: PipelineEvent) -> PipelineEvent | None:
        """Observe canonical pipeline lifecycle events.

        Override when the integration needs stable lifecycle notifications beyond
        the three legacy compression-specific hooks.
        """
        return None

"""Compression hooks and pipeline lifecycle events.

Four hooks at well-defined pipeline stages:

1. pre_compress: modify messages before compression (dedup, filter, inject)
2. compute_biases: set per-message compression aggressiveness (position-aware, phase-aware)
3. protect_messages: veto compression of specific messages outright
4. post_compress: observe results after compression (learning, analytics, logging)

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
    """Context passed to the pre_compress, compute_biases and protect_messages hooks.

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

    def protect_messages(
        self,
        messages: list[dict[str, Any]],
        ctx: CompressContext,
    ) -> set[int]:
        """Return message indices that must not be compressed at all.

        ``compute_biases`` is a *soft* lever: a multiplier on how aggressively
        a compressor prunes. Several strategies clamp it against their own
        floors or ignore it entirely, so no bias — however large — reliably
        means "leave this one alone". A hook that has concluded a specific
        message must survive verbatim needs to say so directly, and this is how.

        The router honours these before any routing decision, so the message is
        passed through untouched whatever its content type or role. Use
        sparingly: a protected message is context the compressor cannot reclaim.

        Args:
            messages: The full message list.
            ctx: Compression context.

        Returns:
            Set of message indices to pass through verbatim. Empty = no vetoes.
        """
        return set()

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


def collect_protected(
    hooks: Any,
    messages: list[dict[str, Any]],
    ctx: CompressContext,
) -> set[int] | None:
    """Ask a hooks object for its per-message vetoes, tolerating older ones.

    ``hooks`` is duck-typed at every call site: nothing requires the object to
    be a :class:`CompressionHooks` subclass, and objects written before
    ``protect_messages`` existed will not have the attribute at all. Calling it
    blind would raise ``AttributeError`` inside the compression block, which the
    proxy catches as a compression failure — so adding this hook would silently
    stop compressing for those callers, turning an additive seam into a
    regression. A missing method means "no vetoes".

    Returns None when the hook cannot be asked, which the router treats the same
    as an empty set.
    """
    fn = getattr(hooks, "protect_messages", None)
    if fn is None:
        return None
    protected = fn(messages, ctx)
    if protected is None:
        return None
    # Normalise here rather than trusting the hook: ``hooks`` is duck-typed, so
    # a set is a convention, not a guarantee, and a list of indices is the
    # obvious thing for an implementer to return.
    return {int(i) for i in protected}

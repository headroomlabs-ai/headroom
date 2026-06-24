"""
OpenAI Cache Optimizer.

This module implements cache optimization for OpenAI's automatic prefix caching.
Unlike Anthropic, OpenAI's caching is fully automatic - users cannot control what
gets cached. The only optimization strategy is to stabilize prefixes to maximize
cache hit rates.

OpenAI Caching Details:
    - Fully automatic - no explicit cache control available
    - 50% discount on cached input tokens
    - Requires prompts > 1024 tokens to activate
    - 5-60 minute TTL (varies based on usage patterns)
    - Cache is prefix-based - changes invalidate downstream cache

Optimization Strategy:
    Since we can't control caching explicitly, we focus on PREFIX_STABILIZATION:
    - Extract dynamic content (dates, timestamps) and move to end
    - Normalize whitespace for consistent hashing
    - Remove random IDs from system prompts
    - Track prefix stability to estimate cache hit probability

Dynamic Content Detection Tiers:
    - Tier 1 (regex): Always on, ~0ms - dates, UUIDs, timestamps
    - Tier 2 (ner): Optional, ~5-10ms - names, money, organizations
    - Tier 3 (semantic): Optional, ~20-50ms - volatile patterns via embeddings

Usage:
    # Default: regex only (fastest)
    optimizer = OpenAICacheOptimizer()

    # With NER (requires spacy)
    optimizer = OpenAICacheOptimizer(
        config=CacheConfig(dynamic_detection_tiers=["regex", "ner"])
    )

    # Full detection (requires spacy + sentence-transformers)
    optimizer = OpenAICacheOptimizer(
        config=CacheConfig(dynamic_detection_tiers=["regex", "ner", "semantic"])
    )
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .base import (
    BaseCacheOptimizer,
    CacheConfig,
    CacheMetrics,
    CacheResult,
    CacheStrategy,
    OptimizationContext,
)
from .dynamic_detector import (
    DetectorConfig,
    DynamicContentDetector,
    DynamicSpan,
)


@dataclass
class PrefixAnalysis:
    """
    Analysis of prefix stability.

    Used to determine likelihood of cache hits and track changes
    between requests.
    """

    # Hash of the stabilized prefix
    prefix_hash: str

    # Estimated token count of stable prefix
    stable_tokens: int

    # Dynamic content that was extracted
    dynamic_spans: list[DynamicSpan] = field(default_factory=list)

    # Whether prefix changed from previous request
    changed_from_previous: bool = False

    # Previous hash for comparison
    previous_hash: str | None = None

    # Detection processing time
    detection_time_ms: float = 0.0


class OpenAICacheOptimizer(BaseCacheOptimizer):
    """
    Cache optimizer for OpenAI's automatic prefix caching.

    OpenAI automatically caches prompt prefixes for requests > 1024 tokens.
    Since caching is automatic, this optimizer focuses on maximizing cache
    hit rates by stabilizing prefixes.

    Key Optimizations:
        1. Extract dynamic content (dates, times) and move to end of messages
        2. Normalize whitespace for consistent formatting
        3. Remove random IDs and timestamps from system prompts
        4. Track prefix changes to estimate cache hit probability

    Usage:
        optimizer = OpenAICacheOptimizer()
        result = optimizer.optimize(messages, context)

        # Check if prefix was stable (likely cache hit)
        if not result.metrics.prefix_changed_from_previous:
            print("Likely cache hit - prefix unchanged")

        # Estimate savings
        savings = result.metrics.estimated_savings_percent
        print(f"Estimated savings: {savings:.1f}%")

    Attributes:
        name: Identifier for this optimizer
        provider: The provider this optimizer targets ("openai")
        strategy: Always CacheStrategy.PREFIX_STABILIZATION
    """

    # OpenAI-specific constants
    MIN_TOKENS_FOR_CACHING = 1024
    CACHE_DISCOUNT_PERCENT = 50.0

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize the OpenAI cache optimizer.

        Args:
            config: Optional cache configuration. If not provided,
                   sensible defaults are used.

        The optimizer uses the DynamicContentDetector with configurable tiers:
            - "regex": Fast pattern matching (~0ms) - always on
            - "ner": Named Entity Recognition (~5-10ms) - requires spacy
            - "semantic": Embedding similarity (~20-50ms) - requires sentence-transformers

        Configure tiers via config.dynamic_detection_tiers.
        """
        super().__init__(config)

        # Initialize the tiered dynamic content detector
        detector_config = DetectorConfig(
            tiers=self.config.dynamic_detection_tiers,  # type: ignore
        )
        self._detector = DynamicContentDetector(detector_config)

    @property
    def name(self) -> str:
        """Name of this optimizer."""
        return "openai-prefix-stabilizer"

    @property
    def provider(self) -> str:
        """Provider this optimizer is for."""
        return "openai"

    @property
    def strategy(self) -> CacheStrategy:
        """The caching strategy this optimizer uses."""
        return CacheStrategy.PREFIX_STABILIZATION

    def optimize(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
        config: CacheConfig | None = None,
    ) -> CacheResult:
        """
        Optimize messages for OpenAI's prefix caching.

        This method stabilizes the message prefix to maximize cache hit rates.
        Since OpenAI caching is automatic, we focus on ensuring the prefix
        remains consistent across requests.

        Args:
            messages: List of message dictionaries in OpenAI format.
            context: Optimization context with request metadata.
            config: Optional configuration override.

        Returns:
            CacheResult containing:
                - Optimized messages with stabilized prefixes
                - Metrics about prefix stability and estimated savings
                - List of transforms applied
                - Any warnings encountered

        Example:
            >>> optimizer = OpenAICacheOptimizer()
            >>> messages = [
            ...     {"role": "system", "content": "Today is Jan 1, 2024. You are helpful."},
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> context = OptimizationContext(provider="openai", model="gpt-4")
            >>> result = optimizer.optimize(messages, context)
            >>> # Date moved to end, prefix stabilized
        """
        effective_config = config or self.config

        # Handle disabled optimization
        if not effective_config.enabled:
            return CacheResult(
                messages=messages,
                metrics=CacheMetrics(),
                transforms_applied=[],
            )

        # Deep copy to avoid mutating input
        optimized_messages = deepcopy(messages)
        transforms_applied: list[str] = []
        warnings: list[str] = []

        # Track all extracted spans across messages
        all_spans: list[DynamicSpan] = []
        total_detection_time = 0.0

        # Process system messages for prefix stabilization
        for i, msg in enumerate(optimized_messages):
            if msg.get("role") == "system":
                content = msg.get("content", "")

                if isinstance(content, str):
                    # Use tiered dynamic content detector
                    result = self._detector.detect(content)
                    all_spans.extend(result.spans)
                    total_detection_time += result.processing_time_ms

                    # Add any detector warnings
                    warnings.extend(result.warnings)

                    if result.spans:
                        transforms_applied.append(f"extracted_{len(result.spans)}_dynamic_elements")
                        transforms_applied.extend(f"tier_{tier}" for tier in result.tiers_used)

                    # Get static content with dynamic parts removed
                    stabilized = result.static_content

                    # Normalize whitespace
                    if effective_config.normalize_whitespace:
                        stabilized = self._normalize_whitespace(
                            stabilized,
                            collapse_blank_lines=effective_config.collapse_blank_lines,
                        )
                        transforms_applied.append("normalized_whitespace")

                    # If we extracted dynamic content, append it at the end
                    if result.dynamic_content:
                        dynamic_section = self._format_dynamic_section(
                            result.dynamic_content,
                            separator=effective_config.dynamic_separator,
                        )
                        stabilized = stabilized.rstrip() + dynamic_section

                    optimized_messages[i]["content"] = stabilized

                elif isinstance(content, list):
                    # Handle content blocks (less common for OpenAI)
                    new_content = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            result = self._detector.detect(text)
                            all_spans.extend(result.spans)
                            total_detection_time += result.processing_time_ms
                            warnings.extend(result.warnings)

                            stabilized = result.static_content

                            if effective_config.normalize_whitespace:
                                stabilized = self._normalize_whitespace(stabilized)

                            if result.dynamic_content:
                                dynamic_section = self._format_dynamic_section(
                                    result.dynamic_content,
                                    separator=effective_config.dynamic_separator,
                                )
                                stabilized = stabilized.rstrip() + dynamic_section

                            new_content.append({**block, "text": stabilized})
                        else:
                            new_content.append(block)

                    optimized_messages[i]["content"] = new_content
                    if all_spans:
                        transforms_applied.append("processed_content_blocks")

        # Analyze prefix stability
        analysis = self._analyze_prefix(optimized_messages, context)

        # Calculate token estimates
        tokens_before = self._estimate_total_tokens(messages)
        tokens_after = self._estimate_total_tokens(optimized_messages)

        # Build metrics
        metrics = CacheMetrics(
            stable_prefix_tokens=analysis.stable_tokens,
            stable_prefix_hash=analysis.prefix_hash,
            prefix_changed_from_previous=analysis.changed_from_previous,
            previous_prefix_hash=analysis.previous_hash,
            estimated_cache_hit=not analysis.changed_from_previous,
            cacheable_tokens=self._calculate_cacheable_tokens(analysis.stable_tokens),
            non_cacheable_tokens=max(0, tokens_after - analysis.stable_tokens),
            estimated_savings_percent=self._calculate_savings_percent(
                analysis.stable_tokens,
                tokens_after,
                likely_cache_hit=not analysis.changed_from_previous,
            ),
        )

        # Add warnings for suboptimal cases
        if tokens_after < self.MIN_TOKENS_FOR_CACHING:
            warnings.append(
                f"Prompt has ~{tokens_after} tokens, below OpenAI's {self.MIN_TOKENS_FOR_CACHING} "
                f"token minimum for caching. Consider adding more static context."
            )

        if analysis.changed_from_previous:
            warnings.append(
                "Prefix changed from previous request - cache miss likely. "
                "Consider reviewing what content is changing between requests."
            )

        # Record metrics and update state
        self._record_metrics(metrics)
        self._previous_prefix_hash = analysis.prefix_hash

        return CacheResult(
            messages=optimized_messages,
            metrics=metrics,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=list(set(transforms_applied)),  # Dedupe
            warnings=warnings,
        )

    def estimate_savings(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
    ) -> float:
        """
        Estimate potential cost savings from caching.

        OpenAI provides 50% discount on cached tokens. This method estimates
        what portion of tokens are likely to be cached based on prefix
        stability and token count.

        Args:
            messages: Messages to analyze.
            context: Optimization context.

        Returns:
            Estimated savings as a percentage (0-100).
            Returns 0 if prompt is below caching threshold.

        Example:
            >>> savings = optimizer.estimate_savings(messages, context)
            >>> print(f"Potential savings: {savings:.1f}%")
        """
        total_tokens = self._estimate_total_tokens(messages)

        # No savings if below threshold
        if total_tokens < self.MIN_TOKENS_FOR_CACHING:
            return 0.0

        # Extract system content for prefix analysis
        system_content = self._extract_system_content(messages)
        system_tokens = self._count_tokens_estimate(system_content)

        # Estimate cacheable portion (system + early messages)
        # OpenAI caches the longest matching prefix
        cacheable_ratio = min(1.0, system_tokens / total_tokens) if total_tokens > 0 else 0.0

        # Check if prefix is stable
        current_hash = self._compute_prefix_hash(system_content)
        likely_hit = (
            self._previous_prefix_hash is not None and current_hash == self._previous_prefix_hash
        )

        if likely_hit:
            # 50% savings on cacheable portion
            return cacheable_ratio * self.CACHE_DISCOUNT_PERCENT
        else:
            # First request or prefix changed - no immediate savings
            # but return expected savings for future requests
            return cacheable_ratio * self.CACHE_DISCOUNT_PERCENT * 0.5

    def _normalize_whitespace(
        self,
        content: str,
        collapse_blank_lines: bool = True,
    ) -> str:
        """
        Normalize whitespace in content.

        Ensures consistent whitespace formatting to improve prefix matching.
        This helps when the same logical content has minor formatting differences.

        Args:
            content: Text to normalize.
            collapse_blank_lines: If True, multiple blank lines become one.

        Returns:
            Content with normalized whitespace.
        """
        # Normalize line endings
        result = content.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse multiple spaces (but preserve indentation)
        lines = result.split("\n")
        normalized_lines = []

        for line in lines:
            # Preserve leading whitespace, normalize trailing
            stripped = line.rstrip()
            if stripped:
                # Find leading whitespace
                leading = len(line) - len(line.lstrip())
                # Collapse multiple spaces in content (not indentation)
                content_part = " ".join(stripped.split())
                normalized_lines.append(
                    " " * leading + content_part[leading:] if leading else content_part
                )
            else:
                normalized_lines.append("")

        result = "\n".join(normalized_lines)

        # Collapse multiple blank lines
        if collapse_blank_lines:
            while "\n\n\n" in result:
                result = result.replace("\n\n\n", "\n\n")

        return result.strip()

    def _format_dynamic_section(
        self,
        dynamic_content: str,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Format extracted dynamic content as a section to append.

        Creates a clearly marked section containing dynamic values,
        appended to the end of the message to preserve prefix stability.

        Args:
            dynamic_content: The dynamic content string to append.
            separator: Separator to use before the dynamic section.

        Returns:
            Formatted dynamic section string.
        """
        if not dynamic_content or not dynamic_content.strip():
            return ""

        # Format as a context section
        return f"{separator}[Current Context]\n{dynamic_content.strip()}\n"

    def _analyze_prefix(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
    ) -> PrefixAnalysis:
        """
        Analyze the prefix for stability metrics.

        Computes hash of the stable prefix portion and compares with
        previous requests to estimate cache hit likelihood.

        Args:
            messages: Messages to analyze.
            context: Optimization context with previous hash.

        Returns:
            PrefixAnalysis with stability metrics.
        """
        # Extract prefix content (system messages + structure)
        prefix_parts = []

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    prefix_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            prefix_parts.append(block.get("text", ""))

        prefix_content = "\n".join(prefix_parts)
        prefix_hash = self._compute_prefix_hash(prefix_content)
        stable_tokens = self._count_tokens_estimate(prefix_content)

        # Check for changes from previous request
        previous_hash = context.previous_prefix_hash or self._previous_prefix_hash
        changed = previous_hash is not None and prefix_hash != previous_hash

        return PrefixAnalysis(
            prefix_hash=prefix_hash,
            stable_tokens=stable_tokens,
            changed_from_previous=changed,
            previous_hash=previous_hash,
        )

    def _calculate_cacheable_tokens(self, stable_prefix_tokens: int) -> int:
        """
        Calculate how many tokens are likely cacheable.

        OpenAI only caches prompts > 1024 tokens, and caches in chunks.

        Args:
            stable_prefix_tokens: Number of tokens in stable prefix.

        Returns:
            Estimated cacheable token count.
        """
        if stable_prefix_tokens < self.MIN_TOKENS_FOR_CACHING:
            return 0

        # OpenAI caches in 128-token chunks (aligned)
        # Return the aligned cacheable amount
        return (stable_prefix_tokens // 128) * 128

    def _calculate_savings_percent(
        self,
        stable_tokens: int,
        total_tokens: int,
        likely_cache_hit: bool,
    ) -> float:
        """
        Calculate estimated savings percentage.

        Args:
            stable_tokens: Tokens in stable prefix.
            total_tokens: Total tokens in request.
            likely_cache_hit: Whether a cache hit is likely.

        Returns:
            Estimated savings as percentage (0-100).
        """
        if total_tokens == 0:
            return 0.0

        cacheable = self._calculate_cacheable_tokens(stable_tokens)
        if cacheable == 0:
            return 0.0

        cacheable_ratio = cacheable / total_tokens

        if likely_cache_hit:
            # Full 50% savings on cacheable portion
            return cacheable_ratio * self.CACHE_DISCOUNT_PERCENT
        else:
            # No savings on first request, but show potential
            return 0.0

    def _estimate_total_tokens(self, messages: list[dict[str, Any]]) -> int:
        """
        Estimate total tokens in messages.

        Args:
            messages: Messages to count.

        Returns:
            Estimated token count.
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._count_tokens_estimate(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total += self._count_tokens_estimate(block.get("text", ""))
                        elif block.get("type") == "image_url":
                            # Rough estimate for images
                            total += 85  # Base cost
        return total

"""Content router for intelligent compression strategy selection.

This module provides the ContentRouter, which analyzes content and routes it
to the optimal compressor. It handles mixed content by splitting, routing
each section to the appropriate compressor, and reassembling.

Supported Compressors:
- CodeAwareCompressor: Source code (AST-preserving)
- SmartCrusher: JSON arrays
- SearchCompressor: grep/ripgrep results
- LogCompressor: Build/test output
- KompressCompressor: Plain text (ML-based)
- Kompress: Plain text (ML-based, requires [ml] extra)

Routing Strategy:
1. Use source hint if available (highest confidence)
2. Check for mixed content (split and route sections)
3. Detect content type (JSON, code, search, logs, text)
4. Route to appropriate compressor
5. Reassemble and return with routing metadata

Usage:
    >>> from headroom.transforms import ContentRouter
    >>> router = ContentRouter()
    >>> result = router.compress(content)  # Auto-routes to best compressor
    >>> print(result.strategy_used)
    >>> print(result.routing_log)

Pipeline Usage:
    >>> pipeline = TransformPipeline([
    ...     ContentRouter(),   # Handles all content types
    ... ])
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import DEFAULT_EXCLUDE_TOOLS, ReadLifecycleConfig, TransformResult
from ..tokenizer import Tokenizer
from .base import Transform
from .content_detector import ContentType, DetectionResult

logger = logging.getLogger(__name__)


def _detect_content(content: str) -> DetectionResult:
    """Detect content type via the Rust detection chain.

    Stage-3d (PR5) wired this through `headroom._core.detect_content_type`,
    which runs the magika→unidiff→PlainText chain. The Python-side
    Magika+regex fallback path was retired here — single detection
    surface, no parallel paths. The Rust extension is a hard dep
    (no Python fallback) per `feedback_no_silent_fallbacks.md`.

    The Rust binding returns the legacy `DetectionResult` shape with
    `confidence=1.0` and an empty metadata dict. Existing callers
    only consumed `.content_type` from it; the strategy mapping in
    `_strategy_from_detection` keys off that field alone.
    """
    from headroom._core import detect_content_type as _rust_detect

    rust_result = _rust_detect(content)
    # Rust's `content_type` is the lowercase string tag (e.g.
    # "json_array"); translate to the Python `ContentType` enum so
    # downstream mapping keys match.
    content_type = ContentType(rust_result.content_type)
    return DetectionResult(
        content_type=content_type,
        confidence=rust_result.confidence,
        metadata={},
    )


def _create_content_signature(
    content_type: str,
    content: str,
    language: str | None = None,
) -> Any:
    """Create a ToolSignature for non-JSON content types.

    This allows TOIN to track compression patterns for code, search results,
    logs, and text - not just JSON arrays.

    Args:
        content_type: The type of content (e.g., "code_aware", "search", "log", "text").
        content: The content being compressed (for structural hints).
        language: Optional language hint for code.

    Returns:
        A ToolSignature for TOIN tracking.
    """
    try:
        from ..telemetry.models import ToolSignature

        # Create a deterministic structure hash based on content type
        # This groups similar content types together for pattern learning
        if language:
            hash_input = f"content:{content_type}:{language}"
        else:
            hash_input = f"content:{content_type}"

        # Add a structural hint from the content (first 100 chars, hashed)
        # This helps differentiate tool outputs of the same type
        content_sample = content[:100] if content else ""
        structure_hint = hashlib.sha256(content_sample.encode()).hexdigest()[:8]
        hash_input = f"{hash_input}:{structure_hint}"

        # Keep SHA256: structure_hash feeds into TOIN which persists to disk.
        # Changing hash function would invalidate all learned patterns.
        structure_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:24]

        return ToolSignature(
            structure_hash=structure_hash,
            field_count=0,  # Not applicable for non-JSON
            has_nested_objects=False,
            has_arrays=False,
            max_depth=0,
        )
    except ImportError:
        return None


class CompressionCache:
    """Two-tier compression cache with TTL.

    Tier 1 (skip set): content hashes that won't compress — instant skip,
    near-zero memory (just ints in a set).

    Tier 2 (result cache): compressed results for content that DID compress —
    reuse the compressed text on subsequent requests.

    Entries expire after TTL (default 30min). No max-entries cap — TTL is the
    natural bound. Memory grows proportional to compressible content × TTL,
    which is bounded by session duration.

    Uses in-process dict for ultra-fast lookups (~100ns). Could be backed
    by memcached/Redis for multi-process deployments.
    """

    def __init__(self, ttl_seconds: int = 1800):
        # Tier 2: compressed results {hash: (text, ratio, strategy, timestamp)}
        self._results: dict[int, tuple[str, float, str, float]] = {}
        # Tier 1: hashes of content that won't compress {hash: timestamp}
        self._skip: dict[int, float] = {}
        self._ttl_seconds = ttl_seconds
        # Metrics
        self._hits = 0
        self._misses = 0
        self._skip_hits = 0
        self._evictions = 0
        self._total_lookup_ns = 0
        self._lookup_count = 0

    def get(self, key: int) -> tuple[str, float, str] | None:
        """Get cached compression result.

        Returns (compressed_text, ratio, strategy) or None if not found/expired.
        Use is_skipped() first to check if content is known non-compressible.
        """
        t0 = time.perf_counter_ns()
        entry = self._results.get(key)
        if entry is not None:
            compressed, ratio, strategy, created_at = entry
            if (time.time() - created_at) < self._ttl_seconds:
                self._hits += 1
                self._total_lookup_ns += time.perf_counter_ns() - t0
                self._lookup_count += 1
                return (compressed, ratio, strategy)
            else:
                del self._results[key]
                self._evictions += 1
        self._misses += 1
        self._total_lookup_ns += time.perf_counter_ns() - t0
        self._lookup_count += 1
        return None

    def is_skipped(self, key: int) -> bool:
        """Check if content is known non-compressible (Tier 1)."""
        ts = self._skip.get(key)
        if ts is not None:
            if (time.time() - ts) < self._ttl_seconds:
                self._skip_hits += 1
                return True
            else:
                del self._skip[key]
                self._evictions += 1
        return False

    def put(self, key: int, compressed: str, ratio: float, strategy: str) -> None:
        """Store a compressed result (Tier 2)."""
        self._results[key] = (compressed, ratio, strategy, time.time())

    def mark_skip(self, key: int) -> None:
        """Mark content as non-compressible (Tier 1)."""
        self._skip[key] = time.time()

    def move_to_skip(self, key: int) -> None:
        """Move a result to skip set (threshold tightened, no longer qualifies)."""
        self._results.pop(key, None)
        self._skip[key] = time.time()

    @property
    def size(self) -> int:
        return len(self._results)

    @property
    def skip_size(self) -> int:
        return len(self._skip)

    @property
    def stats(self) -> dict[str, int | float]:
        avg_ns = self._total_lookup_ns / self._lookup_count if self._lookup_count else 0
        return {
            "cache_hits": self._hits,
            "cache_skip_hits": self._skip_hits,
            "cache_misses": self._misses,
            "cache_evictions": self._evictions,
            "cache_size": len(self._results),
            "cache_skip_size": len(self._skip),
            "cache_avg_lookup_ns": avg_ns,
        }

    def clear(self) -> None:
        """Clear all entries (e.g., on session end)."""
        self._results.clear()
        self._skip.clear()


class CompressionStrategy(Enum):
    """Available compression strategies."""

    CODE_AWARE = "code_aware"
    SMART_CRUSHER = "smart_crusher"
    SEARCH = "search"
    LOG = "log"
    KOMPRESS = "kompress"
    TEXT = "text"
    DIFF = "diff"
    HTML = "html"
    MIXED = "mixed"
    PASSTHROUGH = "passthrough"


@dataclass
class RoutingDecision:
    """Record of a single routing decision."""

    content_type: ContentType
    strategy: CompressionStrategy
    original_tokens: int
    compressed_tokens: int
    confidence: float = 1.0
    section_index: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


@dataclass
class ContentSection:
    """A typed section of content."""

    content: str
    content_type: ContentType
    language: str | None = None
    start_line: int = 0
    end_line: int = 0
    is_code_fence: bool = False


@dataclass
class RouterCompressionResult:
    """Result from ContentRouter with routing metadata.

    Attributes:
        compressed: The compressed content.
        original: Original content before compression.
        strategy_used: Primary strategy used for compression.
        routing_log: List of routing decisions made.
        sections_processed: Number of content sections processed.
    """

    compressed: str
    original: str
    strategy_used: CompressionStrategy
    routing_log: list[RoutingDecision] = field(default_factory=list)
    sections_processed: int = 1

    @property
    def total_original_tokens(self) -> int:
        """Total tokens before compression."""
        return sum(r.original_tokens for r in self.routing_log)

    @property
    def total_compressed_tokens(self) -> int:
        """Total tokens after compression."""
        return sum(r.compressed_tokens for r in self.routing_log)

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio."""
        if self.total_original_tokens == 0:
            return 1.0
        return self.total_compressed_tokens / self.total_original_tokens

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return max(0, self.total_original_tokens - self.total_compressed_tokens)

    @property
    def savings_percentage(self) -> float:
        """Percentage of tokens saved."""
        if self.total_original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.total_original_tokens) * 100

    def summary(self) -> str:
        """Human-readable routing summary."""
        if self.strategy_used == CompressionStrategy.MIXED:
            strategies = {r.strategy.value for r in self.routing_log}
            return (
                f"Mixed content: {self.sections_processed} sections, "
                f"routed to {strategies}. "
                f"{self.total_original_tokens:,}→{self.total_compressed_tokens:,} tokens "
                f"({self.savings_percentage:.0f}% saved)"
            )
        else:
            return (
                f"Pure {self.strategy_used.value}: "
                f"{self.total_original_tokens:,}→{self.total_compressed_tokens:,} tokens "
                f"({self.savings_percentage:.0f}% saved)"
            )


@dataclass
class ContentRouterConfig:
    """Configuration for intelligent content routing.

    Attributes:
        enable_code_aware: Enable AST-based code compression.
        enable_smart_crusher: Enable JSON array compression.
        enable_search_compressor: Enable search result compression.
        enable_log_compressor: Enable build/test log compression.
        enable_image_optimizer: Enable image token optimization.
        prefer_code_aware_for_code: Use CodeAware over Kompress for code.
        mixed_content_threshold: Min distinct types to consider "mixed".
        min_section_tokens: Minimum tokens for a section to compress.
        fallback_strategy: Strategy when no compressor matches.
        skip_user_messages: Never compress user messages (they're the subject).
        skip_recent_messages: Don't compress last N messages (likely the subject).
        protect_analysis_context: Detect "analyze/review" intent, skip compression.
    """

    # Enable/disable specific compressors
    enable_code_aware: bool = False  # Disabled: use code graph MCP tools instead
    enable_kompress: bool = True  # Kompress: ModernBERT token compressor
    enable_smart_crusher: bool = True
    enable_search_compressor: bool = True
    enable_log_compressor: bool = True
    enable_html_extractor: bool = True  # HTML content extraction
    enable_image_optimizer: bool = True  # Image token optimization

    # Routing preferences
    prefer_code_aware_for_code: bool = False  # Disabled: let code pass through unmangled
    mixed_content_threshold: int = 2  # Min types to consider mixed
    min_section_tokens: int = 20  # Min tokens to compress a section

    # Fallback: Kompress handles unknown/mixed content instead of passing through
    fallback_strategy: CompressionStrategy = CompressionStrategy.KOMPRESS

    # Protection: Don't compress content that's likely the subject of analysis
    skip_user_messages: bool = True  # User messages contain what they want analyzed
    protect_recent_code: int = 4  # Don't compress CODE in last N messages (0 = disabled)
    protect_analysis_context: bool = True  # Detect "analyze/review" intent, protect code

    # Adaptive Read protection: fraction of total messages to protect from
    # compression.  At 10 msgs, protects ~5 Reads.  At 100 msgs, protects ~10.
    # Old Reads beyond this window become compressible even though they are
    # in DEFAULT_EXCLUDE_TOOLS.  0.0 = always exclude all (old behavior).
    protect_recent_reads_fraction: float = (
        0.0  # 0.0 = protect ALL excluded-tool outputs (safest for coding agents)
    )

    # Adaptive compression ratio: scales with context pressure.
    # At low pressure (<30% full), use the relaxed threshold (reject marginal).
    # At high pressure (>80% full), use the aggressive threshold (accept anything helpful).
    # Linearly interpolates between the two.
    min_ratio_relaxed: float = 0.85  # when context is mostly empty
    min_ratio_aggressive: float = 0.65  # when context is nearly full

    # CCR (Compress-Cache-Retrieve) settings for SmartCrusher
    ccr_enabled: bool = True  # Enable CCR marker injection for reversible compression
    ccr_inject_marker: bool = True  # Add retrieval markers to compressed content

    # Tag protection: preserve custom/workflow XML tags from text compression.
    # When False (default), entire <custom-tag>content</custom-tag> blocks are
    # protected verbatim.  When True, only the tag markers are protected and
    # the content between them can be compressed.
    compress_tagged_content: bool = False

    # Tools to exclude from compression (output passed through unmodified)
    # Set to None to use DEFAULT_EXCLUDE_TOOLS, or provide custom set
    exclude_tools: set[str] | None = None

    # Read lifecycle management (stale/superseded detection)
    read_lifecycle: ReadLifecycleConfig = field(default_factory=ReadLifecycleConfig)

    # Per-tool compression profiles (tool_name → CompressionProfile)
    # Set to None to use DEFAULT_TOOL_PROFILES from config
    tool_profiles: dict[str, Any] | None = None


# Patterns for detecting mixed content
_CODE_FENCE_PATTERN = re.compile(r"^```(\w*)\s*$", re.MULTILINE)
_JSON_BLOCK_START = re.compile(r"^\s*[\[{]", re.MULTILINE)
_SEARCH_RESULT_PATTERN = re.compile(r"^\S+:\d+:", re.MULTILINE)
_PROSE_PATTERN = re.compile(r"[A-Z][a-z]+\s+\w+\s+\w+")


def is_mixed_content(content: str) -> bool:
    """Detect if content contains multiple distinct types.

    Args:
        content: Content to analyze.

    Returns:
        True if content appears to be mixed (multiple types).
    """
    indicators = {
        "has_code_fences": bool(_CODE_FENCE_PATTERN.search(content)),
        "has_json_blocks": bool(_JSON_BLOCK_START.search(content)),
        "has_prose": len(_PROSE_PATTERN.findall(content)) > 5,
        "has_search_results": bool(_SEARCH_RESULT_PATTERN.search(content)),
    }

    # Mixed if 2+ indicators are true
    return sum(indicators.values()) >= 2


def split_into_sections(content: str) -> list[ContentSection]:
    """Parse mixed content into typed sections.

    Args:
        content: Mixed content to split.

    Returns:
        List of ContentSection objects.
    """
    sections: list[ContentSection] = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code fence: ```language
        if match := _CODE_FENCE_PATTERN.match(line):
            language = match.group(1) or "unknown"
            code_lines = []
            start_line = i
            i += 1

            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1

            sections.append(
                ContentSection(
                    content="\n".join(code_lines),
                    content_type=ContentType.SOURCE_CODE,
                    language=language,
                    start_line=start_line,
                    end_line=i,
                    is_code_fence=True,
                )
            )
            i += 1  # Skip closing ```
            continue

        # JSON block
        if line.strip().startswith(("[", "{")):
            json_content, end_i = _extract_json_block(lines, i)
            if json_content:
                sections.append(
                    ContentSection(
                        content=json_content,
                        content_type=ContentType.JSON_ARRAY,
                        start_line=i,
                        end_line=end_i,
                    )
                )
                i = end_i + 1
                continue

        # Search result lines
        if _SEARCH_RESULT_PATTERN.match(line):
            search_lines = []
            start_line = i
            while i < len(lines) and _SEARCH_RESULT_PATTERN.match(lines[i]):
                search_lines.append(lines[i])
                i += 1
            sections.append(
                ContentSection(
                    content="\n".join(search_lines),
                    content_type=ContentType.SEARCH_RESULTS,
                    start_line=start_line,
                    end_line=i - 1,
                )
            )
            continue

        # Collect text until next special section
        text_lines = [line]
        start_line = i
        i += 1

        while i < len(lines):
            next_line = lines[i]
            # Stop if we hit a special section
            if (
                _CODE_FENCE_PATTERN.match(next_line)
                or next_line.strip().startswith(("[", "{"))
                or _SEARCH_RESULT_PATTERN.match(next_line)
            ):
                break
            text_lines.append(next_line)
            i += 1

        # Only add non-empty text sections
        text_content = "\n".join(text_lines)
        if text_content.strip():
            sections.append(
                ContentSection(
                    content=text_content,
                    content_type=ContentType.PLAIN_TEXT,
                    start_line=start_line,
                    end_line=i - 1,
                )
            )

    return sections


def _extract_json_block(lines: list[str], start: int) -> tuple[str | None, int]:
    """Extract a complete JSON block from lines.

    Args:
        lines: All lines of content.
        start: Starting line index.

    Returns:
        Tuple of (json_content, end_line_index) or (None, start) if invalid.
    """
    bracket_count = 0
    brace_count = 0
    json_lines = []

    for i in range(start, len(lines)):
        line = lines[i]
        json_lines.append(line)

        bracket_count += line.count("[") - line.count("]")
        brace_count += line.count("{") - line.count("}")

        if bracket_count <= 0 and brace_count <= 0 and json_lines:
            return "\n".join(json_lines), i

    # Didn't find complete JSON
    return None, start


class ContentRouter(Transform):
    """Intelligent router that selects optimal compression strategy.

    ContentRouter is the recommended entry point for Headroom's compression.
    It analyzes content and routes it to the most appropriate compressor,
    handling mixed content by splitting and reassembling.

    Key Features:
    - Automatic content type detection
    - Source hint support for high-confidence routing
    - Mixed content handling (split → route → reassemble)
    - Graceful fallback when compressors unavailable
    - Rich routing metadata for debugging

    Example:
        >>> router = ContentRouter()
        >>>
        >>> # Automatically uses CodeAwareCompressor
        >>> result = router.compress(python_code)
        >>> print(result.strategy_used)  # CompressionStrategy.CODE_AWARE
        >>>
        >>> # Automatically uses SmartCrusher
        >>> result = router.compress(json_array)
        >>> print(result.strategy_used)  # CompressionStrategy.SMART_CRUSHER
        >>>
        >>> # Splits and routes each section
        >>> result = router.compress(readme_with_code)
        >>> print(result.strategy_used)  # CompressionStrategy.MIXED

    Pipeline Integration:
        >>> pipeline = TransformPipeline([
        ...     ContentRouter(),   # Handles ALL content types
        ... ])
    """

    name: str = "content_router"

    def __init__(
        self,
        config: ContentRouterConfig | None = None,
        observer: Any = None,
    ):
        """Initialize content router.

        Args:
            config: Router configuration. Uses defaults if None.
            observer: Optional `CompressionObserver` (see
                `headroom.transforms.observability`) called once per
                routing decision after `compress()` finishes. The
                proxy's `PrometheusMetrics` is the production
                implementation — it increments per-strategy counters
                so silent regressions become visible. `None` disables
                observation; pick one explicitly per the no-fallback
                rule in the audit doc.
        """
        self.config = config or ContentRouterConfig()
        self._observer = observer

        # Lazy-loaded compressors
        self._code_compressor: Any = None
        self._smart_crusher: Any = None
        self._search_compressor: Any = None
        self._log_compressor: Any = None
        self._diff_compressor: Any = None
        self._html_extractor: Any = None
        self._kompress: Any = None

        # TOIN integration for cross-strategy learning
        self._toin: Any = None

        self._cache = CompressionCache()

    def _record_to_toin(
        self,
        strategy: CompressionStrategy,
        content: str,
        compressed: str,
        original_tokens: int,
        compressed_tokens: int,
        language: str | None = None,
        context: str = "",
    ) -> None:
        """Record compression to TOIN for cross-user learning.

        This allows TOIN to track compression patterns for ALL content types,
        not just JSON arrays. When the LLM retrieves original content via CCR,
        TOIN learns which compressions users need to expand.

        Args:
            strategy: The compression strategy used.
            content: Original content (for signature generation).
            compressed: Compressed content.
            original_tokens: Token count before compression.
            compressed_tokens: Token count after compression.
            language: Optional language hint for code.
            context: Query context for pattern learning.
        """
        # Skip SmartCrusher - it handles its own TOIN recording
        if strategy == CompressionStrategy.SMART_CRUSHER:
            return

        # Skip if no actual compression happened
        if original_tokens <= compressed_tokens:
            return

        try:
            # Lazy load TOIN
            if self._toin is None:
                from ..telemetry.toin import get_toin

                self._toin = get_toin()

            # Create a content-type signature
            signature = _create_content_signature(
                content_type=strategy.value,
                content=content,
                language=language,
            )

            if signature is None:
                return

            # Record the compression
            self._toin.record_compression(
                tool_signature=signature,
                original_count=1,  # Single content block
                compressed_count=1,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                strategy=strategy.value,
                query_context=context if context else None,
            )

            logger.debug(
                "TOIN: Recorded %s compression: %d → %d tokens",
                strategy.value,
                original_tokens,
                compressed_tokens,
            )

        except Exception as e:
            # TOIN recording should never break compression
            logger.debug("TOIN recording failed (non-fatal): %s", e)

    def _timed_compress(
        self, content: str, context: str, bias: float
    ) -> tuple[RouterCompressionResult, float]:
        """Compress with wall-clock timing.  Used by parallel executor."""
        t0 = time.perf_counter()
        result = self.compress(content, context=context, bias=bias)
        return result, (time.perf_counter() - t0) * 1000

    def compress(
        self,
        content: str,
        context: str = "",
        question: str | None = None,
        bias: float = 1.0,
    ) -> RouterCompressionResult:
        """Compress content using optimal strategy based on content detection.

        Args:
            content: Content to compress.
            context: Optional context for relevance-aware compression.
            question: Optional question for QA-aware compression. When provided,
                tokens relevant to answering this question are preserved.
            bias: Compression bias multiplier (>1 = keep more, <1 = keep fewer).

        Returns:
            RouterCompressionResult with compressed content and routing metadata.
        """
        if not content or not content.strip():
            result = RouterCompressionResult(
                compressed=content,
                original=content,
                strategy_used=CompressionStrategy.PASSTHROUGH,
                routing_log=[],
            )
        else:
            # Determine strategy from content analysis
            strategy = self._determine_strategy(content)

            if strategy == CompressionStrategy.MIXED:
                result = self._compress_mixed(content, context, question, bias=bias)
            else:
                result = self._compress_pure(content, strategy, context, question, bias=bias)

        # One observer call per routing decision; the observer is the
        # forcing function for catching strategy-level regressions.
        # Empty routing_log (passthrough fast path) → no calls.
        self._observe(result)
        return result

    def _observe(self, result: RouterCompressionResult) -> None:
        """Forward each `RoutingDecision` in `result.routing_log` to the
        configured `CompressionObserver`. No-op when no observer is set.

        Observers MUST NOT raise per the protocol contract; if one does
        anyway, swallow at debug level. Compression already succeeded;
        a buggy observer must not turn a 200 into a 500.
        """
        if self._observer is None:
            return
        for d in result.routing_log:
            try:
                self._observer.record_compression(
                    strategy=d.strategy.value,
                    original_tokens=d.original_tokens,
                    compressed_tokens=d.compressed_tokens,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("CompressionObserver raised (non-fatal): %s", e)

    def _determine_strategy(self, content: str) -> CompressionStrategy:
        """Determine the compression strategy from content analysis.

        Args:
            content: Content to analyze.

        Returns:
            Selected compression strategy.
        """
        # 1. Check for mixed content
        if is_mixed_content(content):
            return CompressionStrategy.MIXED

        # 2. Detect content type from content itself
        detection = _detect_content(content)
        return self._strategy_from_detection(detection)

    def _strategy_from_detection(self, detection: Any) -> CompressionStrategy:
        """Get strategy from content detection result.

        Args:
            detection: Result from detect_content_type.

        Returns:
            Selected strategy.
        """
        mapping = {
            ContentType.SOURCE_CODE: CompressionStrategy.CODE_AWARE,
            ContentType.JSON_ARRAY: CompressionStrategy.SMART_CRUSHER,
            ContentType.SEARCH_RESULTS: CompressionStrategy.SEARCH,
            ContentType.BUILD_OUTPUT: CompressionStrategy.LOG,
            ContentType.GIT_DIFF: CompressionStrategy.DIFF,
            ContentType.HTML: CompressionStrategy.HTML,
            ContentType.PLAIN_TEXT: CompressionStrategy.TEXT,
        }

        strategy = mapping.get(detection.content_type, self.config.fallback_strategy)

        # Override: prefer CodeAware for code if configured
        if (
            strategy == CompressionStrategy.CODE_AWARE
            and not self.config.prefer_code_aware_for_code
        ):
            strategy = CompressionStrategy.KOMPRESS

        return strategy

    def _compress_mixed(
        self,
        content: str,
        context: str,
        question: str | None = None,
        bias: float = 1.0,
    ) -> RouterCompressionResult:
        """Compress mixed content by splitting and routing sections.

        Args:
            content: Mixed content to compress.
            context: User context for relevance.
            question: Optional question for QA-aware compression.
            bias: Compression bias multiplier.

        Returns:
            RouterCompressionResult with reassembled content.
        """
        sections = split_into_sections(content)

        if not sections:
            return RouterCompressionResult(
                compressed=content,
                original=content,
                strategy_used=CompressionStrategy.PASSTHROUGH,
            )

        compressed_sections: list[str] = []
        routing_log: list[RoutingDecision] = []

        for i, section in enumerate(sections):
            # Get strategy for this section
            strategy = self._strategy_from_detection_type(section.content_type)

            # Compress section
            original_tokens = len(section.content.split())
            compressed_content, compressed_tokens = self._apply_strategy_to_content(
                section.content,
                strategy,
                context,
                section.language,
                question,
                bias=bias,
            )

            # Preserve code fence markers
            if section.is_code_fence and section.language:
                compressed_content = f"```{section.language}\n{compressed_content}\n```"

            compressed_sections.append(compressed_content)
            routing_log.append(
                RoutingDecision(
                    content_type=section.content_type,
                    strategy=strategy,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    section_index=i,
                )
            )

        return RouterCompressionResult(
            compressed="\n\n".join(compressed_sections),
            original=content,
            strategy_used=CompressionStrategy.MIXED,
            routing_log=routing_log,
            sections_processed=len(sections),
        )

    def _compress_pure(
        self,
        content: str,
        strategy: CompressionStrategy,
        context: str,
        question: str | None = None,
        bias: float = 1.0,
    ) -> RouterCompressionResult:
        """Compress pure (non-mixed) content.

        Args:
            content: Content to compress.
            strategy: Selected strategy.
            context: User context.
            question: Optional question for QA-aware compression.
            bias: Compression bias multiplier.

        Returns:
            RouterCompressionResult.
        """
        original_tokens = len(content.split())

        compressed, compressed_tokens = self._apply_strategy_to_content(
            content, strategy, context, question=question, bias=bias
        )

        return RouterCompressionResult(
            compressed=compressed,
            original=content,
            strategy_used=strategy,
            routing_log=[
                RoutingDecision(
                    content_type=self._content_type_from_strategy(strategy),
                    strategy=strategy,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                )
            ],
        )

    def _apply_strategy_to_content(
        self,
        content: str,
        strategy: CompressionStrategy,
        context: str,
        language: str | None = None,
        question: str | None = None,
        bias: float = 1.0,
    ) -> tuple[str, int]:
        """Apply a compression strategy to content.

        Args:
            content: Content to compress.
            strategy: Strategy to use.
            context: User context.
            language: Language hint for code.
            question: Optional question for QA-aware compression.
            bias: Compression bias multiplier (>1 = keep more, <1 = keep fewer).

        Returns:
            Tuple of (compressed_content, compressed_token_count).
        """
        # Track original tokens for TOIN recording
        original_tokens = len(content.split())
        compressed: str | None = None
        compressed_tokens: int | None = None

        try:
            if strategy == CompressionStrategy.CODE_AWARE:
                if self.config.enable_code_aware:
                    compressor = self._get_code_compressor()
                    if compressor:
                        result = compressor.compress(content, language=language, context=context)
                        compressed, compressed_tokens = result.compressed, result.compressed_tokens
                if compressed is None:
                    # Fallback to Kompress
                    compressed, compressed_tokens = self._try_ml_compressor(
                        content, context, question
                    )
                    strategy = CompressionStrategy.KOMPRESS  # Update for TOIN

            elif strategy == CompressionStrategy.SMART_CRUSHER:
                # SmartCrusher handles its own TOIN recording
                if self.config.enable_smart_crusher:
                    crusher = self._get_smart_crusher()
                    if crusher:
                        result = crusher.crush(content, query=context, bias=bias)
                        return result.compressed, len(result.compressed.split())

            elif strategy == CompressionStrategy.SEARCH:
                if self.config.enable_search_compressor:
                    compressor = self._get_search_compressor()
                    if compressor:
                        result = compressor.compress(content, context=context, bias=bias)
                        compressed, compressed_tokens = (
                            result.compressed,
                            len(result.compressed.split()),
                        )

            elif strategy == CompressionStrategy.LOG:
                if self.config.enable_log_compressor:
                    compressor = self._get_log_compressor()
                    if compressor:
                        result = compressor.compress(content, bias=bias)
                        compressed, compressed_tokens = (
                            result.compressed,
                            result.compressed_line_count,
                        )

            elif strategy == CompressionStrategy.DIFF:
                compressor = self._get_diff_compressor()
                if compressor:
                    result = compressor.compress(content, context=context)
                    compressed, compressed_tokens = (
                        result.compressed,
                        result.compressed_line_count,
                    )

            elif strategy == CompressionStrategy.HTML:
                if self.config.enable_html_extractor:
                    extractor = self._get_html_extractor()
                    if extractor:
                        result = extractor.extract(content)
                        compressed = result.extracted
                        # Estimate tokens from extracted text (simple word count)
                        compressed_tokens = len(compressed.split()) if compressed else 0

            elif strategy == CompressionStrategy.KOMPRESS:
                compressed, compressed_tokens = self._try_ml_compressor(content, context, question)

            elif strategy == CompressionStrategy.TEXT:
                # Prefer Kompress ML compressor for text
                # Passes through unchanged if Kompress not available
                compressed, compressed_tokens = self._try_ml_compressor(content, context, question)

        except Exception as e:
            logger.warning("Compression with %s failed: %s", strategy.value, e)

        # If compression succeeded, record to TOIN
        if compressed is not None and compressed_tokens is not None:
            self._record_to_toin(
                strategy=strategy,
                content=content,
                compressed=compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                language=language,
                context=context,
            )
            return compressed, compressed_tokens

        # Fallback: return unchanged
        return content, original_tokens

    def _try_ml_compressor(
        self, content: str, context: str, question: str | None = None
    ) -> tuple[str, int]:
        """ML-based compression using Kompress.

        Kompress (ModernBERT, trained on 330K structured tool outputs)
        auto-downloads from HuggingFace on first use. No heuristic fallback.

        Custom/workflow XML tags (<system-reminder>, <tool_call>, <thinking>)
        are protected before compression and restored after.  Standard HTML
        tags are left alone (HTMLExtractor handles those separately).

        Args:
            content: Content to compress.
            context: User context.
            question: Optional question for QA-aware compression.

        Returns:
            Tuple of (compressed, token_count).
        """
        from .tag_protector import protect_tags, restore_tags

        # Protect custom tags before any ML compression
        cleaned, protected = protect_tags(
            content,
            compress_tagged_content=self.config.compress_tagged_content,
        )

        # If the entire content is custom tags with nothing to compress
        if protected and not cleaned.strip():
            return content, len(content.split())

        # Use the cleaned (tag-free) text for compression
        text_to_compress = cleaned if protected else content
        compressed: str | None = None
        compressed_tokens: int | None = None

        # Primary: Kompress — downloads from chopratejas/kompress-base on first use
        if self.config.enable_kompress:
            compressor = self._get_kompress()
            if compressor:
                try:
                    result = compressor.compress(
                        text_to_compress,
                        context=context,
                        question=question,
                        target_ratio=getattr(self, "_runtime_target_ratio", None),
                    )
                    compressed = result.compressed
                    compressed_tokens = result.compressed_tokens
                except Exception as e:
                    logger.warning("Kompress failed: %s", e)

        if compressed is None:
            return content, len(content.split())

        # Restore protected tag blocks into the compressed text
        if protected:
            compressed = restore_tags(compressed, protected)
            compressed_tokens = len(compressed.split())

        return compressed, compressed_tokens or len(compressed.split())

    def _strategy_from_detection_type(self, content_type: ContentType) -> CompressionStrategy:
        """Get strategy from ContentType enum."""
        mapping = {
            ContentType.SOURCE_CODE: CompressionStrategy.CODE_AWARE,
            ContentType.JSON_ARRAY: CompressionStrategy.SMART_CRUSHER,
            ContentType.SEARCH_RESULTS: CompressionStrategy.SEARCH,
            ContentType.BUILD_OUTPUT: CompressionStrategy.LOG,
            ContentType.GIT_DIFF: CompressionStrategy.DIFF,
            ContentType.HTML: CompressionStrategy.HTML,
            ContentType.PLAIN_TEXT: CompressionStrategy.TEXT,
        }
        return mapping.get(content_type, self.config.fallback_strategy)

    def _content_type_from_strategy(self, strategy: CompressionStrategy) -> ContentType:
        """Get ContentType from strategy."""
        mapping = {
            CompressionStrategy.CODE_AWARE: ContentType.SOURCE_CODE,
            CompressionStrategy.SMART_CRUSHER: ContentType.JSON_ARRAY,
            CompressionStrategy.SEARCH: ContentType.SEARCH_RESULTS,
            CompressionStrategy.LOG: ContentType.BUILD_OUTPUT,
            CompressionStrategy.DIFF: ContentType.GIT_DIFF,
            CompressionStrategy.HTML: ContentType.HTML,
            CompressionStrategy.TEXT: ContentType.PLAIN_TEXT,
            CompressionStrategy.KOMPRESS: ContentType.PLAIN_TEXT,
            CompressionStrategy.PASSTHROUGH: ContentType.PLAIN_TEXT,
        }
        return mapping.get(strategy, ContentType.PLAIN_TEXT)

    # Lazy compressor getters

    def _get_code_compressor(self) -> Any:
        """Get CodeAwareCompressor (lazy load)."""
        if self._code_compressor is None:
            try:
                from .code_compressor import CodeAwareCompressor, _check_tree_sitter_available

                if _check_tree_sitter_available():
                    self._code_compressor = CodeAwareCompressor()
                else:
                    logger.debug("tree-sitter not available")
            except ImportError:
                logger.debug("CodeAwareCompressor not available")
        return self._code_compressor

    def _get_smart_crusher(self) -> Any:
        """Get SmartCrusher (lazy load) with CCR config."""
        if self._smart_crusher is None:
            try:
                from ..config import CCRConfig
                from .smart_crusher import SmartCrusher

                # Pass CCR config for marker injection
                ccr_config = CCRConfig(
                    enabled=self.config.ccr_enabled,
                    inject_retrieval_marker=self.config.ccr_inject_marker,
                )
                self._smart_crusher = SmartCrusher(ccr_config=ccr_config)
            except ImportError:
                logger.debug("SmartCrusher not available")
        return self._smart_crusher

    def _get_search_compressor(self) -> Any:
        """Get SearchCompressor (lazy load)."""
        if self._search_compressor is None:
            try:
                from .search_compressor import SearchCompressor

                self._search_compressor = SearchCompressor()
            except ImportError:
                logger.debug("SearchCompressor not available")
        return self._search_compressor

    def _get_log_compressor(self) -> Any:
        """Get LogCompressor (lazy load)."""
        if self._log_compressor is None:
            try:
                from .log_compressor import LogCompressor

                self._log_compressor = LogCompressor()
            except ImportError:
                logger.debug("LogCompressor not available")
        return self._log_compressor

    def _get_diff_compressor(self) -> Any:
        """Get DiffCompressor (lazy load). Rust-only — Python implementation
        retired in Stage 3b. The wheel (`headroom._core`) is a hard import.
        """
        if self._diff_compressor is None:
            from .diff_compressor import DiffCompressor

            self._diff_compressor = DiffCompressor()
        return self._diff_compressor

    def _get_html_extractor(self) -> Any:
        """Get HTMLExtractor (lazy load)."""
        if self._html_extractor is None:
            try:
                from .html_extractor import HTMLExtractor

                self._html_extractor = HTMLExtractor()
            except ImportError:
                logger.debug("HTMLExtractor not available (install trafilatura)")
        return self._html_extractor

    def eager_load_compressors(self) -> dict[str, str]:
        """Pre-load compressors at startup to avoid first-request latency.

        Call this during proxy startup to load models and parsers
        before any requests arrive. Eliminates cold-start latency spikes.

        Returns:
            Dict of component name -> status string for logging.
        """
        status: dict[str, str] = {}

        # 1. ML text compressor: Kompress
        if self.config.enable_kompress:
            compressor = self._get_kompress()
            if compressor:
                logger.info("Kompress model pre-loaded at startup")
                status["kompress"] = "enabled"
            else:
                status["kompress"] = "unavailable"

        # 2. Magika content detector (avoids 100-200ms on first content detection)
        try:
            from ..compression.detector import _get_magika, _magika_available

            if _magika_available():
                _get_magika()  # Initializes the singleton
                logger.info("Magika content detector pre-loaded at startup")
                status["magika"] = "enabled"
            else:
                status["magika"] = "not installed"
        except Exception as e:
            logger.debug("Magika pre-load skipped: %s", e)
            status["magika"] = "skipped"

        # 3. CodeAware compressor + common tree-sitter parsers
        if self.config.enable_code_aware:
            code_compressor = self._get_code_compressor()
            if code_compressor:
                status["code_aware"] = "enabled"
                # Pre-load tree-sitter parsers for common languages
                # Each parser is ~50ms to load; doing it here avoids 500ms+ on first code hit
                try:
                    from .code_compressor import _check_tree_sitter_available, _get_parser

                    if _check_tree_sitter_available():
                        common_languages = [
                            "python",
                            "javascript",
                            "typescript",
                            "go",
                            "rust",
                            "java",
                            "c",
                            "cpp",
                        ]
                        loaded = []
                        for lang in common_languages:
                            try:
                                _get_parser(lang)
                                loaded.append(lang)
                            except (ValueError, ImportError):
                                pass  # Language not available, skip
                        if loaded:
                            logger.info("Tree-sitter parsers pre-loaded: %s", ", ".join(loaded))
                            status["tree_sitter"] = f"loaded ({len(loaded)} languages)"
                except Exception as e:
                    logger.debug("Tree-sitter pre-load skipped: %s", e)
                    status["tree_sitter"] = "skipped"
            else:
                status["code_aware"] = "not installed"

        # 4. SmartCrusher (lightweight init, but ensures import + TOIN ready)
        smart_crusher = self._get_smart_crusher()
        if smart_crusher:
            status["smart_crusher"] = "ready"

        return status

    def _get_kompress(self) -> Any:
        """Get KompressCompressor (lazy load). Downloads from HuggingFace on first use.

        Respects runtime kompress_model kwarg:
        - None: use default (chopratejas/kompress-base) — cached on self
        - "disabled": return None (skip ML compression entirely)
        - any model ID string: create compressor with that model
          (model weights are cached at module level in kompress_compressor.py,
          so repeated calls with the same model_id are cheap)
        """
        model_id = getattr(self, "_runtime_kompress_model", None)

        # Explicitly disabled — no ML compression
        if model_id == "disabled":
            return None

        # Custom model — don't touch self._kompress (that's the default cache)
        if model_id:
            try:
                from .kompress_compressor import (
                    KompressCompressor,
                    KompressConfig,
                    is_kompress_available,
                )

                if is_kompress_available():
                    return KompressCompressor(config=KompressConfig(model_id=model_id))
            except ImportError:
                pass
            return None

        # Default path — exactly as before, cached on self
        if self._kompress is None:
            try:
                from .kompress_compressor import KompressCompressor, is_kompress_available

                if is_kompress_available():
                    self._kompress = KompressCompressor()
            except ImportError:
                logger.debug("Kompress dependencies not available")
        return self._kompress

    def _get_image_optimizer(self) -> Any:
        """Create an ImageCompressor for one optimization pass.

        The ImageCompressor handles image token compression using:
        - Trained MiniLM classifier from HuggingFace (chopratejas/technique-router)
        - SigLIP for image analysis
        - Provider-specific compression (OpenAI detail, Anthropic/Google resize)
        """
        try:
            from ..image import ImageCompressor

            return ImageCompressor()
        except ImportError:
            logger.debug("ImageCompressor not available")
            return None

    def optimize_images_in_messages(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        provider: str = "openai",
        user_query: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Optimize images in messages.

        This is a convenience method for image optimization that can be called
        directly or as part of the transform pipeline.

        Uses ImageCompressor with trained MiniLM router from HuggingFace
        (chopratejas/technique-router) + SigLIP for image analysis.

        Args:
            messages: Messages potentially containing images.
            tokenizer: Tokenizer for token counting (unused, kept for API compat).
            provider: LLM provider (openai, anthropic, google).
            user_query: User query for task intent detection (unused, auto-extracted).

        Returns:
            Tuple of (optimized_messages, metrics).
        """
        if not self.config.enable_image_optimizer:
            return messages, {"images_optimized": 0, "tokens_saved": 0}

        compressor = self._get_image_optimizer()
        if compressor is None:
            return messages, {"images_optimized": 0, "tokens_saved": 0}

        try:
            # Check if there are images to compress
            if not compressor.has_images(messages):
                return messages, {"images_optimized": 0, "tokens_saved": 0}

            # Compress images (query is auto-extracted from messages)
            optimized = compressor.compress(messages, provider=provider)

            # Get metrics from last compression
            result = compressor.last_result
            if result:
                metrics = {
                    "images_optimized": result.compressed_tokens < result.original_tokens,
                    "tokens_before": result.original_tokens,
                    "tokens_after": result.compressed_tokens,
                    "tokens_saved": result.original_tokens - result.compressed_tokens,
                    "technique": result.technique.value,
                    "confidence": result.confidence,
                }
            else:
                metrics = {"images_optimized": 0, "tokens_saved": 0}

            return optimized, metrics
        finally:
            if hasattr(compressor, "close"):
                compressor.close()

    # Transform interface

    def _build_tool_name_map(self, messages: list[dict[str, Any]]) -> dict[str, str]:
        """Build mapping from tool_call_id to tool_name.

        Scans assistant messages to find tool calls and extract their names.
        Supports both OpenAI and Anthropic message formats.
        """
        mapping: dict[str, str] = {}

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            # OpenAI format: tool_calls array
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    name = tc.get("function", {}).get("name", "")
                    if tc_id and name:
                        mapping[tc_id] = name

            # Anthropic format: content blocks with type=tool_use
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tc_id = block.get("id", "")
                        name = block.get("name", "")
                        if tc_id and name:
                            mapping[tc_id] = name

        return mapping

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """Apply intelligent routing to messages.

        Args:
            messages: Messages to transform.
            tokenizer: Tokenizer for counting.
            **kwargs: Additional arguments (context).

        Returns:
            TransformResult with routed and compressed messages.
        """
        # Pre-process: Read lifecycle management (stale/superseded detection)
        if self.config.read_lifecycle.enabled:
            from .read_lifecycle import ReadLifecycleManager

            lifecycle_mgr = ReadLifecycleManager(
                self.config.read_lifecycle,
                compression_store=kwargs.get("compression_store"),
            )
            lifecycle_result = lifecycle_mgr.apply(
                messages,
                frozen_message_count=kwargs.get("frozen_message_count", 0),
            )
            messages = lifecycle_result.messages
            # lifecycle transforms tracked separately, merged at the end
            lifecycle_transforms = lifecycle_result.transforms_applied
            lifecycle_ccr_hashes = lifecycle_result.ccr_hashes
        else:
            lifecycle_transforms = []
            lifecycle_ccr_hashes = []

        # Runtime overrides from CompressConfig (via kwargs from compress())
        # These override self.config defaults for this call only.
        skip_user = (
            kwargs.get("compress_user_messages") is not True and self.config.skip_user_messages
        )
        skip_system = kwargs.get("compress_system_messages") is False
        protect_recent = kwargs.get("protect_recent", self.config.protect_recent_code)
        protect_analysis = kwargs.get(
            "protect_analysis_context", self.config.protect_analysis_context
        )
        min_tokens = kwargs.get("min_tokens_to_compress", 50)
        # Store runtime options on self for access by _route_and_compress_block
        self._runtime_target_ratio: float | None = kwargs.get("target_ratio")
        self._runtime_kompress_model: str | None = kwargs.get("kompress_model")

        tokens_before = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
        context = kwargs.get("context", "")
        hook_biases: dict[int, float] = kwargs.get("biases") or {}

        # Build tool name map for exclusion checking
        tool_name_map = self._build_tool_name_map(messages)

        # Compute excluded tool IDs based on config
        exclude_tools = (
            self.config.exclude_tools
            if self.config.exclude_tools is not None
            else DEFAULT_EXCLUDE_TOOLS
        )
        excluded_tool_ids = {
            tool_id for tool_id, name in tool_name_map.items() if name in exclude_tools
        }

        # --- Adaptive parameters based on context pressure ---
        num_messages = len(messages)
        model_limit = kwargs.get("model_limit", 0)

        # Adaptive Read protection: protect a fraction of recent messages
        if self.config.protect_recent_reads_fraction > 0:
            # Scale: at 10 msgs protect 5, at 50 msgs protect 25, at 200 msgs protect 100
            # But cap at a reasonable floor so very short convos still protect everything
            read_protection_window = max(
                4,  # always protect at least last 4 messages
                int(num_messages * self.config.protect_recent_reads_fraction),
            )
        else:
            read_protection_window = num_messages  # 0.0 = protect all (old behavior)

        # Adaptive compression ratio: scale with context pressure
        if model_limit > 0:
            context_pressure = min(1.0, tokens_before / model_limit)
        else:
            context_pressure = 0.5  # default: moderate

        # Linear interpolation between relaxed and aggressive thresholds
        # pressure 0.0 → relaxed, pressure 1.0 → aggressive
        min_ratio = (
            self.config.min_ratio_relaxed
            + (self.config.min_ratio_aggressive - self.config.min_ratio_relaxed) * context_pressure
        )
        # Clamp to [aggressive, relaxed] range
        min_ratio = max(
            self.config.min_ratio_aggressive,
            min(self.config.min_ratio_relaxed, min_ratio),
        )

        if context_pressure > 0.3:
            logger.debug(
                "content_router adaptive: pressure=%.2f, min_ratio=%.2f, "
                "read_protect_window=%d/%d msgs",
                context_pressure,
                min_ratio,
                read_protection_window,
                num_messages,
            )

        transformed_messages: list[dict[str, Any]] = []
        transforms_applied: list[str] = []
        warnings: list[str] = []
        compressor_timing: dict[str, float] = {}  # strategy → cumulative ms

        # Routing reason counters for summary logging
        route_counts: dict[str, int] = {
            "excluded_tool": 0,
            "user_msg": 0,
            "small": 0,
            "recent_code": 0,
            "analysis_ctx": 0,
            "ratio_too_high": 0,
            "non_string": 0,
            "content_blocks": 0,
        }
        compressed_details: list[str] = []  # e.g. ["code_aware:0.72", "kompress:0.65"]

        # Check for analysis intent in the most recent user message
        analysis_intent = False
        if self.config.protect_analysis_context:
            analysis_intent = self._detect_analysis_intent(messages)

        frozen_message_count = kwargs.get("frozen_message_count", 0)

        # ------------------------------------------------------------------
        # Two-pass parallel compression.
        #
        # Pass 1 (sequential): categorise every message — frozen, protected,
        #   cached, small, etc. are resolved immediately.  Cache-miss messages
        #   that need full compression are collected into *pending_tasks*.
        #
        # Pass 2 (parallel): all cache-miss compressions run concurrently in
        #   a thread pool.  Each self.compress() call is independent.
        #
        # Pass 3 (sequential): results are stitched back into message order,
        #   caches updated, and counters incremented.
        # ------------------------------------------------------------------

        # Pre-allocate result slots — None means "pending compression".
        result_slots: list[dict[str, Any] | None] = [None] * num_messages

        # Tasks: list of (slot_index, content, context, bias, content_key)
        _PendingTask = tuple[int, str, str, float, int]
        pending_tasks: list[_PendingTask] = []

        for i, message in enumerate(messages):
            # Skip frozen messages (in provider's prefix cache).
            # Modifying these would invalidate the cache, replacing a 90%
            # read discount with a 25% write penalty (Anthropic).
            if i < frozen_message_count:
                result_slots[i] = message
                continue

            role = message.get("role", "")
            content = message.get("content", "")
            bias = 1.0  # Default bias, may be overridden for tool messages

            messages_from_end = num_messages - i

            # Handle list content (Anthropic format with content blocks)
            if isinstance(content, list):
                transformed_message = self._process_content_blocks(
                    message,
                    content,
                    context,
                    transforms_applied,
                    excluded_tool_ids,
                    tool_name_map=tool_name_map,
                    route_counts=route_counts,
                    compressed_details=compressed_details,
                    min_ratio=min_ratio,
                    read_protection_window=read_protection_window,
                    messages_from_end=messages_from_end,
                    compressor_timing=compressor_timing,
                )
                result_slots[i] = transformed_message
                route_counts["content_blocks"] += 1
                continue

            # Skip non-string content (other types)
            if not isinstance(content, str):
                result_slots[i] = message
                route_counts["non_string"] += 1
                continue

            # Skip OpenAI-style tool messages for excluded tools
            # BUT: allow compression of old excluded-tool outputs beyond the
            # adaptive protection window (age-based decay).
            if role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                if tool_call_id in excluded_tool_ids:
                    if messages_from_end <= read_protection_window:
                        # Recent — protect as before
                        result_slots[i] = message
                        transforms_applied.append("router:excluded:tool")
                        route_counts["excluded_tool"] += 1
                        continue
                    # Old excluded-tool output — fall through to compression
                    # (the LLM is unlikely to need exact content from this far back,
                    # and CCR provides retrieval if it does)
                # Look up tool-specific compression bias for OpenAI tool messages
                tool_name = tool_name_map.get(tool_call_id, "")
                bias = self._get_tool_bias(tool_name) if tool_name else 1.0

            # Protection 1: Never compress user messages (unless overridden)
            if skip_user and role == "user":
                result_slots[i] = message
                transforms_applied.append("router:protected:user_message")
                route_counts["user_msg"] += 1
                continue

            # Protection 1b: Never compress system messages (when disabled)
            if skip_system and role == "system":
                result_slots[i] = message
                transforms_applied.append("router:protected:system_message")
                route_counts.setdefault("system_msg", 0)
                route_counts["system_msg"] += 1
                continue

            if not content or tokenizer.count_text(content) < min_tokens:
                # Skip small content
                result_slots[i] = message
                route_counts["small"] += 1
                continue

            # Detect content type for protection decisions
            detection = _detect_content(content)
            is_code = detection.content_type == ContentType.SOURCE_CODE

            # Protection 2: Don't compress recent CODE
            messages_from_end = num_messages - i
            if protect_recent > 0 and messages_from_end <= protect_recent and is_code:
                result_slots[i] = message
                transforms_applied.append("router:protected:recent_code")
                route_counts["recent_code"] += 1
                continue

            # Protection 3: Don't compress CODE when analysis intent detected
            if protect_analysis and analysis_intent and is_code:
                result_slots[i] = message
                transforms_applied.append("router:protected:analysis_context")
                route_counts["analysis_ctx"] += 1
                continue

            # Compression pinning: if this message was already compressed
            # (contains a CCR retrieval marker), skip recompression.
            # Recompressing would change byte content and break provider
            # prefix caching with no meaningful further reduction.
            if "Retrieve more: hash=" in content or "Retrieve original: hash=" in content:
                result_slots[i] = message
                route_counts.setdefault("already_compressed", 0)
                route_counts["already_compressed"] += 1
                continue

            # Route and compress based on content detection
            # Merge tool-specific bias with hook-provided bias (multiplicative)
            msg_bias = bias if role == "tool" else 1.0
            if i in hook_biases:
                msg_bias *= hook_biases[i]

            # Two-tier compression cache.
            # Tier 1 (skip): known won't-compress → instant skip.
            # Tier 2 (result): known compresses → reuse compressed text.
            content_key = hash(content)

            # Tier 1: skip set — instant rejection
            if self._cache.is_skipped(content_key):
                result_slots[i] = message
                route_counts["ratio_too_high"] += 1
                route_counts.setdefault("cache_hit", 0)
                route_counts["cache_hit"] += 1
                continue

            # Tier 2: result cache — reuse compressed output
            cached = self._cache.get(content_key)
            if cached is not None:
                cached_compressed, cached_ratio, cached_strategy = cached
                # Re-check ratio against current min_ratio (shifts with context pressure)
                if cached_ratio < min_ratio:
                    result_slots[i] = {**message, "content": cached_compressed}
                    transforms_applied.append(f"router:{cached_strategy}:{cached_ratio:.2f}")
                    compressed_details.append(f"{cached_strategy}:{cached_ratio:.2f}")
                else:
                    # Threshold tightened — no longer qualifies. Move to skip.
                    self._cache.move_to_skip(content_key)
                    result_slots[i] = message
                    route_counts["ratio_too_high"] += 1
                route_counts.setdefault("cache_hit", 0)
                route_counts["cache_hit"] += 1
                continue

            # Cache miss — defer to parallel compression pass
            route_counts.setdefault("cache_miss", 0)
            route_counts["cache_miss"] += 1
            pending_tasks.append((i, content, context, msg_bias, content_key))

        # --- Pass 2: Parallel compression of all cache-miss messages ---
        if pending_tasks:
            max_workers = min(
                len(pending_tasks), int(os.environ.get("HEADROOM_COMPRESS_WORKERS", "4"))
            )
            t_parallel_start = time.perf_counter()

            if max_workers <= 1 or len(pending_tasks) == 1:
                # Single task or parallelism disabled — compress inline
                task_results = []
                for _, task_content, task_ctx, task_bias, _ in pending_tasks:
                    t0 = time.perf_counter()
                    r = self.compress(task_content, context=task_ctx, bias=task_bias)
                    task_results.append((r, (time.perf_counter() - t0) * 1000))
            else:
                # Parallel compression via thread pool
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for _, task_content, task_ctx, task_bias, _ in pending_tasks:
                        futures.append(
                            executor.submit(self._timed_compress, task_content, task_ctx, task_bias)
                        )
                    task_results = [f.result() for f in futures]

            parallel_ms = (time.perf_counter() - t_parallel_start) * 1000
            compressor_timing["parallel_compress_total"] = parallel_ms

            # --- Pass 3: Merge results back (sequential, updates caches) ---
            for (slot_idx, _, _, _, content_key), (result, compress_ms) in zip(
                pending_tasks, task_results
            ):
                message = messages[slot_idx]
                strategy_key = f"compressor:{result.strategy_used.value}"
                compressor_timing[strategy_key] = (
                    compressor_timing.get(strategy_key, 0.0) + compress_ms
                )

                if result.compression_ratio < min_ratio:
                    # Compressed — store in result cache
                    self._cache.put(
                        content_key,
                        result.compressed,
                        result.compression_ratio,
                        result.strategy_used.value,
                    )
                    result_slots[slot_idx] = {**message, "content": result.compressed}
                    transforms_applied.append(
                        f"router:{result.strategy_used.value}:{result.compression_ratio:.2f}"
                    )
                    compressed_details.append(
                        f"{result.strategy_used.value}:{result.compression_ratio:.2f}"
                    )
                else:
                    # Didn't compress — add to skip set
                    self._cache.mark_skip(content_key)
                    result_slots[slot_idx] = message
                    route_counts["ratio_too_high"] += 1

        # Build final message list from slots
        transformed_messages = [m for m in result_slots if m is not None]

        tokens_after = sum(
            tokenizer.count_text(str(m.get("content", ""))) for m in transformed_messages
        )

        # Log routing summary
        parts = []
        if compressed_details:
            parts.append(f"{len(compressed_details)} compressed ({', '.join(compressed_details)})")
        if route_counts["excluded_tool"]:
            parts.append(f"{route_counts['excluded_tool']} excluded (Read/Glob)")
        if route_counts["user_msg"]:
            parts.append(f"{route_counts['user_msg']} skipped (user)")
        if route_counts["small"]:
            parts.append(f"{route_counts['small']} skipped (<50 words)")
        if route_counts["recent_code"]:
            parts.append(f"{route_counts['recent_code']} protected (recent code)")
        if route_counts["analysis_ctx"]:
            parts.append(f"{route_counts['analysis_ctx']} protected (analysis ctx)")
        if route_counts.get("already_compressed"):
            parts.append(f"{route_counts['already_compressed']} pinned (already compressed)")
        if route_counts["ratio_too_high"]:
            parts.append(f"{route_counts['ratio_too_high']} unchanged (ratio>={min_ratio:.2f})")
        if route_counts["content_blocks"]:
            parts.append(f"{route_counts['content_blocks']} content-block msgs")
        if route_counts["non_string"]:
            parts.append(f"{route_counts['non_string']} non-string")
        if route_counts.get("cache_hit"):
            parts.append(f"{route_counts['cache_hit']} cache hits")
        if route_counts.get("cache_miss"):
            parts.append(f"{route_counts['cache_miss']} cache misses")
        cs = self._cache.stats
        if cs["cache_size"] > 0 or cs["cache_skip_size"] > 0:
            parts.append(
                f"cache[{cs['cache_size']} results, {cs['cache_skip_size']} skips, "
                f"{cs['cache_avg_lookup_ns']:.0f}ns avg]"
            )
        if parts:
            logger.info(
                "content_router: %d msgs — %s",
                num_messages,
                ", ".join(parts),
            )

        all_transforms = lifecycle_transforms + transforms_applied
        return TransformResult(
            messages=transformed_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=all_transforms if all_transforms else ["router:noop"],
            markers_inserted=lifecycle_ccr_hashes,
            warnings=warnings,
            timing=compressor_timing,
        )

    def _get_tool_bias(self, tool_name: str) -> float:
        """Look up compression bias for a tool name.

        Checks user-configured profiles first, then DEFAULT_TOOL_PROFILES.
        Returns 1.0 (moderate) if no profile is configured.
        """
        from ..config import DEFAULT_TOOL_PROFILES

        # Check user-configured profiles
        if self.config.tool_profiles:
            profile = self.config.tool_profiles.get(tool_name)
            if profile:
                return float(profile.bias)

        # Check default profiles
        profile = DEFAULT_TOOL_PROFILES.get(tool_name)
        if profile:
            return profile.bias

        return 1.0  # Default: moderate

    def _process_content_blocks(
        self,
        message: dict[str, Any],
        content_blocks: list[Any],
        context: str,
        transforms_applied: list[str],
        excluded_tool_ids: set[str],
        tool_name_map: dict[str, str] | None = None,
        route_counts: dict[str, int] | None = None,
        compressed_details: list[str] | None = None,
        min_ratio: float = 0.85,
        read_protection_window: int = 8,
        messages_from_end: int = 0,
        compressor_timing: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Process content blocks (Anthropic format) for tool_result compression.

        Handles tool_result blocks by compressing their string content using
        the appropriate strategy (typically SmartCrusher for JSON arrays).

        Args:
            message: The original message.
            content_blocks: List of content blocks.
            context: Context for compression.
            transforms_applied: List to append transform names to.
            excluded_tool_ids: Tool IDs to skip compression for.
            tool_name_map: Mapping from tool_call_id to tool_name for profile lookup.
            route_counts: Optional routing reason counters to update.
            compressed_details: Optional list to append compression details to.
            min_ratio: Adaptive compression ratio threshold.
            read_protection_window: Messages from end within which excluded tools are protected.
            messages_from_end: How far this message is from the end of the conversation.

        Returns:
            Transformed message with compressed content blocks.
        """
        new_blocks = []
        any_compressed = False

        for block in content_blocks:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue

            block_type = block.get("type")

            # Handle tool_result blocks
            if block_type == "tool_result":
                # Check if tool is excluded from compression
                tool_use_id = block.get("tool_use_id", "")
                if tool_use_id in excluded_tool_ids:
                    if messages_from_end <= read_protection_window:
                        # Recent — protect as before
                        new_blocks.append(block)
                        transforms_applied.append("router:excluded:tool")
                        if route_counts is not None:
                            route_counts["excluded_tool"] += 1
                        continue
                    # Old excluded-tool output — fall through to compression

                # Look up tool-specific compression bias
                tool_name = (tool_name_map or {}).get(tool_use_id, "")
                bias = self._get_tool_bias(tool_name) if tool_name else 1.0

                tool_content = block.get("content", "")

                # Only process string content
                if isinstance(tool_content, str) and len(tool_content) > 500:
                    # Compression pinning: skip already-compressed content
                    if (
                        "Retrieve more: hash=" in tool_content
                        or "Retrieve original: hash=" in tool_content
                    ):
                        new_blocks.append(block)
                        if route_counts is not None:
                            route_counts.setdefault("already_compressed", 0)
                            route_counts["already_compressed"] += 1
                        continue

                    # Two-tier compression cache
                    content_key = hash(tool_content)

                    # Tier 1: skip set — instant rejection
                    if self._cache.is_skipped(content_key):
                        new_blocks.append(block)
                        if route_counts is not None:
                            route_counts["ratio_too_high"] += 1
                            route_counts.setdefault("cache_hit", 0)
                            route_counts["cache_hit"] += 1
                        continue

                    # Tier 2: result cache — reuse compressed output
                    cached = self._cache.get(content_key)
                    if cached is not None:
                        cached_compressed, cached_ratio, cached_strategy = cached
                        if cached_ratio < min_ratio:
                            new_blocks.append({**block, "content": cached_compressed})
                            transforms_applied.append(f"router:tool_result:{cached_strategy}")
                            if compressed_details is not None:
                                compressed_details.append(
                                    f"tool:{cached_strategy}:{cached_ratio:.2f}"
                                )
                            any_compressed = True
                        else:
                            # Threshold tightened — move to skip
                            self._cache.move_to_skip(content_key)
                            new_blocks.append(block)
                            if route_counts is not None:
                                route_counts["ratio_too_high"] += 1
                        if route_counts is not None:
                            route_counts.setdefault("cache_hit", 0)
                            route_counts["cache_hit"] += 1
                        continue

                    # Cache miss — run full compression
                    if route_counts is not None:
                        route_counts.setdefault("cache_miss", 0)
                        route_counts["cache_miss"] += 1
                    t0 = time.perf_counter()
                    result = self.compress(tool_content, context=context, bias=bias)
                    compress_ms = (time.perf_counter() - t0) * 1000
                    if compressor_timing is not None:
                        key = f"compressor:{result.strategy_used.value}"
                        compressor_timing[key] = compressor_timing.get(key, 0.0) + compress_ms
                    if result.compression_ratio < min_ratio:
                        # Compressed — store in result cache
                        self._cache.put(
                            content_key,
                            result.compressed,
                            result.compression_ratio,
                            result.strategy_used.value,
                        )
                        new_blocks.append({**block, "content": result.compressed})
                        transforms_applied.append(
                            f"router:tool_result:{result.strategy_used.value}"
                        )
                        if compressed_details is not None:
                            compressed_details.append(
                                f"tool:{result.strategy_used.value}:{result.compression_ratio:.2f}"
                            )
                        any_compressed = True
                        continue
                    else:
                        # Didn't compress — add to skip set
                        self._cache.mark_skip(content_key)
                        if route_counts is not None:
                            route_counts["ratio_too_high"] += 1
                else:
                    if route_counts is not None:
                        route_counts["small"] += 1

            # Keep block unchanged
            new_blocks.append(block)

        if any_compressed:
            return {**message, "content": new_blocks}
        return message

    def _detect_analysis_intent(self, messages: list[dict[str, Any]]) -> bool:
        """Detect if user wants to analyze/review code.

        Looks at the most recent user message for analysis keywords.

        Args:
            messages: Conversation messages.

        Returns:
            True if analysis intent detected.
        """
        # Analysis keywords that suggest user wants full code details
        analysis_keywords = {
            "analyze",
            "analyse",
            "review",
            "audit",
            "inspect",
            "security",
            "vulnerability",
            "bug",
            "issue",
            "problem",
            "explain",
            "understand",
            "how does",
            "what does",
            "debug",
            "fix",
            "error",
            "wrong",
            "broken",
            "refactor",
            "improve",
            "optimize",
            "clean up",
        }

        # Find most recent user message
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    content_lower = content.lower()
                    for keyword in analysis_keywords:
                        if keyword in content_lower:
                            return True
                break

        return False

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Check if routing should be applied.

        Always returns True - the router handles all content types.
        """
        return True


def route_and_compress(
    content: str,
    context: str = "",
) -> str:
    """Convenience function for one-off routing and compression.

    Args:
        content: Content to compress.
        context: Optional context for relevance-aware compression.

    Returns:
        Compressed content.

    Example:
        >>> compressed = route_and_compress(mixed_content)
    """
    router = ContentRouter()
    result = router.compress(content, context=context)
    return result.compressed

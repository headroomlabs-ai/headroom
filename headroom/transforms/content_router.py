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
import json
import logging
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import DEFAULT_EXCLUDE_TOOLS, ReadLifecycleConfig, TransformResult
from ..tokenizer import Tokenizer
from .base import Transform
from .content_detector import ContentType, DetectionResult
from .content_detector import detect_content_type as _regex_detect_content_type
from .error_detection import content_has_strong_error_indicators

logger = logging.getLogger(__name__)


_detect_backend_warned = False


def _router_debug_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _log_router_debug(event: str, **payload: Any) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    payload = {"event": event, **payload}
    logger.debug("event=%s %s", event, _router_debug_dumps(payload))


def _json_shape(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except Exception as exc:
        return {"is_json": False, "error": type(exc).__name__}
    if isinstance(parsed, dict):
        return {
            "is_json": True,
            "kind": "object",
            "keys": list(parsed.keys()),
            "length": len(parsed),
        }
    if isinstance(parsed, list):
        return {"is_json": True, "kind": "array", "length": len(parsed)}
    return {"is_json": True, "kind": type(parsed).__name__}


def _mixed_indicators(content: str) -> dict[str, bool]:
    return {
        "has_code_fences": bool(_CODE_FENCE_PATTERN.search(content)),
        "has_json_blocks": bool(_JSON_BLOCK_START.search(content)),
        "has_prose": len(_PROSE_PATTERN.findall(content)) > 5,
        "has_search_results": bool(_SEARCH_RESULT_PATTERN.search(content)),
    }


def _section_debug(section: ContentSection, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "content_type": section.content_type.value,
        "language": getattr(section, "language", None),
        "start_line": getattr(section, "start_line", None),
        "end_line": getattr(section, "end_line", None),
        "is_code_fence": getattr(section, "is_code_fence", False),
        "chars": len(section.content),
        "bytes": len(section.content.encode("utf-8", errors="replace")),
        "tokens_estimate": len(section.content.split()),
        "json_shape": _json_shape(section.content),
        "content": section.content,
    }


def _resolve_detect_backend() -> str:
    """Pick the content-detection backend: ``"rust"`` or ``"python"``."""
    backend = os.environ.get("HEADROOM_DETECT_BACKEND", "").strip().lower()
    if backend in ("python", "rust"):
        return backend
    return "python" if sys.platform == "win32" else "rust"


def _detect_content(content: str) -> DetectionResult:
    """Detect content type via the native chain, with a safe Windows default.

    Stage-3d (PR5) wired this through `headroom._core.detect_content_type`,
    which runs the magika→unidiff→PlainText chain. On Windows, native Magika
    initialization can leave an ONNX Runtime thread alive after timeout, so the
    default backend there is the pure-Python regex detector.

    Set `HEADROOM_DETECT_BACKEND=rust` or `python` to force a backend.

    The Rust binding returns the legacy `DetectionResult` shape with
    `confidence=1.0` and an empty metadata dict. Existing callers
    only consumed `.content_type` from it; the strategy mapping in
    `_strategy_from_detection` keys off that field alone.
    """
    global _detect_backend_warned

    backend = _resolve_detect_backend()
    if backend == "python":
        if not _detect_backend_warned:
            _detect_backend_warned = True
            logger.warning(
                "Content detection using pure-Python backend "
                "(native Magika/ONNX detector is unsafe by default on Windows; "
                "override with HEADROOM_DETECT_BACKEND=rust)."
            )
        return _regex_detect_content_type(content)

    from headroom._core import detect_content_type as _rust_detect

    rust_result = _rust_detect(content)
    # Rust's `content_type` is the lowercase string tag (e.g.
    # "json_array"); translate to the Python `ContentType` enum so
    # downstream mapping keys match.
    content_type = ContentType(rust_result.content_type)
    if content_type is ContentType.PLAIN_TEXT:
        regex_result = _regex_detect_content_type(content)
        if regex_result.content_type is not ContentType.PLAIN_TEXT:
            return regex_result
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


def _gain_bucket(gain: float) -> str:
    """Quantize a net-cost gain into a coarse magnitude band for markers.

    The net-cost gate emits a ``netcost:skip:<band>`` transform marker. Using
    the raw rounded gain would make every distinct value a unique marker and
    blow up the cardinality of any ``transforms_applied`` aggregation. Bands
    keep the signal (rough magnitude + sign) while bounding cardinality to a
    handful of values. The exact gain is still logged at INFO for debugging.
    """
    if not math.isfinite(gain):
        return "nan"
    mag = abs(gain)
    if mag < 100:
        band = "lt100"
    elif mag < 1000:
        band = "lt1k"
    elif mag < 10000:
        band = "lt10k"
    else:
        band = "gte10k"
    if gain == 0:
        return "0"
    return ("neg_" if gain < 0 else "") + band


def _netcost_message_tokens(message: dict[str, Any], tokenizer: Tokenizer) -> int:
    """Token count of a message for net-cost suffix (S) estimation.

    String content is counted directly. Anthropic block-list content is
    counted by summing the text-bearing fields (``text`` blocks and
    ``tool_result`` content) rather than stringifying the whole list, which
    would count Python ``repr`` punctuation and type names and badly
    miscount S — the value that drives the break-even gate decision.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return tokenizer.count_text(content)
    if not isinstance(content, list):
        return tokenizer.count_text(str(content))
    total = 0
    for block in content:
        if not isinstance(block, dict):
            total += tokenizer.count_text(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            total += tokenizer.count_text(str(block.get("text", "")))
        elif block_type == "tool_result":
            tc = block.get("content", "")
            if isinstance(tc, str):
                total += tokenizer.count_text(tc)
            elif isinstance(tc, list):
                for sub in tc:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        total += tokenizer.count_text(str(sub.get("text", "")))
                    else:
                        total += tokenizer.count_text(str(sub))
            else:
                total += tokenizer.count_text(str(tc))
        else:
            # Other blocks (image, tool_use input, …) — repr is a rough proxy
            # but bounded; these rarely dominate a suffix.
            total += tokenizer.count_text(str(block))
    return total


class CompressionCache:
    """Two-tier compression cache with TTL.  Thread-safe.

    Tier 1 (skip set): content hashes that won't compress — instant skip,
    near-zero memory (just ints in a set).

    Tier 2 (result cache): compressed results for content that DID compress —
    reuse the compressed text on subsequent requests.

    Entries expire after TTL (default 30min). No max-entries cap — TTL is the
    natural bound. Memory grows proportional to compressible content × TTL,
    which is bounded by session duration.

    Uses in-process dict for ultra-fast lookups (~100ns). Could be backed
    by memcached/Redis for multi-process deployments.

    Thread safety: a ``threading.Lock`` guards all read-modify-write
    operations.  The ``apply()`` path runs compression inside a
    ``ThreadPoolExecutor``; without the lock concurrent cache misses for
    the same content would produce duplicate compression work (correct but
    wasteful) and metrics counters would drift.
    """

    def __init__(self, ttl_seconds: int = 1800):
        import threading

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
        self._lock = threading.Lock()

    def get(self, key: int) -> tuple[str, float, str] | None:
        """Get cached compression result.  Thread-safe.

        Returns (compressed_text, ratio, strategy) or None if not found/expired.
        Use is_skipped() first to check if content is known non-compressible.
        """
        t0 = time.perf_counter_ns()
        with self._lock:
            entry = self._results.get(key)
            if entry is not None:
                compressed, ratio, strategy, created_at = entry
                if (time.monotonic() - created_at) < self._ttl_seconds:
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
        """Check if content is known non-compressible (Tier 1).  Thread-safe."""
        with self._lock:
            ts = self._skip.get(key)
            if ts is not None:
                if (time.monotonic() - ts) < self._ttl_seconds:
                    self._skip_hits += 1
                    return True
                else:
                    del self._skip[key]
                    self._evictions += 1
            return False

    def put(self, key: int, compressed: str, ratio: float, strategy: str) -> None:
        """Store a compressed result (Tier 2).  Thread-safe."""
        with self._lock:
            self._results[key] = (compressed, ratio, strategy, time.monotonic())

    def mark_skip(self, key: int) -> None:
        """Mark content as non-compressible (Tier 1).  Thread-safe."""
        with self._lock:
            self._skip[key] = time.monotonic()

    def move_to_skip(self, key: int) -> None:
        """Move a result to skip set (threshold tightened, no longer qualifies).
        Thread-safe."""
        with self._lock:
            self._results.pop(key, None)
            self._skip[key] = time.monotonic()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._results)

    @property
    def skip_size(self) -> int:
        with self._lock:
            return len(self._skip)

    @property
    def stats(self) -> dict[str, int | float]:
        with self._lock:
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
        """Clear all entries (e.g., on session end).  Thread-safe."""
        with self._lock:
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
        strategy_chain: Every strategy attempted in order. For a direct
            hit it's a single entry; for the SMART_CRUSHER → KOMPRESS →
            LOG fallback chain it's three. Lets log readers see *how*
            we got to the final compressor without parsing the
            decision_reason string.
        cache_hit: True when this result came from the router's
            result_cache (no fresh compression ran). Currently the
            single-content compress() path doesn't populate the cache,
            so this is False in practice — placeholder for the
            cache-wire-up follow-up.
    """

    compressed: str
    original: str
    strategy_used: CompressionStrategy
    routing_log: list[RoutingDecision] = field(default_factory=list)
    sections_processed: int = 1
    strategy_chain: list[str] = field(default_factory=list)
    cache_hit: bool = False

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

    # Protection: failed tool calls / error outputs stay verbatim (issue #847).
    # The model needs exact tracebacks and error text to recover; compressing
    # them measurably hurts agent recovery. Outputs above the size cap still
    # compress — LogCompressor preserves error lines in big logs, so the two
    # features stay complementary.
    protect_error_outputs: bool = True
    error_protection_max_chars: int = 8000  # ~2K tokens; larger errors compress

    # Cache safety: assistant text-block compression.
    # Default OFF. Assistant content is echoed back by the client in
    # subsequent turns and becomes part of the upstream provider's
    # prefix cache (Anthropic cache_control, DeepSeek/OpenAI auto).
    # Compressing it changes the bytes that must match for a cache
    # hit on the next turn. The hash-keyed result cache makes the
    # compressed output deterministic *within* a process, but cache
    # eviction or proxy restart can re-compress with a different
    # output for stochastic compressors — and that miss costs the
    # whole prefix discount. Enable only for deployments routed to
    # backends that don't honor cache_control AND whose compressors
    # are byte-deterministic.
    compress_assistant_text_blocks: bool = False

    # Minimum content length (in chars) at which a text or tool_result
    # block is considered for compression. Below this, the overhead of
    # routing/detecting/caching exceeds any savings, so the block is
    # passed through verbatim.
    min_chars_for_block_compression: int = 500

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
    smart_crusher_max_items_after_crush: int | None = None
    smart_crusher_with_compaction: bool = True

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

    # SmartCrusher configuration override. None → transforms-level
    # SmartCrusherConfig() defaults. Lets deployments tune the lossless
    # dispatch threshold and compaction heuristics without constructing
    # the crusher themselves.
    smart_crusher: Any | None = None

    # Group search-compressor output by file (`rg --heading` style).
    # Default False; the proxy enables it in token mode.
    search_group_by_file: bool = False


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
    in_string = False
    escaped = False

    for i in range(start, len(lines)):
        line = lines[i]
        json_lines.append(line)

        # Count brackets/braces, but ignore any that appear inside a JSON
        # string literal — a naive line.count() treats e.g. the "]" in
        # {"path": "a]b"} as a closing bracket and terminates the block
        # early, splitting one array across multiple sections.
        for ch in line:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                if in_string:
                    escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                bracket_count += 1
            elif ch == "]":
                bracket_count -= 1
            elif ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1

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

        # F2.2: per-request CompressionPolicy, set from
        # ``kwargs["compression_policy"]`` at the start of ``apply()``
        # and read by ``_record_to_toin`` to gate TOIN writes when
        # ``policy.toin_read_only`` is true (Subscription mode).
        # Defaults to ``None`` so direct ``compress()`` callers (e.g.
        # tests, hand-written pipelines that don't go through the
        # proxy) keep pre-F2.2 behaviour: TOIN writes are not gated.
        # Same pattern the existing ``_runtime_target_ratio`` /
        # ``_runtime_kompress_model`` fields below use.
        self._runtime_compression_policy: Any = None

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

        # F2.2 gate: when the active CompressionPolicy says
        # ``toin_read_only=True`` (Subscription auth mode), don't
        # mutate the TOIN learning pool from this request. Direct
        # ``compress()`` callers don't go through ``apply()`` and
        # have ``self._runtime_compression_policy is None`` — those
        # keep their pre-F2.2 write-enabled behaviour.
        policy = self._runtime_compression_policy
        if policy is not None and policy.toin_read_only:
            logger.debug(
                "ContentRouter: skipping TOIN record_compression for %s "
                "— policy.toin_read_only=True (auth_mode resolved as "
                "Subscription, F2.2 gate)",
                strategy.value,
            )
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
        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        request_debug = (
            {
                "chars": len(content),
                "bytes": len(content.encode("utf-8", errors="replace")),
                "tokens_estimate": len(content.split()),
                "json_shape": _json_shape(content),
                "mixed_indicators": _mixed_indicators(content),
                "context_chars": len(context),
                "question": question,
                "bias": bias,
                "content": content,
                "context": context,
            }
            if debug_enabled
            else {}
        )
        if not content or not content.strip():
            if debug_enabled:
                _log_router_debug(
                    "content_router_input",
                    **request_debug,
                    selected_strategy=CompressionStrategy.PASSTHROUGH.value,
                    selection_reason="empty_or_whitespace",
                )
            result = RouterCompressionResult(
                compressed=content,
                original=content,
                strategy_used=CompressionStrategy.PASSTHROUGH,
                routing_log=[],
            )
        else:
            # Determine strategy from content analysis
            mixed = is_mixed_content(content)
            detection = _detect_content(content)
            force_kompress = bool(getattr(self, "_runtime_force_kompress", False))
            strategy = (
                CompressionStrategy.KOMPRESS
                if force_kompress
                else self._determine_strategy(content)
            )
            if debug_enabled:
                _log_router_debug(
                    "content_router_input",
                    **request_debug,
                    detected_content_type=detection.content_type.value,
                    detection_confidence=detection.confidence,
                    selected_strategy=strategy.value,
                    selection_reason=(
                        "runtime_force_kompress"
                        if force_kompress
                        else "mixed_content"
                        if mixed
                        else "content_detection"
                    ),
                )

            if strategy == CompressionStrategy.MIXED:
                result = self._compress_mixed(content, context, question, bias=bias)
            else:
                result = self._compress_pure(content, strategy, context, question, bias=bias)

        # Empty-output guard: compression must NEVER blank out non-empty input.
        # An empty user-message content makes Anthropic reject the whole request
        # with 400 ("messages.N: user messages must have non-empty content").
        # If any transform yields empty/whitespace from non-empty input, fall
        # back to the original content (passthrough) instead of emitting empty.
        if (
            content
            and content.strip()
            and (result.compressed is None or not str(result.compressed).strip())
        ):
            logger.warning(
                "content_router: compression produced EMPTY output from non-empty "
                "input (%d chars, strategy=%s); falling back to original to avoid 400.",
                len(content),
                getattr(result.strategy_used, "value", result.strategy_used),
            )
            result.compressed = content

        # One observer call per routing decision; the observer is the
        # forcing function for catching strategy-level regressions.
        # Empty routing_log (passthrough fast path) → no calls.
        self._observe(result)
        if debug_enabled:
            _log_router_debug(
                "content_router_output",
                selected_strategy=result.strategy_used.value,
                sections_processed=result.sections_processed,
                total_original_tokens=result.total_original_tokens,
                total_compressed_tokens=result.total_compressed_tokens,
                tokens_saved=result.tokens_saved,
                savings_percentage=result.savings_percentage,
                compression_ratio=result.compression_ratio,
                routing_log=[
                    {
                        "content_type": decision.content_type.value,
                        "strategy": decision.strategy.value,
                        "original_tokens": decision.original_tokens,
                        "compressed_tokens": decision.compressed_tokens,
                        "confidence": decision.confidence,
                        "section_index": decision.section_index,
                        "compression_ratio": decision.compression_ratio,
                    }
                    for decision in result.routing_log
                ],
                original=result.original,
                compressed=result.compressed,
            )
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
        if logger.isEnabledFor(logging.DEBUG):
            _log_router_debug(
                "content_router_mixed_sections",
                section_count=len(sections),
                sections=[_section_debug(section, idx) for idx, section in enumerate(sections)],
                content=content,
            )

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
            compressed_content, compressed_tokens, _section_chain = self._apply_strategy_to_content(
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

        compressed, compressed_tokens, strategy_chain = self._apply_strategy_to_content(
            content, strategy, context, question=question, bias=bias
        )

        return RouterCompressionResult(
            compressed=compressed,
            original=content,
            strategy_used=strategy,
            strategy_chain=strategy_chain,
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
    ) -> tuple[str, int, list[str]]:
        """Apply a compression strategy to content.

        Args:
            content: Content to compress.
            strategy: Strategy to use.
            context: User context.
            language: Language hint for code.
            question: Optional question for QA-aware compression.
            bias: Compression bias multiplier (>1 = keep more, <1 = keep fewer).

        Returns:
            Tuple of (compressed_content, compressed_token_count,
            strategy_chain). The chain lists every strategy attempted
            in order — first the requested one, then any fallbacks.
            Single-entry chain means a direct hit; multi-entry means
            the fallback chain fired (e.g. ``[smart_crusher, kompress,
            log]``). Log readers use this to see *how* we got to the
            final compressor without parsing decision_reason strings.
        """
        # Track original tokens for TOIN recording
        original_tokens = len(content.split())
        compressed: str | None = None
        compressed_tokens: int | None = None
        requested_strategy = strategy
        actual_strategy = strategy
        compressor_name = strategy.value
        decision_reason = "strategy_not_enabled_or_unavailable"
        strategy_chain: list[str] = [strategy.value]
        error: str | None = None

        try:
            if strategy == CompressionStrategy.CODE_AWARE:
                if self.config.enable_code_aware:
                    compressor = self._get_code_compressor()
                    if compressor:
                        compressor_name = type(compressor).__name__
                        result = compressor.compress(content, language=language, context=context)
                        compressed, compressed_tokens = result.compressed, result.compressed_tokens
                        decision_reason = "code_aware"
                if compressed is None:
                    # Fallback to Kompress
                    compressed, compressed_tokens = self._try_ml_compressor(
                        content, context, question
                    )
                    strategy = CompressionStrategy.KOMPRESS  # Update for TOIN
                    actual_strategy = strategy
                    compressor_name = "KompressCompressor"
                    decision_reason = "code_aware_unavailable_fallback_kompress"
                    strategy_chain.append(CompressionStrategy.KOMPRESS.value)

            elif strategy == CompressionStrategy.SMART_CRUSHER:
                # SmartCrusher handles its own TOIN recording
                if self.config.enable_smart_crusher:
                    crusher = self._get_smart_crusher()
                    if crusher:
                        compressor_name = type(crusher).__name__
                        result = crusher.crush(content, query=context, bias=bias)
                        compressed, compressed_tokens = (
                            result.compressed,
                            len(result.compressed.split()),
                        )
                        decision_reason = "smart_crusher"
                        # Fallback to Kompress (and possibly Log) is
                        # handled by the unified post-strategy block below
                        # — no inline fallback here to avoid duplicate
                        # Kompress invocations.

            elif strategy == CompressionStrategy.SEARCH:
                if self.config.enable_search_compressor:
                    compressor = self._get_search_compressor()
                    if compressor:
                        compressor_name = type(compressor).__name__
                        result = compressor.compress(content, context=context, bias=bias)
                        compressed, compressed_tokens = (
                            result.compressed,
                            len(result.compressed.split()),
                        )
                        decision_reason = "search_compressor"

            elif strategy == CompressionStrategy.LOG:
                if self.config.enable_log_compressor:
                    compressor = self._get_log_compressor()
                    if compressor:
                        compressor_name = type(compressor).__name__
                        result = compressor.compress(content, bias=bias)
                        # Use the same word-count metric the rest of the
                        # router uses; `compressed_line_count` is in
                        # lines, not tokens — recording it here made
                        # ratios meaningless against `original_tokens`.
                        compressed, compressed_tokens = (
                            result.compressed,
                            len(result.compressed.split()),
                        )
                        decision_reason = "log_compressor"

            elif strategy == CompressionStrategy.DIFF:
                compressor = self._get_diff_compressor()
                if compressor:
                    compressor_name = type(compressor).__name__
                    result = compressor.compress(content, context=context)
                    compressed, compressed_tokens = (
                        result.compressed,
                        len(result.compressed.split()),
                    )
                    decision_reason = "diff_compressor"

            elif strategy == CompressionStrategy.HTML:
                if self.config.enable_html_extractor:
                    extractor = self._get_html_extractor()
                    if extractor:
                        compressor_name = type(extractor).__name__
                        result = extractor.extract(content)
                        compressed = result.extracted
                        # Estimate tokens from extracted text (simple word count)
                        compressed_tokens = len(compressed.split()) if compressed else 0
                        decision_reason = "html_extractor"

            elif strategy == CompressionStrategy.KOMPRESS:
                compressed, compressed_tokens = self._try_ml_compressor(content, context, question)
                compressor_name = "KompressCompressor"
                decision_reason = "kompress"

            elif strategy == CompressionStrategy.TEXT:
                # Prefer Kompress ML compressor for text
                # Passes through unchanged if Kompress not available
                compressed, compressed_tokens = self._try_ml_compressor(content, context, question)
                compressor_name = "KompressCompressor"
                decision_reason = "text_uses_kompress"

            elif strategy == CompressionStrategy.PASSTHROUGH:
                compressed = content
                compressed_tokens = original_tokens
                compressor_name = "Passthrough"
                decision_reason = "explicit_passthrough"

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            decision_reason = "compression_exception"
            logger.warning("Compression with %s failed: %s", strategy.value, e)

        # If compression succeeded, record to TOIN
        if compressed is not None and compressed_tokens is not None:
            fallback_eligible_strategy = strategy in {
                CompressionStrategy.SMART_CRUSHER,
                CompressionStrategy.CODE_AWARE,
            }
            fallback_no_savings = compressed == content or compressed_tokens >= original_tokens
            if fallback_eligible_strategy and fallback_no_savings:
                # Skip if Kompress was already tried by an inline fallback
                # (e.g. CODE_AWARE's code-compressor-unavailable path at
                # line 1249).  Prevents a duplicate strategy_chain entry
                # and a wasted second _try_ml_compressor call.
                already_tried_kompress = CompressionStrategy.KOMPRESS.value in strategy_chain
                if not already_tried_kompress:
                    strategy_chain.append(CompressionStrategy.KOMPRESS.value)
                    fallback_compressed, fallback_tokens = self._try_ml_compressor(
                        content, context, question
                    )
                else:
                    fallback_compressed = compressed
                    fallback_tokens = compressed_tokens
                if fallback_tokens < compressed_tokens:
                    compressed = fallback_compressed
                    compressed_tokens = fallback_tokens
                    actual_strategy = CompressionStrategy.KOMPRESS
                    compressor_name = "KompressCompressor"
                    decision_reason = f"{decision_reason}_fallback_kompress_after_no_savings"
                else:
                    # Last-ditch: line-structured compressors (the proxy's
                    # own log dumps land here — repetitive JSONL that
                    # Kompress can't shrink but the log compressor can).
                    # Only attempted when the strategy was SMART_CRUSHER so
                    # we don't reroute genuine code/diff content.
                    if (
                        strategy == CompressionStrategy.SMART_CRUSHER
                        and self.config.enable_log_compressor
                    ):
                        log_compressor = self._get_log_compressor()
                        if log_compressor is not None:
                            strategy_chain.append(CompressionStrategy.LOG.value)
                            try:
                                log_result = log_compressor.compress(content, bias=bias)
                            except Exception as exc:  # noqa: BLE001
                                logger.debug("Log fallback failed for SMART_CRUSHER: %s", exc)
                            else:
                                log_compressed_tokens = len(log_result.compressed.split())
                                if log_compressed_tokens < compressed_tokens:
                                    compressed = log_result.compressed
                                    compressed_tokens = log_compressed_tokens
                                    actual_strategy = CompressionStrategy.LOG
                                    compressor_name = type(log_compressor).__name__
                                    decision_reason = (
                                        f"{decision_reason}_fallback_log_after_no_savings"
                                    )

            # Re-narrow for mypy: all reassignments above produce str, but
            # mypy 1.14.x widens after nested try/except/else reassignments.
            assert compressed is not None
            if logger.isEnabledFor(logging.DEBUG):
                _log_router_debug(
                    "content_router_strategy_result",
                    requested_strategy=requested_strategy.value,
                    actual_strategy=actual_strategy.value,
                    strategy_chain=strategy_chain,
                    compressor=compressor_name,
                    reason=decision_reason,
                    language=language,
                    question=question,
                    bias=bias,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    tokens_saved=max(0, original_tokens - compressed_tokens),
                    compression_ratio=compressed_tokens / original_tokens
                    if original_tokens
                    else 1.0,
                    json_shape=_json_shape(content),
                    input=content,
                    output=compressed,
                    error=error,
                )
            self._record_to_toin(
                strategy=strategy,
                content=content,
                compressed=compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                language=language,
                context=context,
            )
            return compressed, compressed_tokens, strategy_chain

        # Fallback: return unchanged
        strategy_chain.append(CompressionStrategy.PASSTHROUGH.value)
        if logger.isEnabledFor(logging.DEBUG):
            _log_router_debug(
                "content_router_strategy_result",
                requested_strategy=requested_strategy.value,
                actual_strategy=CompressionStrategy.PASSTHROUGH.value,
                strategy_chain=strategy_chain,
                compressor=None,
                reason=decision_reason,
                language=language,
                question=question,
                bias=bias,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                tokens_saved=0,
                compression_ratio=1.0,
                json_shape=_json_shape(content),
                input=content,
                output=content,
                error=error,
            )
        return content, original_tokens, strategy_chain

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

        # Primary: Kompress — downloads from chopratejas/kompress-v2-base on first use
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
                from .smart_crusher import SmartCrusher, SmartCrusherConfig

                # Pass CCR config for marker injection
                ccr_config = CCRConfig(
                    enabled=self.config.ccr_enabled,
                    inject_retrieval_marker=self.config.ccr_inject_marker,
                )
                # Full config override (smart_crusher) wins as the base;
                # the per-field knobs from savings profiles still apply on top.
                crusher_config = self.config.smart_crusher or SmartCrusherConfig()
                if self.config.smart_crusher_max_items_after_crush is not None:
                    crusher_config.max_items_after_crush = (
                        self.config.smart_crusher_max_items_after_crush
                    )
                self._smart_crusher = SmartCrusher(
                    config=crusher_config,
                    ccr_config=ccr_config,
                    with_compaction=self.config.smart_crusher_with_compaction,
                )
            except ImportError:
                logger.debug("SmartCrusher not available")
        return self._smart_crusher

    def _get_search_compressor(self) -> Any:
        """Get SearchCompressor (lazy load)."""
        if self._search_compressor is None:
            try:
                from .search_compressor import SearchCompressor, SearchCompressorConfig

                self._search_compressor = SearchCompressor(
                    SearchCompressorConfig(group_by_file=self.config.search_group_by_file)
                )
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

        # 1. ML text compressor: Kompress.
        #
        # Eager preload is cache-only (allow_download=False): on a cold cache we
        # must NOT trigger a network download here, because this runs on the
        # blocking startup/lifespan path before the proxy binds its port. A slow
        # download stalls the bind, and a hard crash in the native download/ML
        # stack (uncatchable SIGABRT) kills the interpreter before it ever
        # listens — the proxy then "never opens its port" and the supervisor
        # gives up. When the model isn't cached we defer to first use instead.
        if self.config.enable_kompress:
            from .kompress_compressor import KompressModelNotCached

            compressor = self._get_kompress()
            if compressor:
                if not hasattr(compressor, "preload"):
                    status["kompress"] = "enabled"
                    status["kompress_backend"] = "unknown"
                else:
                    try:
                        backend = compressor.preload(allow_download=False)
                    except KompressModelNotCached:
                        logger.info(
                            "Kompress model not cached; deferring download to "
                            "first use to keep startup non-blocking"
                        )
                        status["kompress"] = "deferred"
                    else:
                        logger.info("Kompress model pre-loaded at startup backend=%s", backend)
                        status["kompress"] = "enabled"
                        status["kompress_backend"] = str(backend)
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
        - None: use default (chopratejas/kompress-v2-base) — cached on self
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

    def _net_cost_allows(
        self,
        *,
        slot_idx: int,
        original_tokens: int,
        compressed_tokens: int,
        suffix_tokens: list[int],
        route_counts: dict[str, int],
        transforms_applied: list[str],
        batch_state: dict[str, int | None] | None = None,
    ) -> bool:
        """Break-even gate for one candidate mutation (#856 P2, flag-gated).

        Consumes ``CompressionPolicy.net_mutation_gain`` with the issue's v1
        estimators: ΔT is the candidate's exact token saving (the compressed
        form is already computed when this runs), S is the token total after
        the slot, and R / P_alive are env-tunable constants
        (``HEADROOM_NET_COST_EXPECTED_READS``, default 10;
        ``HEADROOM_NET_COST_P_ALIVE``, default 1.0 — the conservative
        full-penalty assumption). Every decision is logged with its inputs
        and counted in ``route_counts`` so the flag can be validated from
        telemetry before any default-on.

        #856 P3a (batch deep edits): a mutation at depth K busts the
        provider's cached suffix after K, so every *later* candidate at a
        deeper slot rides that same invalidation for free — mutating it adds
        no incremental cache-bust cost. ``batch_state["floor"]`` tracks the
        shallowest slot already admitted as a net-positive mutation. When the
        current candidate sits strictly deeper than that floor, S is charged
        as 0 (rather than the full invalidated suffix), so the break-even
        formula admits it on the write/read economics alone. Charging S=0 via
        the same ``net_mutation_gain`` (instead of blanket-admitting on
        ``delta_t > 0``) keeps the decision conservative: it never admits a
        mutation the real economics would reject. The floor is only set/lowered
        by full-S admits, so a slot only ever rides free behind a genuinely
        mutated shallower slot. Each batch admission emits the
        ``router:netcost_batch_admit`` marker and the ``netcost_batch_admitted``
        counter for telemetry.
        """
        delta_t = max(0, original_tokens - compressed_tokens)
        # Batch reclaim: if a shallower slot was already admitted, its
        # cache-bust already invalidated everything after it, including this
        # slot — so charge S=0 here. Otherwise S is the full suffix after the
        # candidate (P2 v1 estimator).
        floor = batch_state.get("floor") if batch_state is not None else None
        batch_reclaim = floor is not None and slot_idx > floor
        suffix = 0 if batch_reclaim else suffix_tokens[slot_idx + 1]
        policy = self._runtime_compression_policy
        if policy is None:
            from .compression_policy import policy_default_payg

            policy = policy_default_payg()
        # Malformed env values fall back to defaults with a warning rather
        # than crashing the request path (same posture as the #851 breaker
        # env guard).
        # ``float()`` parses "nan"/"inf" without raising, so a non-finite
        # check is needed in addition to the ValueError guard — otherwise a
        # malformed-but-parseable value would be logged verbatim (misleading
        # telemetry) even though ``net_mutation_gain`` clamps it internally.
        reads, p_alive = 10.0, 1.0
        try:
            _reads = float(os.environ.get("HEADROOM_NET_COST_EXPECTED_READS", "") or 10.0)
            if not math.isfinite(_reads):
                raise ValueError("non-finite")
            reads = _reads
        except ValueError:
            logger.warning("HEADROOM_NET_COST_EXPECTED_READS malformed; using 10")
        try:
            _p_alive = float(os.environ.get("HEADROOM_NET_COST_P_ALIVE", "") or 1.0)
            if not math.isfinite(_p_alive):
                raise ValueError("non-finite")
            p_alive = _p_alive
        except ValueError:
            logger.warning("HEADROOM_NET_COST_P_ALIVE malformed; using 1.0")
        gain = float(policy.net_mutation_gain(delta_t, suffix, reads, p_alive))
        allowed = gain > 0.0
        logger.info(
            "NetCostPolicy slot=%d delta_t=%d suffix=%d reads=%.1f p_alive=%.2f "
            "gain=%.0f batch_reclaim=%s -> %s",
            slot_idx,
            delta_t,
            suffix,
            reads,
            p_alive,
            gain,
            batch_reclaim,
            "mutate" if allowed else "skip",
        )
        if allowed:
            route_counts.setdefault("netcost_allowed", 0)
            route_counts["netcost_allowed"] += 1
            if batch_reclaim:
                # Rode a shallower edit's cache-bust for free — telemetry only;
                # the floor is unchanged (this slot is deeper than the floor).
                route_counts.setdefault("netcost_batch_admitted", 0)
                route_counts["netcost_batch_admitted"] += 1
                transforms_applied.append("router:netcost_batch_admit")
            elif batch_state is not None:
                # First/shallower full-S admit — open (or lower) the batch
                # floor so deeper candidates can reclaim against it.
                current = batch_state.get("floor")
                batch_state["floor"] = slot_idx if current is None else min(current, slot_idx)
        else:
            route_counts.setdefault("netcost_skipped", 0)
            route_counts["netcost_skipped"] += 1
            # Bucket the gain into a coarse magnitude band rather than emitting
            # the raw value: a distinct numeric gain per skip would explode the
            # cardinality of any ``transforms_applied`` aggregation. The exact
            # value is still in the INFO log above for debugging.
            transforms_applied.append(f"netcost:skip:{_gain_bucket(gain)}")
        return allowed

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
        skip_system = kwargs.get("compress_system_messages") is not True
        protect_recent = kwargs.get("protect_recent", self.config.protect_recent_code)
        protect_analysis = kwargs.get(
            "protect_analysis_context", self.config.protect_analysis_context
        )
        min_tokens = kwargs.get("min_tokens_to_compress", 50)
        # Cache-safety knobs for content-block (Anthropic-format) handling:
        compress_assistant_text_blocks = kwargs.get(
            "compress_assistant_text_blocks",
            self.config.compress_assistant_text_blocks,
        )
        min_chars_for_block_compression = kwargs.get(
            "min_chars_for_block_compression",
            self.config.min_chars_for_block_compression,
        )
        # Store runtime options on self for access by _route_and_compress_block
        self._runtime_target_ratio: float | None = kwargs.get("target_ratio")
        self._runtime_force_kompress: bool = bool(kwargs.get("force_kompress", False))
        self._runtime_kompress_model: str | None = kwargs.get("kompress_model")
        # F2.2: capture the per-request CompressionPolicy so
        # ``_record_to_toin`` can gate TOIN writes on
        # ``policy.toin_read_only``. ``None`` when the caller didn't
        # pass a policy — ``_record_to_toin`` treats that as "no gate"
        # to preserve pre-F2.2 behaviour for non-proxy callers.
        self._runtime_compression_policy = kwargs.get("compression_policy")

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
        runtime_read_protection_window = kwargs.get("read_protection_window")
        if runtime_read_protection_window is not None:
            read_protection_window = max(0, int(runtime_read_protection_window))

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

        # #856 P2 (flag-gated, default off): net-cost mutation gate. Suffix
        # token sums are precomputed once (reverse cumulative) so each
        # candidate's S lookup is O(1). v1 estimator per the issue: S is the
        # token total of every message after the candidate.
        netcost_enabled = os.environ.get("HEADROOM_NET_COST_POLICY") == "1"
        netcost_suffix_tokens: list[int] = []
        # #856 P3a: shared batch-reclaim state for this request. ``floor`` is
        # the shallowest slot admitted as a net-positive mutation; once set,
        # deeper candidates charge S=0 (their cache-bust is already paid).
        netcost_batch_state: dict[str, int | None] = {"floor": None}
        if netcost_enabled:
            netcost_suffix_tokens = [0] * (num_messages + 1)
            for j in range(num_messages - 1, -1, -1):
                netcost_suffix_tokens[j] = netcost_suffix_tokens[j + 1] + _netcost_message_tokens(
                    messages[j], tokenizer
                )

        # Tasks: list of (slot_index, content, context, bias, content_key)
        _PendingTask = tuple[int, str, str, float, int]
        pending_tasks: list[_PendingTask] = []

        # #856 P2b (flag-gated, default off): net-cost frozen-floor unlock.
        # Without the flag, every message in the provider's prefix cache
        # (index < frozen_message_count) is unconditionally skipped — mutating
        # one trades a 90% read discount for a 25% write penalty (Anthropic).
        # That binary floor leaves money on the table: a 50K-token stale tool
        # dump with only a 10K cached suffix after it pays for itself many
        # times over. With HEADROOM_NET_COST_POLICY=1 a *string-content*
        # frozen message instead falls through to the normal candidate
        # pipeline, where the P2 break-even gate (_net_cost_allows) decides
        # per candidate: its S is the full invalidated suffix after the slot,
        # so the deep edit proceeds only when ΔT·(w+r(R-1)) still beats the
        # cache-bust penalty. Block-list and non-string frozen content stay
        # frozen — the gate is wired into the string and parallel-merge paths
        # only, and the per-block cache_control contract in
        # _process_content_blocks is not net-cost aware, so opening them here
        # would mutate cached blocks ungated.
        frozen_unlock_slots: set[int] = set()
        for i, message in enumerate(messages):
            if i < frozen_message_count:
                if netcost_enabled and isinstance(message.get("content", ""), str):
                    # Defer to the break-even gate below instead of skipping.
                    frozen_unlock_slots.add(i)
                    route_counts.setdefault("netcost_frozen_considered", 0)
                    route_counts["netcost_frozen_considered"] += 1
                else:
                    # Frozen — byte-identical to preserve the prefix cache.
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
                    min_chars=min_chars_for_block_compression,
                    skip_user=skip_user,
                    skip_system=skip_system,
                    compress_assistant_text_blocks=compress_assistant_text_blocks,
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

            # Protection 1b: Never compress system/developer messages unless
            # explicitly opted in. These are cache-hot instruction bytes.
            if skip_system and role in {"system", "developer"}:
                result_slots[i] = message
                transforms_applied.append(f"router:protected:{role}_message")
                route_counts.setdefault("system_msg", 0)
                route_counts["system_msg"] += 1
                continue

            if not content or tokenizer.count_text(content) < min_tokens:
                # Skip small content
                result_slots[i] = message
                route_counts["small"] += 1
                continue

            # Protection: failed tool calls / error outputs stay verbatim
            # (issue #847). The model needs exact tracebacks to recover.
            # Strong (>=2 distinct indicators) match only — a single
            # keyword false-positives on benign outputs that mention
            # errors. Above the size cap, fall through — LogCompressor
            # preserves error lines in big logs.
            if (
                self.config.protect_error_outputs
                and role == "tool"
                and len(content) <= self.config.error_protection_max_chars
                and content_has_strong_error_indicators(content)
            ):
                result_slots[i] = message
                transforms_applied.append("router:protected:error_output")
                route_counts.setdefault("error_protected", 0)
                route_counts["error_protected"] += 1
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
                    if netcost_enabled and not self._net_cost_allows(
                        slot_idx=i,
                        original_tokens=tokenizer.count_text(content),
                        compressed_tokens=tokenizer.count_text(cached_compressed),
                        suffix_tokens=netcost_suffix_tokens,
                        route_counts=route_counts,
                        transforms_applied=transforms_applied,
                        batch_state=netcost_batch_state,
                    ):
                        # Net-cost gate: mutation would cost more in cache
                        # invalidation than it saves — leave untouched.
                        result_slots[i] = message
                    else:
                        result_slots[i] = {**message, "content": cached_compressed}
                        transforms_applied.append(f"router:{cached_strategy}:{cached_ratio:.2f}")
                        compressed_details.append(f"{cached_strategy}:{cached_ratio:.2f}")
                        if i in frozen_unlock_slots:
                            transforms_applied.append("router:netcost_frozen_unlock")
                            route_counts.setdefault("netcost_frozen_unlocked", 0)
                            route_counts["netcost_frozen_unlocked"] += 1
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
            for (slot_idx, task_content, _, _, content_key), (result, compress_ms) in zip(
                pending_tasks, task_results
            ):
                message = messages[slot_idx]
                strategy_key = f"compressor:{result.strategy_used.value}"
                compressor_timing[strategy_key] = (
                    compressor_timing.get(strategy_key, 0.0) + compress_ms
                )

                if result.compression_ratio < min_ratio:
                    # Compressed — store in result cache. The cache is still
                    # warmed when the net-cost gate blocks the slot: the
                    # gate's verdict is contextual (suffix size), the
                    # compression result is not.
                    self._cache.put(
                        content_key,
                        result.compressed,
                        result.compression_ratio,
                        result.strategy_used.value,
                    )
                    if netcost_enabled and not self._net_cost_allows(
                        slot_idx=slot_idx,
                        original_tokens=tokenizer.count_text(task_content),
                        compressed_tokens=tokenizer.count_text(result.compressed),
                        suffix_tokens=netcost_suffix_tokens,
                        route_counts=route_counts,
                        transforms_applied=transforms_applied,
                        batch_state=netcost_batch_state,
                    ):
                        result_slots[slot_idx] = message
                        continue
                    result_slots[slot_idx] = {**message, "content": result.compressed}
                    transforms_applied.append(
                        f"router:{result.strategy_used.value}:{result.compression_ratio:.2f}"
                    )
                    compressed_details.append(
                        f"{result.strategy_used.value}:{result.compression_ratio:.2f}"
                    )
                    if slot_idx in frozen_unlock_slots:
                        transforms_applied.append("router:netcost_frozen_unlock")
                        route_counts.setdefault("netcost_frozen_unlocked", 0)
                        route_counts["netcost_frozen_unlocked"] += 1
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
        if route_counts.get("error_protected"):
            parts.append(f"{route_counts['error_protected']} protected (error output)")
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
        if route_counts.get("netcost_batch_admitted"):
            parts.append(f"{route_counts['netcost_batch_admitted']} netcost batch-admitted")
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

        # Forward route_counts to the observer so `/stats` can surface a
        # session-level protection breakdown (issue #454). The observer
        # may not implement this method on older versions; ignore
        # AttributeError so a non-conforming observer doesn't poison
        # routing.
        if self._observer is not None and route_counts:
            try:
                self._observer.record_router_route_counts(route_counts)
            except AttributeError:
                pass
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Router observer raised (non-fatal): %s", e)

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
        min_chars: int = 500,
        skip_user: bool = True,
        skip_system: bool = True,
        compress_assistant_text_blocks: bool = False,
    ) -> dict[str, Any]:
        """Process content blocks (Anthropic format) for compression.

        Cache-safety contract:
          1. Any block carrying `cache_control` is the client's explicit
             cache breakpoint. Modifying any byte of such a block changes
             the cache key the upstream provider matches against, turning
             a 90% read discount into a 25% write penalty (Anthropic).
             We never modify cache_control'd blocks, regardless of role
             or block type.
          2. Assistant text blocks are echoed back by the client in
             subsequent turns and become part of the upstream provider's
             auto-prefix cache (DeepSeek, OpenAI). Default-skip; opt in
             via `compress_assistant_text_blocks` when the deployment
             knows the backend doesn't honor cache_control AND
             compression is byte-deterministic.
          3. User and system blocks carry the prompt the model is acting
             on; compressing them silently mutates the request. Always
             skipped per `skip_user` / `skip_system`.
          4. Tool / function blocks are tool outputs — semantically safe
             to compress (the model references them once, then moves on).

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
            min_chars: Minimum block content length (chars) to consider for compression.
            skip_user: If True, never compress text blocks in user-role messages.
            skip_system: If True, never compress text blocks in system-role messages.
            compress_assistant_text_blocks: If True, allow compressing text blocks in
                assistant-role messages. Default False (cache-safe).

        Returns:
            Transformed message with compressed content blocks.
        """
        new_blocks = []
        any_compressed = False
        role = message.get("role", "")

        # Role-based gate for `text` blocks. Tool/function roles are tool
        # outputs and compress freely; assistant defaults to skip (cache
        # safety) with explicit opt-in; unknown roles default to skip.
        if (skip_user and role == "user") or (skip_system and role in {"system", "developer"}):
            protect_text_blocks = True
        elif role == "assistant" and not compress_assistant_text_blocks:
            protect_text_blocks = True
        elif role not in ("assistant", "tool", "function"):
            protect_text_blocks = True
        else:
            protect_text_blocks = False

        for block in content_blocks:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue

            # Defense in depth: cache_control marker is the client's
            # cache breakpoint. Frozen-message-count is a coarse
            # message-level approximation; this is the per-block
            # guarantee that we never bust an explicit cache key.
            if "cache_control" in block:
                new_blocks.append(block)
                if route_counts is not None:
                    route_counts.setdefault("cache_control_protected", 0)
                    route_counts["cache_control_protected"] += 1
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

                # Protection: failed tool calls / error outputs stay verbatim
                # (issue #847). `is_error` is Anthropic's explicit failure
                # flag and suffices alone; the indicator scan catches error
                # text without the flag but requires >=2 distinct keywords
                # so benign outputs mentioning errors don't skip compression.
                # Above the size cap, fall through — LogCompressor preserves
                # error lines in big logs.
                if (
                    self.config.protect_error_outputs
                    and isinstance(tool_content, str)
                    and len(tool_content) <= self.config.error_protection_max_chars
                    and (
                        block.get("is_error") is True
                        or content_has_strong_error_indicators(tool_content)
                    )
                ):
                    new_blocks.append(block)
                    transforms_applied.append("router:protected:error_output")
                    if route_counts is not None:
                        route_counts.setdefault("error_protected", 0)
                        route_counts["error_protected"] += 1
                    continue

                # Only process string content
                if isinstance(tool_content, str) and len(tool_content) > min_chars:
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

                    # Two-tier compression cache → shared helper
                    compressed_content, was_compressed = self._compress_block_content(
                        content=tool_content,
                        content_key=hash(tool_content),
                        context=context,
                        bias=bias,
                        min_ratio=min_ratio,
                        compressor_timing=compressor_timing,
                        transforms_applied=transforms_applied,
                        route_counts=route_counts,
                        compressed_details=compressed_details,
                        strategy_label="tool_result",
                        details_prefix="tool",
                    )
                    if compressed_content is not None:
                        new_blocks.append({**block, "content": compressed_content})
                        any_compressed = True
                    else:
                        new_blocks.append(block)
                    continue
                else:
                    if route_counts is not None:
                        route_counts["small"] += 1

            # Handle text blocks — compress for non-Anthropic clients (e.g.
            # OpenAI/DeepSeek via Cline) whose SDK normalizes content to
            # block-list form. Roles are gated above (user/system always
            # skipped; assistant default-skipped, opt-in via
            # `compress_assistant_text_blocks`).
            elif block_type == "text" and not protect_text_blocks:
                text_content = block.get("text", "")
                if isinstance(text_content, str) and len(text_content) > min_chars:
                    # Pinning: skip already-compressed content
                    if (
                        "Retrieve more: hash=" in text_content
                        or "Retrieve original: hash=" in text_content
                    ):
                        new_blocks.append(block)
                        if route_counts is not None:
                            route_counts.setdefault("already_compressed", 0)
                            route_counts["already_compressed"] += 1
                        continue

                    # Two-tier compression cache → shared helper
                    compressed_content, _was_compressed = self._compress_block_content(
                        content=text_content,
                        content_key=hash(text_content),
                        context=context,
                        bias=1.0,
                        min_ratio=min_ratio,
                        compressor_timing=compressor_timing,
                        transforms_applied=transforms_applied,
                        route_counts=route_counts,
                        compressed_details=compressed_details,
                        strategy_label="text_block",
                        details_prefix="text",
                    )
                    if compressed_content is not None:
                        new_blocks.append({**block, "text": compressed_content})
                        any_compressed = True
                    else:
                        new_blocks.append(block)
                    continue
                else:
                    if route_counts is not None:
                        route_counts["small"] += 1

            # Keep block unchanged
            new_blocks.append(block)

        if any_compressed:
            return {**message, "content": new_blocks}
        return message

    def _compress_block_content(
        self,
        content: str,
        content_key: int,
        context: str,
        bias: float,
        min_ratio: float,
        compressor_timing: dict[str, float] | None,
        transforms_applied: list[str],
        route_counts: dict[str, int] | None,
        compressed_details: list[str] | None,
        strategy_label: str,
        details_prefix: str,
    ) -> tuple[str | None, bool]:
        """Apply two-tier cache lookup + compression to a single content string.

        Encapsulates the shared cache→compress→store logic used by both
        ``tool_result`` and ``text`` block paths in ``_process_content_blocks``.
        Previously this logic was duplicated ~60 lines per path; centralising
        it ensures both paths stay in sync (cache expiry, pinning, ratio gating).

        Args:
            content: The string content to compress.
            content_key: Pre-computed ``hash(content)`` for cache lookups.
            context: User/query context for relevance-aware compression.
            bias: Compression bias multiplier (tool-specific or 1.0).
            min_ratio: Adaptive minimum compression ratio threshold.
            compressor_timing: Optional dict to accumulate per-strategy timing.
            transforms_applied: List mutated in-place with transform labels.
            route_counts: Optional dict mutated in-place with route counters.
            compressed_details: Optional list mutated with compression details.
            strategy_label: Transform label prefix (e.g. ``"tool_result"``).
            details_prefix: Compressed-details prefix (e.g. ``"tool"``).

        Returns:
            Tuple of ``(compressed_content_or_None, was_compressed)``.
            When ``compressed_content`` is ``None`` the caller should keep
            the original block unchanged. When ``was_compressed`` is
            ``True`` the caller should update the block with the returned
            content and set ``any_compressed``.
        """
        # Tier 1: skip set — instant rejection
        if self._cache.is_skipped(content_key):
            if route_counts is not None:
                route_counts["ratio_too_high"] = route_counts.get("ratio_too_high", 0) + 1
                route_counts["cache_hit"] = route_counts.get("cache_hit", 0) + 1
            return None, False

        # Tier 2: result cache — reuse compressed output
        cached = self._cache.get(content_key)
        if cached is not None:
            cached_compressed, cached_ratio, cached_strategy = cached
            if route_counts is not None:
                route_counts["cache_hit"] = route_counts.get("cache_hit", 0) + 1
            if cached_ratio < min_ratio:
                transforms_applied.append(f"router:{strategy_label}:{cached_strategy}")
                if compressed_details is not None:
                    compressed_details.append(
                        f"{details_prefix}:{cached_strategy}:{cached_ratio:.2f}"
                    )
                return cached_compressed, True
            # Threshold tightened — move result to skip set
            self._cache.move_to_skip(content_key)
            if route_counts is not None:
                route_counts["ratio_too_high"] = route_counts.get("ratio_too_high", 0) + 1
            return None, False

        # Cache miss — run full compression
        if route_counts is not None:
            route_counts["cache_miss"] = route_counts.get("cache_miss", 0) + 1
        t0 = time.perf_counter()
        result = self.compress(content, context=context, bias=bias)
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
            transforms_applied.append(f"router:{strategy_label}:{result.strategy_used.value}")
            if compressed_details is not None:
                compressed_details.append(
                    f"{details_prefix}:{result.strategy_used.value}:{result.compression_ratio:.2f}"
                )
            return result.compressed, True
        # Didn't compress enough — add to skip set
        self._cache.mark_skip(content_key)
        if route_counts is not None:
            route_counts["ratio_too_high"] = route_counts.get("ratio_too_high", 0) + 1
        return None, False

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

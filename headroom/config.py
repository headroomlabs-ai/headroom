"""Configuration models for Headroom SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from headroom.models.config import ML_MODEL_DEFAULTS


class HeadroomMode(str, Enum):
    """Operating modes for Headroom."""

    AUDIT = "audit"  # Observe only, no modifications
    OPTIMIZE = "optimize"  # Apply deterministic transforms
    SIMULATE = "simulate"  # Return transform plan without API call


# Model context limits should be provided by the Provider
# This dict allows user overrides only
DEFAULT_MODEL_CONTEXT_LIMITS: dict[str, int] = {}


@dataclass
class CacheAlignerConfig:
    """Configuration for cache alignment.

    Phase 1 Enhancement: Now integrates DynamicContentDetector for comprehensive
    dynamic content detection beyond just dates.

    New Detection Capabilities (when use_dynamic_detector=True):
    - UUIDs: 550e8400-e29b-41d4-a716-446655440000
    - API keys/tokens: sk-abc123..., api_key_xyz...
    - JWT tokens: eyJhbGciOiJIUzI1NiIs...
    - Unix timestamps: 1705312847
    - Request/trace IDs: req_abc123, trace_xyz789
    - Hex hashes: MD5 (32 chars), SHA1 (40 chars), SHA256 (64 chars)
    - Version numbers: v1.2.3, v2.0.0-beta
    - Structural patterns: "Session: abc123", "User: john@example.com"
    - High-entropy strings: Random-looking alphanumeric sequences

    GOTCHAS:
    - Date regex may match non-date content (e.g., version numbers like "2024-01-15")
    - Moving dates to end of system prompt may confuse models if date was
      semantically important in its original position
    - Whitespace normalization may break:
      - Code blocks with significant indentation
      - ASCII art or formatted tables
      - Markdown that relies on specific spacing
    - ISO timestamps in tool outputs may be incorrectly flagged as "dynamic dates"

    SAFE: Only applied to SYSTEM messages, not user/assistant/tool content.
    """

    enabled: bool = False  # Disabled by default — prefix stability gains are marginal in practice

    # === Phase 1: DynamicContentDetector Integration ===
    # When True, uses the full DynamicContentDetector with 15+ patterns
    # When False, uses legacy date_patterns only (backward compatible)
    use_dynamic_detector: bool = True

    # Which detection tiers to use (only when use_dynamic_detector=True)
    # - "regex": Fast structural/universal patterns (~0ms) - RECOMMENDED
    # - "ner": Named Entity Recognition via spaCy (~5-10ms) - optional
    # - "semantic": Embedding similarity (~20-50ms) - optional
    detection_tiers: list[Literal["regex", "ner", "semantic"]] = field(
        default_factory=lambda: ["regex"]
    )

    # Additional dynamic labels to detect (extends default list)
    # These are KEY names that hint the VALUE is dynamic
    # e.g., "session" will detect "session: abc123" and extract "abc123"
    extra_dynamic_labels: list[str] = field(default_factory=list)

    # Entropy threshold for detecting random strings (0-1 scale)
    # Higher = more selective (only very random strings like UUIDs)
    # Lower = more aggressive (may catch non-random content)
    entropy_threshold: float = 0.7

    # === Legacy Configuration (used when use_dynamic_detector=False) ===
    date_patterns: list[str] = field(
        default_factory=lambda: [
            r"Current [Dd]ate:?\s*\d{4}-\d{2}-\d{2}",
            r"Today is \w+,?\s+\w+ \d+",
            r"Today's date:?\s*\d{4}-\d{2}-\d{2}",
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        ]
    )

    # === Whitespace Normalization ===
    normalize_whitespace: bool = True
    collapse_blank_lines: bool = True

    # Separator used to mark where dynamic content begins in system message
    # Content before this separator is cached; content after is dynamic
    dynamic_tail_separator: str = "\n\n---\n[Dynamic Context]\n"


@dataclass
class RelevanceScorerConfig:
    """Configuration for relevance scoring in SmartCrusher.

    Relevance scoring determines which items to keep when compressing
    tool outputs. Uses the pattern: relevance(item, context) -> [0, 1].

    Available tiers:
    - "bm25": BM25 keyword matching (zero dependencies, fast)
    - "embedding": Semantic similarity via sentence-transformers
    - "hybrid": BM25 + embedding with adaptive fusion (RECOMMENDED)

    DEFAULT: "hybrid" - combines exact matching (UUIDs, IDs) with semantic
    understanding. Falls back to BM25 if sentence-transformers not installed.

    For full hybrid support, install: pip install headroom[relevance]

    WHY HYBRID IS DEFAULT:
    - Missing important items during compression is catastrophic
    - BM25 alone gives low scores for single-term matches (e.g., "Alice" = 0.07)
    - Semantic matching catches "errors" -> "failed", "issues", etc.
    - 5-10ms latency is acceptable vs. losing critical data
    """

    tier: Literal["bm25", "embedding", "hybrid"] = "hybrid"

    # BM25 parameters
    bm25_k1: float = 1.5  # Term frequency saturation
    bm25_b: float = 0.75  # Length normalization

    # Embedding parameters
    embedding_model: str = field(default_factory=lambda: ML_MODEL_DEFAULTS.sentence_transformer)

    # Hybrid parameters
    hybrid_alpha: float = 0.5  # BM25 weight (1-alpha = embedding weight)
    adaptive_alpha: bool = True  # Adjust alpha based on query type

    # Scoring thresholds
    # With hybrid/embedding: semantic scores are meaningful (0.3-0.5 for good matches)
    # With BM25 fallback: threshold is still reasonable for multi-term matches
    # Lower threshold = safer (keeps more items), higher = more aggressive compression
    relevance_threshold: float = 0.25  # Keep items above this score


@dataclass
class AnchorConfig:
    """Configuration for dynamic anchor allocation in SmartCrusher.

    Anchor selection determines which array positions are preserved during
    compression. Different data patterns benefit from different anchor strategies:
    - Search results: Front-heavy (top results are most relevant)
    - Logs: Back-heavy (recent entries matter most)
    - Time series: Balanced (need both ends to show trends)
    - Generic: Distributed (no assumption about order importance)

    The anchor budget is a percentage of max_items allocated to position-based
    anchors. The remaining budget goes to relevance-scored items.
    """

    # Base anchor budget as percentage of max_items
    anchor_budget_pct: float = 0.25  # 25% of slots for position anchors

    # Minimum and maximum anchor slots
    min_anchor_slots: int = 3
    max_anchor_slots: int = 12

    # Default distribution weights (sum to 1.0)
    default_front_weight: float = 0.5
    default_back_weight: float = 0.4
    default_middle_weight: float = 0.1

    # Pattern-specific overrides
    search_front_weight: float = 0.75  # Search results: front-heavy
    search_back_weight: float = 0.15
    logs_front_weight: float = 0.15  # Logs: back-heavy (recent)
    logs_back_weight: float = 0.75

    # Query keyword detection for dynamic adjustment
    recency_keywords: tuple[str, ...] = (
        "latest",
        "recent",
        "last",
        "newest",
        "current",
        "now",
    )
    historical_keywords: tuple[str, ...] = (
        "first",
        "oldest",
        "earliest",
        "original",
        "initial",
        "beginning",
    )

    # Information density selection
    use_information_density: bool = True
    candidate_multiplier: int = 3  # Consider 3x candidates per slot
    dedup_identical_items: bool = True  # Don't waste slots on identical items


# Default tools to exclude from compression (local file/code tools)
# Read: Returns exact file content needed for Edit tool's old_string matching.
#   Compressing would break the edit workflow.
# Glob: Returns compact file path lists used for navigation. Low token count,
#   not worth compressing.
# Tool outputs that are reference data and must NOT be compressed.
# Read/Glob/Grep contain exact file contents/search results the agent needs for edits.
# Write/Edit record what changes were made — compressing them causes duplicate/conflicting edits.
# Bash is NOT excluded — its outputs (build logs, test output) are ideal compression targets.
DEFAULT_EXCLUDE_TOOLS: frozenset[str] = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "Write",
        "Edit",
        "Bash",
        # Lowercase variants for case-insensitive matching
        "read",
        "glob",
        "grep",
        "write",
        "edit",
        "bash",
    }
)

# Tool names recognized as Read/Edit/Write for lifecycle tracking
_READ_TOOL_NAMES: frozenset[str] = frozenset({"Read", "read"})
_EDIT_TOOL_NAMES: frozenset[str] = frozenset({"Edit", "edit"})
_WRITE_TOOL_NAMES: frozenset[str] = frozenset({"Write", "write"})
_MUTATING_TOOL_NAMES: frozenset[str] = _EDIT_TOOL_NAMES | _WRITE_TOOL_NAMES


@dataclass
class ReadLifecycleConfig:
    """Event-driven Read lifecycle management.

    Detects stale and superseded Read outputs in the conversation and replaces
    them with compact markers + CCR hashes. Fresh Reads are never touched.

    A Read is STALE when the file was subsequently edited (content is factually
    wrong). A Read is SUPERSEDED when the file was subsequently re-Read (content
    is redundant). Both are provably safe to compress.

    Operates as a pre-processing pass before ContentRouter, independent of
    tool exclusion logic. Read remains in DEFAULT_EXCLUDE_TOOLS — fresh Read
    outputs still bypass ContentRouter compression.
    """

    enabled: bool = True  # On by default: stale/superseded Reads are provably safe to compress
    compress_stale: bool = True  # Replace Reads of files that were later edited
    compress_superseded: bool = False  # Disabled: busts Anthropic prompt cache prefix
    min_size_bytes: int = 512  # Skip tiny Read outputs (not worth the overhead)


@dataclass
class CompressionProfile:
    """Per-tool compression bias applied to statistically-determined K.

    Instead of hardcoding max_items=15, the adaptive sizer computes the optimal K
    via information saturation (Kneedle on unique bigram coverage). This profile
    applies a bias multiplier: >1 keeps more items (conservative), <1 keeps fewer
    (aggressive).
    """

    bias: float = 1.0  # 0.7=aggressive, 1.0=moderate, 1.5=conservative
    min_k: int = 3  # Never keep fewer than this
    max_k: int | None = None  # Cap (None = no cap, let statistics decide)


# Named presets for convenience
PROFILE_PRESETS: dict[str, CompressionProfile] = {
    "conservative": CompressionProfile(bias=1.5, min_k=5),
    "moderate": CompressionProfile(bias=1.0, min_k=3),
    "aggressive": CompressionProfile(bias=0.7, min_k=3),
}

# Default per-tool profiles: tools not listed here use moderate (bias=1.0)
DEFAULT_TOOL_PROFILES: dict[str, CompressionProfile] = {
    # Search results: keep more matches for accuracy
    "Grep": PROFILE_PRESETS["conservative"],
    "grep": PROFILE_PRESETS["conservative"],
    # Logs/output: balanced compression
    "Bash": PROFILE_PRESETS["moderate"],
    "bash": PROFILE_PRESETS["moderate"],
    # Web pages are verbose, compress aggressively
    "WebFetch": PROFILE_PRESETS["aggressive"],
    "webfetch": PROFILE_PRESETS["aggressive"],
}


@dataclass
class SmartCrusherConfig:
    """Configuration for smart statistical crusher (DEFAULT).

    Uses statistical analysis to intelligently compress tool outputs while
    PRESERVING THE ORIGINAL JSON SCHEMA. Output contains only items from
    the original array - no wrappers, no generated text, no metadata.

    Handles ALL JSON types:
    - Arrays of dicts, strings, numbers, mixed types
    - Flat objects with many keys
    - Nested objects (recursive compression)

    Safety guarantees (consistent across all types):
    - First K, last K items always kept (K is adaptive via Kneedle algorithm)
    - Error items never dropped
    - Anomalous numeric items (> 2 std from mean) always kept
    - Items matching query context via RelevanceScorer

    GOTCHAS:
    - Adds ~5-10ms overhead per tool output for statistical analysis
    - Change point detection uses fixed window (5 items) - may miss:
      - Very gradual changes
      - Patterns in smaller arrays
    - TOP_N for search results assumes higher score = more relevant
      (may not be true for all APIs)

    SAFER SETTINGS:
    - Increase max_items_after_crush for critical data
    - Set variance_threshold lower (1.5) to catch more change points
    """

    enabled: bool = True  # Enabled by default — sole tool-output compressor
    min_items_to_analyze: int = 5  # Don't analyze tiny arrays
    min_tokens_to_crush: int = 200  # Only crush if > N tokens
    variance_threshold: float = 2.0  # Std devs for change point detection
    uniqueness_threshold: float = 0.1  # Below this = nearly constant
    similarity_threshold: float = 0.8  # For clustering similar strings
    max_items_after_crush: int = 15  # Target max items in output
    preserve_change_points: bool = True
    factor_out_constants: bool = False  # Disabled - preserves original schema
    include_summaries: bool = False  # Disabled - no generated text

    # Feedback loop integration (TOIN - Tool Output Intelligence Network)
    use_feedback_hints: bool = True  # Use learned patterns to adjust compression

    # LOW FIX #21: Make TOIN confidence threshold configurable
    # Minimum confidence required to apply TOIN recommendations
    toin_confidence_threshold: float = 0.3

    # Relevance scoring configuration
    relevance: RelevanceScorerConfig = field(default_factory=RelevanceScorerConfig)

    # Anchor selection configuration (dynamic position-based preservation)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)

    # Content deduplication - prevents wasting slots on identical items
    # When multiple preservation mechanisms (anchors, anomalies, outliers) add
    # the same item, only one copy is kept. This is critical for arrays where
    # many items have identical content (e.g., repeated status messages).
    dedup_identical_items: bool = True

    # Adaptive K boundary allocation (fraction of total K for first/last items)
    # The remaining fraction is filled by importance scoring (errors, anomalies, etc.)
    first_fraction: float = 0.3  # 30% of K from start of array
    last_fraction: float = 0.15  # 15% of K from end of array


@dataclass
class CacheOptimizerConfig:
    """Configuration for provider-specific cache optimization.

    The CacheOptimizer system provides provider-specific caching strategies:
    - Anthropic: Explicit cache_control breakpoints for prompt caching
    - OpenAI: Prefix stabilization for automatic prefix caching
    - Google: CachedContent API lifecycle management

    This is COMPLEMENTARY to the CacheAligner transform - CacheAligner does
    basic prefix stabilization (date extraction, whitespace normalization),
    while CacheOptimizer applies provider-specific optimizations.

    Enable this for maximum cache hit rates when you know your provider.
    """

    enabled: bool = True  # Enable provider-specific cache optimization
    auto_detect_provider: bool = True  # Auto-detect from HeadroomClient provider
    min_cacheable_tokens: int = 1024  # Minimum tokens for caching (provider may override)
    enable_semantic_cache: bool = False  # Enable query-level semantic caching
    semantic_cache_similarity: float = 0.95  # Similarity threshold for semantic cache
    semantic_cache_max_entries: int = 1000  # Max semantic cache entries
    semantic_cache_ttl_seconds: int = 300  # Semantic cache TTL


@dataclass
class CCRConfig:
    """Configuration for Compress-Cache-Retrieve architecture.

    CCR makes compression REVERSIBLE: when SmartCrusher compresses tool outputs,
    the original data is cached. If the LLM needs more data, it can retrieve it.

    Key insight from research: REVERSIBLE compression beats irreversible compression.
    - Phil Schmid: "Prefer raw > Compaction > Summarization"
    - Factory.ai: "Cutting context too aggressively can backfire"

    How CCR works:
    1. COMPRESS: SmartCrusher compresses array from 1000 to 20 items
    2. CACHE: Original 1000 items stored in CompressionStore
    3. INJECT: Marker added to tell LLM how to retrieve more
    4. RETRIEVE: If LLM needs more, it calls headroom_retrieve(hash, query)

    Benefits:
    - Zero-risk compression: worst case = LLM retrieves what it needs
    - Feedback loop: track what gets retrieved to improve compression
    - Network effect: retrieval patterns improve compression for all users

    GOTCHAS:
    - Cache has TTL (default 5 min) - retrieval fails after expiration
    - Memory usage: ~1KB per cached entry
    - Only works with array compression (not string truncation)
    """

    enabled: bool = True  # Enable CCR (cache + retrieval markers)
    store_max_entries: int = 1000  # Max entries in compression store
    store_ttl_seconds: int = 300  # Cache TTL (5 minutes)
    inject_retrieval_marker: bool = True  # Add retrieval hint to compressed output
    feedback_enabled: bool = True  # Track retrieval events for learning
    min_items_to_cache: int = 20  # Only cache if original had >= N items

    # Tool injection (Phase 3)
    inject_tool: bool = True  # Inject headroom_retrieve tool into tools array
    inject_system_instructions: bool = False  # Add retrieval instructions to system message

    # Retrieval marker format
    # Inserted at end of compressed content to tell LLM how to get more
    marker_template: str = (
        "\n[{original_count} items compressed to {compressed_count}."
        "{summary}"
        " Retrieve more: hash={hash}."
        " Expires in {ttl_minutes}m.]"
    )


@dataclass
class PrefixFreezeConfig:
    """Configuration for cache-aware prefix freezing.

    When enabled, tracks provider prefix cache state across turns and freezes
    already-cached messages so the transform pipeline skips them. This prevents
    Headroom from invalidating the provider's prefix cache (which would replace
    a 90% read discount with a 25% write penalty on Anthropic).

    The force_compress_threshold controls when compression savings are large
    enough to justify busting the cache. For Anthropic (90% read discount),
    compression must save >90% of tokens in the frozen prefix to be worth it.
    """

    enabled: bool = True
    min_cached_tokens: int = 1024  # Min cached tokens to activate freeze
    session_ttl_seconds: int = 600  # Session tracker cleanup TTL
    force_compress_threshold: float = 0.5  # Bust cache if compression saves > this fraction


@dataclass
class HeadroomConfig:
    """Main configuration for HeadroomClient."""

    store_url: str = "sqlite:///headroom.db"
    default_mode: HeadroomMode = HeadroomMode.AUDIT
    model_context_limits: dict[str, int] = field(
        default_factory=lambda: DEFAULT_MODEL_CONTEXT_LIMITS.copy()
    )
    smart_crusher: SmartCrusherConfig = field(default_factory=SmartCrusherConfig)
    cache_aligner: CacheAlignerConfig = field(default_factory=CacheAlignerConfig)
    cache_optimizer: CacheOptimizerConfig = field(default_factory=CacheOptimizerConfig)
    ccr: CCRConfig = field(default_factory=CCRConfig)  # Compress-Cache-Retrieve
    prefix_freeze: PrefixFreezeConfig = field(default_factory=PrefixFreezeConfig)

    # Output buffer reserved for the model's response when sizing the
    # incoming context. Previously lived on RollingWindowConfig; hoisted
    # to the top-level config when PR-B1 retired the rolling-window stage.
    output_buffer_tokens: int = 4000

    # Content Router - intelligent content-type based compression
    # Routes content to appropriate compressor (Kompress for text, SmartCrusher for JSON,
    # CodeCompressor for code, LogCompressor for logs, etc.)
    content_router_enabled: bool = True

    # Tool-result interceptors (ast-grep Read outline, etc.). Opt-in for now.
    # Env var HEADROOM_INTERCEPT_ENABLED=1 also enables (for CLI `--intercept-tool-results`).
    intercept_tool_results: bool = False

    # Debugging - opt-in diff artifact generation
    generate_diff_artifact: bool = False  # Enable to get detailed transform diffs

    # Canonical pipeline lifecycle extensions
    pipeline_extensions: list[Any] = field(default_factory=list)
    discover_pipeline_extensions: bool = True

    def get_context_limit(self, model: str) -> int | None:
        """
        Get context limit for a model from user overrides.

        Args:
            model: Model name.

        Returns:
            Context limit if configured, None otherwise.
            Provider should be consulted if None is returned.
        """
        if model in self.model_context_limits:
            return self.model_context_limits[model]
        # Try prefix matching for versioned model names
        for known_model, limit in self.model_context_limits.items():
            if model.startswith(known_model):
                return limit
        return None


@dataclass
class Block:
    """Atomic unit of context analysis."""

    kind: Literal["system", "user", "assistant", "tool_call", "tool_result", "rag", "unknown"]
    text: str
    tokens_est: int
    content_hash: str
    source_index: int  # Position in original messages
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class WasteSignals:
    """Detected waste signals in a request."""

    json_bloat_tokens: int = 0  # JSON blocks > 500 tokens
    html_noise_tokens: int = 0  # HTML tags/comments
    base64_tokens: int = 0  # Base64 encoded blobs
    whitespace_tokens: int = 0  # Repeated whitespace
    dynamic_date_tokens: int = 0  # Dynamic dates in system prompt
    repetition_tokens: int = 0  # Repeated content

    def total(self) -> int:
        """Total waste tokens detected."""
        return (
            self.json_bloat_tokens
            + self.html_noise_tokens
            + self.base64_tokens
            + self.whitespace_tokens
            + self.dynamic_date_tokens
            + self.repetition_tokens
        )

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for storage."""
        return {
            "json_bloat": self.json_bloat_tokens,
            "html_noise": self.html_noise_tokens,
            "base64": self.base64_tokens,
            "whitespace": self.whitespace_tokens,
            "dynamic_date": self.dynamic_date_tokens,
            "repetition": self.repetition_tokens,
        }


@dataclass
class CachePrefixMetrics:
    """Detailed cache prefix metrics for debugging cache misses.

    Log these per-request to understand why caching is or isn't working.
    Compare stable_prefix_hash across requests - any change means cache miss.
    """

    stable_prefix_bytes: int  # Byte length of static prefix
    stable_prefix_tokens_est: int  # Estimated token count of static prefix
    stable_prefix_hash: str  # Hash of canonicalized prefix (16 chars)
    prefix_changed: bool  # True if hash differs from previous request in session
    previous_hash: str | None = None  # Previous hash for comparison (None = first request)


@dataclass
class TransformResult:
    """Output of a transform operation."""

    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    transforms_applied: list[str]
    markers_inserted: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diff_artifact: DiffArtifact | None = None  # Populated if generate_diff_artifact=True
    cache_metrics: CachePrefixMetrics | None = None  # Populated by CacheAligner
    timing: dict[str, float] = field(default_factory=dict)  # transform_name → ms
    waste_signals: WasteSignals | None = None  # Detected waste in original messages

    @property
    def transforms_summary(self) -> dict[str, int]:
        """Counted summary of transforms_applied (e.g. {'router:tool_result:text': 4})."""
        from collections import Counter

        return dict(Counter(self.transforms_applied))


@dataclass
class TransformDiff:
    """Diff info for a single transform (for debugging/perf)."""

    transform_name: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    items_removed: int = 0
    items_kept: int = 0
    details: str = ""  # Human-readable description of what changed
    duration_ms: float = 0.0  # Wall-clock time for this transform


@dataclass
class DiffArtifact:
    """Complete diff artifact for debugging transform pipeline.

    Opt-in via HeadroomConfig.generate_diff_artifact = True.
    Useful for understanding what each transform did to your messages.
    """

    request_id: str
    original_tokens: int
    optimized_tokens: int
    total_tokens_saved: int
    transforms: list[TransformDiff] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result of a simulation (dry-run)."""

    tokens_before: int
    tokens_after: int
    tokens_saved: int
    transforms: list[str]
    estimated_savings: str  # Human-readable cost estimate
    messages_optimized: list[dict[str, Any]]
    block_breakdown: dict[str, int]
    waste_signals: dict[str, int]
    stable_prefix_hash: str
    cache_alignment_score: float


@dataclass
class RequestMetrics:
    """Comprehensive metrics for a single request."""

    request_id: str
    timestamp: datetime
    model: str
    stream: bool
    mode: str  # audit | optimize | simulate

    # Token breakdown
    tokens_input_before: int
    tokens_input_after: int
    tokens_output: int | None = None  # None if streaming

    # Block breakdown
    block_breakdown: dict[str, int] = field(default_factory=dict)

    # Waste signals
    waste_signals: dict[str, int] = field(default_factory=dict)

    # Cache metrics (basic)
    stable_prefix_hash: str = ""
    cache_alignment_score: float = 0.0
    cached_tokens: int | None = None  # From API response if available

    # Cache optimizer metrics (provider-specific)
    cache_optimizer_used: str | None = None  # e.g., "anthropic-cache-optimizer"
    cache_optimizer_strategy: str | None = None  # e.g., "explicit_breakpoints"
    cacheable_tokens: int = 0  # Tokens eligible for caching
    breakpoints_inserted: int = 0  # Cache breakpoints added (Anthropic)
    estimated_cache_hit: bool = False  # Whether prefix matches previous
    estimated_savings_percent: float = 0.0  # Estimated savings if cached
    semantic_cache_hit: bool = False  # Whether semantic cache was hit

    # Transform details
    transforms_applied: list[str] = field(default_factory=list)
    tool_units_dropped: int = 0
    turns_dropped: int = 0

    # For debugging
    messages_hash: str = ""
    error: str | None = None

"""Configuration models for Headroom SDK."""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from headroom.models.config import ML_MODEL_DEFAULTS
from headroom.rollout import RolloutSnapshot, resolve_rollout


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
# WebSearch/WebFetch results are large reference payloads that must remain verbatim.
# Bash is NOT excluded — its outputs (build logs, test output) are ideal compression targets.
# To protect Bash or other non-excluded tools from lossy compression, use
# HEADROOM_PROTECT_TOOL_RESULTS=Bash or --protect-tool-results Bash.
# headroom_retrieve: its entire contract is returning already-retrieved, original
# CCR content verbatim. Recompressing it writes a new <<ccr:hash>> marker the
# agent can never redeem (#1077).
DEFAULT_EXCLUDE_TOOLS: frozenset[str] = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "Write",
        "Edit",
        "WebSearch",
        "WebFetch",
        "headroom_retrieve",
        # Copilot CLI's file-read tool (its `Read` equivalent): raw file bytes
        # the model byte-patches against.
        "view",
        # Cursor's file-read tool, the same category as `Read` and `view`. Its
        # absence here was not a decision: without it Cursor's raw source took
        # the LOSSY path, which is strictly worse than the fold that
        # DEFAULT_BYTE_EXACT_EXCLUDE_TOOLS protects it from. Protecting a read
        # tool needs both halves -- excluded here so it is never lossily
        # compressed, and named below so it is never folded either -- and a
        # plugin that supplies only the first half cannot be relied on to be
        # installed.
        "read_file",
        # Skill bodies (Claude Code's `Skill` tool) are INSTRUCTIONS, not tool
        # output, and lossy compression on a directive does not degrade it --
        # it inverts it. "do not guess at model names" losing its "not" is a
        # correctness failure, not a quality tradeoff. Measured across 40 real
        # SKILL.md bodies on the lossy path: only 73.5% of negations (not,
        # never, avoid, unless, without) and 65.5% of modals (must, always,
        # only, required) survived; two skills kept 0/2 and 1/6 of theirs.
        # Bodies load on demand and are read once, so the tokens at stake are
        # small and the downside is unbounded. Named below as well, because
        # protecting a tool needs both halves.
        "Skill",
        # Lowercase variants for case-insensitive matching
        "read",
        "glob",
        "grep",
        "skill",
        "write",
        "edit",
        "web_search",
        "web_fetch",
    }
)

# These excluded web-tool results must remain byte-faithful. Even the
# excluded-tool lossless fold rewrites formatted JSON.
# Five consumers key off this frozenset, across two modules -- a name added here
# is protected on every wire shape, not just the one you happened to test:
# transforms/content_router.py has ContentRouter's two per-block excluded-tool
# guards (OpenAI chat + Anthropic blocks) and _cross_turn_dedup_messages's
# verbatim_tool_ids; proxy/handlers/openai.py builds verbatim_excluded_call_ids
# for the Responses excluded-tool guard and for the Responses dedup's
# protected_call_ids. The dedup consumers have no dedicated guard of their own,
# so removing headroom_retrieve from here would silently reopen the retrieval
# loop for those paths with cross-turn dedup on.
DEFAULT_VERBATIM_EXCLUDE_TOOLS: frozenset[str] = frozenset(
    {
        "WebSearch",
        "WebFetch",
        "web_search",
        "web_fetch",
        "headroom_retrieve",
        # `view` (Copilot CLI file read) must stay BYTE-EXACT: the model produces
        # line/byte-precise edits against it, and even "lossless" JSON rewrites
        # or cross-turn dedup folds break old_str matching and force re-reads.
        "view",
    }
)

# File-READ tools whose output the excluded-tool lossless fold must never
# REWRITE. A strictly weaker protection than DEFAULT_VERBATIM_EXCLUDE_TOOLS
# above, and that gap is the whole reason there are two sets: this one gates only
# the fold, so these tools keep cross-turn dedup and the age-based fall-through.
#
# The failure being fixed: a read tool that returns raw file bytes (Codex-style
# `read`, custom agents, /v1/compress callers naming their own tools) hands
# `_lossless_compact_excluded` a pretty-printed JSON file, which json-min's it.
# That fold is data-lossless and the proxy can invert it -- but the inverse runs
# on OUR side, while the copy that has to match on disk is typed by the MODEL out
# of the bytes it was SHOWN. It builds `Edit(old_string=...)` from minified JSON,
# the file on disk is still pretty-printed, the edit misses, and the retry turn
# costs far more than the fold saved. The log fold (repeats rewritten to
# `... (repeated N times)`) and the search fold (path hoisted into a heading)
# destroy an Edit anchor the same way, so this gates the fold as a whole rather
# than the JSON branch alone. DEFAULT_EXCLUDE_TOOLS has claimed since forever
# that Read "Returns exact file content needed for Edit tool's old_string
# matching"; this is what makes that true instead of aspirational.
#
# Why NOT just add these to DEFAULT_VERBATIM_EXCLUDE_TOOLS, which would have been
# the one-line fix: that set also gates cross-turn dedup and the age-based lossy
# fall-through, and neither shares the failure mode. Dedup replaces a repeat with
# an in-context pointer to an EARLIER copy that is never rewritten (the
# keep-earliest invariant in transforms/cross_turn_dedup.py), so the true bytes
# stay physically in the window for the model to copy from; the fold leaves them
# nowhere at all. Measured on one file read three times, dedup is worth ~66% of
# those tokens -- the largest Read-side saving there is, and the one that still
# fires on Claude Code's `cat -n`-shaped Read output, where the fold never fires
# in the first place. Buying byte-exactness with dedup would be paying the large
# bill to plug the small leak. `view` stays in the stricter set above: that was a
# deliberate, documented call for Copilot and loosening it is not this change's
# business.
DEFAULT_BYTE_EXACT_EXCLUDE_TOOLS: frozenset[str] = frozenset(
    {
        "Read",
        "read",
        # Cursor's `Read` equivalent. Named here for a reason worth stating,
        # because it is counter-intuitive: adding a read tool to the proxy's
        # EXCLUDE set to keep it away from lossy compression makes things WORSE
        # on its own. Exclusion is exactly what routes a tool INTO the
        # excluded-tool fold, and the pluggable provider on that path is handed
        # content with no tool name, so it cannot tell a file read from a grep
        # and will happily rewrite raw source. Protecting a read tool takes
        # both halves: excluded from the lossy path, and named here to stay out
        # of the fold. Caught by scripts/verify-lossless-coverage.py in the
        # partner-trial repo, which reported `read_file` reaching the provider
        # seam once the plugin started excluding it.
        "read_file",
        # Skill bodies. The other half of the protection added above: excluding
        # `Skill` from the lossy path routes it INTO the excluded-tool fold, and
        # the fold rewrites what the model is SHOWN. On tool output that is a
        # fair trade; on a directive it is not, because the folds that collapse
        # repeated lines or hoist paths into headings can merge two distinct
        # instructions into one. Skill bodies are prose and fold poorly anyway,
        # so this costs close to nothing and removes the whole class.
        "Skill",
        "skill",
    }
)


def _tool_name_aliases(name: str) -> tuple[str, ...]:
    """Return equivalent spellings for tool exclusion matching."""
    if not isinstance(name, str):
        # Pre-existing fragility (not introduced here): a malformed message can
        # put a non-string value in the tool-name map (see _build_tool_name_map's
        # truthy-only `if tc_id and name:` filter). Fail safe -- no aliases means
        # is_tool_excluded() returns False -- rather than crashing the pipeline.
        return ()
    aliases = [name]
    lname = name.lower()

    if lname.startswith("mcp__"):
        # OpenAI-style MCP wrappers use mcp__server__tool. Custom agents that
        # speak Anthropic sometimes emit the same wrapper as mcp_Server_tool.
        parts = name.split("__", 2)
        if len(parts) == 3 and parts[1] and parts[2]:
            aliases.append(f"mcp_{parts[1]}_{parts[2]}")
            aliases.append(parts[2])
    elif lname.startswith("mcp_"):
        parts = name.split("_", 2)
        if len(parts) == 3 and parts[1] and parts[2]:
            aliases.append(f"mcp__{parts[1]}__{parts[2]}")
            aliases.append(parts[2])

    return tuple(dict.fromkeys(aliases))


# Hermes Agent's deferred-tool bridge. Hermes loads on-demand tools via a
# `tool_search`/`tool_describe`/`tool_call` indirection; on the wire the
# emitted tool call is named `tool_call` and the REAL tool name lives in the
# arguments payload (`{"name": "...", "arguments": {...}}`). Tool exclusion /
# protect lists match on the real name, so we must unwrap this bridge before
# building the tool_call_id -> name map, or whitelists silently no-op for all
# deferred tools.
_HERMES_TOOL_CALL_WRAPPER = "tool_call"


def unwrap_tool_call_name(name: str, arguments: Any) -> str:
    """Extract the real tool name from a Hermes deferred ``tool_call`` wrapper.

    Non-wrapper names pass through unchanged. Malformed/unparseable wrappers
    fail open and return the wrapper name (caller decides what that means).
    """
    if name != _HERMES_TOOL_CALL_WRAPPER:
        return name
    raw = arguments
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return name
    if isinstance(raw, dict):
        inner = raw.get("name")
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
    return name


def is_tool_excluded(name: str, exclude_tools: Iterable[str]) -> bool:
    """Return True if ``name`` matches the tool-exclusion set.

    Plain entries match by exact (case-insensitive) name, so the common case
    stays a set lookup. Entries containing a glob metacharacter (``*``, ``?`` or
    ``[``) are matched with :func:`fnmatch.fnmatchcase`, letting a single pattern
    such as ``mcp__*`` cover every tool an MCP server exposes without listing
    each name (issue #870).

    MCP tool wrappers are also matched through their common aliases. For example,
    ``mcp__Headroom__headroom_retrieve`` and
    ``mcp_Headroom_headroom_retrieve`` both match ``mcp__*`` and the bare
    ``headroom_retrieve`` entry.
    """
    if not exclude_tools:
        return False

    patterns = tuple(exclude_tools)
    if not patterns:
        return False
    aliases = _tool_name_aliases(name)
    exact_patterns = set(patterns)
    lower_exact_patterns = {pat.lower() for pat in exact_patterns}
    if any(alias in exact_patterns or alias.lower() in lower_exact_patterns for alias in aliases):
        return True

    return any(
        fnmatch.fnmatchcase(alias.lower(), pat.lower())
        for alias in aliases
        for pat in patterns
        if "*" in pat or "?" in pat or "[" in pat
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
class ReadMaturationConfig:
    """Mechanism B: hold-back Read maturation (compress before cache entry).

    Motivation (measured by `headroom audit-reads`): the median Read stays
    in context for ~118 assistant turns after it appears, billed at the
    provider's cache-read rate every request — a Read's lifetime cost is
    roughly 13x its size. The only cache-safe moment to shrink it is
    BEFORE it is ever cache-written.

    Mechanics: a fresh large Read is held out of the provider prefix
    cache (the trailing cache breakpoint is relocated to just before it)
    while its file is ACTIVE, stays verbatim the whole time the model is
    working with it, and matures into a CCR-backed marker once the file
    has been quiet for `quiesce_turns`. Only that final compressed form
    ever enters the cache. No cached byte is ever mutated — there is
    nothing to bust.

    Activity-based (not a fixed hold window) because the audit-reads
    simulation showed touch gaps are fat-tailed: next-touch p50 is 4
    turns but p90 is 81 — no fixed window covers the tail, while a
    quiesce rule covers the activity cluster and lets the tail self-heal
    via the model's observed habit of re-reading ranges from disk (95%
    of re-reads in real traffic are partial-range reads made while the
    full text was still in context).

    Disabled by default while the mechanism is validated in pilots.
    """

    enabled: bool = False
    # Mature a held Read once its FILE has had no activity (reads or
    # edits) for this many assistant turns. Simulation: next-touch p50
    # is 4 turns, so 5 covers the median activity cluster.
    quiesce_turns: int = 5
    # Safety valve: mature regardless once held this many turns, bounding
    # the hold-out cost for files that stay active for long stretches.
    max_hold_turns: int = 25
    # Only hold/mature Reads at least this large; small Reads are cached
    # immediately as before (holding them costs more than it saves).
    min_size_bytes: int = 2048


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
    - Use a conservative CompressionProfile bias for critical data
    - Set variance_threshold lower (1.5) to catch more change points
    """

    enabled: bool = True  # Enabled by default — sole tool-output compressor
    min_items_to_analyze: int = 5  # Don't analyze tiny arrays
    min_tokens_to_crush: int = 200  # Only crush if > N tokens
    variance_threshold: float = 2.0  # Std devs for change point detection
    uniqueness_threshold: float = 0.1  # Below this = nearly constant
    similarity_threshold: float = 0.8  # For clustering similar strings
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

    # Lossless-first dispatch: minimum byte-savings ratio for the lossless
    # Table/CSV compaction path to win over the lossy path. Must stay in
    # lockstep with the Rust default (smart_crusher config.rs) and the
    # transforms-level dataclass.
    lossless_min_savings_ratio: float = 0.15

    # Strict lossless mode. When True, lossless tabular compaction still
    # applies, but any path that would emit a CCR marker (lossy row-drop
    # OR opaque-blob offload) leaves the content uncompacted instead, so
    # the output is always marker-free and byte-recoverable. Mirrors the
    # Rust default. See also `CCRConfig` — with this on, no `<<ccr:…>>`
    # markers are produced regardless of CCR settings.
    lossless_only: bool = False

    # Compaction heuristics (mirror Rust CompactConfig). A field is "core"
    # if present in at least this fraction of rows; arrays whose key sets
    # are mostly non-core are bucketed by a discriminator instead.
    compaction_core_field_fraction: float = 0.8
    compaction_heterogeneous_core_ratio: float = 0.6
    compaction_max_flatten_inner_keys: int = 6
    compaction_min_buckets: int = 2
    compaction_max_buckets: int = 8


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
    4. RETRIEVE: If LLM needs more, it calls headroom_retrieve(hash) to get the full original back

    Benefits:
    - Zero-risk compression: worst case = LLM retrieves what it needs
    - Feedback loop: track what gets retrieved to improve compression
    - Network effect: retrieval patterns improve compression for all users

    GOTCHAS:
    - Cache has TTL (default 30 min) - retrieval fails after expiration
    - Memory usage: ~1KB per cached entry
    - Only works with array compression (not string truncation)
    """

    enabled: bool = True  # Enable CCR (cache + retrieval markers)
    store_max_entries: int = 1000  # Max entries in compression store
    # Session-scale TTL. The original 5-minute default predates agentic
    # sessions that routinely run 30+ minutes; an expired entry silently
    # converts "lossless with retrieval" into "lossy", so the TTL is the
    # weakest link in the no-accuracy-loss guarantee. Kept in lockstep
    # with Rust DEFAULT_TTL (crates/headroom-core/src/ccr/mod.rs) and
    # DEFAULT_CCR_TTL_SECONDS (cache/compression_store.py).
    store_ttl_seconds: int = 1800  # Cache TTL (30 minutes)
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

    # Deprecated compatibility argument. ContentRouter is always present
    # in the default pipeline; accepting this avoids breaking old config
    # constructors while keeping it out of runtime state.
    content_router_enabled: InitVar[bool | None] = None

    # Tool-result interceptors (ast-grep Read outline, etc.). Opt-in for now.
    # The legacy env alias and this typed request still obey the canary rollout gate.
    intercept_tool_results: bool = False

    # Immutable runtime rollout state. ``None`` is resolved once here so every
    # pipeline built from this config observes the same decisions even if the
    # process environment later changes.
    rollout: RolloutSnapshot | None = None

    # Debugging - opt-in diff artifact generation
    generate_diff_artifact: bool = False  # Enable to get detailed transform diffs

    # Canonical pipeline lifecycle extensions
    pipeline_extensions: list[Any] = field(default_factory=list)
    discover_pipeline_extensions: bool = True

    def __post_init__(self, content_router_enabled: bool | None = None) -> None:
        if self.rollout is None:
            requested = ("tool_result_interceptors",) if self.intercept_tool_results else ()
            self.rollout = resolve_rollout(requested=requested)

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
    reread_tokens: int = 0  # Tool results re-served after already appearing earlier
    # Subset of reread_tokens whose first serve was compressed away (CCR
    # marker left in its place) — re-reads attributable to over-compression
    # rather than agent behavior (#899). Excluded from total() because the
    # same tokens are already counted in reread_tokens.
    reread_compressed_tokens: int = 0

    def total(self) -> int:
        """Total waste tokens detected."""
        return (
            self.json_bloat_tokens
            + self.html_noise_tokens
            + self.base64_tokens
            + self.whitespace_tokens
            + self.dynamic_date_tokens
            + self.repetition_tokens
            + self.reread_tokens
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
            "reread": self.reread_tokens,
            "reread_compressed": self.reread_compressed_tokens,
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

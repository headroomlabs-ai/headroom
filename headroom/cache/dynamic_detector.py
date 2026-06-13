"""
Dynamic Content Detector for Cache Optimization.

This module provides a scalable, language-agnostic approach to detecting dynamic
content in prompts. Dynamic content (dates, prices, user data, session info) breaks
cache prefixes. By detecting and moving dynamic content to the end, we maximize
cache hits.

Design Philosophy:
    - NO HARDCODED PATTERNS for locale-specific content (no month names, etc.)
    - Structural detection: "Label: value" patterns where LABEL indicates dynamism
    - Entropy-based detection: High entropy = likely dynamic (UUIDs, tokens, hashes)
    - Universal patterns only: ISO 8601, UUIDs, Unix timestamps (truly universal)

Tiers (configurable, each adds latency):
    Tier 1: Regex (~0ms) - Structural patterns, universal formats, entropy-based
    Tier 2: NER (~5-10ms) - Named Entity Recognition for names, money, orgs
    Tier 3: Semantic (~20-50ms) - Embedding similarity to known dynamic patterns

Usage:
    from headroom.cache.dynamic_detector import DynamicContentDetector

    detector = DynamicContentDetector(tiers=["regex", "ner"])
    result = detector.detect("Session: abc123. User: John paid $500.")

    # result.spans = [
    #   DynamicSpan(text="Session: abc123", category="session", tier="regex", ...),
    #   DynamicSpan(text="John", category="person", tier="ner", ...),
    #   DynamicSpan(text="$500", category="money", tier="ner", ...),
    # ]
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from importlib.util import find_spec
from typing import Any, Literal

from headroom.models.config import ML_MODEL_DEFAULTS

# Optional ML dependencies are checked without importing them so this module
# stays cheap to import during proxy startup.
_SPACY_AVAILABLE = find_spec("spacy") is not None
_SENTENCE_TRANSFORMERS_AVAILABLE = (
    find_spec("numpy") is not None and find_spec("sentence_transformers") is not None
)


class DynamicCategory(str, Enum):
    """Categories of dynamic content."""

    # Tier 1: Structural/Regex detectable
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    UUID = "uuid"
    REQUEST_ID = "request_id"
    VERSION = "version"
    SESSION = "session"
    USER_DATA = "user_data"
    IDENTIFIER = "identifier"  # Generic high-entropy ID

    # Tier 2: NER detectable
    PERSON = "person"
    MONEY = "money"
    ORG = "org"
    LOCATION = "location"

    # Tier 3: Semantic
    VOLATILE = "volatile"  # Semantically detected as changing
    REALTIME = "realtime"

    # Fallback
    UNKNOWN = "unknown"


@dataclass
class DynamicSpan:
    """A span of dynamic content detected in text."""

    # The actual text matched
    text: str

    # Position in original content
    start: int
    end: int

    # What category of dynamic content
    category: DynamicCategory

    # Which tier detected it
    tier: Literal["regex", "ner", "semantic"]

    # Confidence score (0-1)
    confidence: float = 1.0

    # Additional metadata (pattern name, entity type, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Result of dynamic content detection."""

    # All detected spans
    spans: list[DynamicSpan]

    # Content with dynamic parts removed
    static_content: str

    # Content that was extracted (for reinsertion at end)
    dynamic_content: str

    # Which tiers were used
    tiers_used: list[str]

    # Processing time in milliseconds
    processing_time_ms: float = 0.0

    # Any warnings (e.g., "spaCy not available, skipping NER")
    warnings: list[str] = field(default_factory=list)


@dataclass
class DetectorConfig:
    """Configuration for the dynamic content detector."""

    # Which tiers to enable (order matters - later tiers can use earlier results)
    tiers: list[Literal["regex", "ner", "semantic"]] = field(default_factory=lambda: ["regex"])

    # Tier 1: Structural labels that indicate dynamic content
    # These are the KEY names that hint the VALUE is dynamic
    # Users can add domain-specific labels
    dynamic_labels: list[str] = field(
        default_factory=lambda: [
            # Time-related
            "date",
            "time",
            "timestamp",
            "datetime",
            "created",
            "updated",
            "modified",
            "expires",
            "last",
            "current",
            "today",
            "now",
            # Identifiers
            "id",
            "uuid",
            "guid",
            "session",
            "request",
            "trace",
            "span",
            "transaction",
            "correlation",
            "token",
            "key",
            "secret",
            # User-related
            "user",
            "username",
            "email",
            "name",
            "phone",
            "address",
            "customer",
            "client",
            "employee",
            "member",
            # System state
            "version",
            "build",
            "commit",
            "branch",
            "revision",
            "status",
            "state",
            "count",
            "total",
            "balance",
            "remaining",
            "load",
            "queue",
            "active",
            "pending",
            # Order/ticket related
            "order",
            "ticket",
            "case",
            "invoice",
            "reference",
        ]
    )

    # Tier 1: Custom regex patterns (user-provided)
    custom_patterns: list[tuple[str, DynamicCategory]] = field(default_factory=list)

    # Entropy threshold for detecting random strings (0-1 scale normalized)
    # Higher = more selective (only very random strings)
    entropy_threshold: float = 0.7

    # Minimum length for entropy-based detection
    min_entropy_length: int = 8

    # Tier 2: NER config
    spacy_model: str = field(default_factory=lambda: ML_MODEL_DEFAULTS.spacy)
    ner_entity_types: list[str] = field(
        default_factory=lambda: ["DATE", "TIME", "MONEY", "PERSON", "ORG", "GPE"]
    )

    # Tier 3: Semantic config
    embedding_model: str = field(default_factory=lambda: ML_MODEL_DEFAULTS.sentence_transformer)
    semantic_threshold: float = 0.7

    # General
    min_span_length: int = 2
    merge_overlapping: bool = True


def calculate_entropy(s: str) -> float:
    """
    Calculate Shannon entropy of a string, normalized to 0-1.

    Higher entropy = more random/unpredictable = likely dynamic.
    - "aaaaaaa" -> ~0 (low entropy, predictable)
    - "a1b2c3d4" -> ~0.7 (medium entropy)
    - "550e8400-e29b-41d4" -> ~0.9 (high entropy, random-looking)

    Returns:
        Normalized entropy (0-1). Higher = more likely dynamic.
    """
    if not s:
        return 0.0

    # Count character frequencies
    freq: dict[str, int] = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1

    # Calculate entropy
    length = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)

    # Normalize: max entropy for string of length n with k unique chars
    # is log2(min(n, alphabet_size)). We'll normalize by log2(length)
    # to get a 0-1 scale
    max_entropy = math.log2(length) if length > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


class RegexDetector:
    """
    Tier 1: Scalable pattern detection.

    Uses THREE strategies (no hardcoded month names!):
    1. Structural: "Label: value" patterns where label indicates dynamic content
    2. Universal: Truly universal formats (ISO 8601, UUID, Unix timestamps)
    3. Entropy: High-entropy strings (tokens, hashes, IDs)
    """

    # Universal patterns (these formats are language-agnostic)
    UNIVERSAL_PATTERNS = [
        # UUID - truly universal format
        (
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            DynamicCategory.UUID,
            "uuid",
        ),
        # ISO 8601 datetime (most universal date format)
        (
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
            DynamicCategory.DATETIME,
            "iso_datetime",
        ),
        # ISO 8601 date only
        (r"\d{4}-\d{2}-\d{2}(?!\d)", DynamicCategory.DATE, "iso_date"),
        # Unix timestamps (10-13 digits, but NOT within longer numbers)
        (r"(?<![0-9])\d{10,13}(?![0-9])", DynamicCategory.TIMESTAMP, "unix_timestamp"),
        # 24-hour time HH:MM:SS or HH:MM
        (
            r"(?<![0-9])\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM|am|pm))?(?![0-9])",
            DynamicCategory.TIME,
            "time",
        ),
        # Version numbers with v prefix (unambiguous)
        (r"\bv\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?", DynamicCategory.VERSION, "version"),
        # API key/token patterns (prefix + random string)
        (
            r"\b(?:sk|pk|api|key|token|bearer|auth)[-_][a-zA-Z0-9]{16,}",
            DynamicCategory.REQUEST_ID,
            "api_key",
        ),
        # Common prefixed IDs (req_, sess_, txn_, etc.)
        (r"\b[a-z]{2,6}_[a-zA-Z0-9]{8,}", DynamicCategory.REQUEST_ID, "prefixed_id"),
        # Hex strings of common ID lengths (32 = MD5, 40 = SHA1, 64 = SHA256)
        (r"\b[a-fA-F0-9]{32}\b", DynamicCategory.IDENTIFIER, "hex_32"),
        (r"\b[a-fA-F0-9]{40}\b", DynamicCategory.IDENTIFIER, "hex_40"),
        (r"\b[a-fA-F0-9]{64}\b", DynamicCategory.IDENTIFIER, "hex_64"),
        # JWT tokens (three base64 sections separated by dots)
        (
            r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
            DynamicCategory.REQUEST_ID,
            "jwt",
        ),
    ]

    def __init__(self, config: DetectorConfig):
        """Initialize regex detector."""
        self.config = config

        # Compile universal patterns
        self._universal_patterns: list[tuple[re.Pattern[str], DynamicCategory, str]] = [
            (re.compile(pattern), category, name)
            for pattern, category, name in self.UNIVERSAL_PATTERNS
        ]

        # Build structural pattern from dynamic labels
        # Pattern: "label" followed by separator then value
        labels_pattern = "|".join(re.escape(label) for label in config.dynamic_labels)
        self._structural_pattern = re.compile(
            rf"(?P<label>(?:{labels_pattern}))(?P<sep>\s*[:=]\s*|\s+)(?P<value>[^\n,;]+)",
            re.IGNORECASE,
        )

        # Compile custom patterns
        self._custom_patterns: list[tuple[re.Pattern[str], DynamicCategory]] = [
            (re.compile(pattern, re.IGNORECASE), category)
            for pattern, category in config.custom_patterns
        ]

    def detect(self, content: str) -> list[DynamicSpan]:
        """Detect dynamic content using structural, universal, and entropy detection."""
        spans: list[DynamicSpan] = []
        seen_ranges: set[tuple[int, int]] = set()

        # 1. Universal patterns first (most specific)
        for pattern, category, pattern_name in self._universal_patterns:
            for match in pattern.finditer(content):
                start, end = match.start(), match.end()
                if self._is_overlapping(start, end, seen_ranges):
                    continue
                if end - start < self.config.min_span_length:
                    continue

                spans.append(
                    DynamicSpan(
                        text=match.group(),
                        start=start,
                        end=end,
                        category=category,
                        tier="regex",
                        confidence=1.0,
                        metadata={"pattern": pattern_name, "method": "universal"},
                    )
                )
                seen_ranges.add((start, end))

        # 2. Structural detection: "Label: value" patterns
        for match in self._structural_pattern.finditer(content):
            # Get the full match range
            start, end = match.start(), match.end()

            # Skip if overlaps with universal patterns
            if self._is_overlapping(start, end, seen_ranges):
                continue

            label = match.group("label").lower()
            value = match.group("value").strip()

            # Determine category from label
            category = self._categorize_label(label)

            # Only add the value portion (keep label as static)
            value_start = match.start("value")
            value_end = match.end("value")

            # Skip if value is too short or empty
            if value_end - value_start < self.config.min_span_length:
                continue
            if not value.strip():
                continue

            spans.append(
                DynamicSpan(
                    text=value,
                    start=value_start,
                    end=value_end,
                    category=category,
                    tier="regex",
                    confidence=0.9,
                    metadata={"pattern": "structural", "method": "structural", "label": label},
                )
            )
            seen_ranges.add((value_start, value_end))

        # 3. Entropy-based detection for remaining potential IDs
        spans.extend(self._detect_high_entropy(content, seen_ranges))

        # 4. Custom patterns
        for pattern, category in self._custom_patterns:
            for match in pattern.finditer(content):
                start, end = match.start(), match.end()
                if self._is_overlapping(start, end, seen_ranges):
                    continue
                if end - start < self.config.min_span_length:
                    continue

                spans.append(
                    DynamicSpan(
                        text=match.group(),
                        start=start,
                        end=end,
                        category=category,
                        tier="regex",
                        confidence=0.8,
                        metadata={"pattern": "custom", "method": "custom"},
                    )
                )
                seen_ranges.add((start, end))

        return sorted(spans, key=lambda s: s.start)

    def _detect_high_entropy(
        self,
        content: str,
        seen_ranges: set[tuple[int, int]],
    ) -> list[DynamicSpan]:
        """
        Detect high-entropy strings that look like IDs/tokens.

        Finds alphanumeric sequences and checks their entropy.
        High entropy = likely random/generated = dynamic.
        """
        spans: list[DynamicSpan] = []

        # Find alphanumeric sequences (potential IDs)
        # Must be at least min_entropy_length chars, mix of letters/numbers
        pattern = re.compile(r"\b[a-zA-Z0-9_-]{8,}\b")

        for match in pattern.finditer(content):
            start, end = match.start(), match.end()
            text = match.group()

            # Skip if already detected
            if self._is_overlapping(start, end, seen_ranges):
                continue

            # Skip if too short
            if len(text) < self.config.min_entropy_length:
                continue

            # Skip if all letters or all numbers (not random-looking)
            if text.isalpha() or text.isdigit():
                continue

            # Skip common words that might look like IDs
            if text.lower() in {"username", "password", "localhost", "undefined"}:
                continue

            # Calculate entropy
            entropy = calculate_entropy(text)

            if entropy >= self.config.entropy_threshold:
                spans.append(
                    DynamicSpan(
                        text=text,
                        start=start,
                        end=end,
                        category=DynamicCategory.IDENTIFIER,
                        tier="regex",
                        confidence=entropy,  # Use entropy as confidence
                        metadata={"pattern": "entropy", "method": "entropy", "entropy": entropy},
                    )
                )
                seen_ranges.add((start, end))

        return spans

    def _is_overlapping(
        self,
        start: int,
        end: int,
        seen_ranges: set[tuple[int, int]],
    ) -> bool:
        """Check if range overlaps with any existing range."""
        return any(not (end <= s or start >= e) for s, e in seen_ranges)

    def _categorize_label(self, label: str) -> DynamicCategory:
        """Categorize based on the label name."""
        label = label.lower()

        # Time-related
        if label in {"date", "datetime", "created", "updated", "modified", "expires", "today"}:
            return DynamicCategory.DATE
        if label in {"time", "timestamp", "now"}:
            return DynamicCategory.TIMESTAMP
        if label == "current":
            return DynamicCategory.DATETIME

        # Identifiers
        if label in {"id", "uuid", "guid"}:
            return DynamicCategory.UUID
        if label in {"session", "request", "trace", "span", "transaction", "correlation"}:
            return DynamicCategory.SESSION
        if label in {"token", "key", "secret"}:
            return DynamicCategory.REQUEST_ID

        # User-related
        if label in {
            "user",
            "username",
            "email",
            "name",
            "phone",
            "address",
            "customer",
            "client",
            "employee",
            "member",
        }:
            return DynamicCategory.USER_DATA

        # System state
        if label in {"version", "build", "commit", "branch", "revision"}:
            return DynamicCategory.VERSION
        if label in {
            "status",
            "state",
            "count",
            "total",
            "balance",
            "remaining",
            "load",
            "queue",
            "active",
            "pending",
        }:
            return DynamicCategory.VOLATILE

        # Order/ticket
        if label in {"order", "ticket", "case", "invoice", "reference"}:
            return DynamicCategory.REQUEST_ID

        return DynamicCategory.UNKNOWN


class NERDetector:
    """Tier 2: spaCy-based Named Entity Recognition."""

    # Map spaCy entity types to our categories
    ENTITY_MAP = {
        "DATE": DynamicCategory.DATE,
        "TIME": DynamicCategory.TIME,
        "MONEY": DynamicCategory.MONEY,
        "PERSON": DynamicCategory.PERSON,
        "ORG": DynamicCategory.ORG,
        "GPE": DynamicCategory.LOCATION,  # Geo-Political Entity
        "LOC": DynamicCategory.LOCATION,
        "FAC": DynamicCategory.LOCATION,  # Facility
        "CARDINAL": DynamicCategory.UNKNOWN,  # Numbers
        "ORDINAL": DynamicCategory.UNKNOWN,
    }

    def __init__(self, config: DetectorConfig):
        """Initialize NER detector, loading spaCy model."""
        self.config = config
        self._nlp = None
        self._load_error: str | None = None

        if not _SPACY_AVAILABLE:
            self._load_error = (
                "spaCy not installed. Install with: "
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
            return

        try:
            # Use centralized registry for shared model instances
            from headroom.models.ml_models import MLModelRegistry

            self._nlp = MLModelRegistry.get_spacy(config.spacy_model)
        except ImportError:
            self._load_error = (
                "spaCy not installed. Install with: "
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        except OSError:
            self._load_error = (
                f"spaCy model '{config.spacy_model}' not found. "
                f"Install with: python -m spacy download {config.spacy_model}"
            )

    @property
    def is_available(self) -> bool:
        """Check if NER is available."""
        return self._nlp is not None

    def detect(
        self,
        content: str,
        existing_spans: list[DynamicSpan] | None = None,
    ) -> tuple[list[DynamicSpan], str | None]:
        """
        Detect dynamic content using NER.

        Args:
            content: Text to analyze.
            existing_spans: Spans already detected (to avoid duplicates).

        Returns:
            Tuple of (new_spans, warning_message).
        """
        if not self.is_available:
            return [], self._load_error

        # Get existing ranges to avoid duplicates
        existing_ranges = set()
        if existing_spans:
            existing_ranges = {(s.start, s.end) for s in existing_spans}

        doc = self._nlp(content)  # type: ignore[misc]
        spans: list[DynamicSpan] = []

        for ent in doc.ents:
            # Skip entity types we don't care about
            if ent.label_ not in self.config.ner_entity_types:
                continue

            # Skip if already detected by regex
            if (ent.start_char, ent.end_char) in existing_ranges:
                continue

            # Check for overlap with existing spans
            overlaps = any(
                not (ent.end_char <= s or ent.start_char >= e) for s, e in existing_ranges
            )
            if overlaps:
                continue

            # Map to our category
            category = self.ENTITY_MAP.get(ent.label_, DynamicCategory.UNKNOWN)

            # Skip unknown categories
            if category == DynamicCategory.UNKNOWN:
                continue

            spans.append(
                DynamicSpan(
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    category=category,
                    tier="ner",
                    confidence=0.9,
                    metadata={"entity_type": ent.label_},
                )
            )
            existing_ranges.add((ent.start_char, ent.end_char))

        return sorted(spans, key=lambda s: s.start), None


class SemanticDetector:
    """Tier 3: Embedding-based semantic detection."""

    # Known phrases that indicate dynamic content
    # These are SEMANTIC patterns, not literal strings to match
    DYNAMIC_EXEMPLARS = [
        # Time-sensitive
        "The current date is",
        "As of today",
        "Updated on",
        "Last refreshed",
        "Real-time data",
        "Live prices",
        "Current stock price",
        # Session-specific
        "Your session ID",
        "Your account balance",
        "Your recent orders",
        "Your conversation history",
        # User-specific
        "Hello [user]",
        "Dear customer",
        "Your name is",
        # System state
        "Server status",
        "System load",
        "Queue length",
        "Active users",
    ]

    def __init__(self, config: DetectorConfig):
        """Initialize semantic detector with embedding model."""
        self.config = config
        self._model = None
        self._exemplar_embeddings = None
        self._load_error: str | None = None

        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            self._load_error = (
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return

        try:
            # Use centralized registry for shared model instances
            from headroom.models.ml_models import MLModelRegistry

            self._model = MLModelRegistry.get_sentence_transformer(config.embedding_model)
            # Pre-compute exemplar embeddings
            self._exemplar_embeddings = self._model.encode(
                self.DYNAMIC_EXEMPLARS,
                convert_to_numpy=True,
            )
        except ImportError:
            self._load_error = (
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            self._load_error = f"Failed to load embedding model: {e}"

    @property
    def is_available(self) -> bool:
        """Check if semantic detection is available."""
        return self._model is not None

    def detect(
        self,
        content: str,
        existing_spans: list[DynamicSpan] | None = None,
    ) -> tuple[list[DynamicSpan], str | None]:
        """
        Detect dynamic content using semantic similarity.

        Splits content into sentences and checks each against known
        dynamic patterns using embedding similarity.

        Args:
            content: Text to analyze.
            existing_spans: Spans already detected (to avoid duplicates).

        Returns:
            Tuple of (new_spans, warning_message).
        """
        if not self.is_available:
            return [], self._load_error

        # Simple sentence splitting (could use spaCy if available)
        sentences = self._split_sentences(content)
        spans: list[DynamicSpan] = []

        # Get existing ranges
        existing_ranges = set()
        if existing_spans:
            existing_ranges = {(s.start, s.end) for s in existing_spans}

        # Encode all sentences
        if not sentences:
            return [], None

        try:
            import numpy as np
        except ImportError:
            return [], "numpy not installed. Install with: pip install numpy"

        sentence_texts = [s[0] for s in sentences]
        sentence_embeddings = self._model.encode(  # type: ignore[union-attr]
            sentence_texts,
            convert_to_numpy=True,
        )

        # Compute similarities
        similarities = np.dot(sentence_embeddings, self._exemplar_embeddings.T)

        for i, (text, start, end) in enumerate(sentences):
            # Get max similarity to any exemplar
            max_sim = float(np.max(similarities[i]))

            if max_sim < self.config.semantic_threshold:
                continue

            # Check overlap with existing spans
            overlaps = any(not (end <= s or start >= e) for s, e in existing_ranges)
            if overlaps:
                continue

            # Find which exemplar matched best
            best_exemplar_idx = int(np.argmax(similarities[i]))
            best_exemplar = self.DYNAMIC_EXEMPLARS[best_exemplar_idx]

            # Determine category based on exemplar
            category = self._categorize_exemplar(best_exemplar)

            spans.append(
                DynamicSpan(
                    text=text,
                    start=start,
                    end=end,
                    category=category,
                    tier="semantic",
                    confidence=max_sim,
                    metadata={
                        "matched_exemplar": best_exemplar,
                        "similarity": max_sim,
                    },
                )
            )
            existing_ranges.add((start, end))

        return sorted(spans, key=lambda s: s.start), None

    def _split_sentences(self, content: str) -> list[tuple[str, int, int]]:
        """Split content into sentences with positions."""
        sentences: list[tuple[str, int, int]] = []
        pattern = r"[^.!?\n]+[.!?\n]?"
        for match in re.finditer(pattern, content):
            text = match.group().strip()
            if len(text) > 10:
                sentences.append((text, match.start(), match.end()))
        return sentences

    def _categorize_exemplar(self, exemplar: str) -> DynamicCategory:
        """Categorize based on which exemplar matched."""
        exemplar_lower = exemplar.lower()

        if any(w in exemplar_lower for w in ["date", "today", "updated", "refreshed"]):
            return DynamicCategory.DATE
        elif any(w in exemplar_lower for w in ["price", "stock", "live", "real-time"]):
            return DynamicCategory.REALTIME
        elif any(w in exemplar_lower for w in ["session", "account", "your"]):
            return DynamicCategory.SESSION
        elif any(w in exemplar_lower for w in ["status", "load", "queue", "active"]):
            return DynamicCategory.VOLATILE
        else:
            return DynamicCategory.VOLATILE


class DynamicContentDetector:
    """
    Unified dynamic content detector with tiered detection.

    Key Design Principles:
    - NO hardcoded locale-specific patterns (no month names)
    - Structural detection: Labels indicate what's dynamic
    - Universal patterns: ISO 8601, UUIDs, Unix timestamps
    - Entropy-based: High entropy = random/generated = dynamic

    Usage:
        # Fast mode (regex only - structural + universal + entropy)
        detector = DynamicContentDetector(DetectorConfig(tiers=["regex"]))

        # Balanced mode (regex + NER for names/money)
        detector = DynamicContentDetector(DetectorConfig(tiers=["regex", "ner"]))

        # Full mode (all tiers)
        detector = DynamicContentDetector(DetectorConfig(
            tiers=["regex", "ner", "semantic"]
        ))

        result = detector.detect("Session: abc123. User: John paid $500.")
    """

    def __init__(self, config: DetectorConfig | None = None):
        """Initialize detector with configuration."""
        self.config = config or DetectorConfig()

        # Initialize detectors based on enabled tiers
        self._regex_detector: RegexDetector | None = None
        self._ner_detector: NERDetector | None = None
        self._semantic_detector: SemanticDetector | None = None

        if "regex" in self.config.tiers:
            self._regex_detector = RegexDetector(self.config)

        if "ner" in self.config.tiers:
            self._ner_detector = NERDetector(self.config)

        if "semantic" in self.config.tiers:
            self._semantic_detector = SemanticDetector(self.config)

    def detect(self, content: str) -> DetectionResult:
        """
        Detect dynamic content in text.

        Runs enabled tiers in order, accumulating spans.
        Each tier can see what previous tiers detected.

        Args:
            content: Text to analyze.

        Returns:
            DetectionResult with spans, static/dynamic content split, etc.
        """
        import time

        start_time = time.perf_counter()

        all_spans: list[DynamicSpan] = []
        tiers_used: list[str] = []
        warnings: list[str] = []

        # Tier 1: Regex (structural + universal + entropy)
        if self._regex_detector:
            regex_spans = self._regex_detector.detect(content)
            all_spans.extend(regex_spans)
            tiers_used.append("regex")

        # Tier 2: NER
        if self._ner_detector:
            ner_spans, ner_warning = self._ner_detector.detect(content, all_spans)
            all_spans.extend(ner_spans)
            if ner_warning:
                warnings.append(ner_warning)
            elif ner_spans or self._ner_detector.is_available:
                tiers_used.append("ner")

        # Tier 3: Semantic
        if self._semantic_detector:
            sem_spans, sem_warning = self._semantic_detector.detect(content, all_spans)
            all_spans.extend(sem_spans)
            if sem_warning:
                warnings.append(sem_warning)
            elif sem_spans or self._semantic_detector.is_available:
                tiers_used.append("semantic")

        # Sort by position
        all_spans = sorted(all_spans, key=lambda s: s.start)

        # Build static and dynamic content
        static_content, dynamic_content = self._split_content(content, all_spans)

        processing_time = (time.perf_counter() - start_time) * 1000

        return DetectionResult(
            spans=all_spans,
            static_content=static_content,
            dynamic_content=dynamic_content,
            tiers_used=tiers_used,
            processing_time_ms=processing_time,
            warnings=warnings,
        )

    def _split_content(
        self,
        content: str,
        spans: list[DynamicSpan],
    ) -> tuple[str, str]:
        """Split content into static and dynamic parts."""
        if not spans:
            return content, ""

        static = content
        dynamic_parts: list[str] = []

        for span in reversed(spans):
            dynamic_parts.append(span.text)
            static = static[: span.start] + static[span.end :]

        static = self._clean_static_content(static)
        dynamic_parts.reverse()
        dynamic = "\n".join(dynamic_parts)

        return static, dynamic

    def _clean_static_content(self, content: str) -> str:
        """Clean up static content after span removal."""
        lines = content.split("\n")
        cleaned_lines: list[str] = []
        prev_blank = False

        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned_lines.append(line.rstrip())
            prev_blank = is_blank

        return "\n".join(cleaned_lines).strip()

    @property
    def available_tiers(self) -> list[str]:
        """Get list of actually available tiers (dependencies installed)."""
        available = []

        if self._regex_detector:
            available.append("regex")

        if self._ner_detector and self._ner_detector.is_available:
            available.append("ner")

        if self._semantic_detector and self._semantic_detector.is_available:
            available.append("semantic")

        return available


# Convenience function
def detect_dynamic_content(
    content: str,
    tiers: list[Literal["regex", "ner", "semantic"]] | None = None,
) -> DetectionResult:
    """
    Detect dynamic content in text.

    Convenience function that creates a detector with specified tiers.

    Args:
        content: Text to analyze.
        tiers: Which tiers to use. Default: ["regex"] for speed.

    Returns:
        DetectionResult with detected spans and split content.

    Example:
        >>> result = detect_dynamic_content(
        ...     "Session: abc123xyz. User: John paid $500.",
        ...     tiers=["regex", "ner"]
        ... )
        >>> print(result.static_content)
        >>> print(result.dynamic_content)
    """
    config = DetectorConfig(tiers=tiers or ["regex"])
    detector = DynamicContentDetector(config)
    return detector.detect(content)

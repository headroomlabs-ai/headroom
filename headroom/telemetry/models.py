"""Data models for privacy-preserving telemetry.

These models capture PATTERNS, not DATA. We never store actual values,
user queries, or identifiable information.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

# Type alias for field semantic types
FieldSemanticType = Literal[
    "unknown",
    "identifier",
    "error_indicator",
    "score",
    "status",
    "temporal",
    "content",
]


@dataclass
class FieldDistribution:
    """Statistics about a field's distribution (no actual values).

    This captures the SHAPE of the data, not the data itself.
    """

    field_name_hash: str  # SHA256[:8] of field name (anonymized)
    field_type: Literal["string", "numeric", "boolean", "array", "object", "null", "mixed"]

    # String field statistics
    avg_length: float | None = None
    unique_ratio: float | None = None  # 0.0 = constant, 1.0 = all unique
    entropy: float | None = None  # Shannon entropy normalized to [0, 1]
    looks_like_id: bool = False  # High entropy + consistent format

    # Numeric field statistics
    has_variance: bool = False
    variance_bucket: Literal["zero", "low", "medium", "high"] | None = None
    has_negative: bool = False
    is_integer: bool = True
    has_outliers: bool = False  # Values > 2σ from mean

    # Array field statistics
    avg_array_length: float | None = None

    # Derived insights
    is_likely_score: bool = False  # Monotonic, bounded, high variance
    is_likely_timestamp: bool = False  # Sequential, numeric, consistent intervals
    is_likely_status: bool = False  # Low cardinality categorical

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "field_name_hash": self.field_name_hash,
            "field_type": self.field_type,
            "avg_length": self.avg_length,
            "unique_ratio": self.unique_ratio,
            "entropy": self.entropy,
            "looks_like_id": self.looks_like_id,
            "has_variance": self.has_variance,
            "variance_bucket": self.variance_bucket,
            "has_negative": self.has_negative,
            "is_integer": self.is_integer,
            "has_outliers": self.has_outliers,
            "avg_array_length": self.avg_array_length,
            "is_likely_score": self.is_likely_score,
            "is_likely_timestamp": self.is_likely_timestamp,
            "is_likely_status": self.is_likely_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldDistribution:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ToolSignature:
    """Anonymized signature of a tool's output structure.

    This identifies SIMILAR tools across users without revealing tool names.
    Two tools with the same field structure will have the same signature.
    """

    # Structural hash (based on field types and names)
    # MEDIUM FIX #15: Uses SHA256[:24] (96 bits) for better collision resistance
    structure_hash: str  # SHA256[:24] of sorted field names + types

    # Schema characteristics
    field_count: int
    has_nested_objects: bool
    has_arrays: bool
    max_depth: int

    # Field type distribution
    string_field_count: int = 0
    numeric_field_count: int = 0
    boolean_field_count: int = 0
    array_field_count: int = 0
    object_field_count: int = 0

    # Pattern indicators (without revealing actual field names)
    has_id_like_field: bool = False
    has_score_like_field: bool = False
    has_timestamp_like_field: bool = False
    has_status_like_field: bool = False
    has_error_like_field: bool = False
    has_message_like_field: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "structure_hash": self.structure_hash,
            "field_count": self.field_count,
            "has_nested_objects": self.has_nested_objects,
            "has_arrays": self.has_arrays,
            "max_depth": self.max_depth,
            "string_field_count": self.string_field_count,
            "numeric_field_count": self.numeric_field_count,
            "boolean_field_count": self.boolean_field_count,
            "array_field_count": self.array_field_count,
            "object_field_count": self.object_field_count,
            "has_id_like_field": self.has_id_like_field,
            "has_score_like_field": self.has_score_like_field,
            "has_timestamp_like_field": self.has_timestamp_like_field,
            "has_status_like_field": self.has_status_like_field,
            "has_error_like_field": self.has_error_like_field,
            "has_message_like_field": self.has_message_like_field,
        }

    @staticmethod
    def _calculate_depth(value: Any, current_depth: int = 1, max_depth_limit: int = 10) -> int:
        """Recursively calculate the depth of a nested structure.

        MEDIUM FIX #12: Actually calculate max_depth instead of hardcoding 1.
        """
        if current_depth >= max_depth_limit:
            return current_depth  # Prevent infinite recursion

        if isinstance(value, dict):
            if not value:
                return current_depth
            return max(
                ToolSignature._calculate_depth(v, current_depth + 1, max_depth_limit)
                for v in value.values()
            )
        elif isinstance(value, list):
            if not value:
                return current_depth
            # Sample first few items in arrays to avoid O(n) traversal
            sample_items = value[:3]
            return max(
                ToolSignature._calculate_depth(item, current_depth + 1, max_depth_limit)
                for item in sample_items
            )
        else:
            return current_depth

    @staticmethod
    def _matches_pattern(
        key_lower: str, patterns: list[str], original_key: str | None = None
    ) -> bool:
        """Check if key matches patterns using word boundary matching.

        MEDIUM FIX #14: Prevent false positives like "hidden" matching "id".
        Uses word boundary logic: pattern must be at start/end or surrounded by
        non-alphanumeric characters (underscore, hyphen, or boundary).

        Args:
            key_lower: The field name in lowercase
            patterns: List of patterns to match against
            original_key: The original field name (for camelCase detection)
        """
        import re

        for pattern in patterns:
            # Exact match
            if key_lower == pattern:
                return True

            # Pattern at start with delimiter: "id_something" or "id-something"
            if key_lower.startswith(pattern + "_") or key_lower.startswith(pattern + "-"):
                return True

            # Pattern at end with delimiter: "user_id" or "user-id"
            if key_lower.endswith("_" + pattern) or key_lower.endswith("-" + pattern):
                return True

            # Pattern in middle with delimiters: "some_id_field"
            if f"_{pattern}_" in key_lower or f"-{pattern}-" in key_lower:
                return True
            if f"_{pattern}-" in key_lower or f"-{pattern}_" in key_lower:
                return True

            # camelCase detection: Look for capitalized pattern in original key
            # e.g., "userId" should match "id" (as "Id")
            if original_key:
                # Pattern capitalized (e.g., "Id" for "id")
                cap_pattern = pattern.capitalize()
                # Look for capital letter at start of pattern, preceded by lowercase
                camel_regex = rf"(?<=[a-z]){re.escape(cap_pattern)}(?=[A-Z]|$)"
                if re.search(camel_regex, original_key):
                    return True

        return False

    @classmethod
    def from_items(cls, items: list[dict[str, Any]]) -> ToolSignature:
        """Create signature from sample items."""
        if not items:
            # HIGH FIX: Generate unique hash for empty outputs to prevent
            # different tools' empty responses from colliding into one pattern.
            # Use a random component to ensure uniqueness across tool types.
            import uuid

            # MEDIUM FIX #15: Use 24 chars (96 bits) instead of 16 (64 bits) to reduce collision risk
            empty_hash = hashlib.sha256(f"empty:{uuid.uuid4()}".encode()).hexdigest()[:24]
            return cls(
                structure_hash=empty_hash,
                field_count=0,
                has_nested_objects=False,
                has_arrays=False,
                max_depth=0,
            )

        # MEDIUM FIX #13: Analyze multiple items (up to 5) to get representative structure
        # This catches cases where items have varying schemas
        sample_items = items[:5] if len(items) >= 5 else items

        # Merge field info from all sampled items
        all_fields: dict[str, set[str]] = {}  # field_name -> set of types seen
        for item in sample_items:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if key not in all_fields:
                    all_fields[key] = set()
                # Determine type
                if isinstance(value, str):
                    all_fields[key].add("string")
                elif isinstance(value, bool):
                    all_fields[key].add("boolean")
                elif isinstance(value, (int, float)):
                    all_fields[key].add("numeric")
                elif isinstance(value, list):
                    all_fields[key].add("array")
                elif isinstance(value, dict):
                    all_fields[key].add("object")
                else:
                    all_fields[key].add("null")

        # Build field_info with most common type per field
        field_info: list[tuple[str, str]] = []
        string_count = 0
        numeric_count = 0
        boolean_count = 0
        array_count = 0
        object_count = 0
        has_nested = False
        has_arrays = False

        # MEDIUM FIX #12: Calculate actual max_depth from sampled items
        max_depth = 1
        for item in sample_items:
            if isinstance(item, dict):
                item_depth = cls._calculate_depth(item)
                max_depth = max(max_depth, item_depth)

        # Pattern detection (heuristic field name matching)
        has_id = False
        has_score = False
        has_timestamp = False
        has_status = False
        has_error = False
        has_message = False

        for key, types in all_fields.items():
            key_lower = key.lower()

            # Use most specific type if multiple seen (prefer non-null)
            types_no_null = types - {"null"}
            if len(types_no_null) == 1:
                field_type = types_no_null.pop()
            elif len(types_no_null) > 1:
                # Multiple types seen - mark as mixed but pick one for counting
                # Priority: object > array > string > numeric > boolean
                for t in ["object", "array", "string", "numeric", "boolean"]:
                    if t in types_no_null:
                        field_type = t
                        break
                else:
                    field_type = "mixed"
            elif types:
                field_type = types.pop()  # Only null seen
            else:
                field_type = "null"

            # Count field types
            if field_type == "string":
                string_count += 1
            elif field_type == "boolean":
                boolean_count += 1
            elif field_type == "numeric":
                numeric_count += 1
            elif field_type == "array":
                array_count += 1
                has_arrays = True
            elif field_type == "object":
                object_count += 1
                has_nested = True

            field_info.append((key, field_type))

            # MEDIUM FIX #14: Pattern detection with word boundary matching
            # Prevents false positives like "hidden" matching "id"
            # Pass original key for camelCase detection
            if cls._matches_pattern(key_lower, ["id", "uuid", "guid"], key) or key_lower.endswith(
                "key"
            ):
                has_id = True
            if cls._matches_pattern(
                key_lower, ["score", "rank", "rating", "relevance", "priority"], key
            ):
                has_score = True
            if (
                cls._matches_pattern(key_lower, ["time", "date", "timestamp"], key)
                or key_lower.endswith("_at")
                or key_lower in ["created", "updated"]
            ):
                has_timestamp = True
            if cls._matches_pattern(key_lower, ["status", "state"], key) or key_lower in [
                "level",
                "type",
                "kind",
            ]:
                has_status = True
            if cls._matches_pattern(key_lower, ["error", "exception", "fail", "warning"], key):
                has_error = True
            if cls._matches_pattern(
                key_lower, ["message", "msg", "text", "content", "body", "description"], key
            ):
                has_message = True

        # Create structure hash
        # MEDIUM FIX #15: Use 24 chars (96 bits) instead of 16 (64 bits) for collision resistance
        sorted_fields = sorted(field_info)
        hash_input = json.dumps(sorted_fields, sort_keys=True)
        structure_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:24]

        return cls(
            structure_hash=structure_hash,
            field_count=len(field_info),
            has_nested_objects=has_nested,
            has_arrays=has_arrays,
            max_depth=max_depth,
            string_field_count=string_count,
            numeric_field_count=numeric_count,
            boolean_field_count=boolean_count,
            array_field_count=array_count,
            object_field_count=object_count,
            has_id_like_field=has_id,
            has_score_like_field=has_score,
            has_timestamp_like_field=has_timestamp,
            has_status_like_field=has_status,
            has_error_like_field=has_error,
            has_message_like_field=has_message,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolSignature:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FieldSemantics:
    """Learned semantics for a field based on retrieval patterns.

    This is the evolution of TOIN - we learn WHAT fields mean
    from HOW users retrieve them. No hardcoded patterns, no assumptions.

    Learning process:
    1. User retrieves items where field X has value Y
    2. TOIN records: field_hash, value_hash, retrieval context
    3. After N retrievals, TOIN infers: "This field behaves like an error indicator"
    4. SmartCrusher uses this learned signal (O(1) lookup, zero latency)

    Privacy: All field names and values are hashed (SHA256[:8]).
    """

    field_hash: str  # SHA256[:8] of field name

    # Inferred semantic type (learned from retrieval patterns, NOT hardcoded)
    # These are behavioral categories, not syntactic patterns:
    # - "identifier": Users query by exact value (e.g., "show me item X")
    # - "error_indicator": Users retrieve when value != most common value
    # - "score": Users retrieve top-N by this field
    # - "status": Low cardinality, specific values trigger retrieval
    # - "temporal": Users query by time ranges
    # - "content": Users do text search on this field
    inferred_type: FieldSemanticType = "unknown"

    confidence: float = 0.0  # 0.0 = no data, 1.0 = high confidence

    # Value patterns (all hashed for privacy)
    # important_value_hashes: values that triggered retrieval
    # default_value_hash: most common value (probably NOT important)
    important_value_hashes: list[str] = field(default_factory=list)
    default_value_hash: str | None = None
    value_retrieval_frequency: dict[str, int] = field(default_factory=dict)  # value_hash -> count

    # Value statistics (for inferring type)
    total_unique_values_seen: int = 0
    total_values_seen: int = 0
    most_common_value_frequency: float = 0.0  # Fraction of items with most common value

    # Query patterns (anonymized)
    # Tracks HOW users query this field (equals, not-equals, greater-than, etc.)
    query_operator_frequency: dict[str, int] = field(default_factory=dict)  # operator -> count

    # Learning metadata
    retrieval_count: int = 0
    compression_count: int = 0  # How many times we've seen this field in compression
    last_updated: float = 0.0

    # Bounds for memory management
    MAX_IMPORTANT_VALUES: int = 50
    MAX_VALUE_FREQUENCY_ENTRIES: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "field_hash": self.field_hash,
            "inferred_type": self.inferred_type,
            "confidence": self.confidence,
            "important_value_hashes": self.important_value_hashes[: self.MAX_IMPORTANT_VALUES],
            "default_value_hash": self.default_value_hash,
            "value_retrieval_frequency": dict(
                sorted(
                    self.value_retrieval_frequency.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[: self.MAX_VALUE_FREQUENCY_ENTRIES]
            ),
            "total_unique_values_seen": self.total_unique_values_seen,
            "total_values_seen": self.total_values_seen,
            "most_common_value_frequency": self.most_common_value_frequency,
            "query_operator_frequency": self.query_operator_frequency,
            "retrieval_count": self.retrieval_count,
            "compression_count": self.compression_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldSemantics:
        """Create from dictionary."""
        # Filter to valid fields only
        valid_fields = {
            "field_hash",
            "inferred_type",
            "confidence",
            "important_value_hashes",
            "default_value_hash",
            "value_retrieval_frequency",
            "total_unique_values_seen",
            "total_values_seen",
            "most_common_value_frequency",
            "query_operator_frequency",
            "retrieval_count",
            "compression_count",
            "last_updated",
        }
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def record_retrieval_value(self, value_hash: str, operator: str = "=") -> None:
        """Record that a value was retrieved for this field.

        Args:
            value_hash: SHA256[:8] hash of the retrieved value.
            operator: Query operator used ("=", "!=", ">", "<", "contains", etc.)
        """
        import time

        self.retrieval_count += 1
        self.last_updated = time.time()

        # Track value frequency
        self.value_retrieval_frequency[value_hash] = (
            self.value_retrieval_frequency.get(value_hash, 0) + 1
        )

        # Bound the frequency dict
        if len(self.value_retrieval_frequency) > self.MAX_VALUE_FREQUENCY_ENTRIES:
            sorted_items = sorted(
                self.value_retrieval_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )[: self.MAX_VALUE_FREQUENCY_ENTRIES]
            self.value_retrieval_frequency = dict(sorted_items)

        # Track important values (values that get retrieved)
        if value_hash not in self.important_value_hashes:
            self.important_value_hashes.append(value_hash)
            if len(self.important_value_hashes) > self.MAX_IMPORTANT_VALUES:
                # Keep most frequently retrieved values
                self.important_value_hashes = sorted(
                    self.important_value_hashes,
                    key=lambda v: self.value_retrieval_frequency.get(v, 0),
                    reverse=True,
                )[: self.MAX_IMPORTANT_VALUES]

        # Track query operators
        self.query_operator_frequency[operator] = self.query_operator_frequency.get(operator, 0) + 1

    def record_compression_stats(
        self,
        unique_values: int,
        total_values: int,
        most_common_value_hash: str | None,
        most_common_frequency: float,
    ) -> None:
        """Record statistics from compression for type inference.

        Args:
            unique_values: Number of unique values seen for this field.
            total_values: Total number of items with this field.
            most_common_value_hash: Hash of the most common value.
            most_common_frequency: Fraction of items with the most common value.
        """
        import time

        self.compression_count += 1
        self.last_updated = time.time()

        # Update rolling statistics
        n = self.compression_count
        self.total_unique_values_seen = int(
            (self.total_unique_values_seen * (n - 1) + unique_values) / n
        )
        self.total_values_seen = int((self.total_values_seen * (n - 1) + total_values) / n)
        self.most_common_value_frequency = (
            self.most_common_value_frequency * (n - 1) + most_common_frequency
        ) / n

        # Track default value (most common)
        if most_common_value_hash and most_common_frequency > 0.5:
            self.default_value_hash = most_common_value_hash

    def infer_type(self) -> None:
        """Infer semantic type from accumulated statistics.

        This is the learning algorithm - purely data-driven, no hardcoded patterns.
        """
        # Need minimum data to infer
        min_retrievals = 3
        min_compressions = 2

        if self.retrieval_count < min_retrievals or self.compression_count < min_compressions:
            self.inferred_type = "unknown"
            self.confidence = 0.0
            return

        # Calculate metrics
        uniqueness_ratio = self.total_unique_values_seen / max(1, self.total_values_seen)
        has_dominant_default = self.most_common_value_frequency > 0.7
        retrieval_diversity = len(self.value_retrieval_frequency) / max(1, self.retrieval_count)

        # Check query operator patterns
        total_ops = sum(self.query_operator_frequency.values())
        equals_ratio = self.query_operator_frequency.get("=", 0) / max(1, total_ops)
        range_ratio = (
            self.query_operator_frequency.get(">", 0)
            + self.query_operator_frequency.get("<", 0)
            + self.query_operator_frequency.get(">=", 0)
            + self.query_operator_frequency.get("<=", 0)
        ) / max(1, total_ops)
        contains_ratio = self.query_operator_frequency.get("contains", 0) / max(1, total_ops)

        # Inference logic (data-driven, no field name patterns)
        inferred: FieldSemanticType = "unknown"
        confidence = 0.0

        # IDENTIFIER: High uniqueness + exact match queries
        if uniqueness_ratio > 0.8 and equals_ratio > 0.7:
            inferred = "identifier"
            confidence = min(0.9, uniqueness_ratio * equals_ratio)

        # ERROR_INDICATOR: Has dominant default + retrievals are for non-default values
        elif has_dominant_default and self.default_value_hash:
            # Check if retrieved values are different from default
            default_retrieval_count = self.value_retrieval_frequency.get(self.default_value_hash, 0)
            non_default_retrieval_ratio = 1 - (
                default_retrieval_count / max(1, self.retrieval_count)
            )
            if non_default_retrieval_ratio > 0.7:
                inferred = "error_indicator"
                confidence = min(
                    0.9, non_default_retrieval_ratio * self.most_common_value_frequency
                )

        # STATUS: Low uniqueness + specific values retrieved
        elif uniqueness_ratio < 0.2 and retrieval_diversity < 0.5:
            inferred = "status"
            confidence = min(0.85, (1 - uniqueness_ratio) * (1 - retrieval_diversity))

        # SCORE: Range queries or sorted access patterns
        elif range_ratio > 0.5:
            inferred = "score"
            confidence = min(0.85, range_ratio)

        # TEMPORAL: Range queries + high uniqueness (likely timestamps)
        elif range_ratio > 0.3 and uniqueness_ratio > 0.7:
            inferred = "temporal"
            confidence = min(0.8, range_ratio * uniqueness_ratio)

        # CONTENT: Contains/text search queries
        elif contains_ratio > 0.5:
            inferred = "content"
            confidence = min(0.85, contains_ratio)

        # Apply minimum confidence threshold
        if confidence < 0.3:
            inferred = "unknown"
            confidence = 0.0

        self.inferred_type = inferred
        self.confidence = confidence

    def is_value_important(self, value_hash: str) -> bool:
        """Check if a specific value is considered important.

        A value is important if:
        1. It's in the important_value_hashes list (has been retrieved)
        2. It's NOT the default value (for error_indicator type)

        Args:
            value_hash: SHA256[:8] hash of the value to check.

        Returns:
            True if this value should be preserved during compression.
        """
        # If we don't have enough data, be conservative
        if self.confidence < 0.3:
            return False

        # For error_indicator: non-default values are important
        if self.inferred_type == "error_indicator":
            if self.default_value_hash and value_hash != self.default_value_hash:
                return True

        # For any type: values that have been retrieved are important
        if value_hash in self.important_value_hashes:
            return True

        # For status: check if this value has been retrieved
        if self.inferred_type == "status":
            return value_hash in self.value_retrieval_frequency

        return False


@dataclass
class CompressionEvent:
    """Record of a single compression decision (anonymized).

    This captures WHAT happened, not WHAT the data was.
    """

    # Tool identification (anonymized)
    tool_signature: ToolSignature

    # Compression metrics
    original_item_count: int
    compressed_item_count: int
    compression_ratio: float  # compressed / original
    original_tokens: int
    compressed_tokens: int
    token_reduction_ratio: float  # 1 - (compressed / original)

    # Strategy used
    strategy: str  # "top_n", "time_series", "smart_sample", "skip", etc.
    strategy_reason: str | None = None  # "high_variance", "has_score_field", etc.

    # Crushability analysis results
    crushability_score: float | None = None  # 0.0 = don't crush, 1.0 = safe to crush
    crushability_reason: str | None = None

    # Field distributions (anonymized)
    field_distributions: list[FieldDistribution] = field(default_factory=list)

    # What was preserved
    kept_first_n: int = 0
    kept_last_n: int = 0
    kept_errors: int = 0
    kept_anomalies: int = 0
    kept_by_relevance: int = 0
    kept_by_score: int = 0

    # Timing
    timestamp: float = 0.0
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_signature": self.tool_signature.to_dict(),
            "original_item_count": self.original_item_count,
            "compressed_item_count": self.compressed_item_count,
            "compression_ratio": self.compression_ratio,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "token_reduction_ratio": self.token_reduction_ratio,
            "strategy": self.strategy,
            "strategy_reason": self.strategy_reason,
            "crushability_score": self.crushability_score,
            "crushability_reason": self.crushability_reason,
            "field_distributions": [f.to_dict() for f in self.field_distributions],
            "kept_first_n": self.kept_first_n,
            "kept_last_n": self.kept_last_n,
            "kept_errors": self.kept_errors,
            "kept_anomalies": self.kept_anomalies,
            "kept_by_relevance": self.kept_by_relevance,
            "kept_by_score": self.kept_by_score,
            "timestamp": self.timestamp,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class RetrievalStats:
    """Aggregated retrieval statistics for a tool signature.

    This tracks how often compression decisions needed correction.
    """

    tool_signature_hash: str  # Reference to ToolSignature.structure_hash

    # Retrieval counts
    total_compressions: int = 0
    total_retrievals: int = 0
    full_retrievals: int = 0  # Retrieved everything
    search_retrievals: int = 0  # Used search filter

    # Derived metrics
    @property
    def retrieval_rate(self) -> float:
        """Fraction of compressions that triggered retrieval."""
        if self.total_compressions == 0:
            return 0.0
        return self.total_retrievals / self.total_compressions

    @property
    def full_retrieval_rate(self) -> float:
        """Fraction of retrievals that were full (not search)."""
        if self.total_retrievals == 0:
            return 0.0
        return self.full_retrievals / self.total_retrievals

    # Query pattern analysis (no actual queries, just patterns)
    query_field_frequency: dict[str, int] = field(default_factory=dict)  # field_hash -> count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_signature_hash": self.tool_signature_hash,
            "total_compressions": self.total_compressions,
            "total_retrievals": self.total_retrievals,
            "full_retrievals": self.full_retrievals,
            "search_retrievals": self.search_retrievals,
            "retrieval_rate": self.retrieval_rate,
            "full_retrieval_rate": self.full_retrieval_rate,
            "query_field_frequency": self.query_field_frequency,
        }


@dataclass
class AnonymizedToolStats:
    """Complete anonymized statistics for a tool type.

    This is what gets aggregated across users to build the data flywheel.
    """

    # Tool identification
    signature: ToolSignature

    # Compression statistics
    total_compressions: int = 0
    total_items_seen: int = 0
    total_items_kept: int = 0
    avg_compression_ratio: float = 0.0
    avg_token_reduction: float = 0.0

    # Strategy distribution
    strategy_counts: dict[str, int] = field(default_factory=dict)  # strategy -> count
    strategy_success_rate: dict[str, float] = field(
        default_factory=dict
    )  # strategy -> success rate

    # Retrieval statistics
    retrieval_stats: RetrievalStats | None = None

    # Learned optimal settings
    recommended_min_items: int | None = None
    recommended_preserve_fields: list[str] = field(default_factory=list)  # field hashes
    skip_compression_recommended: bool = False

    # Confidence in recommendations
    sample_size: int = 0
    confidence: float = 0.0  # 0.0 = no confidence, 1.0 = high confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signature": self.signature.to_dict(),
            "total_compressions": self.total_compressions,
            "total_items_seen": self.total_items_seen,
            "total_items_kept": self.total_items_kept,
            "avg_compression_ratio": self.avg_compression_ratio,
            "avg_token_reduction": self.avg_token_reduction,
            "strategy_counts": self.strategy_counts,
            "strategy_success_rate": self.strategy_success_rate,
            "retrieval_stats": self.retrieval_stats.to_dict() if self.retrieval_stats else None,
            "recommended_min_items": self.recommended_min_items,
            "recommended_preserve_fields": self.recommended_preserve_fields,
            "skip_compression_recommended": self.skip_compression_recommended,
            "sample_size": self.sample_size,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnonymizedToolStats:
        """Create from dictionary.

        Note: This method does not mutate the input dictionary.
        """
        # Use .get() instead of .pop() to avoid mutating input
        signature_data = data.get("signature", {})
        signature = ToolSignature.from_dict(signature_data)

        retrieval_data = data.get("retrieval_stats")
        retrieval_stats = None
        if retrieval_data:
            # Copy query_field_frequency to avoid mutation issues
            query_freq = retrieval_data.get("query_field_frequency", {})
            retrieval_stats = RetrievalStats(
                tool_signature_hash=retrieval_data.get("tool_signature_hash", ""),
                total_compressions=retrieval_data.get("total_compressions", 0),
                total_retrievals=retrieval_data.get("total_retrievals", 0),
                full_retrievals=retrieval_data.get("full_retrievals", 0),
                search_retrievals=retrieval_data.get("search_retrievals", 0),
                query_field_frequency=dict(query_freq) if query_freq else {},
            )

        # Filter to only dataclass fields, excluding signature and retrieval_stats
        # which we've already handled
        excluded_keys = {"signature", "retrieval_stats"}
        filtered_data: dict[str, Any] = {}
        for k, v in data.items():
            if k not in cls.__dataclass_fields__ or k in excluded_keys:
                continue
            # Deep copy mutable values to avoid corruption if caller modifies input
            if isinstance(v, dict):
                filtered_data[k] = dict(v)
            elif isinstance(v, list):
                filtered_data[k] = list(v)  # type: ignore[assignment]
            else:
                filtered_data[k] = v

        return cls(
            signature=signature,
            retrieval_stats=retrieval_stats,
            **filtered_data,  # type: ignore[arg-type]
        )

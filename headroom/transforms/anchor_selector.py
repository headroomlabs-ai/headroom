"""Dynamic anchor selection for SmartCrusher.

This module provides intelligent position-based anchor selection for array
compression. Instead of using fixed first-K/last-K rules, it dynamically
allocates anchor positions based on:

1. Data pattern (search results, logs, time series, generic)
2. Query keywords (recency vs historical context)
3. Information density (rare values, structural uniqueness)
4. Deduplication (skip identical items)

The anchor selection strategy is configurable via AnchorConfig and adapts
to both the data characteristics and the user's query context.

Example usage:
    from headroom.config import AnchorConfig
    from headroom.transforms.anchor_selector import (
        AnchorSelector,
        AnchorStrategy,
        DataPattern,
    )

    config = AnchorConfig()
    selector = AnchorSelector(config)

    # Select anchors for search results
    anchors = selector.select_anchors(
        items=search_results,
        max_items=15,
        pattern=DataPattern.SEARCH_RESULTS,
        query="find the latest error messages",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..config import AnchorConfig

logger = logging.getLogger(__name__)


class DataPattern(Enum):
    """Data patterns detected in tool outputs.

    Each pattern has different characteristics that affect optimal anchor selection:
    - SEARCH_RESULTS: Ordered by relevance, top items most important
    - LOGS: Chronological, recent entries typically most relevant
    - TIME_SERIES: Temporal data, need both ends to show trends
    - GENERIC: No clear ordering, distributed sampling preferred
    """

    SEARCH_RESULTS = "search_results"
    LOGS = "logs"
    TIME_SERIES = "time_series"
    GENERIC = "generic"

    @classmethod
    def from_string(cls, pattern_str: str) -> DataPattern:
        """Convert a pattern string to DataPattern enum.

        Args:
            pattern_str: Pattern string like "search_results", "logs", etc.

        Returns:
            Corresponding DataPattern enum value.
        """
        pattern_map = {
            "search_results": cls.SEARCH_RESULTS,
            "logs": cls.LOGS,
            "time_series": cls.TIME_SERIES,
            "generic": cls.GENERIC,
        }
        return pattern_map.get(pattern_str.lower(), cls.GENERIC)


class AnchorStrategy(Enum):
    """Anchor distribution strategies based on data patterns.

    Each strategy allocates position-based anchors differently:
    - FRONT_HEAVY: Most anchors at start (search results)
    - BACK_HEAVY: Most anchors at end (logs with recent items)
    - BALANCED: Equal distribution (time series)
    - DISTRIBUTED: Evenly spread throughout (generic/unknown)
    """

    FRONT_HEAVY = "front_heavy"  # Search results
    BACK_HEAVY = "back_heavy"  # Logs (recent matters)
    BALANCED = "balanced"  # Time series
    DISTRIBUTED = "distributed"  # Generic/database results


@dataclass
class AnchorWeights:
    """Distribution weights for anchor positions.

    Weights control how many anchors go to each region:
    - front: Beginning of array
    - middle: Center region
    - back: End of array

    Weights should sum to 1.0 for proper distribution.
    """

    front: float = 0.5
    middle: float = 0.1
    back: float = 0.4

    def normalize(self) -> AnchorWeights:
        """Return a copy with weights normalized to sum to 1.0.

        Returns:
            New AnchorWeights with normalized values.
        """
        total = self.front + self.middle + self.back
        if total == 0:
            return AnchorWeights()
        return AnchorWeights(
            front=self.front / total,
            middle=self.middle / total,
            back=self.back / total,
        )


def calculate_information_score(item: dict[str, Any], all_items: list[dict[str, Any]]) -> float:
    """Calculate information density score for an item.

    Information density is based on three factors:
    1. Field value uniqueness: Rare values score higher
    2. Content length: Longer items often more informative
    3. Structural uniqueness: Different fields than typical items

    Args:
        item: Dictionary item to score.
        all_items: All items in the array for comparison.

    Returns:
        Score in range [0.0, 1.0] where higher means more informative.
    """
    if not item or not all_items:
        return 0.0

    if not isinstance(item, dict):
        return 0.0

    score = 0.0
    weights_used = 0.0

    # 1. Field value uniqueness (rare values score higher)
    uniqueness_score = _calculate_value_uniqueness(item, all_items)
    score += uniqueness_score * 0.4
    weights_used += 0.4

    # 2. Content length (longer items often more informative)
    length_score = _calculate_length_score(item, all_items)
    score += length_score * 0.3
    weights_used += 0.3

    # 3. Structural uniqueness (different fields than typical)
    structural_score = _calculate_structural_uniqueness(item, all_items)
    score += structural_score * 0.3
    weights_used += 0.3

    # Normalize in case weights don't sum to 1.0
    if weights_used > 0:
        score /= weights_used

    return min(1.0, max(0.0, score))


def _calculate_value_uniqueness(item: dict[str, Any], all_items: list[dict[str, Any]]) -> float:
    """Calculate how unique the item's field values are.

    Rare field values indicate potentially important or anomalous data.

    Args:
        item: Item to check.
        all_items: All items for comparison.

    Returns:
        Score in [0.0, 1.0] where higher means rarer values.
    """
    if len(all_items) < 2:
        return 0.5

    # Count value frequencies for each field
    field_value_counts: dict[str, Counter[str]] = {}
    for other in all_items:
        if not isinstance(other, dict):
            continue
        for key, value in other.items():
            if key not in field_value_counts:
                field_value_counts[key] = Counter()
            # Convert to string for hashability
            try:
                value_str = (
                    json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
                )
            except (TypeError, ValueError):
                value_str = str(value)
            field_value_counts[key][value_str] += 1

    # Calculate rareness score for this item's values
    rareness_scores: list[float] = []
    total_items = len(all_items)

    for key, value in item.items():
        if key not in field_value_counts:
            continue
        try:
            value_str = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        except (TypeError, ValueError):
            value_str = str(value)

        count = field_value_counts[key].get(value_str, 0)
        if count > 0:
            # Inverse frequency: rare values get higher scores
            frequency = count / total_items
            rareness = 1.0 - frequency
            rareness_scores.append(rareness)

    if not rareness_scores:
        return 0.5

    # Return average rareness
    return sum(rareness_scores) / len(rareness_scores)


def _calculate_length_score(item: dict[str, Any], all_items: list[dict[str, Any]]) -> float:
    """Calculate score based on content length.

    Longer items often contain more information, but we normalize
    against the corpus to identify relatively detailed items.

    Args:
        item: Item to check.
        all_items: All items for comparison.

    Returns:
        Score in [0.0, 1.0] where higher means more content.
    """
    if len(all_items) < 2:
        return 0.5

    # Calculate serialized lengths
    def get_length(i: dict[str, Any]) -> int:
        try:
            return len(json.dumps(i))
        except (TypeError, ValueError):
            return len(str(i))

    item_length = get_length(item)
    all_lengths = [get_length(i) for i in all_items if isinstance(i, dict)]

    if not all_lengths:
        return 0.5

    max_length = max(all_lengths)
    min_length = min(all_lengths)

    if max_length == min_length:
        return 0.5

    # Normalize to [0, 1]
    return (item_length - min_length) / (max_length - min_length)


def _calculate_structural_uniqueness(
    item: dict[str, Any], all_items: list[dict[str, Any]]
) -> float:
    """Calculate how structurally unique an item is.

    Items with unusual field sets may contain error information,
    special cases, or other important data.

    Args:
        item: Item to check.
        all_items: All items for comparison.

    Returns:
        Score in [0.0, 1.0] where higher means more unique structure.
    """
    if len(all_items) < 2:
        return 0.5

    # Get common field set (fields in 80%+ of items)
    field_counts: Counter[str] = Counter()
    valid_items = [i for i in all_items if isinstance(i, dict)]
    n = len(valid_items)

    if n < 2:
        return 0.5

    for other in valid_items:
        for key in other.keys():
            field_counts[key] += 1

    common_fields = {k for k, v in field_counts.items() if v >= n * 0.8}
    rare_fields = {k for k, v in field_counts.items() if v < n * 0.2}

    item_fields = set(item.keys())

    # Score based on presence of rare fields or absence of common fields
    has_rare = len(item_fields & rare_fields)
    missing_common = len(common_fields - item_fields)

    # More rare fields or missing common fields = more unique
    uniqueness = 0.0
    if rare_fields:
        uniqueness += 0.5 * (has_rare / max(len(rare_fields), 1))
    if common_fields:
        uniqueness += 0.5 * (missing_common / max(len(common_fields), 1))

    return min(1.0, uniqueness)


def compute_item_hash(item: dict[str, Any]) -> str:
    """Compute a hash for an item for deduplication.

    Args:
        item: Dictionary item to hash.

    Returns:
        SHA256 hash (first 16 characters) of the item's content.
    """
    try:
        content = json.dumps(item, sort_keys=True, default=str)
    except (TypeError, ValueError):
        content = str(item)
    return hashlib.md5(content.encode()).hexdigest()[:16]  # nosec B324


class AnchorSelector:
    """Dynamic anchor selection for array compression.

    This class determines which array positions should be preserved
    during compression based on data patterns, query context, and
    information density.

    The selection process:
    1. Calculate anchor budget based on array size and max_items
    2. Determine strategy from data pattern
    3. Adjust weights based on query keywords
    4. Distribute anchors across front/middle/back regions
    5. Apply information density scoring for middle candidates
    6. Deduplicate identical items
    """

    def __init__(self, config: AnchorConfig | None = None) -> None:
        """Initialize anchor selector.

        Args:
            config: Anchor configuration. Uses defaults if not provided.
        """
        self.config = config or AnchorConfig()

    def calculate_anchor_budget(self, array_size: int, max_items: int) -> int:
        """Calculate anchor budget based on array size and target.

        The budget is proportional to max_items but bounded by
        min_anchor_slots and max_anchor_slots.

        Args:
            array_size: Total number of items in the array.
            max_items: Target maximum items after compression.

        Returns:
            Number of anchor slots to allocate.
        """
        if array_size <= max_items:
            # No compression needed, no anchors required
            return 0

        # Calculate proportional budget
        raw_budget = int(max_items * self.config.anchor_budget_pct)

        # Clamp to configured bounds
        budget = max(self.config.min_anchor_slots, raw_budget)
        budget = min(self.config.max_anchor_slots, budget)

        # Don't exceed what we can actually use
        budget = min(budget, array_size)

        return budget

    def get_strategy_for_pattern(self, pattern: DataPattern) -> AnchorStrategy:
        """Map data pattern to anchor strategy.

        Args:
            pattern: Detected data pattern.

        Returns:
            Appropriate anchor strategy for the pattern.
        """
        strategy_map = {
            DataPattern.SEARCH_RESULTS: AnchorStrategy.FRONT_HEAVY,
            DataPattern.LOGS: AnchorStrategy.BACK_HEAVY,
            DataPattern.TIME_SERIES: AnchorStrategy.BALANCED,
            DataPattern.GENERIC: AnchorStrategy.DISTRIBUTED,
        }
        return strategy_map.get(pattern, AnchorStrategy.DISTRIBUTED)

    def get_base_weights_for_strategy(self, strategy: AnchorStrategy) -> AnchorWeights:
        """Get base anchor weights for a strategy.

        Args:
            strategy: Anchor distribution strategy.

        Returns:
            AnchorWeights with appropriate distribution.
        """
        if strategy == AnchorStrategy.FRONT_HEAVY:
            return AnchorWeights(
                front=self.config.search_front_weight,
                middle=1.0 - self.config.search_front_weight - self.config.search_back_weight,
                back=self.config.search_back_weight,
            )
        elif strategy == AnchorStrategy.BACK_HEAVY:
            return AnchorWeights(
                front=self.config.logs_front_weight,
                middle=1.0 - self.config.logs_front_weight - self.config.logs_back_weight,
                back=self.config.logs_back_weight,
            )
        elif strategy == AnchorStrategy.BALANCED:
            # Equal front and back, small middle
            return AnchorWeights(front=0.45, middle=0.1, back=0.45)
        else:  # DISTRIBUTED
            return AnchorWeights(
                front=self.config.default_front_weight,
                middle=self.config.default_middle_weight,
                back=self.config.default_back_weight,
            )

    def adjust_weights_for_query(
        self,
        base_weights: AnchorWeights,
        query: str | None,
    ) -> AnchorWeights:
        """Adjust anchor weights based on query keywords.

        If the query contains recency keywords, shift weight toward the back.
        If it contains historical keywords, shift weight toward the front.

        Args:
            base_weights: Starting weight distribution.
            query: User query text (may be None).

        Returns:
            Adjusted AnchorWeights based on query analysis.
        """
        if not query:
            return base_weights

        query_lower = query.lower()

        # Check for recency keywords
        has_recency = any(kw in query_lower for kw in self.config.recency_keywords)

        # Check for historical keywords
        has_historical = any(kw in query_lower for kw in self.config.historical_keywords)

        if has_recency and not has_historical:
            # Shift weight toward back (recent items)
            shift = 0.15
            new_front = max(0.1, base_weights.front - shift)
            new_back = min(0.8, base_weights.back + shift)
            return AnchorWeights(
                front=new_front,
                middle=base_weights.middle,
                back=new_back,
            ).normalize()

        elif has_historical and not has_recency:
            # Shift weight toward front (older items)
            shift = 0.15
            new_front = min(0.8, base_weights.front + shift)
            new_back = max(0.1, base_weights.back - shift)
            return AnchorWeights(
                front=new_front,
                middle=base_weights.middle,
                back=new_back,
            ).normalize()

        # No adjustment needed
        return base_weights

    def select_anchors(
        self,
        items: list[dict[str, Any]],
        max_items: int,
        pattern: DataPattern,
        query: str | None = None,
    ) -> set[int]:
        """Select anchor indices for array compression.

        This is the main method for anchor selection. It:
        1. Calculates the anchor budget
        2. Determines the appropriate strategy
        3. Adjusts weights based on query
        4. Distributes anchors across regions
        5. Applies information density scoring
        6. Deduplicates identical items

        Args:
            items: List of dictionary items to select anchors from.
            max_items: Target maximum items after compression.
            pattern: Detected data pattern.
            query: Optional user query for context-aware selection.

        Returns:
            Set of indices that should be preserved as anchors.
        """
        array_size = len(items)

        if array_size == 0:
            return set()

        if array_size <= max_items:
            # No compression needed, keep all items
            return set(range(array_size))

        # Calculate budget
        budget = self.calculate_anchor_budget(array_size, max_items)

        if budget == 0:
            return set()

        # Get strategy and weights
        strategy = self.get_strategy_for_pattern(pattern)
        base_weights = self.get_base_weights_for_strategy(strategy)
        weights = self.adjust_weights_for_query(base_weights, query)
        weights = weights.normalize()

        # Calculate slots per region
        front_slots = max(1, int(budget * weights.front))
        back_slots = max(1, int(budget * weights.back))
        middle_slots = max(0, budget - front_slots - back_slots)

        # Ensure we don't exceed budget
        total_slots = front_slots + middle_slots + back_slots
        if total_slots > budget:
            # Reduce middle first, then back
            excess = total_slots - budget
            middle_reduction = min(middle_slots, excess)
            middle_slots -= middle_reduction
            excess -= middle_reduction
            if excess > 0:
                back_slots = max(1, back_slots - excess)

        logger.debug(
            f"Anchor selection: budget={budget}, front={front_slots}, "
            f"middle={middle_slots}, back={back_slots}, strategy={strategy.value}"
        )

        anchors: set[int] = set()
        seen_hashes: set[str] = set()

        # Select front anchors
        front_anchors = self._select_region_anchors(
            items=items,
            start_idx=0,
            end_idx=min(front_slots * 2, array_size // 3),  # Front third
            num_slots=front_slots,
            seen_hashes=seen_hashes,
            use_density=False,  # Front is always position-based
        )
        anchors.update(front_anchors)

        # Select back anchors
        back_start = max(array_size - back_slots * 2, (2 * array_size) // 3)
        back_anchors = self._select_region_anchors(
            items=items,
            start_idx=back_start,
            end_idx=array_size,
            num_slots=back_slots,
            seen_hashes=seen_hashes,
            use_density=False,  # Back is always position-based
        )
        anchors.update(back_anchors)

        # Select middle anchors (with information density if enabled)
        if middle_slots > 0:
            middle_start = len(front_anchors)
            middle_end = array_size - len(back_anchors)
            if middle_end > middle_start:
                middle_anchors = self._select_region_anchors(
                    items=items,
                    start_idx=middle_start,
                    end_idx=middle_end,
                    num_slots=middle_slots,
                    seen_hashes=seen_hashes,
                    use_density=self.config.use_information_density,
                )
                anchors.update(middle_anchors)

        return anchors

    def _select_region_anchors(
        self,
        items: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        num_slots: int,
        seen_hashes: set[str],
        use_density: bool,
    ) -> set[int]:
        """Select anchors from a specific region of the array.

        Args:
            items: Full list of items.
            start_idx: Start index of region (inclusive).
            end_idx: End index of region (exclusive).
            num_slots: Number of anchors to select.
            seen_hashes: Set of already-seen item hashes (for dedup).
            use_density: Whether to use information density scoring.

        Returns:
            Set of selected indices from this region.
        """
        if num_slots <= 0 or start_idx >= end_idx:
            return set()

        selected: set[int] = set()
        region_size = end_idx - start_idx

        if not use_density:
            # Position-based selection (evenly distributed within region)
            if num_slots >= region_size:
                # Select all from region (with dedup)
                for idx in range(start_idx, end_idx):
                    if self._should_include(items, idx, seen_hashes):
                        selected.add(idx)
            else:
                # Select evenly spaced indices
                step = region_size / (num_slots + 1)
                for i in range(num_slots):
                    idx = start_idx + int((i + 1) * step)
                    idx = min(idx, end_idx - 1)
                    if self._should_include(items, idx, seen_hashes):
                        selected.add(idx)
                    else:
                        # Try adjacent indices if this one is a duplicate
                        for offset in [1, -1, 2, -2]:
                            alt_idx = idx + offset
                            if start_idx <= alt_idx < end_idx:
                                if self._should_include(items, alt_idx, seen_hashes):
                                    selected.add(alt_idx)
                                    break
        else:
            # Information density-based selection
            selected = self._select_by_density(
                items=items,
                start_idx=start_idx,
                end_idx=end_idx,
                num_slots=num_slots,
                seen_hashes=seen_hashes,
            )

        return selected

    def _select_by_density(
        self,
        items: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        num_slots: int,
        seen_hashes: set[str],
    ) -> set[int]:
        """Select anchors based on information density.

        Considers more candidates than slots and selects the most
        informative ones.

        Args:
            items: Full list of items.
            start_idx: Start index of region (inclusive).
            end_idx: End index of region (exclusive).
            num_slots: Number of anchors to select.
            seen_hashes: Set of already-seen item hashes (for dedup).

        Returns:
            Set of selected indices.
        """
        # Determine candidate pool (multiplier of slots)
        num_candidates = min(
            num_slots * self.config.candidate_multiplier,
            end_idx - start_idx,
        )

        # Select evenly spaced candidates
        region_size = end_idx - start_idx
        step = region_size / (num_candidates + 1) if num_candidates > 0 else 1

        candidates: list[tuple[int, float]] = []
        region_items = items[start_idx:end_idx]

        for i in range(num_candidates):
            idx = start_idx + int((i + 1) * step)
            idx = min(idx, end_idx - 1)

            # Skip if duplicate
            if not self._should_include(items, idx, seen_hashes, check_only=True):
                continue

            # Calculate information score
            item = items[idx]
            if isinstance(item, dict):
                score = calculate_information_score(item, region_items)
            else:
                score = 0.5

            candidates.append((idx, score))

        # Sort by score (descending) and select top slots
        candidates.sort(key=lambda x: x[1], reverse=True)

        selected: set[int] = set()
        for idx, _score in candidates[:num_slots]:
            if self._should_include(items, idx, seen_hashes):
                selected.add(idx)

        return selected

    def _should_include(
        self,
        items: list[dict[str, Any]],
        idx: int,
        seen_hashes: set[str],
        check_only: bool = False,
    ) -> bool:
        """Check if an item should be included (deduplication check).

        Args:
            items: Full list of items.
            idx: Index to check.
            seen_hashes: Set of already-seen hashes.
            check_only: If True, don't add to seen_hashes.

        Returns:
            True if item should be included, False if duplicate.
        """
        if not self.config.dedup_identical_items:
            if not check_only:
                # Still need to track for potential future dedup
                pass
            return True

        if idx < 0 or idx >= len(items):
            return False

        item = items[idx]
        if not isinstance(item, dict):
            return True

        item_hash = compute_item_hash(item)

        if item_hash in seen_hashes:
            return False

        if not check_only:
            seen_hashes.add(item_hash)

        return True

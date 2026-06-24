# Dynamic SmartCrusher Preservation Plan

## Problem Statement

Current SmartCrusher uses **static "First 3 + Last 2" preservation** regardless of:
- Array size (100 items vs 10,000 items get same treatment)
- Data pattern (time series vs search results vs logs)
- Query context (user asking about "latest" vs "oldest")
- Position importance (first items might all be identical/wasteful)
- Learned retrieval patterns (which positions do users actually need?)

This is too simplistic for a production-grade compression system.

---

## Current Implementation Analysis

**Location:** `headroom/transforms/smart_crusher.py`

**Current Logic (lines 2273-2279, 2353-2359, 2561-2567):**
```python
# Always keep first 3
for i in range(min(3, n)):
    keep_indices.add(i)

# Always keep last 2
for i in range(max(0, n - 2), n):
    keep_indices.add(i)
```

**Problems:**
1. **Fixed slots waste budget** - If first 3 items are identical, we've wasted 3 slots
2. **No size adaptation** - 20-item array loses 25% to anchors; 1000-item array loses 0.5%
3. **Pattern-agnostic** - Search results don't need "last 2"; time series might need more recency
4. **No learning** - Doesn't adapt based on what users actually retrieve

---

## Proposed Solution: Adaptive Slot Allocation

### Idea 1: Size-Proportional Anchor Budget

Instead of fixed counts, allocate a **percentage budget** for position-based anchors:

```python
def calculate_anchor_budget(array_size: int, max_items: int) -> AnchorBudget:
    """Allocate slots proportionally, with floors and ceilings."""

    # Base: 20% of output budget for position anchors
    total_anchor_slots = max(3, min(10, int(max_items * 0.20)))

    # Distribution: 60% front, 40% back (front-weighted for context)
    front_slots = max(1, int(total_anchor_slots * 0.6))
    back_slots = max(1, total_anchor_slots - front_slots)

    return AnchorBudget(front=front_slots, back=back_slots)
```

**Example:**
| Array Size | Max Items | Anchor Budget | Front | Back |
|------------|-----------|---------------|-------|------|
| 50         | 10        | 3             | 2     | 1    |
| 200        | 15        | 3             | 2     | 1    |
| 1000       | 20        | 4             | 3     | 1    |
| 5000       | 30        | 6             | 4     | 2    |

---

### Idea 2: Pattern-Aware Anchor Weighting

Different data patterns need different position importance:

```python
class AnchorStrategy(Enum):
    FRONT_HEAVY = "front_heavy"     # Search results: top items matter most
    BACK_HEAVY = "back_heavy"       # Logs: recent items matter most
    BALANCED = "balanced"           # Time series: both ends matter
    MIDDLE_AWARE = "middle_aware"   # Database: order might be arbitrary

def get_anchor_strategy(pattern: DataPattern) -> AnchorStrategy:
    return {
        DataPattern.SEARCH_RESULTS: AnchorStrategy.FRONT_HEAVY,  # Top N by score
        DataPattern.LOGS: AnchorStrategy.BACK_HEAVY,             # Recency matters
        DataPattern.TIME_SERIES: AnchorStrategy.BALANCED,        # Both ends for trend
        DataPattern.GENERIC: AnchorStrategy.MIDDLE_AWARE,        # Don't assume order
    }.get(pattern, AnchorStrategy.BALANCED)
```

**FRONT_HEAVY (Search Results):**
- Front: 80% of anchor budget
- Back: 20% of anchor budget
- Rationale: Top search results are ranked by relevance

**BACK_HEAVY (Logs):**
- Front: 20% of anchor budget
- Back: 80% of anchor budget
- Rationale: Most recent logs are usually most relevant

**BALANCED (Time Series):**
- Front: 50% of anchor budget
- Back: 50% of anchor budget
- Rationale: Need both start and end for trend analysis

**MIDDLE_AWARE (Generic/Database):**
- Front: 30% of anchor budget
- Back: 30% of anchor budget
- Middle sample: 40% of anchor budget (stratified)
- Rationale: Order might be arbitrary; sample across positions

---

### Idea 3: Query-Aware Dynamic Weighting

Adjust anchor strategy based on user's query:

```python
def adjust_for_query(base_strategy: AnchorStrategy, query: str) -> AnchorWeights:
    """Shift anchor weights based on query intent."""

    weights = base_strategy.default_weights()

    # Recency signals
    recency_keywords = ["latest", "recent", "last", "newest", "current"]
    if any(kw in query.lower() for kw in recency_keywords):
        weights.back_weight *= 1.5
        weights.front_weight *= 0.7

    # Historical signals
    historical_keywords = ["first", "oldest", "earliest", "original", "initial"]
    if any(kw in query.lower() for kw in historical_keywords):
        weights.front_weight *= 1.5
        weights.back_weight *= 0.7

    # Range signals
    range_keywords = ["all", "every", "complete", "full"]
    if any(kw in query.lower() for kw in range_keywords):
        weights.middle_weight *= 1.3  # Better coverage

    return weights.normalize()
```

---

### Idea 4: Information-Density Anchor Selection

Don't blindly take first N - select **most informative** items from anchor regions:

```python
def select_informative_anchors(
    items: list[dict],
    region: str,  # "front", "back", "middle"
    slots: int,
    all_items_hash: set[str]
) -> list[int]:
    """Select most informative items from a region."""

    if region == "front":
        candidates = list(range(min(slots * 3, len(items))))  # Consider 3x candidates
    elif region == "back":
        start = max(0, len(items) - slots * 3)
        candidates = list(range(start, len(items)))
    else:  # middle
        step = len(items) // (slots * 3 + 1)
        candidates = [i * step for i in range(1, slots * 3 + 1)]

    # Score each candidate by information content
    scored = []
    for idx in candidates:
        item = items[idx]
        item_hash = hash_item(item)

        # Skip if we've seen identical item
        if item_hash in all_items_hash:
            continue

        score = calculate_information_score(item, items)
        scored.append((idx, score, item_hash))

    # Select top N by information score
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = []
    for idx, _, item_hash in scored[:slots]:
        selected.append(idx)
        all_items_hash.add(item_hash)

    return sorted(selected)


def calculate_information_score(item: dict, all_items: list[dict]) -> float:
    """Score item by how much unique information it contributes."""

    score = 0.0

    # 1. Field uniqueness - rare field values score higher
    for field, value in item.items():
        field_values = [i.get(field) for i in all_items if field in i]
        value_frequency = field_values.count(value) / len(field_values)
        score += (1 - value_frequency)  # Rare values score higher

    # 2. Structural uniqueness - different fields than typical
    typical_fields = get_typical_fields(all_items)
    unique_fields = set(item.keys()) - typical_fields
    score += len(unique_fields) * 0.5

    # 3. Content length - longer items often more informative
    content_length = len(json.dumps(item))
    avg_length = sum(len(json.dumps(i)) for i in all_items) / len(all_items)
    if content_length > avg_length:
        score += 0.3

    return score
```

---

### Idea 5: TOIN-Learned Position Importance

Track which positions users actually retrieve and learn from it:

```python
@dataclass
class PositionRetrievalPattern:
    """Learned position importance from retrieval data."""
    tool_name: str
    total_compressions: int
    position_retrievals: dict[str, int]  # "front_10%", "middle", "back_10%"

    def get_position_weights(self) -> dict[str, float]:
        """Convert retrieval counts to weights."""
        total = sum(self.position_retrievals.values())
        if total == 0:
            return {"front": 0.5, "middle": 0.0, "back": 0.5}

        return {
            position: count / total
            for position, count in self.position_retrievals.items()
        }


class TOINPositionLearning:
    """Learn position importance from retrieval patterns."""

    def record_retrieval(
        self,
        tool_name: str,
        original_size: int,
        retrieved_indices: list[int]
    ):
        """Record which positions were retrieved."""
        for idx in retrieved_indices:
            position = self._classify_position(idx, original_size)
            self._increment_position_count(tool_name, position)

    def _classify_position(self, idx: int, size: int) -> str:
        """Classify index into position bucket."""
        relative_pos = idx / size
        if relative_pos < 0.1:
            return "front_10%"
        elif relative_pos < 0.3:
            return "front_30%"
        elif relative_pos > 0.9:
            return "back_10%"
        elif relative_pos > 0.7:
            return "back_30%"
        else:
            return "middle"

    def get_anchor_recommendation(self, tool_name: str) -> AnchorWeights:
        """Get learned anchor weights for a tool."""
        pattern = self._get_pattern(tool_name)
        if pattern.total_compressions < 10:
            return AnchorWeights.default()  # Not enough data

        weights = pattern.get_position_weights()
        return AnchorWeights(
            front=weights.get("front_10%", 0.3) + weights.get("front_30%", 0.1),
            middle=weights.get("middle", 0.2),
            back=weights.get("back_10%", 0.3) + weights.get("back_30%", 0.1),
        )
```

---

### Idea 6: Stratified Sampling for Middle Positions

For large arrays, sample strategically from middle:

```python
def stratified_middle_sample(
    items: list[dict],
    num_samples: int,
    analysis: ArrayAnalysis
) -> list[int]:
    """Sample middle positions using stratified approach."""

    n = len(items)
    front_boundary = int(n * 0.1)
    back_boundary = int(n * 0.9)
    middle_items = list(range(front_boundary, back_boundary))

    if not middle_items or num_samples <= 0:
        return []

    # Strategy 1: Cluster-based sampling
    if analysis.has_clusterable_field:
        clusters = cluster_by_field(items, analysis.cluster_field)
        return sample_from_clusters(clusters, num_samples, middle_items)

    # Strategy 2: Variance-based sampling (pick high-variance points)
    if analysis.numeric_fields:
        variance_scores = calculate_position_variance(items, analysis.numeric_fields)
        sorted_by_variance = sorted(
            middle_items,
            key=lambda i: variance_scores.get(i, 0),
            reverse=True
        )
        return sorted(sorted_by_variance[:num_samples])

    # Strategy 3: Even spacing (fallback)
    step = len(middle_items) // (num_samples + 1)
    return [middle_items[i * step] for i in range(1, num_samples + 1)]
```

---

## Testing Strategy

### Test Category 1: Adversarial Position Tests

Test cases where important data is NOT at expected positions:

```python
class TestAdversarialPositions:
    """Test scenarios that break 'first 3 + last 2' assumption."""

    def test_important_data_in_middle(self):
        """Critical error at position 50 of 100-item array."""
        items = [{"status": "ok", "value": i} for i in range(100)]
        items[50] = {"status": "error", "error_code": "CRITICAL", "value": 50}

        result = smart_crusher.crush(items, max_items=10)

        # Error item MUST be preserved regardless of position
        assert any(item.get("error_code") == "CRITICAL" for item in result)

    def test_spike_not_at_boundaries(self):
        """Numeric spike at position 75 of 100-item array."""
        items = [{"metric": 10.0 + random.random()} for _ in range(100)]
        items[75]["metric"] = 1000.0  # Spike

        result = smart_crusher.crush(items, max_items=10)

        # Spike MUST be preserved as anomaly
        assert any(item["metric"] > 500 for item in result)

    def test_first_items_identical(self):
        """First 10 items are identical - shouldn't waste slots."""
        items = [{"id": "same", "value": 0}] * 10 + [
            {"id": f"unique_{i}", "value": i} for i in range(90)
        ]

        result = smart_crusher.crush(items, max_items=10)

        # Should NOT have multiple identical items
        ids = [item["id"] for item in result]
        # At most 1-2 of the identical items, not 3
        assert ids.count("same") <= 2

    def test_last_items_identical(self):
        """Last 10 items are identical - shouldn't waste slots."""
        items = [{"id": f"unique_{i}", "value": i} for i in range(90)] + [
            {"id": "same", "value": 100}
        ] * 10

        result = smart_crusher.crush(items, max_items=10)

        ids = [item["id"] for item in result]
        assert ids.count("same") <= 2

    def test_relevant_item_in_middle(self):
        """Item matching user query is in middle of array."""
        items = [{"name": f"item_{i}", "status": "active"} for i in range(100)]
        items[42]["name"] = "target_item"
        items[42]["description"] = "This is what user asked about"

        result = smart_crusher.crush(
            items,
            max_items=10,
            query="find target_item"
        )

        # Query-matched item MUST be preserved
        assert any("target_item" in item.get("name", "") for item in result)
```

### Test Category 2: Size Adaptation Tests

```python
class TestSizeAdaptation:
    """Test that anchor allocation scales with array size."""

    @pytest.mark.parametrize("size,expected_min_anchors", [
        (20, 3),    # Small array: at least 3 anchors
        (100, 4),   # Medium array: at least 4 anchors
        (500, 5),   # Large array: at least 5 anchors
        (2000, 6),  # Very large: at least 6 anchors
    ])
    def test_anchor_count_scales(self, size, expected_min_anchors):
        """Anchor count should increase with array size."""
        items = [{"id": i, "value": i * 10} for i in range(size)]

        result = smart_crusher.crush(items, max_items=20)

        # Count items from first 10% and last 10%
        anchor_count = sum(
            1 for item in result
            if item["id"] < size * 0.1 or item["id"] > size * 0.9
        )

        assert anchor_count >= expected_min_anchors

    def test_small_array_high_preservation(self):
        """Small arrays should preserve higher percentage."""
        items = [{"id": i} for i in range(15)]

        result = smart_crusher.crush(items, max_items=20)

        # Should preserve most/all of small array
        assert len(result) >= 10  # At least 66%

    def test_large_array_efficient_sampling(self):
        """Large arrays should sample efficiently."""
        items = [{"id": i, "value": i} for i in range(5000)]

        result = smart_crusher.crush(items, max_items=20)

        # Should have good distribution across positions
        positions = [item["id"] for item in result]

        has_front = any(p < 500 for p in positions)
        has_middle = any(500 < p < 4500 for p in positions)
        has_back = any(p > 4500 for p in positions)

        assert has_front and has_back
        # Middle should be represented if array is large enough
        assert has_middle
```

### Test Category 3: Pattern-Specific Tests

```python
class TestPatternAwareAnchoring:
    """Test pattern-specific anchor strategies."""

    def test_search_results_front_heavy(self):
        """Search results should preserve more from front."""
        items = [
            {"title": f"Result {i}", "score": 1.0 - (i * 0.01)}
            for i in range(100)
        ]

        result = smart_crusher.crush(items, max_items=10)

        # More items should be from front (high scores)
        front_count = sum(1 for item in result if item["score"] > 0.9)
        back_count = sum(1 for item in result if item["score"] < 0.1)

        assert front_count > back_count

    def test_logs_back_heavy(self):
        """Logs should preserve more from back (recent)."""
        items = [
            {"timestamp": f"2024-01-{i:02d}", "level": "INFO", "message": f"Log {i}"}
            for i in range(1, 31)
        ]

        result = smart_crusher.crush(items, max_items=10)

        # More items should be from back (recent logs)
        timestamps = [item["timestamp"] for item in result]
        recent_count = sum(1 for ts in timestamps if int(ts[-2:]) > 20)
        old_count = sum(1 for ts in timestamps if int(ts[-2:]) < 10)

        assert recent_count >= old_count

    def test_time_series_balanced(self):
        """Time series should have balanced front/back."""
        items = [
            {"timestamp": f"2024-01-01T{i:02d}:00:00", "value": 100 + i}
            for i in range(24)
        ]

        result = smart_crusher.crush(items, max_items=8)

        hours = [int(item["timestamp"][11:13]) for item in result]
        front_count = sum(1 for h in hours if h < 8)
        back_count = sum(1 for h in hours if h > 16)

        # Should be roughly balanced
        assert abs(front_count - back_count) <= 2
```

### Test Category 4: Query-Aware Tests

```python
class TestQueryAwareAnchoring:
    """Test query-based anchor adjustment."""

    def test_latest_query_shifts_to_back(self):
        """'Latest' in query should preserve more recent items."""
        items = [{"id": i, "created": f"2024-01-{i:02d}"} for i in range(1, 31)]

        result = smart_crusher.crush(
            items,
            max_items=8,
            query="Show me the latest entries"
        )

        ids = [item["id"] for item in result]
        recent_count = sum(1 for id in ids if id > 20)

        assert recent_count >= 3  # At least 3 recent items

    def test_first_query_shifts_to_front(self):
        """'First' in query should preserve earlier items."""
        items = [{"id": i, "created": f"2024-01-{i:02d}"} for i in range(1, 31)]

        result = smart_crusher.crush(
            items,
            max_items=8,
            query="Show me the first entries"
        )

        ids = [item["id"] for item in result]
        early_count = sum(1 for id in ids if id < 10)

        assert early_count >= 3

    def test_specific_id_query_finds_item(self):
        """Query for specific ID should find it regardless of position."""
        items = [{"id": f"item_{i:04d}", "value": i} for i in range(1000)]

        result = smart_crusher.crush(
            items,
            max_items=10,
            query="Find item_0567"
        )

        assert any(item["id"] == "item_0567" for item in result)
```

### Test Category 5: Coverage Metrics Tests

```python
class TestCoverageMetrics:
    """Test that preserved items represent the full distribution."""

    def test_value_range_coverage(self):
        """Preserved items should cover the value range."""
        items = [{"value": i} for i in range(100)]

        result = smart_crusher.crush(items, max_items=10)

        values = [item["value"] for item in result]

        # Should cover most of the range
        assert min(values) < 10  # Has low values
        assert max(values) > 90  # Has high values

        # Should have some middle values too
        middle_count = sum(1 for v in values if 30 < v < 70)
        assert middle_count >= 1

    def test_category_coverage(self):
        """Preserved items should represent all categories."""
        items = [
            {"category": cat, "id": i}
            for i, cat in enumerate(["A"] * 30 + ["B"] * 30 + ["C"] * 40)
        ]

        result = smart_crusher.crush(items, max_items=10)

        categories = set(item["category"] for item in result)

        # Should have at least 2 of 3 categories
        assert len(categories) >= 2

    def test_temporal_coverage(self):
        """Preserved items should span the time range."""
        items = [
            {"timestamp": f"2024-{m:02d}-15", "event": f"event_{i}"}
            for i, m in enumerate(range(1, 13))
        ]

        result = smart_crusher.crush(items, max_items=5)

        months = [int(item["timestamp"][5:7]) for item in result]

        # Should span at least 6 months of the year
        assert max(months) - min(months) >= 6
```

### Test Category 6: Retrieval Simulation Tests

```python
class TestRetrievalSimulation:
    """Simulate user retrieval patterns to measure effectiveness."""

    def test_retrieval_hit_rate_random_queries(self):
        """Measure how often preserved items satisfy random queries."""
        items = [
            {"id": i, "name": f"Item {i}", "category": f"cat_{i % 5}"}
            for i in range(100)
        ]

        compressed = smart_crusher.crush(items, max_items=15)
        compressed_ids = {item["id"] for item in compressed}

        # Simulate 100 random "queries" (random item lookups)
        hits = 0
        for _ in range(100):
            target_id = random.randint(0, 99)
            if target_id in compressed_ids:
                hits += 1

        # Should hit at least 15% (we keep 15 of 100)
        assert hits >= 15

    def test_retrieval_hit_rate_weighted_queries(self):
        """Measure hits for queries weighted toward common patterns."""
        items = [{"id": i, "value": i * 10} for i in range(100)]

        compressed = smart_crusher.crush(items, max_items=15)
        compressed_ids = {item["id"] for item in compressed}

        # Weight queries toward front (30%), back (30%), anomalies (40%)
        hits = 0
        queries = (
            list(range(10)) * 3 +  # Front queries
            list(range(90, 100)) * 3 +  # Back queries
            [50] * 4  # Middle anomaly queries
        )

        for target_id in queries:
            if target_id in compressed_ids:
                hits += 1

        # Should hit more often than random due to anchor strategy
        assert hits >= 20  # At least 20% hit rate
```

---

## Implementation Phases

### Phase 1: Refactor Anchor Logic (Foundation)

1. Extract anchor selection into `AnchorSelector` class
2. Make slot counts configurable via `AnchorConfig`
3. Add size-proportional allocation
4. Maintain backward compatibility with current defaults

### Phase 2: Pattern-Aware Anchoring

1. Map `DataPattern` to `AnchorStrategy`
2. Implement front-heavy, back-heavy, balanced, middle-aware strategies
3. Add pattern-specific anchor weight configs

### Phase 3: Information-Density Selection

1. Add `calculate_information_score()` for items
2. Select from candidate region instead of fixed positions
3. Deduplicate identical items across regions

### Phase 4: Query-Aware Adjustment

1. Parse query for position intent keywords
2. Adjust anchor weights dynamically
3. Add query-position relevance scoring

### Phase 5: TOIN Position Learning

1. Track retrieval positions in TOIN
2. Learn per-tool position importance
3. Use learned weights to adjust anchor strategy

### Phase 6: Comprehensive Testing

1. Implement all adversarial tests
2. Add coverage metric tests
3. Add retrieval simulation tests
4. Performance benchmarks

---

## Configuration Schema

```python
@dataclass
class AnchorConfig:
    """Configuration for dynamic anchor allocation."""

    # Base anchor budget as percentage of max_items
    anchor_budget_pct: float = 0.20  # 20% of slots for position anchors

    # Minimum and maximum anchor slots
    min_anchor_slots: int = 3
    max_anchor_slots: int = 10

    # Default distribution (overridden by pattern)
    default_front_weight: float = 0.5
    default_back_weight: float = 0.5
    default_middle_weight: float = 0.0

    # Pattern-specific overrides
    search_front_weight: float = 0.8
    logs_back_weight: float = 0.8
    time_series_balance: float = 0.5

    # Query keyword detection
    recency_keywords: list[str] = field(default_factory=lambda: [
        "latest", "recent", "last", "newest", "current"
    ])
    historical_keywords: list[str] = field(default_factory=lambda: [
        "first", "oldest", "earliest", "original", "initial"
    ])

    # Information density selection
    use_information_density: bool = True
    candidate_multiplier: int = 3  # Consider 3x candidates per slot

    # TOIN learning
    use_learned_positions: bool = True
    min_samples_for_learning: int = 10
```

---

## Success Metrics

1. **Retrieval Coverage**: % of user retrievals that hit preserved items (target: >80%)
2. **Information Density**: Unique information per preserved slot (target: no duplicate items)
3. **Distribution Coverage**: Preserved items span full value/time/category ranges
4. **Adversarial Robustness**: All adversarial tests pass
5. **Backward Compatibility**: Existing tests still pass
6. **Performance**: <5ms additional latency for anchor selection

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Information density calculation is expensive | Cache scores, limit candidate pool |
| Query keyword detection is brittle | Use as soft signal, not hard rule |
| TOIN learning needs cold start | Fall back to pattern-based defaults |
| Breaking existing behavior | Feature flag, A/B testing |
| Middle sampling misses important items | Always include anomalies/errors regardless |

---

## Next Steps

1. Review and approve this plan
2. Write failing tests first (TDD approach)
3. Implement Phase 1 (refactor foundation)
4. Iterate through phases with test validation
5. Benchmark against current implementation
6. A/B test in production with telemetry

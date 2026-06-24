"""Tests for diversity-aware compute_optimal_k in adaptive_sizer."""

from __future__ import annotations

import json

from headroom.transforms.adaptive_sizer import compute_optimal_k


def _make_unique_items(n: int) -> list[str]:
    """Create n completely unique JSON items (high diversity)."""
    return [
        json.dumps(
            {
                "id": i,
                "title": f"Unique topic number {i} about subject area {chr(65 + i % 26)}",
                "content": (
                    f"This is document {i} discussing a completely different subject. "
                    f"It covers concepts like {chr(65 + i % 26)}-theory, "
                    f"methodology-{i * 7 % 100}, and framework-{i * 13 % 50}. "
                    f"The key finding is result-{i} which has implications for field-{i % 10}."
                ),
                "source": f"source_{i}.pdf",
                "score": round(0.99 - i * 0.03, 2),
            }
        )
        for i in range(n)
    ]


def _make_repetitive_items(n: int, templates: int = 3) -> list[str]:
    """Create n items from a few templates (low diversity)."""
    base_templates = [
        {
            "status": "ok",
            "message": "Health check passed",
            "latency_ms": 12,
            "service": "api-gateway",
        },
        {
            "status": "ok",
            "message": "Health check passed",
            "latency_ms": 15,
            "service": "auth-service",
        },
        {
            "status": "ok",
            "message": "Health check passed",
            "latency_ms": 8,
            "service": "db-proxy",
        },
    ]
    return [
        json.dumps({**base_templates[i % templates], "timestamp": f"2026-03-25T10:{i:02d}:00Z"})
        for i in range(n)
    ]


def _make_mixed_items(n: int, unique_fraction: float) -> list[str]:
    """Create items where unique_fraction are unique, rest are duplicates."""
    unique_count = int(n * unique_fraction)
    dup_count = n - unique_count
    items = _make_unique_items(unique_count)
    if dup_count > 0:
        template = json.dumps(
            {
                "status": "ok",
                "message": "Routine health check passed successfully",
                "latency_ms": 10,
            }
        )
        items.extend([template] * dup_count)
    return items


class TestSmallArrays:
    def test_small_array_returns_n(self):
        """Arrays with n <= 8 should always return n (unchanged)."""
        items = _make_unique_items(5)
        assert compute_optimal_k(items) == 5

    def test_eight_items_returns_eight(self):
        items = _make_unique_items(8)
        assert compute_optimal_k(items) == 8


class TestNearTotalRedundancy:
    def test_identical_items_returns_min(self):
        """20 identical items should return ~3 (near-total redundancy)."""
        items = [json.dumps({"status": "ok", "msg": "healthy"})] * 20
        k = compute_optimal_k(items)
        assert k <= 3

    def test_two_groups_returns_small_k(self):
        """Items from 2 groups should return small k."""
        items = [json.dumps({"type": "A", "val": 1})] * 10 + [
            json.dumps({"type": "B", "val": 2})
        ] * 10
        k = compute_optimal_k(items)
        assert k <= 5


class TestHighDiversity:
    def test_all_unique_keeps_most(self):
        """15 completely unique items → should keep >= 10 (not 4 like before)."""
        items = _make_unique_items(15)
        k = compute_optimal_k(items)
        assert k >= 10, f"Expected k >= 10 for 15 unique items, got k={k}"

    def test_twenty_unique_keeps_most(self):
        """20 unique items → should keep >= 14."""
        items = _make_unique_items(20)
        k = compute_optimal_k(items)
        assert k >= 14, f"Expected k >= 14 for 20 unique items, got k={k}"

    def test_twelve_unique_rag_chunks(self):
        """12 unique RAG chunks → should keep >= 8."""
        items = _make_unique_items(12)
        k = compute_optimal_k(items)
        assert k >= 8, f"Expected k >= 8 for 12 unique RAG chunks, got k={k}"


class TestLowDiversity:
    def test_repetitive_items_unchanged(self):
        """15 items from 3 templates → k should stay small (same as before)."""
        items = _make_repetitive_items(15, templates=3)
        k = compute_optimal_k(items)
        assert k <= 8, f"Expected k <= 8 for repetitive items, got k={k}"

    def test_twenty_repetitive_stays_small(self):
        """20 items from 3 templates → k stays small."""
        items = _make_repetitive_items(20, templates=3)
        k = compute_optimal_k(items)
        assert k <= 10, f"Expected k <= 10 for 20 repetitive items, got k={k}"


class TestModerateDiversity:
    def test_half_unique_scales(self):
        """20 items, 50% unique → k should be in middle range."""
        items = _make_mixed_items(20, unique_fraction=0.5)
        k = compute_optimal_k(items)
        assert 6 <= k <= 16, f"Expected 6 <= k <= 16 for 50% unique, got k={k}"


class TestKneeInteraction:
    def test_knee_with_high_diversity_gets_floor(self):
        """Even if knee is found at low value, high diversity boosts k."""
        # Create items that have a weak bigram knee but are all unique via SimHash
        items = _make_unique_items(15)
        k = compute_optimal_k(items)
        # With diversity_ratio ~1.0, diversity_floor should boost k
        assert k >= 10, f"Expected k >= 10 with high diversity floor, got k={k}"

    def test_knee_with_low_diversity_stays(self):
        """Low diversity + knee found → k stays at knee."""
        items = _make_repetitive_items(15, templates=3)
        k = compute_optimal_k(items)
        assert k <= 8, f"Expected knee-derived k <= 8 for low diversity, got k={k}"


class TestBiasAndCaps:
    def test_bias_increases_k(self):
        """Bias > 1 should increase k."""
        items = _make_unique_items(15)
        k_normal = compute_optimal_k(items, bias=1.0)
        k_biased = compute_optimal_k(items, bias=1.5)
        assert k_biased >= k_normal

    def test_bias_decreases_k(self):
        """Bias < 1 should decrease k."""
        items = _make_unique_items(15)
        k_normal = compute_optimal_k(items, bias=1.0)
        k_biased = compute_optimal_k(items, bias=0.5)
        assert k_biased <= k_normal

    def test_max_k_cap_respected(self):
        """Even with high diversity, max_k cap is honored."""
        items = _make_unique_items(20)
        k = compute_optimal_k(items, max_k=5)
        assert k <= 5

    def test_min_k_floor_respected(self):
        """Even with low diversity, min_k floor is honored."""
        items = [json.dumps({"x": 1})] * 20
        k = compute_optimal_k(items, min_k=3)
        assert k >= 3

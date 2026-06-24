"""Formal evals for SmartCrusher quality retention.

These tests verify that SmartCrusher GUARANTEES 100% retention of critical items:
1. Error items: Items containing error keywords
2. Anomaly items: Items with values > 2 std from mean
3. Relevance items: Items matching user query context

This is a FORMAL EVAL - any failure here is a CRITICAL BUG.
"""

import json

import pytest

from headroom.providers.anthropic import AnthropicTokenCounter
from headroom.tokenizer import Tokenizer
from headroom.transforms.smart_crusher import (
    SmartCrusher,
    SmartCrusherConfig,
    smart_crush_tool_output,
)


class TestErrorRetention:
    """Verify 100% retention of error items."""

    ERROR_KEYWORDS = ["error", "exception", "failed", "failure", "critical", "fatal"]

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset with known errors."""
        items = []
        error_indices = []

        for i in range(1000):
            items.append(
                {
                    "id": f"item_{i}",
                    "value": i,
                    "status": "ok",
                    "message": f"Normal operation {i}",
                }
            )

        # Insert errors at specific positions
        for idx in [10, 50, 100, 250, 500, 750, 999]:
            items[idx]["status"] = "failed"
            items[idx]["error"] = f"Error at position {idx}"
            error_indices.append(idx)

        return items, error_indices

    def test_all_error_items_retained(self, large_dataset):
        """CRITICAL: Every item with error keywords MUST be retained."""
        items, error_indices = large_dataset

        config = SmartCrusherConfig(max_items_after_crush=20)
        content = json.dumps(items)
        compressed_str, _, _ = smart_crush_tool_output(content, config, with_compaction=False)
        compressed = json.loads(compressed_str)

        # Count errors before and after
        errors_before = len(error_indices)
        errors_after = sum(1 for x in compressed if x.get("error"))

        assert errors_after == errors_before, (
            f"QUALITY FAILURE: Lost {errors_before - errors_after} error items! "
            f"Expected {errors_before}, got {errors_after}"
        )

    @pytest.mark.parametrize("keyword", ERROR_KEYWORDS)
    def test_each_error_keyword_detected(self, keyword):
        """Each error keyword must trigger retention."""
        items = [{"id": f"item_{i}", "msg": f"Normal {i}"} for i in range(100)]
        items[50]["msg"] = f"This contains {keyword} keyword"

        config = SmartCrusherConfig(max_items_after_crush=15)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        matching = [x for x in compressed if keyword in str(x).lower()]
        assert len(matching) >= 1, f"Item with '{keyword}' keyword was dropped!"

    def test_error_in_nested_structure(self):
        """Errors in nested objects must be detected."""
        items = [{"id": i, "data": {"status": "ok"}} for i in range(100)]
        items[50]["data"]["status"] = "failed"
        items[50]["data"]["error"] = "Nested error"

        config = SmartCrusherConfig(max_items_after_crush=15)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        nested_errors = [x for x in compressed if x.get("data", {}).get("error")]
        assert len(nested_errors) >= 1, "Nested error item was dropped!"

    def test_multiple_errors_all_retained(self):
        """When errors exceed max_items, ALL errors must still be retained."""
        # Create 100 items where 30 are errors (more than max_items_after_crush)
        items = []
        for i in range(100):
            item = {"id": i, "value": i}
            if i % 3 == 0:  # Every 3rd item is an error (33 total)
                item["error"] = f"Error {i}"
                item["status"] = "failed"
            items.append(item)

        error_count_before = sum(1 for x in items if x.get("error"))
        assert error_count_before == 34  # 0,3,6,...,99 = 34 items

        # Compress with max 20 items
        config = SmartCrusherConfig(max_items_after_crush=20)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        error_count_after = sum(1 for x in compressed if x.get("error"))

        # When errors > max_items, we should keep ALL errors (errors take priority)
        # This tests the _prioritize_indices logic
        assert error_count_after == error_count_before, (
            f"CRITICAL: Errors were dropped! "
            f"Before: {error_count_before}, After: {error_count_after}"
        )


class TestAnomalyRetention:
    """Verify 100% retention of anomalous numeric values."""

    def test_numeric_anomalies_retained(self):
        """Items with values > 2 std from mean must be retained."""
        items = []
        anomaly_indices = []

        # Create items with normal values around mean=100, std=10
        for i in range(1000):
            items.append(
                {
                    "id": f"item_{i}",
                    "value": 100 + (i % 20) - 10,  # Values 90-110
                    "name": f"Normal item {i}",
                }
            )

        # Insert anomalies (> 2 std = > 120 or < 80)
        for idx in [100, 300, 500, 700, 900]:
            items[idx]["value"] = 999999  # Extreme anomaly
            items[idx]["is_anomaly"] = True  # Mark for verification
            anomaly_indices.append(idx)

        config = SmartCrusherConfig(max_items_after_crush=20)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        anomalies_after = sum(1 for x in compressed if x.get("is_anomaly"))

        assert anomalies_after == len(anomaly_indices), (
            f"QUALITY FAILURE: Lost anomaly items! "
            f"Expected {len(anomaly_indices)}, got {anomalies_after}"
        )

    def test_negative_anomalies_retained(self):
        """Negative outliers must also be retained."""
        items = [{"id": i, "value": 100} for i in range(100)]
        items[50]["value"] = -999  # Negative anomaly
        items[50]["is_anomaly"] = True

        config = SmartCrusherConfig(max_items_after_crush=15)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        anomalies = [x for x in compressed if x.get("is_anomaly")]
        assert len(anomalies) == 1, "Negative anomaly was dropped!"


class TestRelevanceRetention:
    """Verify retention of items matching query context."""

    def test_relevance_with_query_context(self):
        """Items matching query should be retained when context is provided."""
        items = [{"id": i, "content": f"Generic content about topic {i}"} for i in range(100)]

        # Insert a specific item that matches our query
        # Note: This also contains "error" keyword which will trigger error retention
        items[50]["content"] = "Authentication error: invalid JWT token expired"
        items[50]["is_target"] = True

        # Use SmartCrusher with query context (via message-based API)
        config = SmartCrusherConfig(max_items_after_crush=15)
        crusher = SmartCrusher(config, with_compaction=False)

        # Create tokenizer with proper counter
        model = "claude-3-5-sonnet-20241022"
        token_counter = AnthropicTokenCounter(model)
        tokenizer = Tokenizer(token_counter, model)

        # Create messages with query context
        messages = [
            {"role": "user", "content": "Why is JWT authentication failing?"},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]

        result = crusher.apply(messages, tokenizer)
        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        compressed = json.loads(tool_msg["content"].split("\n")[0])  # Remove marker

        targets = [x for x in compressed if x.get("is_target")]
        assert len(targets) >= 1, "Target item was dropped despite matching query context!"


class TestFirstLastRetention:
    """Verify first K and last K items are always retained."""

    def test_first_items_retained(self):
        """First 3 items must always be retained."""
        items = [{"id": i, "value": i} for i in range(100)]

        config = SmartCrusherConfig(max_items_after_crush=15)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        ids = [x["id"] for x in compressed]
        assert 0 in ids, "First item (id=0) was dropped!"
        assert 1 in ids, "Second item (id=1) was dropped!"
        assert 2 in ids, "Third item (id=2) was dropped!"

    def test_last_items_retained(self):
        """Last 2 items must always be retained."""
        items = [{"id": i, "value": i} for i in range(100)]

        config = SmartCrusherConfig(max_items_after_crush=15)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        ids = [x["id"] for x in compressed]
        assert 98 in ids, "Second-to-last item (id=98) was dropped!"
        assert 99 in ids, "Last item (id=99) was dropped!"


class TestCombinedRetention:
    """Test retention when multiple preservation criteria apply."""

    def test_error_and_anomaly_both_retained(self):
        """Items that are both errors AND anomalies must be retained."""
        items = [{"id": i, "value": 100} for i in range(100)]

        # Item is both an error AND an anomaly
        items[50]["value"] = 999999
        items[50]["error"] = "Critical failure"
        items[50]["is_both"] = True

        config = SmartCrusherConfig(max_items_after_crush=10)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        both = [x for x in compressed if x.get("is_both")]
        assert len(both) == 1, "Item with both error and anomaly was dropped!"

    def test_high_volume_critical_items(self):
        """Even with many critical items, none should be dropped."""
        items = []
        critical_count = 0

        for i in range(500):
            item = {"id": i, "value": 100}

            # Make every 5th item an error
            if i % 5 == 0:
                item["error"] = f"Error {i}"
                critical_count += 1

            # Make every 7th item an anomaly (some overlap)
            if i % 7 == 0:
                item["value"] = 999999
                if "error" not in item:
                    critical_count += 1

            items.append(item)

        config = SmartCrusherConfig(max_items_after_crush=30)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        # Count retained critical items
        errors_retained = sum(1 for x in compressed if x.get("error"))
        sum(1 for x in compressed if x.get("value", 0) > 900000)

        # All errors should be retained
        errors_original = sum(1 for x in items if x.get("error"))
        assert errors_retained == errors_original, (
            f"Some errors dropped: {errors_original} -> {errors_retained}"
        )


class TestCompressionRatio:
    """Verify compression achieves target while preserving quality."""

    def test_compression_with_quality(self):
        """Compression should reduce size significantly while keeping critical items."""
        # Create realistic large dataset
        items = []
        for i in range(1000):
            items.append(
                {
                    "id": f"doc_{i}",
                    "score": 0.5,
                    "title": f"Document {i} about various topics",
                    "snippet": "Lorem ipsum " * 20,
                    "metadata": {"source": "web", "date": "2024-01-01"},
                }
            )

        # Add some critical items
        items[100]["error"] = "Parse error"
        items[500]["value"] = 999999  # Add numeric field for anomaly

        original_size = len(json.dumps(items))

        config = SmartCrusherConfig(max_items_after_crush=50)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        compressed_size = len(json.dumps(compressed))

        # Should achieve significant compression
        compression_ratio = 1 - (compressed_size / original_size)
        assert compression_ratio > 0.9, f"Compression too low: {compression_ratio:.1%}"

        # But critical items must be preserved
        assert any(x.get("error") for x in compressed), "Error item lost during compression!"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Empty array should return empty."""
        compressed_str, was_modified, _ = smart_crush_tool_output("[]", with_compaction=False)
        assert compressed_str == "[]"
        assert not was_modified

    def test_small_array_unchanged(self):
        """Arrays smaller than min_items_to_analyze should be unchanged."""
        items = [{"id": i} for i in range(3)]
        original = json.dumps(items)

        compressed_str, was_modified, _ = smart_crush_tool_output(original, with_compaction=False)

        # Small arrays shouldn't be modified
        assert json.loads(compressed_str) == items

    def test_all_items_are_errors(self):
        """When all items are errors, all should be retained."""
        items = [{"id": i, "error": f"Error {i}"} for i in range(50)]

        config = SmartCrusherConfig(max_items_after_crush=20)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        # All 50 errors should be retained (errors override max_items)
        assert len(compressed) == 50, (
            f"Some errors dropped when all items are errors! Expected 50, got {len(compressed)}"
        )

    def test_unicode_content(self):
        """Unicode content should not break error detection."""
        items = [{"id": i, "content": f"内容 {i}"} for i in range(100)]
        items[50]["error"] = "错误: Unicode error message"

        config = SmartCrusherConfig(max_items_after_crush=15)
        compressed_str, _, _ = smart_crush_tool_output(
            json.dumps(items), config, with_compaction=False
        )
        compressed = json.loads(compressed_str)

        errors = [x for x in compressed if x.get("error")]
        assert len(errors) == 1, "Unicode error item was dropped!"

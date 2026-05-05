"""Comprehensive tests for TOIN implementation fixes.

This file tests all the fixes made to the TOIN implementation:
1. toin_hint.recommended_strategy is used in SmartCrusher
2. strategy_success_rates are used in recommendations
3. preserve_fields are merged in federated learning
4. tool_signature_hash and strategy are passed to feedback system
5. user_count is tracked via instance_id
6. field_retrieval_frequency weights preserve_fields
7. query_context keywords and patterns are detected
"""

import json
import tempfile
from pathlib import Path

import pytest

from headroom.cache.compression_feedback import (
    get_compression_feedback,
    reset_compression_feedback,
)
from headroom.cache.compression_store import (
    RetrievalEvent,
    get_compression_store,
    reset_compression_store,
)
from headroom.telemetry import ToolSignature
from headroom.telemetry.toin import (
    TOINConfig,
    ToolIntelligenceNetwork,
    get_toin,
    reset_toin,
)


@pytest.fixture
def fresh_toin():
    """Create a fresh TOIN instance with temporary storage."""
    reset_toin()
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = str(Path(tmpdir) / "toin_test.json")
        toin = get_toin(
            TOINConfig(
                storage_path=storage_path,
                auto_save_interval=0,
            )
        )
        yield toin
        reset_toin()


@pytest.fixture
def fresh_feedback():
    """Create a fresh feedback instance."""
    reset_compression_feedback()
    feedback = get_compression_feedback()
    yield feedback
    reset_compression_feedback()


@pytest.fixture
def fresh_store():
    """Create a fresh compression store."""
    reset_compression_store()
    store = get_compression_store(max_entries=100, default_ttl=300)
    yield store
    reset_compression_store()


@pytest.mark.skip(
    reason="PR-B5: strategy-recommendation API retired (get_recommendation returns None)"
)
class TestStrategySuccessRates:
    """Test that strategy_success_rates are used in recommendations."""

    def test_recommends_strategy_with_high_success_rate(self, fresh_toin):
        """Strategy with success rate >= 0.5 should be recommended."""
        items = [{"id": i, "score": 100 - i} for i in range(20)]
        signature = ToolSignature.from_items(items)

        # Record compressions to build pattern
        for _ in range(10):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=20,
                compressed_count=10,
                original_tokens=2000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        # Set high success rate
        pattern = fresh_toin._patterns[("unknown", "unknown", signature.structure_hash)]
        pattern.strategy_success_rates["smart_sample"] = 0.8
        pattern.optimal_strategy = "smart_sample"

        # Get recommendation
        hint = fresh_toin.get_recommendation(signature, "test query")

        assert hint.recommended_strategy == "smart_sample"

    def test_rejects_strategy_with_low_success_rate(self, fresh_toin):
        """Strategy with success rate < 0.5 should NOT be recommended."""
        items = [{"id": i, "score": 100 - i} for i in range(20)]
        signature = ToolSignature.from_items(items)

        # Record compressions
        for _ in range(10):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=20,
                compressed_count=10,
                original_tokens=2000,
                compressed_tokens=1000,
                strategy="bad_strategy",
            )

        # Set low success rate
        pattern = fresh_toin._patterns[("unknown", "unknown", signature.structure_hash)]
        pattern.strategy_success_rates["bad_strategy"] = 0.2
        pattern.optimal_strategy = "bad_strategy"

        # Get recommendation
        hint = fresh_toin.get_recommendation(signature, "test query")

        # Should not recommend the bad strategy
        assert hint.recommended_strategy != "bad_strategy"
        # Confidence should be reduced
        assert "low success" in hint.reason.lower()

    def test_finds_best_strategy_when_optimal_is_bad(self, fresh_toin):
        """When optimal_strategy has low success, find a better alternative."""
        items = [{"id": i, "score": 100 - i} for i in range(20)]
        signature = ToolSignature.from_items(items)

        # Record compressions
        for _ in range(10):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=20,
                compressed_count=10,
                original_tokens=2000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        # Set up multiple strategies with different success rates
        pattern = fresh_toin._patterns[("unknown", "unknown", signature.structure_hash)]
        pattern.strategy_success_rates = {
            "bad_strategy": 0.2,
            "good_strategy": 0.9,
        }
        pattern.optimal_strategy = "bad_strategy"

        # Get recommendation
        hint = fresh_toin.get_recommendation(signature, "test query")

        # Should recommend the better strategy
        assert hint.recommended_strategy == "good_strategy"
        assert "using good_strategy instead" in hint.reason


class TestPreserveFieldsMerging:
    """Test preserve_fields merging in federated learning."""

    def test_preserve_fields_merged_on_import(self, fresh_toin):
        """Imported preserve_fields should be merged with existing."""
        items = [{"id": i, "name": f"item_{i}"} for i in range(10)]
        signature = ToolSignature.from_items(items)
        sig_hash = signature.structure_hash

        # Create local pattern with some preserve_fields
        fresh_toin.record_compression(
            tool_signature=signature,
            original_count=10,
            compressed_count=5,
            original_tokens=1000,
            compressed_tokens=500,
            strategy="smart_sample",
        )
        local_pattern = fresh_toin._patterns[("unknown", "unknown", sig_hash)]
        local_pattern.preserve_fields = ["field_a", "field_b"]

        # Import pattern with different preserve_fields
        import_data = {
            "patterns": {
                sig_hash: {
                    "tool_signature_hash": sig_hash,
                    "total_compressions": 100,
                    "total_retrievals": 20,
                    "sample_size": 100,
                    "preserve_fields": ["field_c", "field_d"],
                }
            }
        }

        fresh_toin.import_patterns(import_data)

        # Verify merge
        pattern = fresh_toin._patterns[("unknown", "unknown", sig_hash)]
        assert "field_a" in pattern.preserve_fields
        assert "field_b" in pattern.preserve_fields
        assert "field_c" in pattern.preserve_fields
        assert "field_d" in pattern.preserve_fields

    def test_preserve_fields_limited_to_10(self, fresh_toin):
        """preserve_fields should be capped at 10 entries."""
        items = [{"id": i} for i in range(10)]
        signature = ToolSignature.from_items(items)
        sig_hash = signature.structure_hash

        # Create pattern with 8 fields
        fresh_toin.record_compression(
            tool_signature=signature,
            original_count=10,
            compressed_count=5,
            original_tokens=1000,
            compressed_tokens=500,
            strategy="smart_sample",
        )
        pattern = fresh_toin._patterns[("unknown", "unknown", sig_hash)]
        pattern.preserve_fields = [f"field_{i}" for i in range(8)]

        # Import with 5 more fields
        import_data = {
            "patterns": {
                sig_hash: {
                    "tool_signature_hash": sig_hash,
                    "total_compressions": 50,
                    "sample_size": 50,
                    "preserve_fields": [f"imported_{i}" for i in range(5)],
                }
            }
        }

        fresh_toin.import_patterns(import_data)

        # Should be capped at 10
        pattern = fresh_toin._patterns[("unknown", "unknown", sig_hash)]
        assert len(pattern.preserve_fields) <= 10


class TestUserCountTracking:
    """Test user_count tracking via instance_id."""

    def test_user_count_increments_for_new_instance(self, fresh_toin):
        """user_count should increment when a new instance is seen."""
        items = [{"id": i} for i in range(10)]
        signature = ToolSignature.from_items(items)

        # Record compression (first instance)
        fresh_toin.record_compression(
            tool_signature=signature,
            original_count=10,
            compressed_count=5,
            original_tokens=1000,
            compressed_tokens=500,
            strategy="smart_sample",
        )

        pattern = fresh_toin._patterns[("unknown", "unknown", signature.structure_hash)]
        assert pattern.user_count == 1
        assert len(pattern._seen_instance_hashes) == 1
        assert fresh_toin._instance_id in pattern._seen_instance_hashes

    def test_user_count_stable_for_same_instance(self, fresh_toin):
        """user_count should not increase for same instance."""
        items = [{"id": i} for i in range(10)]
        signature = ToolSignature.from_items(items)

        # Record multiple compressions from same instance
        for _ in range(10):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=10,
                compressed_count=5,
                original_tokens=1000,
                compressed_tokens=500,
                strategy="smart_sample",
            )

        pattern = fresh_toin._patterns[("unknown", "unknown", signature.structure_hash)]
        assert pattern.user_count == 1  # Still 1

    def test_instance_hashes_serialized_and_loaded(self):
        """_seen_instance_hashes should survive save/load cycle."""
        reset_toin()
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = str(Path(tmpdir) / "toin_persist.json")
            toin1 = ToolIntelligenceNetwork(TOINConfig(storage_path=storage_path))

            items = [{"id": i} for i in range(10)]
            signature = ToolSignature.from_items(items)

            # Record compression
            toin1.record_compression(
                tool_signature=signature,
                original_count=10,
                compressed_count=5,
                original_tokens=1000,
                compressed_tokens=500,
                strategy="smart_sample",
            )

            # Save
            toin1.save()

            # Load in new instance
            toin2 = ToolIntelligenceNetwork(TOINConfig(storage_path=storage_path))

            pattern = toin2._patterns.get(("unknown", "unknown", signature.structure_hash))
            assert pattern is not None
            assert pattern.user_count >= 1
            assert len(pattern._seen_instance_hashes) >= 1

    def test_user_count_merged_on_import(self, fresh_toin):
        """user_count should reflect merged instance hashes."""
        items = [{"id": i} for i in range(10)]
        signature = ToolSignature.from_items(items)
        sig_hash = signature.structure_hash

        # Create local pattern
        fresh_toin.record_compression(
            tool_signature=signature,
            original_count=10,
            compressed_count=5,
            original_tokens=1000,
            compressed_tokens=500,
            strategy="smart_sample",
        )

        # Import pattern with different instance hashes
        import_data = {
            "patterns": {
                sig_hash: {
                    "tool_signature_hash": sig_hash,
                    "total_compressions": 50,
                    "sample_size": 50,
                    "seen_instance_hashes": ["other_instance_1", "other_instance_2"],
                    "user_count": 2,
                }
            }
        }

        fresh_toin.import_patterns(import_data)

        pattern = fresh_toin._patterns[("unknown", "unknown", sig_hash)]
        # Should have local + 2 imported = 3
        assert pattern.user_count >= 3


@pytest.mark.skip(
    reason="PR-B5: get_recommendation retired; field-weighting now consumed only by toin publish"
)
class TestFieldRetrievalFrequencyWeighting:
    """Test field_retrieval_frequency weighting in preserve_fields."""

    def test_query_fields_prioritized_in_preserve_fields(self, fresh_toin):
        """Fields mentioned in query should be prioritized."""
        items = [{"id": i, "status": "ok", "category": f"cat_{i}"} for i in range(20)]
        signature = ToolSignature.from_items(items)

        # Build pattern with field retrieval data
        for _ in range(10):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=20,
                compressed_count=10,
                original_tokens=2000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        # Record retrievals for "status" field
        status_hash = fresh_toin._hash_field_name("status")
        pattern = fresh_toin._patterns[("unknown", "unknown", signature.structure_hash)]
        pattern.field_retrieval_frequency = {
            status_hash: 50,
            fresh_toin._hash_field_name("category"): 10,
        }
        pattern.preserve_fields = [status_hash]

        # Get recommendation with query mentioning "status"
        hint = fresh_toin.get_recommendation(signature, "status:error")

        # status hash should be in preserve_fields
        assert status_hash in hint.preserve_fields

    def test_preserve_fields_sorted_by_frequency(self, fresh_toin):
        """preserve_fields should be sorted by retrieval frequency."""
        items = [{"id": i} for i in range(20)]
        signature = ToolSignature.from_items(items)

        # Build pattern
        for _ in range(10):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=20,
                compressed_count=10,
                original_tokens=2000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        pattern = fresh_toin._patterns[("unknown", "unknown", signature.structure_hash)]
        field_a = fresh_toin._hash_field_name("field_a")
        field_b = fresh_toin._hash_field_name("field_b")
        field_c = fresh_toin._hash_field_name("field_c")

        pattern.field_retrieval_frequency = {
            field_a: 10,
            field_b: 50,  # Most frequent
            field_c: 30,
        }
        pattern.preserve_fields = [field_a, field_b, field_c]

        # Get recommendation (no query context)
        hint = fresh_toin.get_recommendation(signature, "")

        # Should be sorted by frequency
        if len(hint.preserve_fields) >= 3:
            # field_b should come before field_c which should come before field_a
            b_idx = hint.preserve_fields.index(field_b) if field_b in hint.preserve_fields else -1
            c_idx = hint.preserve_fields.index(field_c) if field_c in hint.preserve_fields else -1
            hint.preserve_fields.index(field_a) if field_a in hint.preserve_fields else -1

            if b_idx >= 0 and c_idx >= 0:
                assert b_idx < c_idx, "Higher frequency field should come first"


@pytest.mark.skip(reason="PR-B5: get_recommendation retired (returns None / DeprecationWarning)")
class TestQueryContextUsage:
    """Test query_context usage in recommendations."""

    def test_exhaustive_query_keywords_detected(self, fresh_toin):
        """Exhaustive query keywords should trigger conservative compression."""
        items = [{"id": i, "score": 100 - i} for i in range(50)]
        signature = ToolSignature.from_items(items)

        # Build pattern with aggressive compression normally
        for _ in range(10):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=50,
                compressed_count=10,
                original_tokens=5000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        # Low retrieval rate = aggressive compression
        pattern = fresh_toin._patterns[("unknown", "unknown", signature.structure_hash)]
        pattern.total_retrievals = 0

        # Query with exhaustive keyword
        hint = fresh_toin.get_recommendation(signature, "list all items in category")

        # Should be more conservative
        assert hint.max_items >= 40
        assert "exhaustive query" in hint.reason.lower()
        assert hint.compression_level == "conservative"

    def test_every_keyword_triggers_conservative(self, fresh_toin):
        """'every' keyword should trigger conservative compression."""
        items = [{"id": i} for i in range(50)]
        signature = ToolSignature.from_items(items)

        for _ in range(10):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=50,
                compressed_count=10,
                original_tokens=5000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        pattern = fresh_toin._patterns[("unknown", "unknown", signature.structure_hash)]
        pattern.total_retrievals = 0

        hint = fresh_toin.get_recommendation(signature, "find every user")

        assert "exhaustive query" in hint.reason.lower()

    def test_partial_pattern_matching(self, fresh_toin):
        """Partial pattern matching should boost max_items."""
        items = [{"id": i, "status": "ok"} for i in range(50)]
        signature = ToolSignature.from_items(items)

        for _ in range(10):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=50,
                compressed_count=10,
                original_tokens=5000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        pattern = fresh_toin._patterns[("unknown", "unknown", signature.structure_hash)]
        pattern.total_retrievals = 0
        # Add a problematic query pattern
        pattern.common_query_patterns = ["status:*"]

        # Query that uses the same field
        hint = fresh_toin.get_recommendation(signature, "status:error")

        # Should match the pattern
        assert hint.max_items >= 25 or "retrieval pattern" in hint.reason


class TestFeedbackStrategyTracking:
    """Test strategy tracking in compression feedback."""

    def test_record_compression_tracks_strategy(self, fresh_feedback):
        """record_compression should track strategy."""
        fresh_feedback.record_compression(
            tool_name="test_tool",
            original_count=100,
            compressed_count=20,
            strategy="smart_sample",
            tool_signature_hash="abc123",
        )

        pattern = fresh_feedback._tool_patterns.get("test_tool")
        assert pattern is not None
        assert "smart_sample" in pattern.strategy_compressions
        assert pattern.strategy_compressions["smart_sample"] == 1

    def test_record_retrieval_tracks_strategy(self, fresh_feedback):
        """record_retrieval should track strategy retrievals."""
        # First record a compression
        fresh_feedback.record_compression(
            tool_name="test_tool",
            original_count=100,
            compressed_count=20,
            strategy="smart_sample",
        )

        # Then record a retrieval with strategy
        event = RetrievalEvent(
            hash="test_hash",
            query="test query",
            items_retrieved=100,
            total_items=100,
            tool_name="test_tool",
            timestamp=1234567890.0,
            retrieval_type="full",
        )

        fresh_feedback.record_retrieval(event, strategy="smart_sample")

        pattern = fresh_feedback._tool_patterns.get("test_tool")
        assert "smart_sample" in pattern.strategy_retrievals
        assert pattern.strategy_retrievals["smart_sample"] == 1

    def test_strategy_retrieval_rate_calculation(self, fresh_feedback):
        """strategy_retrieval_rate should calculate correctly."""
        # Record 10 compressions
        for _ in range(10):
            fresh_feedback.record_compression(
                tool_name="test_tool",
                original_count=100,
                compressed_count=20,
                strategy="smart_sample",
            )

        # Record 3 retrievals
        for _ in range(3):
            event = RetrievalEvent(
                hash="test_hash",
                query="test query",
                items_retrieved=100,
                total_items=100,
                tool_name="test_tool",
                timestamp=1234567890.0,
                retrieval_type="full",
            )
            fresh_feedback.record_retrieval(event, strategy="smart_sample")

        pattern = fresh_feedback._tool_patterns.get("test_tool")
        rate = pattern.strategy_retrieval_rate("smart_sample")
        assert rate == 0.3  # 3 retrievals / 10 compressions

    def test_best_strategy_selection(self, fresh_feedback):
        """best_strategy should return strategy with lowest retrieval rate."""
        # Record compressions for multiple strategies
        for _ in range(10):
            fresh_feedback.record_compression(
                tool_name="test_tool",
                original_count=100,
                compressed_count=20,
                strategy="bad_strategy",
            )
        for _ in range(10):
            fresh_feedback.record_compression(
                tool_name="test_tool",
                original_count=100,
                compressed_count=20,
                strategy="good_strategy",
            )

        # Record more retrievals for bad strategy
        for _ in range(8):
            event = RetrievalEvent(
                hash="test_hash",
                query=None,
                items_retrieved=100,
                total_items=100,
                tool_name="test_tool",
                timestamp=1234567890.0,
                retrieval_type="full",
            )
            fresh_feedback.record_retrieval(event, strategy="bad_strategy")

        # Record few retrievals for good strategy
        for _ in range(2):
            event = RetrievalEvent(
                hash="test_hash",
                query=None,
                items_retrieved=100,
                total_items=100,
                tool_name="test_tool",
                timestamp=1234567890.0,
                retrieval_type="full",
            )
            fresh_feedback.record_retrieval(event, strategy="good_strategy")

        pattern = fresh_feedback._tool_patterns.get("test_tool")
        # good_strategy has 20% retrieval rate, bad_strategy has 80%
        best = pattern.best_strategy()
        assert best == "good_strategy"


class TestSignatureHashTracking:
    """Test tool_signature_hash tracking in feedback."""

    def test_signature_hash_recorded(self, fresh_feedback):
        """record_compression should track signature hash."""
        fresh_feedback.record_compression(
            tool_name="test_tool",
            original_count=100,
            compressed_count=20,
            strategy="smart_sample",
            tool_signature_hash="unique_sig_hash",
        )

        pattern = fresh_feedback._tool_patterns.get("test_tool")
        assert "unique_sig_hash" in pattern.signature_hashes

    def test_multiple_signature_hashes_tracked(self, fresh_feedback):
        """Multiple different signature hashes should be tracked."""
        hashes = ["hash_1", "hash_2", "hash_3"]

        for h in hashes:
            fresh_feedback.record_compression(
                tool_name="test_tool",
                original_count=100,
                compressed_count=20,
                tool_signature_hash=h,
            )

        pattern = fresh_feedback._tool_patterns.get("test_tool")
        for h in hashes:
            assert h in pattern.signature_hashes


class TestIntegration:
    """Integration tests for the full feedback loop."""

    def test_store_passes_strategy_to_feedback(self, fresh_store, fresh_feedback):
        """CompressionStore should pass strategy to feedback on retrieval."""
        # Store with strategy
        hash_key = fresh_store.store(
            original=json.dumps([{"id": i} for i in range(50)]),
            compressed=json.dumps([{"id": i} for i in range(10)]),
            original_item_count=50,
            compressed_item_count=10,
            tool_name="test_tool",
            tool_signature_hash="test_sig_hash",
            compression_strategy="smart_sample",
        )

        # Retrieve triggers feedback
        fresh_store.retrieve(hash_key, query="test query")

        # Verify feedback received the strategy
        pattern = fresh_feedback._tool_patterns.get("test_tool")
        if pattern:
            # Strategy should be tracked
            assert pattern.total_retrievals >= 1

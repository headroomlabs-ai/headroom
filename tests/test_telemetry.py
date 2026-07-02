"""Tests for telemetry module (data flywheel)."""

import os
import tempfile

import pytest

from headroom.telemetry import (
    AnonymizedToolStats,
    FieldDistribution,
    RetrievalStats,
    TelemetryCollector,
    TelemetryConfig,
    ToolSignature,
    get_telemetry_collector,
    reset_telemetry_collector,
)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before each test."""
    reset_telemetry_collector()
    yield
    reset_telemetry_collector()


class TestFieldDistribution:
    """Test FieldDistribution data model."""

    def test_to_dict(self):
        """to_dict serializes all fields."""
        dist = FieldDistribution(
            field_name_hash="abc12345",
            field_type="string",
            avg_length=50.5,
            unique_ratio=0.8,
            looks_like_id=True,
        )

        d = dist.to_dict()

        assert d["field_name_hash"] == "abc12345"
        assert d["field_type"] == "string"
        assert d["avg_length"] == 50.5
        assert d["unique_ratio"] == 0.8
        assert d["looks_like_id"] is True

    def test_from_dict(self):
        """from_dict deserializes correctly."""
        data = {
            "field_name_hash": "xyz789",
            "field_type": "numeric",
            "has_variance": True,
            "variance_bucket": "high",
        }

        dist = FieldDistribution.from_dict(data)

        assert dist.field_name_hash == "xyz789"
        assert dist.field_type == "numeric"
        assert dist.has_variance is True
        assert dist.variance_bucket == "high"


class TestToolSignature:
    """Test ToolSignature data model."""

    def test_from_items_empty_list(self):
        """Empty list produces valid signature with unique hash.

        HIGH FIX #5: Empty lists now get a proper hash instead of 'empty'
        to prevent hash collisions between different empty-list scenarios.
        """
        sig = ToolSignature.from_items([])

        # Should get a proper hash, not 'empty' (which could cause collisions)
        assert sig.structure_hash != "empty"
        assert len(sig.structure_hash) == 24  # Our hash length
        assert sig.field_count == 0

    def test_from_items_single_item(self):
        """Single item produces valid signature."""
        items = [{"id": "123", "name": "test", "score": 0.95}]

        sig = ToolSignature.from_items(items)

        assert sig.field_count == 3
        assert sig.string_field_count == 2  # id, name
        assert sig.numeric_field_count == 1  # score
        assert sig.has_id_like_field is True
        assert sig.has_score_like_field is True

    def test_from_items_with_nested_objects(self):
        """Nested objects are detected."""
        items = [{"data": {"nested": "value"}}]

        sig = ToolSignature.from_items(items)

        assert sig.has_nested_objects is True
        assert sig.object_field_count == 1

    def test_from_items_with_arrays(self):
        """Arrays are detected."""
        items = [{"tags": ["a", "b", "c"]}]

        sig = ToolSignature.from_items(items)

        assert sig.has_arrays is True
        assert sig.array_field_count == 1

    def test_structure_hash_consistency(self):
        """Same structure produces same hash."""
        items1 = [{"id": "123", "name": "alice"}]
        items2 = [{"id": "456", "name": "bob"}]

        sig1 = ToolSignature.from_items(items1)
        sig2 = ToolSignature.from_items(items2)

        assert sig1.structure_hash == sig2.structure_hash

    def test_structure_hash_differs_for_different_structure(self):
        """Different structure produces different hash."""
        items1 = [{"id": "123", "name": "alice"}]
        items2 = [{"id": "123", "score": 0.5}]  # Different fields

        sig1 = ToolSignature.from_items(items1)
        sig2 = ToolSignature.from_items(items2)

        assert sig1.structure_hash != sig2.structure_hash

    def test_pattern_detection_timestamp(self):
        """Timestamp-like fields are detected."""
        items = [{"created_at": 1234567890, "updated_at": 1234567891}]

        sig = ToolSignature.from_items(items)

        assert sig.has_timestamp_like_field is True

    def test_pattern_detection_status(self):
        """Status-like fields are detected."""
        items = [{"status": "pending", "state": "active"}]

        sig = ToolSignature.from_items(items)

        assert sig.has_status_like_field is True

    def test_pattern_detection_error(self):
        """Error-like fields are detected."""
        items = [{"error": "Not found", "error_code": 404}]

        sig = ToolSignature.from_items(items)

        assert sig.has_error_like_field is True

    def test_pattern_detection_message(self):
        """Message-like fields are detected."""
        items = [{"message": "Success", "description": "Task completed"}]

        sig = ToolSignature.from_items(items)

        assert sig.has_message_like_field is True


class TestTelemetryCollector:
    """Test TelemetryCollector class."""

    def test_record_compression(self):
        """Recording compression updates stats."""
        collector = TelemetryCollector()

        items = [{"id": "1", "name": "test"}, {"id": "2", "name": "test2"}]
        collector.record_compression(
            items=items,
            original_count=100,
            compressed_count=10,
            original_tokens=5000,
            compressed_tokens=500,
            strategy="top_n",
        )

        stats = collector.get_stats()
        assert stats["total_compressions"] == 1
        assert stats["total_tokens_saved"] == 4500

    def test_record_compression_disabled(self):
        """Disabled telemetry does not record."""
        config = TelemetryConfig(enabled=False)
        collector = TelemetryCollector(config)

        items = [{"id": "1"}]
        collector.record_compression(
            items=items,
            original_count=100,
            compressed_count=10,
            original_tokens=5000,
            compressed_tokens=500,
            strategy="top_n",
        )

        stats = collector.get_stats()
        assert stats["total_compressions"] == 0

    def test_record_retrieval(self):
        """Recording retrieval updates stats."""
        collector = TelemetryCollector()

        # First record a compression to create the signature
        items = [{"id": "1", "name": "test"}]
        collector.record_compression(
            items=items,
            original_count=100,
            compressed_count=10,
            original_tokens=5000,
            compressed_tokens=500,
            strategy="top_n",
        )

        # Get the signature hash
        all_stats = collector.get_all_tool_stats()
        sig_hash = list(all_stats.keys())[0]

        # Record retrieval
        collector.record_retrieval(
            tool_signature_hash=sig_hash,
            retrieval_type="full",
        )

        stats = collector.get_stats()
        assert stats["total_retrievals"] == 1

    def test_tool_stats_aggregation(self):
        """Multiple compressions aggregate correctly."""
        collector = TelemetryCollector()

        items = [{"id": "1", "name": "test"}]

        # Record 5 compressions
        for i in range(5):
            collector.record_compression(
                items=items,
                original_count=100,
                compressed_count=10 + i,  # Vary slightly
                original_tokens=5000,
                compressed_tokens=500 + i * 10,
                strategy="top_n",
            )

        # Check aggregation
        all_stats = collector.get_all_tool_stats()
        assert len(all_stats) == 1  # Same structure, same signature

        sig_hash = list(all_stats.keys())[0]
        tool_stats = all_stats[sig_hash]
        assert tool_stats.total_compressions == 5
        assert tool_stats.sample_size == 5

    def test_different_tools_tracked_separately(self):
        """Different tool structures are tracked separately."""
        collector = TelemetryCollector()

        # Tool A structure
        items_a = [{"id": "1", "name": "test"}]
        collector.record_compression(
            items=items_a,
            original_count=100,
            compressed_count=10,
            original_tokens=5000,
            compressed_tokens=500,
            strategy="top_n",
        )

        # Tool B structure (different fields)
        items_b = [{"code": 200, "result": {"data": "value"}}]
        collector.record_compression(
            items=items_b,
            original_count=50,
            compressed_count=5,
            original_tokens=2500,
            compressed_tokens=250,
            strategy="smart_sample",
        )

        all_stats = collector.get_all_tool_stats()
        assert len(all_stats) == 2

    def test_strategy_counts(self):
        """Strategy usage is tracked."""
        collector = TelemetryCollector()

        items = [{"id": "1"}]

        # Different strategies
        collector.record_compression(
            items=items,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )
        collector.record_compression(
            items=items,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="smart_sample",
        )
        collector.record_compression(
            items=items,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        all_stats = collector.get_all_tool_stats()
        sig_hash = list(all_stats.keys())[0]
        tool_stats = all_stats[sig_hash]

        assert tool_stats.strategy_counts["top_n"] == 2
        assert tool_stats.strategy_counts["smart_sample"] == 1

    def test_recommendations_insufficient_samples(self):
        """No recommendations with insufficient samples."""
        config = TelemetryConfig(min_samples_for_recommendation=10)
        collector = TelemetryCollector(config)

        items = [{"id": "1"}]
        for _ in range(5):  # Less than 10
            collector.record_compression(
                items=items,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="top_n",
            )

        all_stats = collector.get_all_tool_stats()
        sig_hash = list(all_stats.keys())[0]

        recommendations = collector.get_recommendations(sig_hash)
        assert recommendations is None

    def test_recommendations_with_sufficient_samples(self):
        """Recommendations provided with sufficient samples."""
        config = TelemetryConfig(min_samples_for_recommendation=5)
        collector = TelemetryCollector(config)

        items = [{"id": "1"}]
        for _ in range(10):
            collector.record_compression(
                items=items,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="top_n",
            )

        all_stats = collector.get_all_tool_stats()
        sig_hash = list(all_stats.keys())[0]

        recommendations = collector.get_recommendations(sig_hash)
        assert recommendations is not None
        assert "signature_hash" in recommendations
        assert "confidence" in recommendations

    def test_export_stats(self):
        """Export produces complete telemetry data."""
        collector = TelemetryCollector()

        items = [{"id": "1", "name": "test"}]
        collector.record_compression(
            items=items,
            original_count=100,
            compressed_count=10,
            original_tokens=5000,
            compressed_tokens=500,
            strategy="top_n",
        )

        export = collector.export_stats()

        assert "version" in export
        assert "export_timestamp" in export
        assert "summary" in export
        assert "tool_stats" in export
        assert export["summary"]["total_compressions"] == 1

    def test_import_stats(self):
        """Import merges telemetry data."""
        collector1 = TelemetryCollector()
        collector2 = TelemetryCollector()

        items = [{"id": "1"}]

        # Collector 1 records some compressions
        for _ in range(5):
            collector1.record_compression(
                items=items,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="top_n",
            )

        # Export from collector 1
        export_data = collector1.export_stats()

        # Collector 2 records different compressions
        for _ in range(3):
            collector2.record_compression(
                items=items,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="smart_sample",
            )

        # Import into collector 2
        collector2.import_stats(export_data)

        # Check merged data
        all_stats = collector2.get_all_tool_stats()
        sig_hash = list(all_stats.keys())[0]
        tool_stats = all_stats[sig_hash]

        assert tool_stats.sample_size == 8  # 5 + 3

    def test_clear_resets_state(self):
        """clear() removes all telemetry data."""
        collector = TelemetryCollector()

        items = [{"id": "1"}]
        collector.record_compression(
            items=items,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        collector.clear()

        stats = collector.get_stats()
        assert stats["total_compressions"] == 0
        assert stats["tool_signatures_tracked"] == 0

    def test_field_distribution_analysis(self):
        """Field distributions are analyzed correctly."""
        config = TelemetryConfig(include_field_distributions=True)
        collector = TelemetryCollector(config)

        items = [
            {"id": "abc123", "score": 0.95, "tags": ["a", "b"]},
            {"id": "xyz789", "score": 0.80, "tags": ["c"]},
            {"id": "def456", "score": 0.70, "tags": ["d", "e", "f"]},
        ]

        collector.record_compression(
            items=items,
            original_count=100,
            compressed_count=10,
            original_tokens=5000,
            compressed_tokens=500,
            strategy="top_n",
        )

        export = collector.export_stats()
        tool_stats_dict = list(export["tool_stats"].values())[0]

        # Field distributions should be captured in events
        # (Note: We don't store events in export by default, just stats)
        assert tool_stats_dict["avg_compression_ratio"] > 0

    def test_max_events_limit(self):
        """Events are limited to max_events_in_memory."""
        config = TelemetryConfig(max_events_in_memory=5)
        collector = TelemetryCollector(config)

        items = [{"id": "1"}]

        # Record more than max events
        for i in range(10):
            collector.record_compression(
                items=items,
                original_count=100 + i,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="top_n",
            )

        # Events should be limited (internal detail)
        assert len(collector._events) <= 5


class TestTelemetryPersistence:
    """Test telemetry persistence to disk."""

    def test_save_and_load(self):
        """Save and load preserves telemetry data."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            storage_path = f.name

        try:
            # Create and populate collector
            config = TelemetryConfig(storage_path=storage_path)
            collector = TelemetryCollector(config)

            items = [{"id": "1", "name": "test"}]
            for _ in range(3):
                collector.record_compression(
                    items=items,
                    original_count=100,
                    compressed_count=10,
                    original_tokens=1000,
                    compressed_tokens=100,
                    strategy="top_n",
                )

            collector.save()

            # Create new collector that loads from disk
            collector2 = TelemetryCollector(config)

            stats = collector2.get_stats()
            assert stats["total_compressions"] == 3

        finally:
            os.unlink(storage_path)


class TestGlobalTelemetryCollector:
    """Test global telemetry collector singleton."""

    def test_singleton_returns_same_instance(self):
        """get_telemetry_collector returns same instance."""
        collector1 = get_telemetry_collector()
        collector2 = get_telemetry_collector()

        assert collector1 is collector2

    def test_reset_clears_singleton(self):
        """reset_telemetry_collector creates new instance."""
        collector1 = get_telemetry_collector()
        items = [{"id": "1"}]
        collector1.record_compression(
            items=items,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        reset_telemetry_collector()

        collector2 = get_telemetry_collector()
        stats = collector2.get_stats()
        assert stats["total_compressions"] == 0

    def test_env_var_disables_telemetry(self, monkeypatch):
        """HEADROOM_TELEMETRY_DISABLED environment variable disables telemetry."""
        reset_telemetry_collector()
        monkeypatch.setenv("HEADROOM_TELEMETRY_DISABLED", "1")

        collector = get_telemetry_collector()

        assert collector._config.enabled is False

    @pytest.mark.parametrize("off_value", ["off", "false", "0", "no", "disable", "disabled"])
    def test_headroom_telemetry_off_disables_collector(self, monkeypatch, off_value):
        """HEADROOM_TELEMETRY=off (and other documented opt-out values) disables
        the collector — closes #390.

        Pre-#390 the collector only honoured HEADROOM_TELEMETRY_DISABLED, which
        is undocumented. Users following the docs set HEADROOM_TELEMETRY=off and
        watched /v1/telemetry continue to report enabled=true. The collector now
        consults `is_telemetry_enabled()` (the documented opt-in predicate),
        so both env vars take effect.
        """
        reset_telemetry_collector()
        monkeypatch.delenv("HEADROOM_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("HEADROOM_TELEMETRY", off_value)

        collector = get_telemetry_collector()

        assert collector._config.enabled is False, (
            f"HEADROOM_TELEMETRY={off_value!r} must disable the collector — "
            "this is the documented opt-out path. If this assertion fails the "
            "collector is silently ignoring the user's opt-out and /v1/telemetry "
            "will report enabled=true even when telemetry is supposed to be off."
        )

    def test_headroom_telemetry_on_keeps_collector_enabled(self, monkeypatch):
        """Sanity check: the explicit opt-in path (HEADROOM_TELEMETRY=on) leaves
        the collector enabled. Telemetry is off by default, so this requires the
        user to have turned it on."""
        reset_telemetry_collector()
        monkeypatch.delenv("HEADROOM_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")

        collector = get_telemetry_collector()

        assert collector._config.enabled is True


class TestRetrievalStatsModel:
    """Test RetrievalStats data model."""

    def test_retrieval_rate_calculation(self):
        """Retrieval rate is calculated correctly."""
        stats = RetrievalStats(
            tool_signature_hash="abc123",
            total_compressions=100,
            total_retrievals=30,
        )

        assert stats.retrieval_rate == 0.3

    def test_retrieval_rate_zero_compressions(self):
        """Retrieval rate is 0 when no compressions."""
        stats = RetrievalStats(
            tool_signature_hash="abc123",
            total_compressions=0,
        )

        assert stats.retrieval_rate == 0.0

    def test_full_retrieval_rate_calculation(self):
        """Full retrieval rate is calculated correctly."""
        stats = RetrievalStats(
            tool_signature_hash="abc123",
            total_retrievals=20,
            full_retrievals=15,
        )

        assert stats.full_retrieval_rate == 0.75

    def test_to_dict(self):
        """to_dict includes derived properties."""
        stats = RetrievalStats(
            tool_signature_hash="abc123",
            total_compressions=100,
            total_retrievals=50,
            full_retrievals=40,
            search_retrievals=10,
        )

        d = stats.to_dict()

        assert d["retrieval_rate"] == 0.5
        assert d["full_retrieval_rate"] == 0.8


class TestAnonymizedToolStats:
    """Test AnonymizedToolStats data model."""

    def test_to_dict(self):
        """to_dict serializes all fields."""
        sig = ToolSignature(
            structure_hash="abc123",
            field_count=3,
            has_nested_objects=False,
            has_arrays=False,
            max_depth=1,
        )
        stats = AnonymizedToolStats(
            signature=sig,
            total_compressions=100,
            total_items_seen=10000,
            total_items_kept=500,
            avg_compression_ratio=0.05,
        )

        d = stats.to_dict()

        assert d["signature"]["structure_hash"] == "abc123"
        assert d["total_compressions"] == 100
        assert d["avg_compression_ratio"] == 0.05

    def test_from_dict(self):
        """from_dict deserializes correctly."""
        data = {
            "signature": {
                "structure_hash": "xyz789",
                "field_count": 5,
                "has_nested_objects": True,
                "has_arrays": False,
                "max_depth": 2,
            },
            "total_compressions": 50,
            "sample_size": 50,
            "confidence": 0.5,
        }

        stats = AnonymizedToolStats.from_dict(data)

        assert stats.signature.structure_hash == "xyz789"
        assert stats.total_compressions == 50
        assert stats.confidence == 0.5

    def test_from_dict_does_not_mutate_input(self):
        """from_dict does not modify the input dictionary."""
        data = {
            "signature": {
                "structure_hash": "abc123",
                "field_count": 3,
                "has_nested_objects": False,
                "has_arrays": False,
                "max_depth": 1,
            },
            "total_compressions": 10,
            "strategy_counts": {"top_n": 5, "smart_sample": 5},
            "recommended_preserve_fields": ["field1", "field2"],
        }

        # Make a deep copy to compare after
        import copy

        original_data = copy.deepcopy(data)

        stats = AnonymizedToolStats.from_dict(data)

        # Modify the stats object
        stats.strategy_counts["new_strategy"] = 10
        stats.recommended_preserve_fields.append("field3")

        # Original data should be unchanged
        assert data == original_data
        assert "new_strategy" not in data["strategy_counts"]
        assert "field3" not in data["recommended_preserve_fields"]

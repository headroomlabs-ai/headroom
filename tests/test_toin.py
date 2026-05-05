"""Tests for Tool Output Intelligence Network (TOIN).

PR-B5 retired the request-time hint API. Tests that exercised the old
`get_recommendation()` / `CompressionHint` shape are skipped at module
level — the new observation-only contract is covered by
`tests/test_toin_observation_only.py` and `tests/test_toin_publish.py`.
"""

import os
import tempfile
import time

import pytest

from headroom.telemetry import (
    TOINConfig,
    ToolIntelligenceNetwork,
    ToolPattern,
    ToolSignature,
    get_toin,
    reset_toin,
)


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch, tmp_path):
    """Reset global state before each test.

    Also disables disk persistence by setting HEADROOM_TOIN_PATH to a temp file
    to avoid loading stale data from ~/.headroom/toin.json.
    """
    # Use a unique temp file for each test to avoid cross-test contamination
    temp_toin_path = str(tmp_path / "toin_test.json")
    monkeypatch.setenv("HEADROOM_TOIN_PATH", temp_toin_path)
    reset_toin()
    yield
    reset_toin()


class TestToolPattern:
    """Test ToolPattern data model."""

    def test_to_dict(self):
        """to_dict serializes all fields."""
        pattern = ToolPattern(
            tool_signature_hash="abc12345",
            total_compressions=100,
            total_items_seen=5000,
            total_items_kept=500,
            avg_compression_ratio=0.1,
            avg_token_reduction=0.8,
            total_retrievals=20,
            full_retrievals=15,
            search_retrievals=5,
            commonly_retrieved_fields=["field1", "field2"],
            optimal_strategy="top_n",
            optimal_max_items=25,
            sample_size=100,
            confidence=0.75,
        )

        d = pattern.to_dict()

        assert d["tool_signature_hash"] == "abc12345"
        assert d["total_compressions"] == 100
        assert d["total_items_seen"] == 5000
        assert d["avg_compression_ratio"] == 0.1
        assert d["retrieval_rate"] == 0.2  # 20/100
        assert d["full_retrieval_rate"] == 0.75  # 15/20
        assert d["commonly_retrieved_fields"] == ["field1", "field2"]
        assert d["optimal_strategy"] == "top_n"

    def test_from_dict(self):
        """from_dict deserializes correctly."""
        data = {
            "tool_signature_hash": "xyz789",
            "total_compressions": 50,
            "total_retrievals": 10,
            "full_retrievals": 8,
            "commonly_retrieved_fields": ["field_a"],
            "optimal_max_items": 30,
            "confidence": 0.6,
        }

        pattern = ToolPattern.from_dict(data)

        assert pattern.tool_signature_hash == "xyz789"
        assert pattern.total_compressions == 50
        assert pattern.total_retrievals == 10
        assert pattern.full_retrievals == 8
        assert pattern.commonly_retrieved_fields == ["field_a"]
        assert pattern.optimal_max_items == 30
        assert pattern.confidence == 0.6

    def test_from_dict_ignores_unknown_fields(self):
        """from_dict ignores unknown fields."""
        data = {
            "tool_signature_hash": "abc123",
            "total_compressions": 10,
            "unknown_field": "should be ignored",
            "another_unknown": 12345,
        }

        pattern = ToolPattern.from_dict(data)

        assert pattern.tool_signature_hash == "abc123"
        assert not hasattr(pattern, "unknown_field")

    def test_retrieval_rate_property(self):
        """retrieval_rate is calculated correctly."""
        pattern = ToolPattern(
            tool_signature_hash="test",
            total_compressions=100,
            total_retrievals=30,
        )

        assert pattern.retrieval_rate == 0.3

    def test_retrieval_rate_zero_compressions(self):
        """retrieval_rate is 0 when no compressions."""
        pattern = ToolPattern(
            tool_signature_hash="test",
            total_compressions=0,
        )

        assert pattern.retrieval_rate == 0.0

    def test_full_retrieval_rate_property(self):
        """full_retrieval_rate is calculated correctly."""
        pattern = ToolPattern(
            tool_signature_hash="test",
            total_retrievals=20,
            full_retrievals=15,
        )

        assert pattern.full_retrieval_rate == 0.75

    def test_full_retrieval_rate_zero_retrievals(self):
        """full_retrieval_rate is 0 when no retrievals."""
        pattern = ToolPattern(
            tool_signature_hash="test",
            total_retrievals=0,
        )

        assert pattern.full_retrieval_rate == 0.0


@pytest.mark.skip(
    reason=(
        "PR-B5: CompressionHint is now private (_CompressionHint) and "
        "the request-time hint API is retired. See "
        "tests/test_toin_observation_only.py for the replacement contract."
    )
)
class TestCompressionHint:
    """Retired: CompressionHint was the public envelope for the
    request-time hint API removed in PR-B5."""

    def test_default_values(self):
        pass

    def test_custom_values(self):
        pass


class TestTOINConfig:
    """Test TOINConfig data model."""

    def test_default_values(self):
        """Default config values."""
        config = TOINConfig()

        assert config.enabled is True
        # Storage path comes from HEADROOM_TOIN_PATH env var (set by fixture) or default
        # Just verify it's a non-empty string
        assert isinstance(config.storage_path, str)
        assert len(config.storage_path) > 0
        assert config.auto_save_interval == 600
        assert config.min_samples_for_recommendation == 10
        assert config.min_users_for_network_effect == 3
        assert config.high_retrieval_threshold == 0.5
        assert config.medium_retrieval_threshold == 0.2
        assert config.anonymize_queries is True

    def test_custom_values(self):
        """Custom config values."""
        config = TOINConfig(
            enabled=False,
            storage_path="/tmp/toin.json",
            min_samples_for_recommendation=5,
            high_retrieval_threshold=0.7,
        )

        assert config.enabled is False
        assert config.storage_path == "/tmp/toin.json"
        assert config.min_samples_for_recommendation == 5
        assert config.high_retrieval_threshold == 0.7


class TestToolIntelligenceNetwork:
    """Test ToolIntelligenceNetwork class."""

    def test_record_compression(self):
        """Recording compression updates pattern."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1", "name": "test"}])
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=5000,
            compressed_tokens=500,
            strategy="top_n",
        )

        pattern = toin.get_pattern(sig.structure_hash)
        assert pattern is not None
        assert pattern.total_compressions == 1
        assert pattern.total_items_seen == 100
        assert pattern.total_items_kept == 10
        assert pattern.avg_compression_ratio == 0.1

    def test_record_compression_disabled(self):
        """Disabled TOIN does not record."""
        config = TOINConfig(enabled=False)
        toin = ToolIntelligenceNetwork(config)

        sig = ToolSignature.from_items([{"id": "1"}])
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        pattern = toin.get_pattern(sig.structure_hash)
        assert pattern is None

    def test_record_compression_multiple(self):
        """Multiple compressions update rolling averages."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1"}])

        # Record 5 compressions with varying ratios
        for i in range(5):
            toin.record_compression(
                tool_signature=sig,
                original_count=100,
                compressed_count=10 + i * 5,  # 10, 15, 20, 25, 30
                original_tokens=1000,
                compressed_tokens=100 + i * 50,
                strategy="top_n",
            )

        pattern = toin.get_pattern(sig.structure_hash)
        assert pattern.total_compressions == 5
        assert pattern.sample_size == 5
        assert pattern.total_items_seen == 500  # 100 * 5
        # Average compression ratio: (0.1 + 0.15 + 0.2 + 0.25 + 0.3) / 5 = 0.2
        assert 0.19 < pattern.avg_compression_ratio < 0.21

    def test_record_retrieval(self):
        """Recording retrieval updates pattern."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1"}])
        sig_hash = sig.structure_hash

        # First record compression
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        # Then record retrieval
        toin.record_retrieval(
            tool_signature_hash=sig_hash,
            retrieval_type="full",
        )

        pattern = toin.get_pattern(sig_hash)
        assert pattern.total_retrievals == 1
        assert pattern.full_retrievals == 1
        assert pattern.search_retrievals == 0
        assert pattern.retrieval_rate == 1.0  # 1/1

    def test_record_retrieval_search(self):
        """Search retrievals are tracked separately."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1"}])
        sig_hash = sig.structure_hash

        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        # Record search retrieval with query
        toin.record_retrieval(
            tool_signature_hash=sig_hash,
            retrieval_type="search",
            query="status:error",
            query_fields=["status"],
        )

        pattern = toin.get_pattern(sig_hash)
        assert pattern.total_retrievals == 1
        assert pattern.full_retrievals == 0
        assert pattern.search_retrievals == 1

    def test_record_retrieval_tracks_query_fields(self):
        """Query fields are tracked (anonymized)."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1", "status": "ok"}])
        sig_hash = sig.structure_hash

        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        # Record multiple retrievals for same field
        for _ in range(5):
            toin.record_retrieval(
                tool_signature_hash=sig_hash,
                retrieval_type="search",
                query_fields=["status"],
            )

        pattern = toin.get_pattern(sig_hash)
        # Field should be in commonly_retrieved_fields after 3+ retrievals
        assert len(pattern.commonly_retrieved_fields) > 0

    # PR-B5: the following tests exercised the request-time hint API
    # that's now retired. They're skipped wholesale; the new contract
    # ("get_recommendation always returns None and emits a deprecation
    # warning") is covered by tests/test_toin_observation_only.py.

    @pytest.mark.skip(
        reason="PR-B5: get_recommendation retired — see test_toin_observation_only.py"
    )
    def test_get_recommendation_no_data(self):
        pass

    @pytest.mark.skip(
        reason="PR-B5: get_recommendation retired — see test_toin_observation_only.py"
    )
    def test_get_recommendation_insufficient_samples(self):
        pass

    @pytest.mark.skip(
        reason="PR-B5: get_recommendation retired — see test_toin_observation_only.py"
    )
    def test_get_recommendation_aggressive_compression(self):
        pass

    @pytest.mark.skip(
        reason="PR-B5: get_recommendation retired — see test_toin_observation_only.py"
    )
    def test_get_recommendation_conservative_compression(self):
        pass

    @pytest.mark.skip(
        reason="PR-B5: get_recommendation retired — see test_toin_observation_only.py"
    )
    def test_get_recommendation_skip_compression(self):
        pass

    @pytest.mark.skip(
        reason="PR-B5: get_recommendation retired — see test_toin_observation_only.py"
    )
    def test_get_recommendation_disabled(self):
        pass

    def test_get_stats(self):
        """get_stats returns overall statistics."""
        toin = ToolIntelligenceNetwork()

        sig1 = ToolSignature.from_items([{"id": "1", "name": "test"}])
        sig2 = ToolSignature.from_items([{"code": 200, "data": {"x": 1}}])

        # Record compressions for two different tool types
        for _ in range(5):
            toin.record_compression(
                tool_signature=sig1,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="top_n",
            )

        for _ in range(3):
            toin.record_compression(
                tool_signature=sig2,
                original_count=50,
                compressed_count=5,
                original_tokens=500,
                compressed_tokens=50,
                strategy="smart_sample",
            )

        # Record some retrievals
        toin.record_retrieval(sig1.structure_hash, "full")
        toin.record_retrieval(sig2.structure_hash, "search")

        stats = toin.get_stats()
        assert stats["patterns_tracked"] == 2
        assert stats["total_compressions"] == 8  # 5 + 3
        assert stats["total_retrievals"] == 2
        assert stats["enabled"] is True

    def test_clear(self):
        """clear() removes all patterns."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1"}])
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        toin.clear()

        stats = toin.get_stats()
        assert stats["patterns_tracked"] == 0
        assert stats["total_compressions"] == 0


class TestTOINExportImport:
    """Test TOIN export/import for federated learning."""

    def test_export_patterns(self):
        """export_patterns produces complete data."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1", "name": "test"}])
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        export = toin.export_patterns()

        assert "version" in export
        assert "export_timestamp" in export
        assert "instance_id" in export
        assert "patterns" in export
        assert len(export["patterns"]) == 1
        # PR-B5: keys are now serialized "auth|model|hash" tuples; default
        # auth/model produce the "unknown|unknown|<hash>" string.
        assert f"unknown|unknown|{sig.structure_hash}" in export["patterns"]

    def test_import_patterns_new_pattern(self):
        """import_patterns adds new patterns."""
        toin = ToolIntelligenceNetwork()

        # Import pattern data
        import_data = {
            "version": "1.0",
            "export_timestamp": time.time(),
            "instance_id": "other_instance",
            "patterns": {
                "abc123": {
                    "tool_signature_hash": "abc123",
                    "total_compressions": 50,
                    "total_retrievals": 10,
                    "sample_size": 50,
                    "confidence": 0.5,
                },
            },
        }

        toin.import_patterns(import_data)

        pattern = toin.get_pattern("abc123")
        assert pattern is not None
        assert pattern.total_compressions == 50
        assert pattern.user_count >= 1

    def test_import_patterns_merge_existing(self):
        """import_patterns merges with existing patterns."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1"}])

        # Record local compressions
        for _ in range(10):
            toin.record_compression(
                tool_signature=sig,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="top_n",
            )

        # Import similar pattern from another instance
        import_data = {
            "version": "1.0",
            "export_timestamp": time.time(),
            "instance_id": "other_instance",
            "patterns": {
                sig.structure_hash: {
                    "tool_signature_hash": sig.structure_hash,
                    "total_compressions": 20,
                    "total_retrievals": 5,
                    "total_items_seen": 2000,
                    "total_items_kept": 200,
                    "sample_size": 20,
                    "avg_compression_ratio": 0.15,
                },
            },
        }

        toin.import_patterns(import_data)

        pattern = toin.get_pattern(sig.structure_hash)
        assert pattern.total_compressions == 30  # 10 + 20
        assert pattern.sample_size == 30
        assert pattern.user_count >= 1

    def test_import_patterns_disabled(self):
        """Import disabled does nothing."""
        config = TOINConfig(enabled=False)
        toin = ToolIntelligenceNetwork(config)

        import_data = {
            "version": "1.0",
            "patterns": {
                "abc123": {"tool_signature_hash": "abc123", "total_compressions": 50},
            },
        }

        toin.import_patterns(import_data)

        pattern = toin.get_pattern("abc123")
        assert pattern is None

    def test_round_trip_export_import(self):
        """Export from one TOIN imports to another."""
        toin1 = ToolIntelligenceNetwork()
        toin2 = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1", "score": 0.5}])

        # Populate toin1
        for _ in range(15):
            toin1.record_compression(
                tool_signature=sig,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="top_n",
            )

        # Record retrievals
        for _ in range(3):
            toin1.record_retrieval(
                sig.structure_hash,
                "search",
                query="score>0.8",
                query_fields=["score"],
            )

        # Export and import
        export = toin1.export_patterns()
        toin2.import_patterns(export)

        # Verify import
        pattern = toin2.get_pattern(sig.structure_hash)
        assert pattern is not None
        assert pattern.total_compressions == 15
        assert pattern.total_retrievals == 3


class TestTOINPersistence:
    """Test TOIN persistence to disk."""

    def test_save_and_load(self):
        """Save and load preserves TOIN data."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            storage_path = f.name

        try:
            # Create and populate TOIN
            config = TOINConfig(storage_path=storage_path)
            toin = ToolIntelligenceNetwork(config)

            sig = ToolSignature.from_items([{"id": "1", "name": "test"}])
            for _ in range(5):
                toin.record_compression(
                    tool_signature=sig,
                    original_count=100,
                    compressed_count=10,
                    original_tokens=1000,
                    compressed_tokens=100,
                    strategy="top_n",
                )

            toin.save()

            # Verify file exists
            assert os.path.exists(storage_path)

            # Create new TOIN that loads from disk
            toin2 = ToolIntelligenceNetwork(config)

            stats = toin2.get_stats()
            assert stats["total_compressions"] == 5

        finally:
            os.unlink(storage_path)

    def test_load_corrupted_file(self):
        """Corrupted file is handled gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json {{{")
            storage_path = f.name

        try:
            config = TOINConfig(storage_path=storage_path)
            toin = ToolIntelligenceNetwork(config)

            # Should not raise, starts fresh
            stats = toin.get_stats()
            assert stats["patterns_tracked"] == 0

        finally:
            os.unlink(storage_path)

    def test_load_nonexistent_file(self):
        """Nonexistent file is handled gracefully."""
        config = TOINConfig(storage_path="/nonexistent/path/toin.json")
        toin = ToolIntelligenceNetwork(config)

        # Should not raise, starts fresh
        stats = toin.get_stats()
        assert stats["patterns_tracked"] == 0


class TestGlobalTOIN:
    """Test global TOIN singleton."""

    def test_singleton_returns_same_instance(self):
        """get_toin returns same instance."""
        toin1 = get_toin()
        toin2 = get_toin()

        assert toin1 is toin2

    def test_reset_clears_singleton(self):
        """reset_toin creates new instance."""
        toin1 = get_toin()

        sig = ToolSignature.from_items([{"id": "1"}])
        toin1.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        reset_toin()

        toin2 = get_toin()
        stats = toin2.get_stats()
        assert stats["total_compressions"] == 0

    def test_get_toin_with_config(self):
        """First call to get_toin accepts config."""
        reset_toin()

        config = TOINConfig(min_samples_for_recommendation=5)
        toin = get_toin(config)

        assert toin._config.min_samples_for_recommendation == 5


class TestTOINQueryAnonymization:
    """Test query pattern anonymization."""

    def test_anonymize_query_pattern(self):
        """Query values are anonymized."""
        toin = ToolIntelligenceNetwork()

        # Test internal method
        pattern = toin._anonymize_query_pattern("status:error AND user:john")
        assert pattern is not None
        assert "error" not in pattern.lower()
        assert "john" not in pattern.lower()
        # Should have structure preserved
        assert "status:*" in pattern or "*" in pattern

    def test_anonymize_empty_query(self):
        """Empty query returns None."""
        toin = ToolIntelligenceNetwork()

        pattern = toin._anonymize_query_pattern("")
        assert pattern is None

    def test_hash_field_name(self):
        """Field names are hashed consistently."""
        toin = ToolIntelligenceNetwork()

        hash1 = toin._hash_field_name("status")
        hash2 = toin._hash_field_name("status")
        hash3 = toin._hash_field_name("different")

        assert hash1 == hash2  # Same input = same hash
        assert hash1 != hash3  # Different input = different hash
        assert len(hash1) == 8  # SHA256[:8]


class TestTOINConfidence:
    """Test confidence calculation."""

    def test_confidence_increases_with_samples(self):
        """More samples increase confidence."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1"}])

        confidences = []
        for i in range(50):
            toin.record_compression(
                tool_signature=sig,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="top_n",
            )
            if (i + 1) % 10 == 0:
                pattern = toin.get_pattern(sig.structure_hash)
                confidences.append(pattern.confidence)

        # Confidence should generally increase (or at least not decrease significantly)
        assert confidences[-1] >= confidences[0]

    def test_confidence_capped_at_max(self):
        """Confidence never exceeds maximum."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1"}])

        # Record many compressions
        for _ in range(500):
            toin.record_compression(
                tool_signature=sig,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="top_n",
            )

        pattern = toin.get_pattern(sig.structure_hash)
        assert pattern.confidence <= 0.95


class TestTOINRecommendationUpdates:
    """Test that recommendations update based on retrieval patterns."""

    def test_optimal_max_items_updates(self):
        """optimal_max_items updates based on retrieval rate."""
        config = TOINConfig(
            min_samples_for_recommendation=5,
            high_retrieval_threshold=0.5,
        )
        toin = ToolIntelligenceNetwork(config)

        sig = ToolSignature.from_items([{"id": "1"}])
        sig_hash = sig.structure_hash

        # Low retrieval rate - aggressive compression OK
        for _ in range(20):
            toin.record_compression(
                tool_signature=sig,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="top_n",
            )

        pattern1 = toin.get_pattern(sig_hash)
        initial_max = pattern1.optimal_max_items

        # Now add many retrievals (high retrieval rate)
        for _ in range(15):  # 15/20 = 75% retrieval rate
            toin.record_retrieval(sig_hash, "search")

        pattern2 = toin.get_pattern(sig_hash)
        # Should recommend more items due to high retrieval
        assert pattern2.optimal_max_items > initial_max

    def test_preserve_fields_populated(self):
        """preserve_fields populated from retrieval patterns."""
        toin = ToolIntelligenceNetwork()

        sig = ToolSignature.from_items([{"id": "1", "status": "ok", "score": 0.5}])
        sig_hash = sig.structure_hash

        # Record compression
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="top_n",
        )

        # Repeatedly retrieve by same field
        for _ in range(10):
            toin.record_retrieval(
                sig_hash,
                "search",
                query_fields=["status"],
            )

        pattern = toin.get_pattern(sig_hash)
        # Field should be marked to preserve
        assert len(pattern.preserve_fields) > 0

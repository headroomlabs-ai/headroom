"""Tests for TOIN feedback loop: headroom_retrieve calls flow back to TOIN."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from headroom.telemetry import (
    TOINConfig,
    ToolIntelligenceNetwork,
    ToolPattern,
    ToolSignature,
    reset_toin,
)


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch, tmp_path):
    """Reset global state before each test."""
    temp_toin_path = str(tmp_path / "toin_test.json")
    monkeypatch.setenv("HEADROOM_TOIN_PATH", temp_toin_path)
    reset_toin()
    yield
    reset_toin()


def _make_config(min_samples: int = 5) -> TOINConfig:
    """Create a TOIN config for testing."""
    return TOINConfig(
        enabled=True,
        min_samples_for_recommendation=min_samples,
        storage_path="",  # disable persistence
    )


def _make_signature(structure_hash: str = "test_hash_123") -> ToolSignature:
    """Create a minimal ToolSignature for testing."""
    return ToolSignature(
        structure_hash=structure_hash,
        field_count=3,
        has_nested_objects=False,
        has_arrays=True,
        max_depth=1,
    )


@pytest.mark.skip(reason="PR-B5: observations counter and request-time hint API retired")
class TestGetRecommendationObservations:
    """Bug 1: get_recommendation() should increment observations counter."""

    def test_increments_observations_when_pattern_exists(self):
        """get_recommendation() should increment observations when pattern exists."""
        config = _make_config(min_samples=5)
        toin = ToolIntelligenceNetwork(config=config)
        sig = _make_signature("obs_test_hash")

        # Record enough compressions to create a pattern with sufficient samples
        for _ in range(15):
            toin.record_compression(
                tool_signature=sig,
                original_count=100,
                compressed_count=20,
                original_tokens=5000,
                compressed_tokens=1000,
                strategy="top_n",
            )

        # Get recommendation
        toin.get_recommendation(sig)

        # Check observations incremented
        pattern = toin._patterns[("unknown", "unknown", sig.structure_hash)]
        assert pattern.observations == 1

        # Call again
        toin.get_recommendation(sig)
        assert pattern.observations == 2

    def test_increments_observations_even_below_min_samples(self):
        """observations increments even when sample_size < min_samples."""
        config = _make_config(min_samples=100)
        toin = ToolIntelligenceNetwork(config=config)
        sig = _make_signature("low_sample_hash")

        # Record just a few compressions (below min_samples)
        for _ in range(3):
            toin.record_compression(
                tool_signature=sig,
                original_count=50,
                compressed_count=10,
                original_tokens=2000,
                compressed_tokens=500,
                strategy="top_n",
            )

        result = toin.get_recommendation(sig)
        assert result.source == "local"  # Not enough samples

        pattern = toin._patterns[("unknown", "unknown", sig.structure_hash)]
        assert pattern.observations == 1

    def test_no_increment_for_unknown_pattern(self):
        """get_recommendation() should NOT increment for unknown patterns."""
        config = _make_config()
        toin = ToolIntelligenceNetwork(config=config)
        sig = _make_signature("nonexistent_hash")

        result = toin.get_recommendation(sig)
        assert result.source == "default"
        assert result.reason == "No pattern data for this tool type"

        # No pattern exists, nothing to increment
        assert "nonexistent_hash" not in toin._patterns

    def test_observations_survives_serialization(self):
        """observations field should serialize and deserialize correctly."""
        pattern = ToolPattern(
            tool_signature_hash="serial_test",
            total_compressions=10,
            observations=42,
        )

        d = pattern.to_dict()
        assert d["observations"] == 42

        restored = ToolPattern.from_dict(d)
        assert restored.observations == 42

    def test_observations_defaults_to_zero(self):
        """observations defaults to 0 for new patterns."""
        pattern = ToolPattern(tool_signature_hash="new_pattern")
        assert pattern.observations == 0


class TestRecordRetrievalPopulatesFields:
    """Bug 1 related: record_retrieval with query_fields populates field data."""

    def test_record_retrieval_populates_fields(self):
        """record_retrieval with query_fields should populate field_retrieval_frequency."""
        config = _make_config()
        toin = ToolIntelligenceNetwork(config=config)
        sig_hash = "retrieval_field_test"

        # Record some compressions first to create the pattern
        sig = _make_signature(sig_hash)
        for _ in range(5):
            toin.record_compression(
                tool_signature=sig,
                original_count=50,
                compressed_count=10,
                original_tokens=2000,
                compressed_tokens=500,
                strategy="top_n",
            )

        # Record multiple retrievals with same field
        for _ in range(5):
            toin.record_retrieval(
                sig_hash,
                "search",
                query="error_message:timeout",
                query_fields=["error_message"],
            )

        pattern = toin._patterns[("unknown", "unknown", sig_hash)]
        assert pattern.total_retrievals == 5
        assert pattern.search_retrievals == 5
        assert len(pattern.field_retrieval_frequency) > 0


class TestCCRFeedbackExtraction:
    """Bug 2: _record_ccr_feedback_from_response extracts headroom_retrieve calls."""

    def test_extract_headroom_retrieve_from_response(self):
        """Should detect headroom_retrieve tool_use blocks in response content."""
        response = {
            "content": [
                {"type": "text", "text": "Let me retrieve that."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "headroom_retrieve",
                    "input": {"hash": "abc123def456", "query": "error fields"},
                },
            ]
        }

        # Extract tool calls the same way _record_ccr_feedback_from_response does
        content = response.get("content", [])
        retrieve_calls = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == "headroom_retrieve":
                input_data = block.get("input", {})
                if input_data.get("hash"):
                    retrieve_calls.append(input_data)

        assert len(retrieve_calls) == 1
        assert retrieve_calls[0]["hash"] == "abc123def456"
        assert retrieve_calls[0]["query"] == "error fields"

    def test_ignore_non_retrieve_tool_calls(self):
        """Should ignore tool_use blocks that are not headroom_retrieve."""
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_456",
                    "name": "some_other_tool",
                    "input": {"data": "something"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_789",
                    "name": "headroom_retrieve",
                    "input": {"hash": "xyz789", "query": None},
                },
            ]
        }

        content = response.get("content", [])
        retrieve_calls = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == "headroom_retrieve":
                input_data = block.get("input", {})
                if input_data.get("hash"):
                    retrieve_calls.append(input_data)

        assert len(retrieve_calls) == 1
        assert retrieve_calls[0]["hash"] == "xyz789"

    def test_empty_content_does_not_crash(self):
        """Should handle empty or missing content gracefully."""
        for response in [
            {"content": []},
            {"content": "not a list"},
            {},
        ]:
            content = response.get("content", [])
            if not isinstance(content, list):
                continue
            # Should not raise
            for _block in content:
                pass

    def test_missing_hash_skipped(self):
        """Should skip headroom_retrieve calls without a hash."""
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_000",
                    "name": "headroom_retrieve",
                    "input": {"query": "some query"},  # No hash
                },
            ]
        }

        content = response.get("content", [])
        retrieve_calls = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == "headroom_retrieve":
                input_data = block.get("input", {})
                if input_data.get("hash"):
                    retrieve_calls.append(input_data)

        assert len(retrieve_calls) == 0


class TestStreamingFeedbackIntegration:
    """Bug 2: Full feedback loop — streaming headroom_retrieve reaches TOIN."""

    def test_record_ccr_feedback_calls_store_search(self):
        """_record_ccr_feedback_from_response should call store.search for queries."""
        from headroom.proxy.server import HeadroomProxy

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "headroom_retrieve",
                    "input": {"hash": "feedbackhash1", "query": "error details"},
                },
            ]
        }

        mock_store = MagicMock()
        with patch(
            "headroom.cache.compression_store.get_compression_store",
            return_value=mock_store,
        ):
            # Create a minimal proxy to test the method
            proxy = HeadroomProxy.__new__(HeadroomProxy)
            proxy.config = MagicMock()
            proxy.config.ccr_inject_tool = True

            proxy._record_ccr_feedback_from_response(response, "anthropic", "req-test-001")

        mock_store.search.assert_called_once_with("feedbackhash1", "error details")

    def test_record_ccr_feedback_calls_store_retrieve_no_query(self):
        """_record_ccr_feedback_from_response should call store.retrieve when no query."""
        from headroom.proxy.server import HeadroomProxy

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_002",
                    "name": "headroom_retrieve",
                    "input": {"hash": "feedbackhash2"},
                },
            ]
        }

        mock_store = MagicMock()
        with patch(
            "headroom.cache.compression_store.get_compression_store",
            return_value=mock_store,
        ):
            proxy = HeadroomProxy.__new__(HeadroomProxy)
            proxy.config = MagicMock()
            proxy.config.ccr_inject_tool = True

            proxy._record_ccr_feedback_from_response(response, "anthropic", "req-test-002")

        mock_store.retrieve.assert_called_once_with("feedbackhash2", query=None)

    def test_record_ccr_feedback_handles_store_exception(self):
        """_record_ccr_feedback_from_response should not raise on store errors."""
        from headroom.proxy.server import HeadroomProxy

        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_003",
                    "name": "headroom_retrieve",
                    "input": {"hash": "feedbackhash3", "query": "test"},
                },
            ]
        }

        mock_store = MagicMock()
        mock_store.search.side_effect = RuntimeError("store unavailable")
        with patch(
            "headroom.cache.compression_store.get_compression_store",
            return_value=mock_store,
        ):
            proxy = HeadroomProxy.__new__(HeadroomProxy)
            proxy.config = MagicMock()
            proxy.config.ccr_inject_tool = True

            # Should not raise
            proxy._record_ccr_feedback_from_response(response, "anthropic", "req-test-003")


class TestParseSSEToolUse:
    """Bug 3: _parse_sse_to_response correctly handles tool_use blocks."""

    def test_parse_sse_extracts_tool_use(self):
        """SSE with tool_use content_block should be parsed correctly."""
        from headroom.proxy.server import HeadroomProxy

        sse_data = (
            'data: {"type":"message_start","message":{"id":"msg_01","model":"claude-3-5-sonnet-20241022","role":"assistant","stop_reason":null,"usage":{"input_tokens":100,"output_tokens":0}}}\n'
            "\n"
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
            "\n"
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me retrieve that."}}\n'
            "\n"
            'data: {"type":"content_block_stop","index":0}\n'
            "\n"
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_abc","name":"headroom_retrieve"}}\n'
            "\n"
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"hash\\": \\"abc123\\", \\"query\\": \\"error\\"}"}}\n'
            "\n"
            'data: {"type":"content_block_stop","index":1}\n'
            "\n"
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":50}}\n'
        )

        proxy = HeadroomProxy.__new__(HeadroomProxy)
        result = proxy._parse_sse_to_response(sse_data, "anthropic")

        assert result is not None
        assert len(result["content"]) == 2

        text_block = result["content"][0]
        assert text_block["type"] == "text"
        assert "retrieve" in text_block["text"]

        tool_block = result["content"][1]
        assert tool_block["type"] == "tool_use"
        assert tool_block["name"] == "headroom_retrieve"
        assert tool_block["id"] == "toolu_abc"
        assert tool_block["input"]["hash"] == "abc123"
        assert tool_block["input"]["query"] == "error"

    def test_parse_sse_non_anthropic_returns_none(self):
        """Non-anthropic provider should return None."""
        from headroom.proxy.server import HeadroomProxy

        proxy = HeadroomProxy.__new__(HeadroomProxy)
        result = proxy._parse_sse_to_response("data: {}", "openai")
        assert result is None

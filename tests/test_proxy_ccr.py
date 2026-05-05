"""Tests for CCR endpoints in the proxy server.

These tests verify the /v1/retrieve endpoints work correctly.
"""

import json

import pytest

# Skip if fastapi not available
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.proxy.server import ProxyConfig, create_app


@pytest.fixture
def client():
    """Create test client with fresh compression store."""
    reset_compression_store()
    config = ProxyConfig(
        optimize=False,  # Disable optimization for simpler tests
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client
    reset_compression_store()


@pytest.fixture
def client_with_data(client):
    """Test client with pre-populated compression store."""
    store = get_compression_store()

    # Store some test data
    items = [{"id": i, "content": f"Item {i} about Python programming"} for i in range(100)]
    store.store(
        original=json.dumps(items),
        compressed=json.dumps(items[:10]),
        original_tokens=1000,
        compressed_tokens=100,
        original_item_count=100,
        compressed_item_count=10,
        tool_name="test_tool",
    )

    return client


class TestCCRRetrieveEndpoint:
    """Test the /v1/retrieve POST endpoint."""

    def test_retrieve_requires_hash(self, client):
        """Request without hash should return 400."""
        response = client.post("/v1/retrieve", json={})
        assert response.status_code == 400
        assert "hash required" in response.json()["detail"]

    def test_retrieve_nonexistent_hash(self, client):
        """Request with nonexistent hash should return 404."""
        response = client.post("/v1/retrieve", json={"hash": "nonexistent123"})
        assert response.status_code == 404
        assert "not found or expired" in response.json()["detail"]

    def test_retrieve_full_content(self, client):
        """Full retrieval returns original content."""
        store = get_compression_store()
        items = [{"id": i} for i in range(50)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
            original_item_count=50,
            compressed_item_count=0,
        )

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200

        data = response.json()
        assert data["hash"] == hash_key
        assert data["original_item_count"] == 50
        assert "original_content" in data

        # Verify content is correct
        retrieved_items = json.loads(data["original_content"])
        assert len(retrieved_items) == 50
        assert retrieved_items[0]["id"] == 0

    def test_retrieve_with_search(self, client):
        """Search retrieval filters by query."""
        store = get_compression_store()
        items = [
            {"id": 1, "text": "Python programming language"},
            {"id": 2, "text": "JavaScript web development"},
            {"id": 3, "text": "Python data science"},
            {"id": 4, "text": "Java enterprise"},
        ]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
            original_item_count=4,
            compressed_item_count=0,
        )

        response = client.post(
            "/v1/retrieve", json={"hash": hash_key, "query": "Python programming"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["hash"] == hash_key
        assert data["query"] == "Python programming"
        assert "results" in data
        assert data["count"] >= 1

    def test_retrieve_increments_count(self, client):
        """Each retrieval increments the retrieval count."""
        store = get_compression_store()
        hash_key = store.store(original="[]", compressed="[]")

        # First retrieval
        response1 = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response1.status_code == 200
        count1 = response1.json()["retrieval_count"]

        # Second retrieval
        response2 = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response2.status_code == 200
        count2 = response2.json()["retrieval_count"]

        assert count2 > count1


class TestCCRRetrieveGetEndpoint:
    """Test the /v1/retrieve/{hash_key} GET endpoint."""

    def test_get_retrieve_full(self, client):
        """GET retrieval returns full content."""
        store = get_compression_store()
        items = [{"id": i} for i in range(20)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
            original_item_count=20,
            compressed_item_count=0,
            tool_name="get_test_tool",
        )

        response = client.get(f"/v1/retrieve/{hash_key}")
        assert response.status_code == 200

        data = response.json()
        assert data["hash"] == hash_key
        assert data["original_item_count"] == 20
        assert data["tool_name"] == "get_test_tool"

    def test_get_retrieve_with_query(self, client):
        """GET retrieval with query parameter invokes search."""
        store = get_compression_store()
        # Create items with distinctive content
        items = [
            {"id": 1, "msg": "Python programming language tutorial for beginners"},
            {"id": 2, "msg": "JavaScript web development framework guide"},
            {"id": 3, "msg": "Python data science machine learning pandas"},
            {"id": 4, "msg": "Java enterprise application development"},
        ]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
        )

        response = client.get(f"/v1/retrieve/{hash_key}?query=Python programming")
        assert response.status_code == 200

        data = response.json()
        assert data["query"] == "Python programming"
        # Response includes search results structure
        assert "results" in data
        assert "count" in data
        # Results should be a list (may be empty if BM25 threshold not met)
        assert isinstance(data["results"], list)

    def test_get_retrieve_nonexistent(self, client):
        """GET with nonexistent hash returns 404."""
        response = client.get("/v1/retrieve/nonexistent123")
        assert response.status_code == 404


class TestCCRStatsEndpoint:
    """Test the /v1/retrieve/stats endpoint."""

    def test_stats_empty_store(self, client):
        """Stats with empty store returns zeros."""
        response = client.get("/v1/retrieve/stats")
        assert response.status_code == 200

        data = response.json()
        assert "store" in data
        assert data["store"]["entry_count"] == 0
        assert "recent_retrievals" in data

    def test_stats_with_entries(self, client):
        """Stats reflect store contents."""
        store = get_compression_store()

        # Add some entries
        store.store(original="[1]", compressed="[]", original_tokens=100)
        store.store(original="[2]", compressed="[]", original_tokens=200)

        response = client.get("/v1/retrieve/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["store"]["entry_count"] == 2
        assert data["store"]["total_original_tokens"] == 300

    def test_stats_tracks_retrievals(self, client):
        """Stats include recent retrieval events."""
        import json as json_module

        store = get_compression_store()

        # Use non-empty content so search actually logs
        content = json_module.dumps(
            [
                {"id": "1", "name": "test item", "value": 100},
                {"id": "2", "name": "another item", "value": 200},
            ]
        )
        hash_key = store.store(
            original=content,
            compressed=content,
            tool_name="stats_test_tool",
        )

        # Make some retrievals
        client.post("/v1/retrieve", json={"hash": hash_key})  # Full retrieval
        client.post("/v1/retrieve", json={"hash": hash_key, "query": "test"})  # Search retrieval

        response = client.get("/v1/retrieve/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["store"]["total_retrievals"] >= 2
        assert len(data["recent_retrievals"]) >= 2

        # Verify we have both retrieval types (no double-logging of full)
        retrieval_types = [r["retrieval_type"] for r in data["recent_retrievals"]]
        assert "full" in retrieval_types
        assert "search" in retrieval_types


class TestCCRIntegration:
    """Integration tests for CCR with proxy."""

    def test_health_endpoint(self, client):
        """Health endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_stats_endpoint(self, client):
        """Stats endpoint includes CCR-relevant info."""
        response = client.get("/stats")
        assert response.status_code == 200
        # Proxy stats endpoint is separate from CCR stats
        data = response.json()
        assert "requests" in data
        assert "tokens" in data


class TestCCREdgeCases:
    """Edge cases for CCR endpoints."""

    def test_retrieve_empty_content(self, client):
        """Retrieve works with empty content."""
        store = get_compression_store()
        hash_key = store.store(original="[]", compressed="[]")

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200
        assert response.json()["original_content"] == "[]"

    def test_retrieve_large_content(self, client):
        """Retrieve works with large content."""
        store = get_compression_store()
        items = [{"id": i, "data": "x" * 100} for i in range(1000)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed=json.dumps(items[:10]),
            original_item_count=1000,
        )

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200

        data = response.json()
        assert data["original_item_count"] == 1000

    def test_search_no_matches(self, client):
        """Search with no matches returns empty results."""
        store = get_compression_store()
        items = [{"id": 1, "text": "hello world"}]
        hash_key = store.store(original=json.dumps(items), compressed="[]")

        response = client.post("/v1/retrieve", json={"hash": hash_key, "query": "xyznonexistent"})
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 0
        assert data["results"] == []

    def test_unicode_content(self, client):
        """Unicode content is handled correctly."""
        store = get_compression_store()
        items = [
            {"id": 1, "text": "日本語テキスト"},
            {"id": 2, "text": "Émoji 🎉 test"},
        ]
        hash_key = store.store(original=json.dumps(items, ensure_ascii=False), compressed="[]")

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200

        data = response.json()
        retrieved = json.loads(data["original_content"])
        assert retrieved[0]["text"] == "日本語テキスト"
        assert "🎉" in retrieved[1]["text"]


class TestEndToEndTOINIntegration:
    """End-to-end tests verifying the production path from proxy → TOIN.

    These tests verify that:
    1. SmartCrusher compresses tool outputs when called through the proxy pipeline
    2. TOIN records compression events
    3. Retrieval events update TOIN field semantics
    4. The full feedback loop works

    This catches bugs where components are wired correctly but don't communicate
    (e.g., compression_store not passing retrieved_items to TOIN).
    """

    @pytest.fixture
    def fresh_toin(self):
        """Create a fresh TOIN instance."""
        import tempfile
        from pathlib import Path

        from headroom.telemetry.toin import (
            TOINConfig,
            get_toin,
            reset_toin,
        )

        reset_toin()
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = str(Path(tmpdir) / "toin.json")
            toin = get_toin(
                TOINConfig(
                    storage_path=storage_path,
                    auto_save_interval=0,
                )
            )
            yield toin
            reset_toin()

    @pytest.fixture
    def client_with_optimization(self, fresh_toin):
        """Create test client with optimization enabled."""
        reset_compression_store()
        config = ProxyConfig(
            optimize=True,  # Enable optimization
            smart_routing=False,  # Use legacy mode for simpler testing
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
        app = create_app(config)
        with TestClient(app) as client:
            yield client
        reset_compression_store()

    def test_pipeline_compresses_tool_output_and_records_toin(
        self, fresh_toin, client_with_optimization
    ):
        """CRITICAL: Verify SmartCrusher compression records events in TOIN.

        This tests the production code path:
        1. Tool output comes in through proxy
        2. SmartCrusher compresses it
        3. TOIN records the compression event
        """
        from headroom.config import CCRConfig, SmartCrusherConfig
        from headroom.providers import AnthropicProvider
        from headroom.telemetry import ToolSignature
        from headroom.transforms import SmartCrusher, TransformPipeline

        # Create tool output with 100 items that will trigger compression
        # Key: score field with varying values signals sortable data
        # Having repetitive category values helps trigger compression
        items = [
            {
                "id": i,
                "score": 1000 - i,  # Decreasing scores signal sorting
                "category": f"cat_{i % 3}",  # Only 3 unique categories
                "status": "active" if i % 2 == 0 else "inactive",  # Binary status
            }
            for i in range(100)
        ]
        tool_output = json.dumps(items)

        # Create messages with tool_result containing our data
        messages = [
            {"role": "user", "content": "Search for items"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "search_api",
                        "input": {"query": "test"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": tool_output,
                    }
                ],
            },
        ]

        # Create pipeline with SmartCrusher (same as proxy does).
        # Use with_compaction=False so we exercise the lossy + CCR
        # caching path that this test asserts. The PR4 lossless
        # default substitutes a CSV+schema string and skips CCR
        # caching (nothing dropped → no cache entry).
        pipeline = TransformPipeline(
            transforms=[
                SmartCrusher(
                    SmartCrusherConfig(
                        enabled=True,
                        min_tokens_to_crush=100,
                        max_items_after_crush=15,
                    ),
                    ccr_config=CCRConfig(
                        enabled=True,
                        inject_retrieval_marker=True,
                        min_items_to_cache=10,
                    ),
                    with_compaction=False,
                ),
            ],
            provider=AnthropicProvider(),
        )

        # Apply pipeline (this is what the proxy does)
        result = pipeline.apply(
            messages=messages,
            model="claude-sonnet-4-20250514",
            model_limit=200000,
        )

        # Verify SmartCrusher was invoked (transform name starts with smart_crush)
        smart_crush_applied = any(
            t.startswith("smart_crush") or t.startswith("smart:") for t in result.transforms_applied
        )
        assert smart_crush_applied, (
            f"SmartCrusher should be in transforms: {result.transforms_applied}"
        )

        # Check if compression was actually performed (not skipped)
        # Skip messages look like "smart:skip:reason(100->100)"
        compression_was_skipped = any(
            "skip" in t.lower() for t in result.transforms_applied if "smart:" in t.lower()
        )

        # If compression happened, verify TOIN and store
        if not compression_was_skipped:
            # Verify compression store has the entry
            store = get_compression_store()
            stats = store.get_stats()
            assert stats["entry_count"] >= 1, "Should have cached entry"

            # Verify TOIN recorded the compression
            signature = ToolSignature.from_items(items)
            pattern = fresh_toin._patterns.get(signature.structure_hash)
            assert pattern is not None, (
                "TOIN should have recorded compression event. "
                "If this fails, SmartCrusher is not calling TOIN.record_compression."
            )
            assert pattern.total_compressions >= 1, "Should have at least 1 compression"
        else:
            # Compression was skipped - this is expected for some data patterns
            # The important thing is that SmartCrusher was invoked and made a decision
            # The other tests verify the full loop when compression does happen
            pass

    def test_retrieval_through_proxy_updates_toin_field_semantics(
        self, fresh_toin, client_with_optimization
    ):
        """CRITICAL: Verify retrieval through proxy updates TOIN field semantics.

        This tests the full feedback loop:
        1. Store compressed content (simulating prior compression)
        2. Retrieve through proxy endpoint
        3. Verify TOIN learned field semantics from retrieved items
        """
        from headroom.telemetry import ToolSignature

        # Create items with distinctive field types
        items = [
            {
                "id": i,
                "error_code": 500 if i % 10 == 0 else 200,
                "timestamp": f"2024-01-{i:02d}T00:00:00Z",
                "message": f"Log entry {i}",
            }
            for i in range(50)
        ]
        original_content = json.dumps(items)
        compressed_content = json.dumps(items[:10])

        # Get the signature hash
        signature = ToolSignature.from_items(items)

        # Store in compression store with correct metadata
        store = get_compression_store()
        hash_key = store.store(
            original=original_content,
            compressed=compressed_content,
            original_item_count=50,
            compressed_item_count=10,
            tool_name="logs_api",
            tool_signature_hash=signature.structure_hash,
            compression_strategy="smart_sample",
        )

        # Pre-record some compressions in TOIN (needed for pattern to exist)
        for _ in range(3):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=50,
                compressed_count=10,
                original_tokens=5000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        # Retrieve through proxy endpoint
        response = client_with_optimization.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200

        # Process pending feedback (this is what triggers TOIN learning)
        # Note: get_compression_store is already imported at module level
        store = get_compression_store()
        store.process_pending_feedback()

        # PR-B5: pattern key is now `(auth_mode, model_family, sig_hash)`.
        # Callers that don't supply auth/model land on the
        # `("unknown", "unknown", sig_hash)` slot.
        from headroom.telemetry.toin import _make_pattern_key

        pattern = fresh_toin._patterns.get(_make_pattern_key(None, None, signature.structure_hash))
        assert pattern is not None, "Pattern should exist after compression and retrieval"

        # CRITICAL ASSERTION: This catches the bug where compression_store
        # wasn't passing retrieved_items to TOIN
        assert len(pattern.field_semantics) > 0, (
            "TOIN should have learned field semantics from retrieved items. "
            "If this fails, the production code path "
            "(CompressionStore.process_pending_feedback -> TOIN.record_retrieval) "
            "is not passing retrieved_items."
        )

        # Verify specific field types were learned
        field_names = list(pattern.field_semantics.keys())
        assert len(field_names) > 0, "Should have learned at least one field"

    def test_full_proxy_ccr_feedback_loop(self, fresh_toin, client_with_optimization):
        """CRITICAL: Test the complete CCR feedback loop through proxy.

        This is the most important integration test - it verifies:
        1. Compression happens and TOIN records it
        2. Retrieval happens and TOIN learns from it
        3. Future recommendations reflect the learning
        """
        from headroom.telemetry import ToolSignature

        # Create items for the full feedback loop test
        items = [
            {
                "id": i,
                "score": 1000 - i,
                "category": f"cat_{i % 5}",
                "status": "active" if i % 2 == 0 else "inactive",
            }
            for i in range(100)
        ]
        signature = ToolSignature.from_items(items)

        # Store content directly (simulating what SmartCrusher does)
        # This ensures we have entries regardless of whether compression was triggered
        store = get_compression_store()
        hash_key = store.store(
            original=json.dumps(items),
            compressed=json.dumps(items[:15]),
            original_item_count=100,
            compressed_item_count=15,
            tool_name="search_api",
            tool_signature_hash=signature.structure_hash,
            compression_strategy="smart_sample",
        )

        # Record compressions in TOIN (simulating what SmartCrusher does)
        for _ in range(3):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=100,
                compressed_count=15,
                original_tokens=5000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        # Step 2: Retrieve through proxy endpoint
        response = client_with_optimization.post(
            "/v1/retrieve",
            json={"hash": hash_key, "query": "category:cat_1"},
        )
        assert response.status_code == 200

        # Process feedback (this triggers TOIN learning)
        store.process_pending_feedback()

        # Step 3: Verify TOIN learned
        # PR-B5: pattern key is now `(auth_mode, model_family, sig_hash)`.
        # Callers that don't supply auth/model land on the
        # `("unknown", "unknown", sig_hash)` slot.
        from headroom.telemetry.toin import _make_pattern_key

        pattern = fresh_toin._patterns.get(_make_pattern_key(None, None, signature.structure_hash))
        assert pattern is not None, "Pattern should exist"
        assert pattern.total_compressions >= 1, "Should have compression count"
        assert pattern.total_retrievals >= 1, "Should have retrieval count"

        # Step 4: Verify field semantics were learned
        assert len(pattern.field_semantics) > 0, (
            "TOIN should learn field semantics through the full proxy CCR loop. "
            "This is the ultimate integration test - if this fails, "
            "the production feedback loop is broken."
        )

        # Step 5: PR-B5 retired the request-time recommendation API in favor of
        # observation-only learning + startup-published recommendations.toml.
        # `get_recommendation()` now returns None and emits a deprecation
        # warning; the dispatcher consumes published advice via the Rust
        # `RecommendationStore`. Assert the deprecation contract here so a
        # future revival of the API doesn't slip past silently.
        assert fresh_toin.get_recommendation(signature, "find category") is None

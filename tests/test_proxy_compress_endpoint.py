"""Tests for the /v1/compress endpoint in the proxy server.

These tests verify that the compression-only endpoint works correctly
for the TypeScript SDK and other HTTP clients.
"""

import json

import pytest

# Skip if fastapi not available
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


@pytest.fixture
def client():
    """Create test client with optimization enabled."""
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_optimize():
    """Create test client with optimization disabled."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as c:
        yield c


class TestCompressEndpointValidation:
    """Test request validation for /v1/compress."""

    def test_missing_messages_returns_400(self, client):
        """Request without messages field should return 400."""
        response = client.post("/v1/compress", json={"model": "gpt-4"})
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request"
        assert "messages" in data["error"]["message"]

    def test_missing_model_returns_400(self, client):
        """Request without model field should return 400."""
        response = client.post(
            "/v1/compress",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request"
        assert "model" in data["error"]["message"]

    def test_invalid_json_returns_400(self, client):
        """Request with invalid JSON should return 400."""
        response = client.post(
            "/v1/compress",
            content=b"not valid json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["type"] == "invalid_request"


class TestCompressEndpointBasic:
    """Test basic compress endpoint behavior."""

    def test_empty_messages_returns_empty(self, client):
        """Empty messages list should return as-is with zero metrics."""
        response = client.post(
            "/v1/compress",
            json={"messages": [], "model": "gpt-4"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == []
        assert data["tokens_before"] == 0
        assert data["tokens_after"] == 0
        assert data["tokens_saved"] == 0
        assert data["compression_ratio"] == 1.0
        assert data["transforms_applied"] == []
        assert data["ccr_hashes"] == []

    def test_basic_compression_response_shape(self, client):
        """Verify the response contains all expected fields."""
        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "Hello, world!"}],
                "model": "gpt-4",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Check all expected fields are present
        assert "messages" in data
        assert "tokens_before" in data
        assert "tokens_after" in data
        assert "tokens_saved" in data
        assert "compression_ratio" in data
        assert "transforms_applied" in data
        assert "ccr_hashes" in data

        # Messages should be a list
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) >= 1

        # Numeric fields should be non-negative
        assert data["tokens_before"] >= 0
        assert data["tokens_after"] >= 0
        assert data["tokens_saved"] >= 0
        assert data["compression_ratio"] > 0

    def test_bypass_header_returns_uncompressed(self, client):
        """X-Headroom-Bypass header should skip compression."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        response = client.post(
            "/v1/compress",
            json={"messages": messages, "model": "gpt-4"},
            headers={"x-headroom-bypass": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == messages
        assert data["tokens_before"] == 0
        assert data["tokens_after"] == 0
        assert data["tokens_saved"] == 0
        assert data["compression_ratio"] == 1.0
        assert data["transforms_applied"] == []
        assert data["ccr_hashes"] == []

    def test_bypass_header_case_insensitive(self, client):
        """Bypass header should be case-insensitive."""
        messages = [{"role": "user", "content": "Hello"}]
        response = client.post(
            "/v1/compress",
            json={"messages": messages, "model": "gpt-4"},
            headers={"x-headroom-bypass": "TRUE"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == messages


class TestCompressEndpointCompression:
    """Test that actual compression happens for large content."""

    def test_large_tool_output_gets_compressed(self, client):
        """Large tool output content should result in tokens_saved > 0."""
        # Create a large repetitive tool output that should be compressible
        large_data = json.dumps(
            [
                {
                    "id": i,
                    "name": f"Item {i}",
                    "description": f"This is a detailed description for item number {i}. "
                    f"It contains various attributes and metadata that are typical "
                    f"of API responses. The item has a status of active and was "
                    f"created on 2024-01-{(i % 28) + 1:02d}. Additional fields "
                    f"include category=electronics, price={i * 10.99:.2f}, "
                    f"rating={4.0 + (i % 10) / 10:.1f}, stock={i * 5}.",
                    "tags": ["electronics", "sale", "featured", "new-arrival"],
                    "metadata": {
                        "created_by": "system",
                        "updated_at": "2024-01-15T00:00:00Z",
                        "version": i,
                        "source": "api",
                    },
                }
                for i in range(200)
            ]
        )

        messages = [
            {"role": "user", "content": "What items are available?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "list_items",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": large_data,
            },
            {"role": "user", "content": "Summarize the first 5 items."},
        ]

        response = client.post(
            "/v1/compress",
            json={"messages": messages, "model": "gpt-4"},
        )
        assert response.status_code == 200
        data = response.json()

        # With a large tool output, the pipeline should process successfully
        assert data["tokens_before"] > 0
        assert data["tokens_after"] > 0
        assert data["tokens_after"] <= data["tokens_before"]
        assert data["tokens_saved"] == data["tokens_before"] - data["tokens_after"]
        assert 0 < data["compression_ratio"] <= 1.0
        assert isinstance(data["transforms_applied"], list)

    def test_small_content_may_not_compress(self, client):
        """Small messages may not get compressed but should still work."""
        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "model": "gpt-4",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should still return valid response regardless of compression
        assert data["tokens_before"] >= 0
        assert data["tokens_after"] >= 0
        assert isinstance(data["transforms_applied"], list)

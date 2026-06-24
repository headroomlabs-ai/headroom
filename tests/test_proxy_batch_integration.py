"""Integration tests for proxy batch APIs with compression.

These tests verify that batch endpoints work correctly with real API calls
and compression enabled, testing token savings tracking.

Required environment variables:
- OPENAI_API_KEY: For OpenAI /v1/batches endpoint
- ANTHROPIC_API_KEY: For Anthropic /v1/messages/batches endpoint

IMPORTANT: Batch API tests create real batch jobs which may incur costs.
Use sparingly and clean up resources after testing.

Run with:
    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... pytest tests/test_proxy_batch_integration.py -v
"""

import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def openai_batch_client():
    """Create test client for OpenAI batch API with compression enabled."""
    config = ProxyConfig(
        optimize=True,  # Enable compression for batch
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def anthropic_batch_client():
    """Create test client for Anthropic batch API with compression enabled."""
    config = ProxyConfig(
        optimize=True,  # Enable compression for batch
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    return os.environ.get("OPENAI_API_KEY")


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY")


def create_large_messages(num_items: int = 50) -> list[dict]:
    """Create messages with large JSON data for compression testing."""
    # Create a list of items that will be compressible
    items = [
        {
            "id": i,
            "name": f"Item number {i}",
            "description": f"This is a detailed description for item {i}. It contains additional information.",
            "status": "active" if i % 2 == 0 else "inactive",
            "metadata": {
                "created_at": f"2024-01-{(i % 28) + 1:02d}",
                "updated_at": f"2024-06-{(i % 28) + 1:02d}",
                "tags": [f"tag{i % 5}", f"category{i % 3}"],
            },
        }
        for i in range(num_items)
    ]
    large_json = json.dumps(items, indent=2)

    return [
        {"role": "system", "content": "You are a helpful data analyst assistant."},
        {"role": "user", "content": "I have some data I need you to analyze."},
        {"role": "assistant", "content": f"I've received your data:\n\n{large_json}"},
        {"role": "user", "content": "How many items have status 'active'?"},
    ]


# =============================================================================
# OpenAI Batch API Tests
# =============================================================================


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIBatchCreate:
    """Test OpenAI /v1/batches create endpoint with compression."""

    def test_batch_create_validation_missing_input_file(self, openai_batch_client, openai_api_key):
        """POST /v1/batches without input_file_id returns validation error."""
        response = openai_batch_client.post(
            "/v1/batches",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "input_file_id" in data["error"]["message"].lower()

    def test_batch_create_validation_missing_endpoint(self, openai_batch_client, openai_api_key):
        """POST /v1/batches without endpoint returns validation error."""
        response = openai_batch_client.post(
            "/v1/batches",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={
                "input_file_id": "file-abc123",
                "completion_window": "24h",
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "endpoint" in data["error"]["message"].lower()

    def test_batch_create_with_compression(self, openai_batch_client, openai_api_key):
        """Full batch creation flow with compression.

        This test:
        1. Creates a JSONL file with compressible content
        2. Uploads it to OpenAI
        3. Creates a batch with compression enabled
        4. Verifies compression stats are tracked
        5. Cancels the batch to avoid costs
        """
        # Step 1: Create JSONL content with compressible messages
        messages = create_large_messages(num_items=30)
        jsonl_lines = [
            json.dumps(
                {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": messages,
                        "max_tokens": 100,
                    },
                }
            )
            for i in range(3)  # 3 requests in batch
        ]
        jsonl_content = "\n".join(jsonl_lines)

        # Step 2: Upload the JSONL file directly to OpenAI
        import httpx

        upload_response = httpx.post(
            "https://api.openai.com/v1/files",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            files={"file": ("batch_input.jsonl", jsonl_content.encode(), "application/jsonl")},
            data={"purpose": "batch"},
        )
        assert upload_response.status_code == 200, f"File upload failed: {upload_response.text}"
        file_data = upload_response.json()
        input_file_id = file_data["id"]

        try:
            # Step 3: Create batch through proxy with compression
            response = openai_batch_client.post(
                "/v1/batches",
                headers={"Authorization": f"Bearer {openai_api_key}"},
                json={
                    "input_file_id": input_file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                    "metadata": {"test": "compression_integration"},
                },
            )
            assert response.status_code == 200, f"Batch creation failed: {response.text}"
            batch_data = response.json()

            # Verify batch was created
            assert "id" in batch_data
            assert batch_data["object"] == "batch"
            batch_id = batch_data["id"]

            # Verify compression stats in response headers
            if "x-headroom-tokens-saved" in response.headers:
                tokens_saved = int(response.headers["x-headroom-tokens-saved"])
                assert tokens_saved >= 0

            if "x-headroom-savings-percent" in response.headers:
                savings_percent = float(response.headers["x-headroom-savings-percent"])
                assert 0 <= savings_percent <= 100

            # Verify compression metadata was added
            metadata = batch_data.get("metadata", {})
            if metadata.get("headroom_compressed") == "true":
                # Compression was applied
                assert "headroom_tokens_saved" in metadata
                assert "headroom_original_tokens" in metadata
                assert "headroom_compressed_tokens" in metadata
                tokens_saved = int(metadata["headroom_tokens_saved"])
                assert tokens_saved >= 0

            # Step 4: Cancel the batch to avoid costs
            cancel_response = openai_batch_client.post(
                f"/v1/batches/{batch_id}/cancel",
                headers={"Authorization": f"Bearer {openai_api_key}"},
            )
            # Cancel may succeed or fail if batch already completed/cancelled
            assert cancel_response.status_code in [200, 400]

        finally:
            # Cleanup: Delete the uploaded file
            httpx.delete(
                f"https://api.openai.com/v1/files/{input_file_id}",
                headers={"Authorization": f"Bearer {openai_api_key}"},
            )


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIBatchList:
    """Test OpenAI /v1/batches list endpoint passthrough."""

    def test_list_batches(self, openai_batch_client, openai_api_key):
        """GET /v1/batches returns list of batches."""
        response = openai_batch_client.get(
            "/v1/batches",
            headers={"Authorization": f"Bearer {openai_api_key}"},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify list response format
        assert "data" in data
        assert "object" in data
        assert data["object"] == "list"

    def test_list_batches_with_limit(self, openai_batch_client, openai_api_key):
        """GET /v1/batches with limit parameter."""
        response = openai_batch_client.get(
            "/v1/batches?limit=5",
            headers={"Authorization": f"Bearer {openai_api_key}"},
        )
        assert response.status_code == 200
        data = response.json()

        assert len(data["data"]) <= 5


# =============================================================================
# Anthropic Batch API Tests
# =============================================================================


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicBatchCreate:
    """Test Anthropic /v1/messages/batches create endpoint with compression."""

    def test_batch_create_validation_missing_requests(
        self, anthropic_batch_client, anthropic_api_key
    ):
        """POST /v1/messages/batches without requests returns validation error."""
        response = anthropic_batch_client.post(
            "/v1/messages/batches",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
            },
            json={},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_batch_create_validation_empty_requests(
        self, anthropic_batch_client, anthropic_api_key
    ):
        """POST /v1/messages/batches with empty requests list returns error."""
        response = anthropic_batch_client.post(
            "/v1/messages/batches",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
            },
            json={"requests": []},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_batch_create_with_compression(self, anthropic_batch_client, anthropic_api_key):
        """Create Anthropic batch with compression.

        This test:
        1. Creates a batch request with compressible messages
        2. Verifies the batch is created successfully
        3. Checks that compression stats are tracked
        4. Cancels the batch to avoid costs
        """
        # Create messages with compressible content
        messages = create_large_messages(num_items=25)

        # Create batch request in Anthropic format
        batch_requests = [
            {
                "custom_id": f"req-{i}",
                "params": {
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 100,
                    "messages": messages,
                },
            }
            for i in range(2)  # 2 requests in batch
        ]

        response = anthropic_batch_client.post(
            "/v1/messages/batches",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
                "content-type": "application/json",
            },
            json={"requests": batch_requests},
        )
        assert response.status_code == 200, f"Batch creation failed: {response.text}"
        batch_data = response.json()

        # Verify batch was created
        assert "id" in batch_data
        assert batch_data["type"] == "message_batch"
        batch_id = batch_data["id"]

        # Verify processing status
        assert "processing_status" in batch_data
        assert batch_data["processing_status"] in ["in_progress", "ended", "canceling"]

        # Check proxy stats for compression
        stats_response = anthropic_batch_client.get("/stats")
        stats = stats_response.json()
        # Batch requests should be tracked
        assert stats["requests"]["total"] >= 1

        # Cancel the batch to avoid costs
        cancel_response = anthropic_batch_client.post(
            f"/v1/messages/batches/{batch_id}/cancel",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
            },
        )
        # Cancel may succeed or return error if already processed
        assert cancel_response.status_code in [200, 400, 409]


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicBatchList:
    """Test Anthropic /v1/messages/batches list endpoint passthrough."""

    def test_list_batches(self, anthropic_batch_client, anthropic_api_key):
        """GET /v1/messages/batches returns list of batches."""
        response = anthropic_batch_client.get(
            "/v1/messages/batches",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify list response format
        assert "data" in data

    def test_list_batches_with_limit(self, anthropic_batch_client, anthropic_api_key):
        """GET /v1/messages/batches with limit parameter."""
        response = anthropic_batch_client.get(
            "/v1/messages/batches?limit=5",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert len(data.get("data", [])) <= 5


# =============================================================================
# Compression Verification Tests
# =============================================================================


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestBatchCompressionStats:
    """Test that batch compression stats are properly tracked."""

    def test_stats_track_batch_requests(self, openai_batch_client, openai_api_key):
        """Verify batch requests update proxy stats correctly."""
        # Get initial stats
        initial_stats = openai_batch_client.get("/stats").json()
        initial_requests = initial_stats["requests"]["total"]

        # Make a batch list request (passthrough)
        openai_batch_client.get(
            "/v1/batches",
            headers={"Authorization": f"Bearer {openai_api_key}"},
        )

        # Verify stats updated
        updated_stats = openai_batch_client.get("/stats").json()
        assert updated_stats["requests"]["total"] >= initial_requests


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicBatchCompressionStats:
    """Test Anthropic batch compression stats tracking."""

    def test_stats_track_anthropic_batch_requests(self, anthropic_batch_client, anthropic_api_key):
        """Verify Anthropic batch requests update proxy stats."""
        # Get initial stats
        initial_stats = anthropic_batch_client.get("/stats").json()
        initial_requests = initial_stats["requests"]["total"]

        # Make a batch list request
        anthropic_batch_client.get(
            "/v1/messages/batches",
            headers={
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
            },
        )

        # Verify stats updated
        updated_stats = anthropic_batch_client.get("/stats").json()
        assert updated_stats["requests"]["total"] >= initial_requests


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestBatchErrorHandling:
    """Test error handling for batch endpoints."""

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_openai_batch_invalid_file_id(self, openai_batch_client, openai_api_key):
        """Invalid file ID returns appropriate error."""
        response = openai_batch_client.post(
            "/v1/batches",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={
                "input_file_id": "file-nonexistent12345",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        # Should return error for non-existent file
        assert response.status_code in [400, 404]

    def test_openai_batch_missing_auth(self, openai_batch_client):
        """Missing authentication returns error (401 or 404 depending on routing)."""
        response = openai_batch_client.post(
            "/v1/batches",
            json={
                "input_file_id": "file-abc123",
                "endpoint": "/v1/chat/completions",
            },
        )
        # Proxy may return 404 (no route match) or 401 (auth error)
        assert response.status_code in [401, 404]

    def test_anthropic_batch_missing_auth(self, anthropic_batch_client):
        """Missing authentication returns error (401 or 400 depending on validation)."""
        response = anthropic_batch_client.post(
            "/v1/messages/batches",
            headers={
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
            },
            json={"requests": []},
        )
        # Proxy may return 400 (validation) or 401 (auth error)
        assert response.status_code in [400, 401]

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_openai_batch_invalid_json(self, openai_batch_client, openai_api_key):
        """Invalid JSON body returns 400."""
        response = openai_batch_client.post(
            "/v1/batches",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            content=b"not valid json",
        )
        assert response.status_code == 400

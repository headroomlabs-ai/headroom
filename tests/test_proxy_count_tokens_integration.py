"""Integration tests for Gemini countTokens endpoint with compression.

These tests verify that the Gemini /v1beta/models/{model}:countTokens endpoint
works correctly with compression enabled, properly counting tokens after
compression is applied.

Required environment variables:
- GEMINI_API_KEY: For Gemini countTokens endpoint

Run with:
    GEMINI_API_KEY=... pytest tests/test_proxy_count_tokens_integration.py -v
"""

import json
import os

import pytest

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gemini_client_optimized():
    """Create test client with optimization enabled for Gemini."""
    config = ProxyConfig(
        optimize=True,  # Enable compression
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def gemini_client_passthrough():
    """Create test client with optimization disabled (passthrough mode)."""
    config = ProxyConfig(
        optimize=False,  # Disable compression
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_key():
    """Get Gemini API key from environment."""
    return os.environ.get("GEMINI_API_KEY")


def create_large_content(num_items: int = 50) -> list[dict]:
    """Create Gemini-format contents with large compressible data."""
    # Create JSON data that can be compressed
    items = [
        {
            "id": i,
            "name": f"Product Item {i}",
            "description": f"This is a detailed description for product item {i}. "
            f"It includes various specifications and features.",
            "price": 99.99 + i * 0.5,
            "category": f"category_{i % 5}",
            "in_stock": i % 2 == 0,
            "metadata": {
                "sku": f"SKU-{i:05d}",
                "weight": f"{i * 0.1:.2f}kg",
                "dimensions": f"{10 + i}x{15 + i}x{5 + i}cm",
            },
        }
        for i in range(num_items)
    ]
    large_json = json.dumps(items, indent=2)

    return [
        {
            "role": "user",
            "parts": [{"text": "I have product data to analyze."}],
        },
        {
            "role": "model",
            "parts": [{"text": f"Here is the product data:\n\n{large_json}"}],
        },
        {
            "role": "user",
            "parts": [{"text": "How many products are in stock?"}],
        },
    ]


def create_simple_content() -> list[dict]:
    """Create simple Gemini-format contents for basic testing."""
    return [
        {
            "role": "user",
            "parts": [{"text": "What is 2 + 2?"}],
        }
    ]


# =============================================================================
# Basic countTokens Tests
# =============================================================================


class TestGeminiCountTokensBasic:
    """Test basic Gemini countTokens functionality."""

    def test_count_tokens_simple_content(self, gemini_client_optimized, api_key):
        """Basic token counting works correctly."""
        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": create_simple_content()},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify response format
        assert "totalTokens" in data
        assert isinstance(data["totalTokens"], int)
        assert data["totalTokens"] > 0

    def test_count_tokens_with_system_instruction(self, gemini_client_optimized, api_key):
        """Token counting includes system instruction."""
        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={
                "contents": create_simple_content(),
                "systemInstruction": {"parts": [{"text": "You are a helpful math assistant."}]},
            },
        )
        # Note: systemInstruction may not be supported by all models/versions
        # Accept both success and 400 (if not supported)
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert "totalTokens" in data
            assert data["totalTokens"] > 0

    def test_count_tokens_multi_turn(self, gemini_client_optimized, api_key):
        """Token counting for multi-turn conversation."""
        contents = [
            {"role": "user", "parts": [{"text": "Hello, my name is Alice."}]},
            {"role": "model", "parts": [{"text": "Nice to meet you, Alice!"}]},
            {"role": "user", "parts": [{"text": "What is my name?"}]},
        ]

        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": contents},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["totalTokens"] > 0


# =============================================================================
# Compression Tests
# =============================================================================


class TestGeminiCountTokensCompression:
    """Test that compression reduces token count."""

    def test_compression_reduces_token_count(
        self, gemini_client_optimized, gemini_client_passthrough, api_key
    ):
        """Verify compression reduces token count for large content.

        This test compares token counts between:
        - Passthrough mode (no compression)
        - Optimized mode (compression enabled)
        """
        large_contents = create_large_content(num_items=40)

        # Get token count without compression
        passthrough_response = gemini_client_passthrough.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": large_contents},
        )
        assert passthrough_response.status_code == 200
        passthrough_tokens = passthrough_response.json()["totalTokens"]

        # Get token count with compression
        optimized_response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": large_contents},
        )
        assert optimized_response.status_code == 200
        optimized_tokens = optimized_response.json()["totalTokens"]

        # Compression should reduce token count (or at least not increase it)
        # Note: compression effect depends on content and may vary
        assert optimized_tokens <= passthrough_tokens * 1.1  # Allow 10% margin

        # For large content, we expect some savings
        if passthrough_tokens > 1000:
            assert optimized_tokens < passthrough_tokens, (
                f"Expected compression to reduce tokens from {passthrough_tokens} "
                f"but got {optimized_tokens}"
            )

    def test_compression_stats_tracked(self, gemini_client_optimized, api_key):
        """Verify compression stats are tracked in proxy stats."""
        large_contents = create_large_content(num_items=30)

        # Make countTokens request with large content
        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": large_contents},
        )
        assert response.status_code == 200

        # Check proxy stats
        stats_response = gemini_client_optimized.get("/stats")
        assert stats_response.status_code == 200
        stats = stats_response.json()

        # Verify Gemini requests are tracked
        assert stats["requests"]["total"] >= 1
        assert "gemini" in stats["requests"]["by_provider"]


class TestGeminiCountTokensLargeContent:
    """Test countTokens with large content that benefits from compression."""

    def test_very_large_json_content(self, gemini_client_optimized, api_key):
        """Token counting handles very large JSON content."""
        large_contents = create_large_content(num_items=100)

        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": large_contents},
        )
        assert response.status_code == 200
        data = response.json()

        assert "totalTokens" in data
        assert data["totalTokens"] > 0

    def test_repeated_data_compression(self, gemini_client_optimized, api_key):
        """Content with repeated patterns compresses well."""
        # Create content with highly repetitive data
        repeated_items = [{"id": i, "status": "active", "type": "item"} for i in range(200)]
        repeated_json = json.dumps(repeated_items)

        contents = [
            {"role": "user", "parts": [{"text": "Analyze this data."}]},
            {"role": "model", "parts": [{"text": f"Data:\n{repeated_json}"}]},
            {"role": "user", "parts": [{"text": "Count the items."}]},
        ]

        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": contents},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["totalTokens"] > 0

    def test_code_content_compression(self, gemini_client_optimized, api_key):
        """Token counting handles code content."""
        code_sample = '''
def calculate_statistics(data):
    """Calculate statistics for the given data."""
    if not data:
        return {"count": 0, "sum": 0, "average": 0}

    count = len(data)
    total = sum(data)
    average = total / count

    return {
        "count": count,
        "sum": total,
        "average": average,
        "min": min(data),
        "max": max(data),
    }

# Example usage
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = calculate_statistics(numbers)
print(result)
'''

        contents = [
            {"role": "user", "parts": [{"text": "Can you explain this code?"}]},
            {
                "role": "model",
                "parts": [{"text": f"Here's the code:\n\n```python\n{code_sample}\n```"}],
            },
            {"role": "user", "parts": [{"text": "What does calculate_statistics return?"}]},
        ]

        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": contents},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["totalTokens"] > 0


# =============================================================================
# Model Variant Tests
# =============================================================================


class TestGeminiCountTokensModels:
    """Test countTokens with different Gemini models."""

    def test_gemini_flash_model(self, gemini_client_optimized, api_key):
        """countTokens works with gemini-2.0-flash model."""
        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": create_simple_content()},
        )
        assert response.status_code == 200
        assert "totalTokens" in response.json()

    def test_gemini_flash_lite_model(self, gemini_client_optimized, api_key):
        """countTokens works with gemini-2.0-flash-lite model."""
        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash-lite:countTokens?key={api_key}",
            json={"contents": create_simple_content()},
        )
        # Model may or may not be available
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            assert "totalTokens" in response.json()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestGeminiCountTokensErrors:
    """Test error handling for countTokens endpoint."""

    def test_invalid_api_key(self, gemini_client_optimized):
        """Invalid API key returns authentication error."""
        response = gemini_client_optimized.post(
            "/v1beta/models/gemini-2.0-flash:countTokens?key=invalid-key-12345",
            json={"contents": create_simple_content()},
        )
        assert response.status_code in [400, 401, 403]

    def test_invalid_model(self, gemini_client_optimized, api_key):
        """Invalid model name returns error."""
        response = gemini_client_optimized.post(
            f"/v1beta/models/nonexistent-model-xyz:countTokens?key={api_key}",
            json={"contents": create_simple_content()},
        )
        assert response.status_code >= 400

    def test_empty_contents(self, gemini_client_optimized, api_key):
        """Empty contents may return error or zero tokens."""
        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": []},
        )
        # May return error or success with 0 tokens
        if response.status_code == 200:
            data = response.json()
            assert "totalTokens" in data

    def test_invalid_json_body(self, gemini_client_optimized, api_key):
        """Invalid JSON body returns 400 error."""
        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            headers={"Content-Type": "application/json"},
            content=b"not valid json",
        )
        assert response.status_code == 400

    def test_missing_contents_field(self, gemini_client_optimized, api_key):
        """Missing contents field handled gracefully."""
        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={},
        )
        # May return error or handle empty contents
        assert response.status_code in [200, 400]


# =============================================================================
# Stats Tracking Tests
# =============================================================================


class TestGeminiCountTokensStats:
    """Test proxy stats tracking for countTokens requests."""

    def test_stats_track_gemini_provider(self, gemini_client_optimized, api_key):
        """Stats correctly track Gemini provider."""
        # Clear stats by getting a fresh client
        gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": create_simple_content()},
        )

        stats = gemini_client_optimized.get("/stats").json()
        assert "gemini" in stats["requests"]["by_provider"]
        assert stats["requests"]["by_provider"]["gemini"] >= 1

    def test_stats_track_model(self, gemini_client_optimized, api_key):
        """Stats correctly track model used."""
        gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": create_simple_content()},
        )

        stats = gemini_client_optimized.get("/stats").json()
        # Model should be tracked in by_model
        assert len(stats["requests"]["by_model"]) >= 1

    def test_stats_track_tokens_saved(self, gemini_client_optimized, api_key):
        """Stats track tokens saved from compression."""
        # Make request with large compressible content
        large_contents = create_large_content(num_items=30)
        gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": large_contents},
        )

        stats = gemini_client_optimized.get("/stats").json()
        # tokens.saved should be tracked (may be 0 if content wasn't compressed)
        assert "tokens" in stats
        assert "saved" in stats["tokens"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestGeminiCountTokensIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, gemini_client_optimized, api_key):
        """Test complete workflow: count tokens, verify compression, check stats."""
        # Step 1: Count tokens with large content
        large_contents = create_large_content(num_items=35)

        initial_stats = gemini_client_optimized.get("/stats").json()
        initial_tokens_saved = initial_stats["tokens"]["saved"]

        # Step 2: Make countTokens request
        response = gemini_client_optimized.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": large_contents},
        )
        assert response.status_code == 200
        token_count = response.json()["totalTokens"]
        assert token_count > 0

        # Step 3: Verify stats updated
        updated_stats = gemini_client_optimized.get("/stats").json()
        assert updated_stats["requests"]["total"] > initial_stats["requests"]["total"]

        # Step 4: Verify tokens saved is tracked (may be negative for small overhead)
        # Allow for some compression overhead
        assert updated_stats["tokens"]["saved"] >= initial_tokens_saved - 100

    def test_multiple_requests_accumulate_stats(self, gemini_client_optimized, api_key):
        """Multiple requests correctly accumulate stats."""
        initial_stats = gemini_client_optimized.get("/stats").json()
        initial_total = initial_stats["requests"]["total"]

        # Make several requests
        for _ in range(3):
            gemini_client_optimized.post(
                f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
                json={"contents": create_simple_content()},
            )

        updated_stats = gemini_client_optimized.get("/stats").json()
        assert updated_stats["requests"]["total"] >= initial_total + 3

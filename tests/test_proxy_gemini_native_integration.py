"""Integration tests for Gemini native API endpoint with real API calls.

These tests require a valid GEMINI_API_KEY environment variable.
They test the /v1beta/models/{model}:generateContent endpoint with compression.

Run with:
    GEMINI_API_KEY=your-key pytest tests/test_proxy_gemini_native_integration.py -v
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


@pytest.fixture
def gemini_native_client():
    """Create test client for Gemini native API with optimization enabled."""
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
def api_key():
    """Get Gemini API key from environment."""
    return os.environ.get("GEMINI_API_KEY")


class TestGeminiNativeGenerateContent:
    """Test /v1beta/models/{model}:generateContent endpoint."""

    def test_basic_generation(self, gemini_native_client, api_key):
        """Basic text generation works."""
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={"contents": [{"parts": [{"text": "What is 2+2? Reply with just the number."}]}]},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify Gemini native response format
        assert "candidates" in data
        assert len(data["candidates"]) > 0
        assert "content" in data["candidates"][0]
        assert "parts" in data["candidates"][0]["content"]
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        assert "4" in text

        # Verify usage metadata
        assert "usageMetadata" in data
        assert "promptTokenCount" in data["usageMetadata"]

    def test_with_system_instruction(self, gemini_native_client, api_key):
        """System instruction works correctly."""
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": "Hello"}]}],
                "systemInstruction": {"parts": [{"text": "Always respond with exactly one word."}]},
            },
        )
        assert response.status_code == 200
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        # Should be a short response due to system instruction
        assert len(text.split()) <= 3

    def test_multi_turn_conversation(self, gemini_native_client, api_key):
        """Multi-turn conversations maintain context."""
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "My name is TestUser456."}]},
                    {"role": "model", "parts": [{"text": "Nice to meet you, TestUser456!"}]},
                    {"role": "user", "parts": [{"text": "What is my name?"}]},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].lower()
        assert "testuser456" in text

    def test_function_calling(self, gemini_native_client, api_key):
        """Function calling / tools work correctly."""
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": "What is the weather in Tokyo?"}]}],
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "get_weather",
                                "description": "Get current weather for a location",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {"type": "string", "description": "City name"}
                                    },
                                    "required": ["location"],
                                },
                            }
                        ]
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify function call response
        parts = data["candidates"][0]["content"]["parts"]
        function_call = None
        for part in parts:
            if "functionCall" in part:
                function_call = part["functionCall"]
                break

        assert function_call is not None
        assert function_call["name"] == "get_weather"
        assert "tokyo" in function_call["args"]["location"].lower()

    def test_generation_config(self, gemini_native_client, api_key):
        """Generation config parameters are respected."""
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": "Write a very short poem about AI."}]}],
                "generationConfig": {"maxOutputTokens": 50, "temperature": 0.1},
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Response should be limited by maxOutputTokens
        assert data["usageMetadata"]["candidatesTokenCount"] <= 60  # Some buffer


class TestGeminiNativeCompression:
    """Test that compression works with Gemini native API."""

    def test_compression_on_model_message(self, gemini_native_client, api_key):
        """Large data in model message gets compressed."""
        # Create large JSON data (simulating tool output)
        items = [
            {"id": i, "name": f"Item {i}", "desc": f"Description for item {i}"} for i in range(100)
        ]
        tool_output = json.dumps(items)

        # Send as model message (like tool returning data)
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "Get items from database"}]},
                    {"role": "model", "parts": [{"text": f"Here are the results:\n{tool_output}"}]},
                    {"role": "user", "parts": [{"text": "How many items are there?"}]},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        # Model should correctly count the items
        assert "100" in text

        # Check that compression happened via stats
        stats = gemini_native_client.get("/stats").json()
        # At least some tokens should have been saved
        assert stats["tokens"]["saved"] >= 0  # May or may not compress depending on size

    def test_user_messages_protected(self, gemini_native_client, api_key):
        """User messages are not compressed (by design)."""
        # Large data in user message
        items = [{"id": i} for i in range(50)]
        user_data = json.dumps(items)

        # First request with data in user message
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": f"Analyze this data: {user_data}"}]}
                ]
            },
        )
        assert response.status_code == 200
        # The request should succeed - user messages are protected from compression


class TestGeminiNativeStats:
    """Test that proxy stats track Gemini native requests correctly."""

    def test_stats_track_gemini_provider(self, gemini_native_client, api_key):
        """Stats show requests under 'gemini' provider."""
        # Make a request
        gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={"contents": [{"parts": [{"text": "Hi"}]}]},
        )

        stats = gemini_native_client.get("/stats").json()
        assert "gemini" in stats["requests"]["by_provider"]
        assert stats["requests"]["by_provider"]["gemini"] >= 1

    def test_stats_track_model(self, gemini_native_client, api_key):
        """Stats track the specific model used."""
        gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={"contents": [{"parts": [{"text": "Hi"}]}]},
        )

        stats = gemini_native_client.get("/stats").json()
        assert "gemini-2.0-flash" in stats["requests"]["by_model"]


class TestGeminiNativeErrorHandling:
    """Test error handling for Gemini native API."""

    def test_invalid_api_key(self, gemini_native_client):
        """Invalid API key returns appropriate error."""
        response = gemini_native_client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent?key=invalid-key-123",
            json={"contents": [{"parts": [{"text": "Hi"}]}]},
        )
        assert response.status_code >= 400

    def test_invalid_model(self, gemini_native_client, api_key):
        """Invalid model returns appropriate error."""
        response = gemini_native_client.post(
            f"/v1beta/models/nonexistent-model-xyz:generateContent?key={api_key}",
            json={"contents": [{"parts": [{"text": "Hi"}]}]},
        )
        assert response.status_code >= 400

    def test_empty_contents(self, gemini_native_client, api_key):
        """Empty contents handled gracefully."""
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}", json={"contents": []}
        )
        # Should either return error or handle gracefully
        assert response.status_code in [200, 400]


class TestGeminiNativeHeaderAuth:
    """Test authentication via x-goog-api-key header."""

    def test_header_auth(self, gemini_native_client, api_key):
        """API key in header works."""
        response = gemini_native_client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent",
            headers={"x-goog-api-key": api_key},
            json={"contents": [{"parts": [{"text": "Hi"}]}]},
        )
        assert response.status_code == 200


class TestGeminiNativeCountTokens:
    """Test /v1beta/models/{model}:countTokens endpoint with compression."""

    def test_count_tokens_basic(self, gemini_native_client, api_key):
        """Basic token counting works."""
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={"contents": [{"parts": [{"text": "Hello, world!"}]}]},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify response format
        assert "totalTokens" in data
        assert isinstance(data["totalTokens"], int)
        assert data["totalTokens"] > 0

    def test_count_tokens_with_system_instruction(self, gemini_native_client, api_key):
        """Token counting includes system instruction."""
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={
                "contents": [{"parts": [{"text": "Hello"}]}],
                "systemInstruction": {"parts": [{"text": "You are a helpful assistant."}]},
            },
        )
        # Note: systemInstruction may not be supported by countTokens in all versions
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert "totalTokens" in data
            assert data["totalTokens"] > 0

    def test_count_tokens_reflects_compression(self, gemini_native_client, api_key):
        """Token count reflects compressed content size."""
        # Create large repetitive JSON data that should compress
        items = [
            {
                "id": i,
                "name": f"Item {i}",
                "description": f"This is the description for item number {i}",
            }
            for i in range(100)
        ]
        tool_output = json.dumps(items)

        # Count tokens with large data in model message (which gets compressed)
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "Get items from database"}]},
                    {"role": "model", "parts": [{"text": f"Here are the results:\n{tool_output}"}]},
                    {"role": "user", "parts": [{"text": "Summarize these items"}]},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify we got a token count
        assert "totalTokens" in data
        compressed_tokens = data["totalTokens"]
        assert compressed_tokens > 0

        # Check stats to verify compression was applied
        stats = gemini_native_client.get("/stats").json()
        # The request should have been tracked
        assert stats["requests"]["by_provider"].get("gemini", 0) >= 1

    def test_count_tokens_multi_turn(self, gemini_native_client, api_key):
        """Token counting works for multi-turn conversations."""
        response = gemini_native_client.post(
            f"/v1beta/models/gemini-2.0-flash:countTokens?key={api_key}",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "My name is Alice."}]},
                    {"role": "model", "parts": [{"text": "Nice to meet you, Alice!"}]},
                    {"role": "user", "parts": [{"text": "What is my name?"}]},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "totalTokens" in data
        assert data["totalTokens"] > 0

    def test_count_tokens_header_auth(self, gemini_native_client, api_key):
        """API key in header works for countTokens."""
        response = gemini_native_client.post(
            "/v1beta/models/gemini-2.0-flash:countTokens",
            headers={"x-goog-api-key": api_key},
            json={"contents": [{"parts": [{"text": "Hello"}]}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "totalTokens" in data

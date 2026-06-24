"""Integration tests for the proxy with real Gemini API calls.

These tests require a valid GEMINI_API_KEY environment variable.
They test the actual /v1/chat/completions endpoint with real API calls.

Run with:
    GEMINI_API_KEY=your-key pytest tests/test_proxy_gemini_integration.py -v
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

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


@pytest.fixture
def gemini_client():
    """Create test client configured to forward to Gemini."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        openai_api_url=GEMINI_BASE_URL,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_key():
    """Get Gemini API key from environment."""
    return os.environ.get("GEMINI_API_KEY")


class TestGeminiChatCompletions:
    """Test /v1/chat/completions with real Gemini API."""

    def test_basic_completion(self, gemini_client, api_key):
        """Basic chat completion works."""
        response = gemini_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gemini-2.0-flash",
                "messages": [
                    {"role": "user", "content": "What is 2+2? Reply with just the number."}
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI-compatible response format
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        assert "4" in data["choices"][0]["message"]["content"]

        # Verify usage stats
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]

    def test_multi_turn_conversation(self, gemini_client, api_key):
        """Multi-turn conversations maintain context."""
        response = gemini_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gemini-2.0-flash",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Be concise."},
                    {"role": "user", "content": "My name is TestUser123."},
                    {"role": "assistant", "content": "Nice to meet you, TestUser123!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()

        content = data["choices"][0]["message"]["content"].lower()
        assert "testuser123" in content

    def test_streaming(self, gemini_client, api_key):
        """Streaming responses work correctly."""
        response = gemini_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gemini-2.0-flash",
                "stream": True,
                "messages": [{"role": "user", "content": "Count from 1 to 3."}],
            },
        )
        assert response.status_code == 200

        # Parse SSE stream
        chunks = []
        for line in response.text.strip().split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                chunks.append(chunk)

        assert len(chunks) > 0

        # Verify chunk format
        for chunk in chunks:
            assert "choices" in chunk
            assert "delta" in chunk["choices"][0]
            assert chunk["object"] == "chat.completion.chunk"

    def test_function_calling(self, gemini_client, api_key):
        """Function calling / tools work correctly."""
        response = gemini_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gemini-2.0-flash",
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string", "description": "City name"}
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify tool call response
        message = data["choices"][0]["message"]
        assert "tool_calls" in message
        assert len(message["tool_calls"]) > 0

        tool_call = message["tool_calls"][0]
        assert tool_call["function"]["name"] == "get_weather"
        assert "paris" in tool_call["function"]["arguments"].lower()

    def test_json_mode(self, gemini_client, api_key):
        """JSON response format works."""
        response = gemini_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gemini-2.0-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": "Return a JSON object with keys 'name' and 'age' for a person named Alice who is 30 years old.",
                    }
                ],
                "response_format": {"type": "json_object"},
            },
        )
        assert response.status_code == 200
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        # Parse the response as JSON
        parsed = json.loads(content)
        assert "name" in parsed or "Name" in parsed
        assert "age" in parsed or "Age" in parsed


class TestGeminiModels:
    """Test /v1/models endpoint with Gemini."""

    def test_list_models(self, gemini_client, api_key):
        """Can list available models."""
        response = gemini_client.get("/v1/models", headers={"Authorization": f"Bearer {api_key}"})
        # This goes through passthrough handler
        assert response.status_code == 200
        data = response.json()

        assert "data" in data or "object" in data


class TestProxyStats:
    """Test that proxy stats track Gemini requests correctly."""

    def test_stats_track_requests(self, gemini_client, api_key):
        """Proxy stats track Gemini requests."""
        # Make a request
        gemini_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "gemini-2.0-flash", "messages": [{"role": "user", "content": "Hi"}]},
        )

        # Check stats
        stats_response = gemini_client.get("/stats")
        assert stats_response.status_code == 200
        stats = stats_response.json()

        assert stats["requests"]["total"] >= 1
        assert stats["requests"]["by_provider"]["openai"] >= 1
        assert "gemini" in str(stats["requests"]["by_model"]).lower()


class TestErrorHandling:
    """Test error handling with Gemini."""

    def test_invalid_api_key(self, gemini_client):
        """Invalid API key returns appropriate error."""
        response = gemini_client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer invalid-key-123"},
            json={"model": "gemini-2.0-flash", "messages": [{"role": "user", "content": "Hi"}]},
        )
        # Should return 4xx error
        assert response.status_code >= 400

    def test_invalid_model(self, gemini_client, api_key):
        """Invalid model returns appropriate error."""
        response = gemini_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "nonexistent-model-xyz",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        # Should return 4xx error
        assert response.status_code >= 400

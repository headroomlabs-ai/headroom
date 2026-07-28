"""Integration tests for Vertex native API endpoint with real API calls through the proxy.

Run with:
    GCP_ACCESS_TOKEN="$(gcloud auth print-access-token)" GCP_PROJECT_ID="your-project" pytest tests/test_proxy_vertex_native_integration.py -v
"""

import json
import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GCP_ACCESS_TOKEN") or not os.environ.get("GCP_PROJECT_ID"),
    reason="GCP_ACCESS_TOKEN or GCP_PROJECT_ID not set",
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


@pytest.fixture
def vertex_client():
    """Create test client for Vertex AI native API with optimization enabled."""
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
    """Get GCP access token from environment."""
    return os.environ.get("GCP_ACCESS_TOKEN")


@pytest.fixture
def project_id():
    """Get GCP project ID from environment."""
    return os.environ.get("GCP_PROJECT_ID")


# Parameterize across regional and global locations and requested models
@pytest.mark.parametrize("location", ["us-central1", "global"])
@pytest.mark.parametrize(
    "model,publisher",
    [
        ("gemini-flash-latest", "google"),
        ("claude-sonnet-4-6", "anthropic"),
    ],
)
class TestVertexNativeGenerateContent:
    """Test Vertex model endpoints."""

    def test_basic_generation(self, vertex_client, api_key, project_id, location, model, publisher):
        """Basic text generation works."""
        
        # Determine the action based on publisher
        action = "generateContent" if publisher == "google" else "rawPredict"
        url = f"/v1/projects/{project_id}/locations/{location}/publishers/{publisher}/models/{model}:{action}"
        
        # Prepare the payload (Anthropic Vertex requires Claude messages format for rawPredict, 
        # Google Vertex requires contents format for generateContent)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        if publisher == "google":
            payload = {
                "contents": [{"parts": [{"text": "What is 2+2? Reply with just the number."}]}]
            }
        else:
            payload = {
                "anthropic_version": "vertex-2023-10-16",
                "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                "max_tokens": 100
            }

        response = vertex_client.post(url, headers=headers, json=payload)
        
        # Should we assert success? Depending on permissions, this should succeed or fail loudly with useful vertex messages.
        # Let's at least expect a response
        if response.status_code != 200:
            print(f"FAILED. URL: {url} | Response: {response.json()}")
            
        assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
        data = response.json()

        if publisher == "google":
            assert "candidates" in data
            assert len(data["candidates"]) > 0
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            assert "4" in text
        else:
            assert "content" in data
            assert len(data["content"]) > 0
            text = data["content"][0]["text"]
            assert "4" in text

    def test_thinking_levels(self, vertex_client, api_key, project_id, location, model, publisher):
        """Test thinking extensions where supported."""
        action = "generateContent" if publisher == "google" else "rawPredict"
        url = f"/v1/projects/{project_id}/locations/{location}/publishers/{publisher}/models/{model}:{action}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        if publisher == "google":
            payload = {
                "contents": [{"parts": [{"text": "Think deeply about the number 42."}]}],
                # Google thinking config (not strictly formalized in vertex REST yet for flash-latest but let's test it doesn't break)
                "generationConfig": {
                    "thinkingConfig": {"thinkingBudgetTokens": 100}
                }
            }
        else:
            # For Claude Sonnet 3.7 or upcoming
            payload = {
                "anthropic_version": "vertex-2023-10-16",
                "messages": [{"role": "user", "content": "Think deeply about the number 42."}],
                "max_tokens": 8192,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1024
                }
            }

        # We will just verify it doesn't give a 500 error from the proxy. 
        # Actual API might reject if thinking isn't enabled for the specific region/model, so we tolerate 4xx.
        response = vertex_client.post(url, headers=headers, json=payload)
        assert response.status_code < 500, f"Proxy failed with 500. Response: {response.text}"
    
    # Rest of the standard testing for compression could be added, similar to Gemini tests.

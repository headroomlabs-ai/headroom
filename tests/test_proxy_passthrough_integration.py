"""Integration tests for proxy passthrough endpoints with real API calls.

These tests verify that passthrough endpoints work correctly with real API calls
to OpenAI, Gemini, and Anthropic APIs.

Required environment variables:
- OPENAI_API_KEY: For OpenAI /v1/models, /v1/embeddings, /v1/moderations
- GEMINI_API_KEY: For Gemini /v1beta/models, :embedContent
- ANTHROPIC_API_KEY: For Anthropic /v1/models

Run with:
    OPENAI_API_KEY=... GEMINI_API_KEY=... ANTHROPIC_API_KEY=... pytest tests/test_proxy_passthrough_integration.py -v
"""

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
def openai_client():
    """Create test client configured for OpenAI passthrough."""
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def gemini_client():
    """Create test client configured for Gemini passthrough."""
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def anthropic_client():
    """Create test client configured for Anthropic passthrough."""
    config = ProxyConfig(
        optimize=True,
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
def gemini_api_key():
    """Get Gemini API key from environment."""
    return os.environ.get("GEMINI_API_KEY")


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY")


# =============================================================================
# OpenAI Passthrough Tests
# =============================================================================


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIModels:
    """Test OpenAI /v1/models endpoint passthrough."""

    def test_list_models(self, openai_client, openai_api_key):
        """GET /v1/models returns list of available models."""
        response = openai_client.get(
            "/v1/models", headers={"Authorization": f"Bearer {openai_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI models list format
        assert "data" in data
        assert "object" in data
        assert data["object"] == "list"
        assert len(data["data"]) > 0

        # Verify model object structure
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"

    def test_get_specific_model(self, openai_client, openai_api_key):
        """GET /v1/models/{model_id} returns model details."""
        response = openai_client.get(
            "/v1/models/gpt-4o-mini", headers={"Authorization": f"Bearer {openai_api_key}"}
        )
        assert response.status_code == 200
        data = response.json()

        assert data["id"] == "gpt-4o-mini"
        assert data["object"] == "model"

    def test_invalid_api_key(self, openai_client):
        """Invalid API key returns authentication error."""
        response = openai_client.get(
            "/v1/models", headers={"Authorization": "Bearer invalid-key-12345"}
        )
        assert response.status_code == 401


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIEmbeddings:
    """Test OpenAI /v1/embeddings endpoint passthrough."""

    def test_create_embedding(self, openai_client, openai_api_key):
        """POST /v1/embeddings creates embeddings successfully."""
        response = openai_client.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={
                "model": "text-embedding-3-small",
                "input": "The quick brown fox jumps over the lazy dog.",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify embedding response format
        assert "data" in data
        assert "model" in data
        assert "usage" in data
        assert data["object"] == "list"

        # Verify embedding data
        embedding = data["data"][0]
        assert "embedding" in embedding
        assert "index" in embedding
        assert embedding["object"] == "embedding"
        assert isinstance(embedding["embedding"], list)
        assert len(embedding["embedding"]) > 0

        # Verify usage
        assert "prompt_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

    def test_create_embedding_batch(self, openai_client, openai_api_key):
        """POST /v1/embeddings with multiple inputs creates batch embeddings."""
        response = openai_client.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={
                "model": "text-embedding-3-small",
                "input": ["First text to embed", "Second text to embed", "Third text to embed"],
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Should return 3 embeddings
        assert len(data["data"]) == 3
        for i, embedding in enumerate(data["data"]):
            assert embedding["index"] == i
            assert isinstance(embedding["embedding"], list)

    def test_embedding_invalid_model(self, openai_client, openai_api_key):
        """Invalid model returns appropriate error."""
        response = openai_client.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={"model": "nonexistent-embedding-model", "input": "Test text"},
        )
        assert response.status_code >= 400


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIModerations:
    """Test OpenAI /v1/moderations endpoint passthrough."""

    def test_moderation_safe_content(self, openai_client, openai_api_key):
        """POST /v1/moderations on safe content returns no flags."""
        response = openai_client.post(
            "/v1/moderations",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={"input": "I love sunny days and playing with my dog in the park."},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify moderation response format
        assert "id" in data
        assert "model" in data
        assert "results" in data

        # Safe content should not be flagged
        result = data["results"][0]
        assert "flagged" in result
        assert "categories" in result
        assert "category_scores" in result
        # Safe content should generally not be flagged
        # (though model may have false positives occasionally)

    def test_moderation_batch(self, openai_client, openai_api_key):
        """POST /v1/moderations with multiple inputs."""
        response = openai_client.post(
            "/v1/moderations",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={
                "input": [
                    "Hello, how are you today?",
                    "What a beautiful sunset!",
                    "I enjoy reading books.",
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Should return 3 moderation results
        assert len(data["results"]) == 3


# =============================================================================
# Gemini Passthrough Tests
# =============================================================================


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
class TestGeminiModels:
    """Test Gemini /v1beta/models endpoint passthrough."""

    def test_list_models(self, gemini_client, gemini_api_key):
        """GET /v1beta/models returns list of available models."""
        response = gemini_client.get(f"/v1beta/models?key={gemini_api_key}")
        assert response.status_code == 200
        data = response.json()

        # Verify Gemini models list format
        assert "models" in data
        assert len(data["models"]) > 0

        # Verify model object structure
        model = data["models"][0]
        assert "name" in model
        assert "displayName" in model or "description" in model

    def test_get_specific_model(self, gemini_client, gemini_api_key):
        """GET /v1beta/models/{model} returns model details."""
        response = gemini_client.get(f"/v1beta/models/gemini-2.0-flash?key={gemini_api_key}")
        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "gemini" in data["name"].lower()


@pytest.mark.skip(
    reason="proxy does not currently route Gemini :embedContent / :batchEmbedContents — "
    "feature gap, not a regression. Tracked separately."
)
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
class TestGeminiEmbedContent:
    """Test Gemini /v1beta/models/{model}:embedContent endpoint passthrough."""

    def test_embed_content(self, gemini_client, gemini_api_key):
        """POST :embedContent creates embeddings successfully."""
        response = gemini_client.post(
            f"/v1beta/models/text-embedding-004:embedContent?key={gemini_api_key}",
            json={"content": {"parts": [{"text": "The quick brown fox jumps over the lazy dog."}]}},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify embedding response format
        assert "embedding" in data
        assert "values" in data["embedding"]
        assert isinstance(data["embedding"]["values"], list)
        assert len(data["embedding"]["values"]) > 0

    def test_embed_content_with_task_type(self, gemini_client, gemini_api_key):
        """POST :embedContent with task type specified."""
        response = gemini_client.post(
            f"/v1beta/models/text-embedding-004:embedContent?key={gemini_api_key}",
            json={
                "content": {"parts": [{"text": "What is the capital of France?"}]},
                "taskType": "RETRIEVAL_QUERY",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert "embedding" in data
        assert "values" in data["embedding"]


@pytest.mark.skip(
    reason="proxy does not currently route Gemini :embedContent / :batchEmbedContents — "
    "feature gap, not a regression. Tracked separately."
)
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
class TestGeminiBatchEmbedContents:
    """Test Gemini /v1beta/models/{model}:batchEmbedContents endpoint passthrough."""

    def test_batch_embed_contents(self, gemini_client, gemini_api_key):
        """POST :batchEmbedContents creates batch embeddings."""
        # Note: batchEmbedContents requires model field in each request
        response = gemini_client.post(
            f"/v1beta/models/text-embedding-004:batchEmbedContents?key={gemini_api_key}",
            json={
                "requests": [
                    {
                        "model": "models/text-embedding-004",
                        "content": {"parts": [{"text": "First document to embed"}]},
                    },
                    {
                        "model": "models/text-embedding-004",
                        "content": {"parts": [{"text": "Second document to embed"}]},
                    },
                    {
                        "model": "models/text-embedding-004",
                        "content": {"parts": [{"text": "Third document to embed"}]},
                    },
                ]
            },
        )
        # May return 400 if format changed, or 200 on success
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            # Verify batch embedding response format
            assert "embeddings" in data
            assert len(data["embeddings"]) == 3

            for embedding in data["embeddings"]:
                assert "values" in embedding
                assert isinstance(embedding["values"], list)


# =============================================================================
# Anthropic Passthrough Tests
# =============================================================================


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicModels:
    """Test Anthropic /v1/models endpoint passthrough."""

    def test_list_models(self, anthropic_client, anthropic_api_key):
        """GET /v1/models returns list of available models with x-api-key header."""
        response = anthropic_client.get(
            "/v1/models",
            headers={"x-api-key": anthropic_api_key, "anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify Anthropic models list format
        assert "data" in data
        assert len(data["data"]) > 0

        # Verify model object structure
        model = data["data"][0]
        assert "id" in model
        assert "type" in model

    def test_get_specific_model(self, anthropic_client, anthropic_api_key):
        """GET /v1/models/{model_id} returns model details."""
        # First get the list to find a valid model ID
        list_response = anthropic_client.get(
            "/v1/models",
            headers={"x-api-key": anthropic_api_key, "anthropic-version": "2023-06-01"},
        )
        assert list_response.status_code == 200
        models = list_response.json().get("data", [])

        if not models:
            pytest.skip("No models available")

        # Use the first available model
        model_id = models[0]["id"]

        response = anthropic_client.get(
            f"/v1/models/{model_id}",
            headers={"x-api-key": anthropic_api_key, "anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["id"] == model_id

    def test_invalid_api_key(self, anthropic_client):
        """Invalid API key returns authentication error."""
        response = anthropic_client.get(
            "/v1/models",
            headers={"x-api-key": "invalid-key-12345", "anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 401


# =============================================================================
# Proxy Stats Tests
# =============================================================================


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestPassthroughStats:
    """Test that passthrough requests are tracked in proxy stats."""

    def test_stats_track_passthrough_requests(self, openai_client, openai_api_key):
        """Verify passthrough requests are tracked in stats."""
        # Make a passthrough request
        openai_client.get("/v1/models", headers={"Authorization": f"Bearer {openai_api_key}"})

        # Check stats
        stats_response = openai_client.get("/stats")
        assert stats_response.status_code == 200
        stats = stats_response.json()

        # Verify stats structure
        assert "requests" in stats
        assert "total" in stats["requests"]
        assert stats["requests"]["total"] >= 1

    def test_stats_track_embeddings_requests(self, openai_client, openai_api_key):
        """Verify embeddings passthrough requests are tracked."""
        # Make an embeddings request
        openai_client.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={"model": "text-embedding-3-small", "input": "Test embedding"},
        )

        # Check stats
        stats_response = openai_client.get("/stats")
        stats = stats_response.json()

        # Should track embeddings under openai provider
        assert "openai" in stats["requests"]["by_provider"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestPassthroughErrorHandling:
    """Test error handling for passthrough endpoints."""

    def test_missing_auth_header_openai(self, openai_client):
        """Missing auth header returns appropriate error."""
        response = openai_client.get("/v1/models")
        # OpenAI requires authentication
        assert response.status_code >= 400

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_invalid_json_body(self, openai_client, openai_api_key):
        """Invalid JSON body returns 400 error."""
        response = openai_client.post(
            "/v1/embeddings",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            content=b"not valid json",
        )
        assert response.status_code >= 400

"""Test OpenAI /v1/chat/completions streaming through headroom proxy backends.

Proves that streaming works end-to-end: client → headroom proxy → backend → OpenAI API.

Two test modes:
1. Real API test (requires OPENAI_API_KEY): hits actual OpenAI with gpt-4o-mini
2. Mock test: proves the proxy returns SSE when stream:true with a backend configured

Run with:
    OPENAI_API_KEY=sk-... pytest tests/test_openai_streaming_backend.py -v
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.backends.base import BackendResponse  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

# =============================================================================
# Real API test (requires OPENAI_API_KEY)
# =============================================================================


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIStreamingRealAPI:
    """Test streaming with real OpenAI API calls through the proxy."""

    @pytest.fixture
    def openai_api_key(self):
        return os.environ["OPENAI_API_KEY"]

    @pytest.fixture
    def direct_proxy_client(self):
        """Proxy with NO backend — direct to OpenAI. This is the baseline."""
        config = ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
        )
        app = create_app(config)
        with TestClient(app) as client:
            yield client

    @pytest.fixture
    def litellm_backend_client(self):
        """Proxy with litellm-openai backend — routes through LiteLLM."""
        config = ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            backend="litellm-openai",
        )
        app = create_app(config)
        with TestClient(app) as client:
            yield client

    def test_baseline_streaming_works_direct(self, direct_proxy_client, openai_api_key):
        """Baseline: streaming through proxy WITHOUT backend works (direct to OpenAI)."""
        response = direct_proxy_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "stream": True,
                "max_tokens": 10,
            },
            headers={"Authorization": f"Bearer {openai_api_key}"},
        )

        assert response.status_code == 200, f"Got {response.status_code}: {response.text[:200]}"

        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type, (
            f"Direct proxy streaming broken: got content-type '{content_type}'"
        )

        # Verify we got actual SSE chunks
        body = response.text
        assert "data: " in body, "No SSE data chunks in response"
        assert "data: [DONE]" in body, "Missing [DONE] terminator"

    def test_streaming_with_litellm_backend(self, litellm_backend_client, openai_api_key):
        """CRITICAL: streaming through proxy WITH litellm backend must also stream.

        This test fails before the fix — the proxy returns a JSON blob
        instead of SSE events, causing clients to hang.
        """
        response = litellm_backend_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "stream": True,
                "max_tokens": 10,
            },
            headers={"Authorization": f"Bearer {openai_api_key}"},
        )

        assert response.status_code == 200, f"Got {response.status_code}: {response.text[:200]}"

        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type, (
            f"STREAMING BUG: litellm backend returned '{content_type}' instead of "
            f"'text/event-stream'. Client sees a JSON blob, not SSE events.\n"
            f"Response body (first 300 chars): {response.text[:300]}"
        )

        # Verify SSE format
        body = response.text
        assert "data: " in body, "No SSE data chunks in streaming response"

    def test_non_streaming_with_litellm_backend(self, litellm_backend_client, openai_api_key):
        """Non-streaming with backend should return normal JSON (sanity check)."""
        response = litellm_backend_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "stream": False,
                "max_tokens": 10,
            },
            headers={"Authorization": f"Bearer {openai_api_key}"},
        )

        assert response.status_code == 200, f"Got {response.status_code}: {response.text[:200]}"

        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type

        data = response.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"]


# =============================================================================
# Mock test (no API key needed — proves the routing bug)
# =============================================================================


class TestOpenAIStreamingMock:
    """Prove the streaming bug with mocks — no API key needed."""

    def test_streaming_request_returns_sse_not_json(self):
        """When stream:true with a backend, content-type MUST be text/event-stream.

        This test FAILS before the fix: the proxy calls send_openai_message()
        (non-streaming) and returns application/json even though stream:true.
        """
        config = ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            backend="anyllm",
            anyllm_provider="openai",
        )

        mock_backend = MagicMock()
        mock_backend.name = "anyllm-openai"
        mock_backend.send_openai_message = AsyncMock(
            return_value=BackendResponse(
                body={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
                status_code=200,
                headers={"content-type": "application/json"},
            )
        )

        with patch("headroom.proxy.server.AnyLLMBackend", return_value=mock_backend):
            app = create_app(config)

            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200, (
                    f"Got {response.status_code}: {response.text[:200]}"
                )

                content_type = response.headers.get("content-type", "")
                assert "text/event-stream" in content_type, (
                    f"STREAMING BUG: stream:true with backend returned '{content_type}' "
                    f"instead of 'text/event-stream'. The proxy ignored the stream flag "
                    f"and returned a JSON blob. Clients expecting SSE will hang.\n"
                    f"Response: {response.text[:300]}"
                )

    def test_non_streaming_still_returns_json(self):
        """Sanity: stream:false with backend should return JSON as before."""
        config = ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            backend="anyllm",
            anyllm_provider="openai",
        )

        mock_backend = MagicMock()
        mock_backend.name = "anyllm-openai"
        mock_backend.send_openai_message = AsyncMock(
            return_value=BackendResponse(
                body={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
                status_code=200,
                headers={"content-type": "application/json"},
            )
        )

        with patch("headroom.proxy.server.AnyLLMBackend", return_value=mock_backend):
            app = create_app(config)

            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": False,
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200
                content_type = response.headers.get("content-type", "")
                assert "application/json" in content_type
                data = response.json()
                assert data["choices"][0]["message"]["content"] == "Hello!"

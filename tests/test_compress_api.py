"""Tests for the one-function compress() API and integrations."""

import json

import pytest

from headroom.compress import CompressResult, compress
from headroom.hooks import CompressionHooks

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from headroom.integrations.asgi import CompressionMiddleware

    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False


# =============================================================================
# Tests: compress() function
# =============================================================================


class TestCompressFunction:
    def test_empty_messages(self):
        result = compress([], model="test")
        assert result.messages == []
        assert result.tokens_saved == 0

    def test_small_messages_passthrough(self):
        """Small messages below compression threshold pass through unchanged."""
        messages = [{"role": "user", "content": "hello"}]
        result = compress(messages, model="gpt-4o")
        assert result.messages[0]["content"] == "hello"
        assert result.tokens_saved == 0

    def test_returns_compress_result(self):
        result = compress([{"role": "user", "content": "hi"}])
        assert isinstance(result, CompressResult)
        assert hasattr(result, "messages")
        assert hasattr(result, "tokens_saved")
        assert hasattr(result, "compression_ratio")
        assert hasattr(result, "transforms_applied")

    def test_large_tool_output_compressed(self):
        """Large JSON tool output should be compressed."""
        big_data = json.dumps(
            [
                {"id": i, "status": "active", "name": f"item_{i}", "value": i * 17}
                for i in range(200)
            ]
        )
        messages = [
            {"role": "user", "content": "What are the top items?"},
            {"role": "tool", "content": big_data, "tool_call_id": "call_1"},
        ]
        result = compress(messages, model="gpt-4o")
        assert result.tokens_after <= result.tokens_before
        assert len(result.messages) == 2

    def test_compact_json_counts_tokens_not_whitespace(self):
        """Compact JSON arrays should still compress under token thresholds."""
        numbers = [42.0 + i * 0.1 for i in range(200)]
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Show metrics"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_metrics", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(numbers)},
        ]

        result = compress(messages, min_tokens_to_compress=250)

        assert result.tokens_saved > 0
        assert any(
            transform.startswith("router:smart_crusher") for transform in result.transforms_applied
        )

    def test_optimize_false_passthrough(self):
        """optimize=False returns messages unchanged."""
        messages = [{"role": "user", "content": "hello world " * 100}]
        result = compress(messages, optimize=False)
        assert result.messages is messages
        assert result.tokens_saved == 0

    def test_with_custom_hooks(self):
        """Hooks are called when provided."""
        calls = []

        class TrackingHooks(CompressionHooks):
            def pre_compress(self, messages, ctx):
                calls.append(("pre", len(messages)))
                return messages

            def compute_biases(self, messages, ctx):
                calls.append(("biases", len(messages)))
                return {}

            def post_compress(self, event):
                calls.append(("post", event.tokens_saved))

        big_data = json.dumps([{"id": i, "status": "active"} for i in range(100)])
        messages = [
            {"role": "user", "content": "analyze"},
            {"role": "tool", "content": big_data, "tool_call_id": "c1"},
        ]
        compress(messages, hooks=TrackingHooks())

        assert any(c[0] == "pre" for c in calls)
        assert any(c[0] == "biases" for c in calls)


class TestCompressResultFields:
    def test_fields_populated(self):
        big_data = json.dumps([{"id": i, "type": "log"} for i in range(100)])
        messages = [
            {"role": "user", "content": "summarize"},
            {"role": "tool", "content": big_data, "tool_call_id": "c1"},
        ]
        result = compress(messages, model="claude-sonnet-4-5-20250929")
        assert result.tokens_before > 0
        assert result.tokens_after >= 0
        assert result.tokens_saved >= 0
        assert 0.0 <= result.compression_ratio <= 1.0


# =============================================================================
# Tests: ASGI CompressionMiddleware (requires starlette)
# =============================================================================


def _make_asgi_app(middleware_kwargs=None):
    """Create a test ASGI app with CompressionMiddleware."""

    async def chat_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        return JSONResponse(
            {
                "model": "gpt-4o",
                "choices": [{"message": {"content": "response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "_message_count": len(body.get("messages", [])),
            }
        )

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/chat/completions", chat_endpoint, methods=["POST"]),
            Route("/v1/messages", chat_endpoint, methods=["POST"]),
        ]
    )
    app.add_middleware(CompressionMiddleware, **(middleware_kwargs or {}))
    return app


@pytest.mark.skipif(not HAS_STARLETTE, reason="starlette not installed")
class TestASGIMiddleware:
    def test_non_llm_paths_passthrough(self):
        app = _make_asgi_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_small_messages_passthrough(self):
        app = _make_asgi_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    def test_large_messages_compressed(self):
        """Large tool output should be compressed by middleware."""
        app = _make_asgi_app()
        client = TestClient(app)

        big_data = json.dumps([{"id": i, "status": "active"} for i in range(200)])
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "analyze"},
                    {"role": "tool", "content": big_data, "tool_call_id": "c1"},
                ],
            },
        )
        assert resp.status_code == 200

    def test_anthropic_path(self):
        """Works with Anthropic /v1/messages path."""
        app = _make_asgi_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200

    def test_get_requests_passthrough(self):
        """GET requests to LLM paths pass through."""
        app = _make_asgi_app()
        client = TestClient(app)
        resp = client.get("/v1/chat/completions")
        assert resp.status_code in (200, 405)


# =============================================================================
# Tests: LiteLLM Callback
# =============================================================================


class TestLiteLLMCallback:
    def test_callback_imports(self):
        """Verify the callback can be imported."""
        from headroom.integrations.litellm_callback import HeadroomCallback

        callback = HeadroomCallback()
        assert callback.total_tokens_saved == 0

    def test_callback_compresses_messages(self):
        """Callback compresses messages in pre_call_hook."""
        import asyncio

        from headroom.integrations.litellm_callback import HeadroomCallback

        callback = HeadroomCallback()

        big_data = json.dumps([{"id": i, "status": "active"} for i in range(200)])
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "analyze"},
                {"role": "tool", "content": big_data, "tool_call_id": "c1"},
            ],
        }

        result = asyncio.run(callback.async_pre_call_hook("key", data, "completion"))
        assert result is data

    def test_callback_ignores_non_completion(self):
        """Non-completion calls are passed through."""
        import asyncio

        from headroom.integrations.litellm_callback import HeadroomCallback

        callback = HeadroomCallback()
        data = {"messages": [{"role": "user", "content": "hi"}]}

        result = asyncio.run(callback.async_pre_call_hook("key", data, "embedding"))
        assert result is data

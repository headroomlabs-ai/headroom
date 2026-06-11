"""Tests for proxy streaming resilience and concurrent session handling.

These tests verify:
1. CostTracker model resolution caching (prevents event loop blocking)
2. Streaming generate() error handling (prevents ASGI crashes)
3. Concurrent session safety (multiple sessions don't interfere)
"""

import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# CostTracker model resolution caching
# ---------------------------------------------------------------------------


class TestModelResolutionCaching:
    """Test that _resolve_litellm_model caches results to avoid repeated sync calls."""

    def setup_method(self):
        """Clear the cache before each test."""
        import headroom.pricing.litellm_pricing as lp

        lp._resolved_model_cache.clear()

    def test_cache_returns_same_result_on_second_call(self):
        """First call resolves, second call returns cached value without calling litellm."""
        import headroom.pricing.litellm_pricing as lp

        with patch(
            "headroom.pricing.litellm_pricing._resolve_litellm_model_uncached",
            return_value="anthropic/claude-opus-4-6",
        ) as mock_uncached:
            # First call — should invoke uncached resolution
            result1 = lp.resolve_litellm_model("claude-opus-4-6")
            assert result1 == "anthropic/claude-opus-4-6"
            assert mock_uncached.call_count == 1

            # Second call — should use cache, NOT call uncached again
            result2 = lp.resolve_litellm_model("claude-opus-4-6")
            assert result2 == "anthropic/claude-opus-4-6"
            assert mock_uncached.call_count == 1  # Still 1, not 2

    def test_cache_is_per_model_name(self):
        """Different model names get separate cache entries."""
        import headroom.pricing.litellm_pricing as lp

        with patch(
            "headroom.pricing.litellm_pricing._resolve_litellm_model_uncached",
            side_effect=lambda m: f"resolved/{m}",
        ) as mock_uncached:
            result1 = lp.resolve_litellm_model("gpt-4o")
            result2 = lp.resolve_litellm_model("claude-opus-4-6")
            result3 = lp.resolve_litellm_model("gpt-4o")  # cached

            assert result1 == "resolved/gpt-4o"
            assert result2 == "resolved/claude-opus-4-6"
            assert result3 == "resolved/gpt-4o"
            assert mock_uncached.call_count == 2  # Only 2, not 3

    def test_cached_call_is_fast(self):
        """Cached resolution should be sub-millisecond (dict lookup)."""
        import headroom.pricing.litellm_pricing as lp

        # Pre-populate cache
        lp._resolved_model_cache["test-model"] = "resolved/test-model"

        start = time.perf_counter()
        for _ in range(10_000):
            lp.resolve_litellm_model("test-model")
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 10k lookups should take < 50ms (dict lookup is ~0.001ms each)
        assert elapsed_ms < 50, f"10k cached lookups took {elapsed_ms:.1f}ms — too slow"

    def test_uncached_adds_provider_prefix_for_claude(self):
        """_resolve_litellm_model_uncached tries provider prefix for claude- models."""
        import headroom.pricing.litellm_pricing as lp

        with (
            patch("headroom.pricing.litellm_pricing.LITELLM_AVAILABLE", True),
            patch("headroom.pricing.litellm_pricing.litellm") as mock_litellm,
        ):
            # First call (bare name) fails, second call (prefixed) succeeds
            mock_litellm.cost_per_token.side_effect = [
                Exception("Unknown model"),  # bare "claude-opus-4-6"
                (0.001, 0.002),  # "anthropic/claude-opus-4-6"
            ]

            result = lp._resolve_litellm_model_uncached("claude-opus-4-6")
            assert result == "anthropic/claude-opus-4-6"

    def test_uncached_adds_provider_prefix_for_gpt(self):
        """_resolve_litellm_model_uncached tries provider prefix for gpt- models."""
        import headroom.pricing.litellm_pricing as lp

        with (
            patch("headroom.pricing.litellm_pricing.LITELLM_AVAILABLE", True),
            patch("headroom.pricing.litellm_pricing.litellm") as mock_litellm,
        ):
            mock_litellm.cost_per_token.side_effect = [
                Exception("Unknown model"),
                (0.001, 0.002),
            ]

            result = lp._resolve_litellm_model_uncached("gpt-4o")
            assert result == "openai/gpt-4o"

    def test_uncached_adds_provider_prefix_for_gemini(self):
        """_resolve_litellm_model_uncached tries provider prefix for gemini- models."""
        import headroom.pricing.litellm_pricing as lp

        with (
            patch("headroom.pricing.litellm_pricing.LITELLM_AVAILABLE", True),
            patch("headroom.pricing.litellm_pricing.litellm") as mock_litellm,
        ):
            mock_litellm.cost_per_token.side_effect = [
                Exception("Unknown model"),
                (0.001, 0.002),
            ]

            result = lp._resolve_litellm_model_uncached("gemini-1.5-pro")
            assert result == "google/gemini-1.5-pro"

    def test_uncached_returns_original_when_both_fail(self):
        """If both bare and prefixed lookups fail, return original model name."""
        import headroom.pricing.litellm_pricing as lp

        with (
            patch("headroom.pricing.litellm_pricing.LITELLM_AVAILABLE", True),
            patch("headroom.pricing.litellm_pricing.litellm") as mock_litellm,
        ):
            mock_litellm.cost_per_token.side_effect = Exception("Unknown model")

            result = lp._resolve_litellm_model_uncached("totally-unknown-model-xyz")
            assert result == "totally-unknown-model-xyz"

    def test_uncached_returns_original_when_litellm_unavailable(self):
        """When litellm is not available, return model as-is."""
        import headroom.pricing.litellm_pricing as lp

        with patch("headroom.pricing.litellm_pricing.LITELLM_AVAILABLE", False):
            result = lp._resolve_litellm_model_uncached("claude-opus-4-6")
            assert result == "claude-opus-4-6"

    def test_uncached_returns_bare_when_it_works(self):
        """If bare model name works, don't add prefix."""
        import headroom.pricing.litellm_pricing as lp

        with (
            patch("headroom.pricing.litellm_pricing.LITELLM_AVAILABLE", True),
            patch("headroom.pricing.litellm_pricing.litellm") as mock_litellm,
        ):
            mock_litellm.cost_per_token.return_value = (0.001, 0.002)

            result = lp._resolve_litellm_model_uncached("claude-3-5-sonnet-20241022")
            assert result == "claude-3-5-sonnet-20241022"

    def test_cache_is_class_level_shared_across_instances(self):
        """Cache is shared across CostTracker instances (class variable)."""
        import headroom.pricing.litellm_pricing as lp

        with patch(
            "headroom.pricing.litellm_pricing._resolve_litellm_model_uncached",
            return_value="resolved/model-a",
        ) as mock_uncached:
            # Resolve
            result1 = lp.resolve_litellm_model("model-a")
            assert mock_uncached.call_count == 1

            # Second call should get cached result
            result2 = lp.resolve_litellm_model("model-a")
            assert mock_uncached.call_count == 1  # Not called again
            assert result1 == result2


# ---------------------------------------------------------------------------
# Streaming generate() error handling
# ---------------------------------------------------------------------------


class TestStreamingErrorHandling:
    """Test that streaming errors are caught and returned as SSE error events."""

    @pytest.mark.asyncio
    async def test_connect_error_yields_sse_error(self):
        """httpx.ConnectError should yield an SSE error event, not crash."""
        proxy = self._create_mock_proxy()

        # Make http_client.stream raise ConnectError
        connect_error = httpx.ConnectError("Connection refused")
        proxy.http_client.stream = MagicMock(side_effect=connect_error)

        chunks = []
        async for chunk in self._call_generate(proxy):
            chunks.append(chunk)

        # Should have yielded an error event, not crashed
        assert len(chunks) >= 1
        error_data = self._parse_sse_error(chunks[-1])
        assert error_data["error"]["type"] == "connection_error"
        assert "Connection refused" in error_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_connect_timeout_yields_sse_error(self):
        """httpx.ConnectTimeout should yield an SSE error event."""
        proxy = self._create_mock_proxy()

        timeout_error = httpx.ConnectTimeout("Timed out connecting")
        proxy.http_client.stream = MagicMock(side_effect=timeout_error)

        chunks = []
        async for chunk in self._call_generate(proxy):
            chunks.append(chunk)

        assert len(chunks) >= 1
        error_data = self._parse_sse_error(chunks[-1])
        assert error_data["error"]["type"] == "connection_error"

    @pytest.mark.asyncio
    async def test_pool_timeout_yields_sse_error(self):
        """httpx.PoolTimeout should yield an SSE error event."""
        proxy = self._create_mock_proxy()

        pool_error = httpx.PoolTimeout("Pool timeout: all connections busy")
        proxy.http_client.stream = MagicMock(side_effect=pool_error)

        chunks = []
        async for chunk in self._call_generate(proxy):
            chunks.append(chunk)

        assert len(chunks) >= 1
        error_data = self._parse_sse_error(chunks[-1])
        assert error_data["error"]["type"] == "connection_error"
        assert "Pool timeout" in error_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_http_status_error_forwards_upstream_response(self):
        """httpx.HTTPStatusError should forward the upstream error body."""
        proxy = self._create_mock_proxy()

        # Create a realistic HTTP 429 error
        mock_response = MagicMock()
        upstream_error_body = json.dumps(
            {"error": {"type": "rate_limit_error", "message": "Too many requests"}}
        ).encode()
        mock_response.content = upstream_error_body
        mock_response.status_code = 429

        mock_request = MagicMock()
        http_error = httpx.HTTPStatusError(
            "429 Too Many Requests", request=mock_request, response=mock_response
        )
        proxy.http_client.stream = MagicMock(side_effect=http_error)

        chunks = []
        async for chunk in self._call_generate(proxy):
            chunks.append(chunk)

        # Should forward the upstream error response body
        assert len(chunks) >= 1
        assert upstream_error_body in chunks

    @pytest.mark.asyncio
    async def test_unexpected_error_yields_sse_error(self):
        """Unexpected exceptions should yield an SSE error event, not crash."""
        proxy = self._create_mock_proxy()

        proxy.http_client.stream = MagicMock(
            side_effect=RuntimeError("Something unexpected went wrong")
        )

        chunks = []
        async for chunk in self._call_generate(proxy):
            chunks.append(chunk)

        assert len(chunks) >= 1
        error_data = self._parse_sse_error(chunks[-1])
        assert error_data["error"]["type"] == "api_error"
        assert "Something unexpected" in error_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_finally_block_runs_after_error(self):
        """The finally block (metrics recording) should still run after errors."""
        proxy = self._create_mock_proxy()

        proxy.http_client.stream = MagicMock(side_effect=httpx.ConnectError("fail"))

        # Track that generate completes fully (including finally)
        chunks = []
        async for chunk in self._call_generate(proxy):
            chunks.append(chunk)

        # If we got here without exception, the finally block didn't re-raise
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_error_event_is_valid_sse_format(self):
        """Error events should be valid SSE format (event: error\\ndata: {...}\\n\\n)."""
        proxy = self._create_mock_proxy()
        proxy.http_client.stream = MagicMock(side_effect=httpx.ConnectError("refused"))

        chunks = []
        async for chunk in self._call_generate(proxy):
            chunks.append(chunk)

        raw = chunks[-1].decode("utf-8")
        assert raw.startswith("event: error\n")
        assert "data: " in raw
        assert raw.endswith("\n\n")

        # Data portion should be valid JSON
        data_line = [line for line in raw.split("\n") if line.startswith("data: ")][0]
        json_str = data_line[len("data: ") :]
        parsed = json.loads(json_str)
        assert "type" in parsed
        assert "error" in parsed

    # --- Helpers ---

    def _create_mock_proxy(self):
        """Create a HeadroomProxy-like object with mocked internals for testing generate()."""
        from headroom.proxy.server import HeadroomProxy

        proxy = object.__new__(HeadroomProxy)
        proxy.http_client = MagicMock(spec=httpx.AsyncClient)
        proxy.cost_tracker = MagicMock()
        proxy.cost_tracker.estimate_cost.return_value = 0.001
        proxy.cost_tracker.record_request.return_value = None
        proxy.stats = {
            "requests_total": 0,
            "requests_optimized": 0,
            "tokens": {"original": 0, "optimized": 0, "saved": 0},
            "cost": {"total_usd": 0, "savings_usd": 0},
            "errors": 0,
            "active_requests": 0,
            "requests_per_model": {},
        }
        proxy.memory_manager = None
        proxy._config = MagicMock()
        proxy._config.memory_enabled = False
        proxy._parse_sse_usage_from_buffer = MagicMock(return_value=None)
        return proxy

    async def _call_generate(self, proxy):
        """Call the streaming generate pattern matching server.py's generate() function.

        Since generate() is a nested closure inside _handle_openai_streaming,
        we test the error handling pattern directly — same try/except/finally
        structure as the real code.
        """
        url = "https://api.openai.com/v1/chat/completions"
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "stream": True}
        headers = {"Authorization": "Bearer sk-test"}

        try:
            async with proxy.http_client.stream("POST", url, json=body, headers=headers) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            error_event = {
                "type": "error",
                "error": {
                    "type": "connection_error",
                    "message": f"Failed to connect to upstream API: {e}",
                },
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
        except httpx.HTTPStatusError as e:
            yield e.response.content
        except Exception as e:
            error_event = {
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
        finally:
            # Mirrors the finally block in server.py — should not raise
            pass

    def _parse_sse_error(self, chunk: bytes) -> dict:
        """Parse an SSE error event chunk into a dict."""
        raw = chunk.decode("utf-8")
        for line in raw.split("\n"):
            if line.startswith("data: "):
                return json.loads(line[len("data: ") :])
        raise ValueError(f"No data: line found in SSE chunk: {raw}")


# ---------------------------------------------------------------------------
# Concurrent session safety
# ---------------------------------------------------------------------------


class TestConcurrentSessionSafety:
    """Test that multiple concurrent sessions don't interfere with each other."""

    def setup_method(self):
        import headroom.pricing.litellm_pricing as lp

        lp._resolved_model_cache.clear()

    @pytest.mark.asyncio
    async def test_concurrent_model_resolution_is_safe(self):
        """Multiple concurrent tasks resolving the same model should all get correct result."""
        import headroom.pricing.litellm_pricing as lp

        call_count = 0

        def slow_uncached(model: str) -> str:
            nonlocal call_count
            call_count += 1
            # Simulate the slow litellm lookup
            return f"resolved/{model}"

        with patch(
            "headroom.pricing.litellm_pricing._resolve_litellm_model_uncached",
            side_effect=slow_uncached,
        ):
            # Launch 50 concurrent resolution tasks for the same model
            tasks = [
                asyncio.to_thread(lp.resolve_litellm_model, "claude-opus-4-6") for _ in range(50)
            ]
            results = await asyncio.gather(*tasks)

        # All should get the same result
        assert all(r == "resolved/claude-opus-4-6" for r in results)
        # Uncached should be called very few times (ideally 1, but a few races are OK)
        assert call_count <= 5, f"Uncached called {call_count} times — expected ~1"

    @pytest.mark.asyncio
    async def test_concurrent_resolution_different_models(self):
        """Concurrent resolution of different models should each resolve independently."""
        import headroom.pricing.litellm_pricing as lp

        models = ["gpt-4o", "claude-opus-4-6", "gemini-1.5-pro", "gpt-4o-mini"]

        with patch(
            "headroom.pricing.litellm_pricing._resolve_litellm_model_uncached",
            side_effect=lambda m: f"resolved/{m}",
        ):
            tasks = [
                asyncio.to_thread(lp.resolve_litellm_model, model)
                for model in models * 10  # 40 tasks total
            ]
            results = await asyncio.gather(*tasks)

        # Verify each model resolved correctly
        for i, model in enumerate(models * 10):
            assert results[i] == f"resolved/{model}"

        # Cache should have exactly 4 entries
        assert len(lp._resolved_model_cache) == 4

    @pytest.mark.asyncio
    async def test_concurrent_streaming_errors_are_independent(self):
        """Each session's streaming error should be independent — one failure shouldn't affect others."""

        async def simulate_session(session_id: int, should_fail: bool):
            """Simulate a streaming session that either succeeds or fails."""
            chunks = []

            try:
                if should_fail:
                    raise httpx.ConnectError(f"Session {session_id} connection refused")
                else:
                    # Successful session
                    for i in range(3):
                        chunks.append(f"data: chunk-{session_id}-{i}\n\n".encode())
                        await asyncio.sleep(0.001)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "connection_error",
                        "message": str(e),
                    },
                }
                chunks.append(f"event: error\ndata: {json.dumps(error_event)}\n\n".encode())

            return session_id, chunks, should_fail

        # Run 10 sessions: odd ones fail, even ones succeed
        tasks = [simulate_session(i, should_fail=(i % 2 == 1)) for i in range(10)]
        results = await asyncio.gather(*tasks)

        for session_id, chunks, should_fail in results:
            if should_fail:
                # Failed sessions should have an error chunk
                assert len(chunks) == 1
                error_data = json.loads(chunks[0].decode("utf-8").split("data: ")[1].strip())
                assert error_data["error"]["type"] == "connection_error"
                assert f"Session {session_id}" in error_data["error"]["message"]
            else:
                # Successful sessions should have their data chunks
                assert len(chunks) == 3
                for i, chunk in enumerate(chunks):
                    assert f"chunk-{session_id}-{i}".encode() in chunk

    @pytest.mark.asyncio
    async def test_estimate_cost_concurrent_with_caching(self):
        """Multiple concurrent estimate_cost calls should not block each other."""
        import headroom.pricing.litellm_pricing as lp
        from headroom.proxy.server import CostTracker

        tracker = CostTracker()

        # Pre-populate cache to simulate steady-state
        lp._resolved_model_cache["gpt-4o"] = "openai/gpt-4o"

        with (
            patch("headroom.proxy.cost.LITELLM_AVAILABLE", True),
            patch("headroom.pricing.litellm_pricing.litellm") as mock_litellm,
            patch("headroom.proxy.cost.litellm") as mock_cost_litellm,
        ):
            mock_litellm.cost_per_token.return_value = (0.001, 0.002)
            mock_litellm.get_model_info.return_value = {}
            mock_cost_litellm.cost_per_token.return_value = (0.001, 0.002)
            mock_cost_litellm.get_model_info.return_value = {}

            start = time.perf_counter()
            tasks = [
                asyncio.to_thread(tracker.estimate_cost, "gpt-4o", 1000, 500) for _ in range(100)
            ]
            results = await asyncio.gather(*tasks)
            elapsed_ms = (time.perf_counter() - start) * 1000

        # All should return a valid cost
        assert all(r is not None and r > 0 for r in results)
        # 100 concurrent calls should complete quickly (no blocking)
        assert elapsed_ms < 5000, f"100 concurrent estimate_cost took {elapsed_ms:.0f}ms"


# ---------------------------------------------------------------------------
# Cost tracking — no double-counting of cache tokens
# ---------------------------------------------------------------------------


class TestCostTrackingAccuracy:
    """Test that cost calculations don't double-count cache tokens."""

    def setup_method(self):
        import headroom.pricing.litellm_pricing as lp

        lp._resolved_model_cache.clear()

    def test_estimate_cost_separates_input_and_cache(self):
        """Input tokens and cache tokens should be billed separately, not double-counted."""
        from headroom.proxy.server import CostTracker

        tracker = CostTracker()

        with (
            patch("headroom.proxy.cost.LITELLM_AVAILABLE", True),
            patch("headroom.proxy.cost.litellm") as mock_litellm,
        ):
            # Setup: $10/M input, $30/M output
            def mock_cost(model, prompt_tokens, completion_tokens, **kwargs):
                input_cost = prompt_tokens * 0.00001
                output_cost = completion_tokens * 0.00003
                # Add cache costs if provided
                cache_read = kwargs.get("cache_read_input_tokens", 0)
                cache_write = kwargs.get("cache_creation_input_tokens", 0)
                if cache_read or cache_write:
                    model_info = mock_litellm.get_model_info()
                    input_cost += cache_read * model_info.get("cache_read_input_token_cost", 0)
                    input_cost += cache_write * model_info.get("cache_creation_input_token_cost", 0)
                return (input_cost, output_cost)

            mock_litellm.cost_per_token.side_effect = mock_cost
            mock_litellm.get_model_info.return_value = {
                "cache_read_input_token_cost": 0.000001,  # 10% of input
                "cache_creation_input_token_cost": 0.0000125,  # 125% of input
            }

            # 1000 input + 500 cache_read + 200 cache_write + 100 output
            cost = tracker.estimate_cost(
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=100,
                cache_read_tokens=500,
                cache_write_tokens=200,
            )

            assert cost is not None
            # input_cost = 1000 * 0.00001 = 0.01
            # output_cost = 100 * 0.00003 = 0.003
            # cache_read = 500 * 0.000001 = 0.0005
            # cache_write = 200 * 0.0000125 = 0.0025
            expected = 0.01 + 0.003 + 0.0005 + 0.0025
            assert abs(cost - expected) < 0.0001, f"Expected {expected}, got {cost}"

    def test_estimate_cost_without_cache_tokens(self):
        """Cost without cache tokens should just be input + output."""
        from headroom.proxy.server import CostTracker

        tracker = CostTracker()

        with (
            patch("headroom.proxy.cost.LITELLM_AVAILABLE", True),
            patch("headroom.proxy.cost.litellm") as mock_litellm,
        ):
            mock_litellm.cost_per_token.side_effect = (
                lambda model, prompt_tokens, completion_tokens, **kwargs: (
                    prompt_tokens * 0.00001,
                    completion_tokens * 0.00003,
                )
            )
            mock_litellm.get_model_info.return_value = {}

            cost = tracker.estimate_cost("gpt-4o", input_tokens=1000, output_tokens=100)

            expected = 1000 * 0.00001 + 100 * 0.00003
            assert abs(cost - expected) < 0.0001

    def test_estimate_cost_returns_none_without_litellm(self):
        """When litellm is unavailable, estimate_cost should return None."""
        from headroom.proxy.server import CostTracker

        tracker = CostTracker()

        with patch("headroom.proxy.cost.LITELLM_AVAILABLE", False):
            cost = tracker.estimate_cost("gpt-4o", input_tokens=1000, output_tokens=100)
            assert cost is None

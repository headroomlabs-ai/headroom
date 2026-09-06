"""Tests for the /v1/compress endpoint in the proxy server.

These tests verify that the compression-only endpoint works correctly
for the TypeScript SDK and other HTTP clients.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
    # /v1/compress is loopback-gated (#1227).
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as c:
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
    # /v1/compress is loopback-gated (#1227).
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as c:
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

    @pytest.mark.parametrize("biases", [None, [], "0:2.0"])
    def test_biases_must_be_an_object(self, client, biases):
        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "biases": biases,
            },
        )

        assert response.status_code == 400
        assert response.json()["error"] == {
            "type": "invalid_request",
            "message": "biases must be an object mapping message indices to positive numbers",
        }

    @pytest.mark.parametrize("index", ["-1", "01", "message"])
    def test_bias_keys_must_be_canonical_non_negative_indices(self, client, index):
        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "biases": {index: 1.5},
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request"
        assert "canonical non-negative integer" in response.json()["error"]["message"]

    def test_bias_index_must_reference_an_existing_message(self, client):
        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "biases": {"1": 1.5},
            },
        )

        assert response.status_code == 400
        assert response.json()["error"] == {
            "type": "invalid_request",
            "message": "bias index 1 is out of range for 1 messages",
        }

    @pytest.mark.parametrize("bias", [None, True, "high", 0, -0.5])
    def test_bias_values_must_be_finite_positive_numbers(self, client, bias):
        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "biases": {"0": bias},
            },
        )

        assert response.status_code == 400
        assert response.json()["error"] == {
            "type": "invalid_request",
            "message": "bias for message index 0 must be a finite positive number",
        }


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

    def test_response_ccr_hashes_extracts_only_retrievable_hashes(self):
        """Embedded CCR markers are reported without unrelated transform metadata."""
        from headroom.proxy.handlers.openai import _response_ccr_hashes

        messages = [
            {
                "role": "tool",
                "content": ("[100 rows compressed. Retrieve more: hash=abc123def4567890abc123de]"),
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<<ccr:feedface00112233 10_rows_offloaded>>",
                    },
                    {
                        "type": "text",
                        "text": "Retrieve original: hash=ABC123DEF4567890ABC123DE",
                    },
                ],
            },
        ]

        hashes = _response_ccr_hashes(
            messages,
            [
                "deadbeef0000000000000000",
                "<headroom:tool_digest sha256=1234567890abcdef>",
                "stable_prefix_hash:feedface00112233",
            ],
        )

        assert hashes == [
            "deadbeef0000000000000000",
            "abc123def4567890abc123de",
            "feedface00112233",
        ]

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

    def test_biases_are_normalized_and_forwarded_to_pipeline(self, client, monkeypatch):
        proxy = client.app.state.proxy
        result = SimpleNamespace(
            messages=[{"role": "user", "content": "compressed"}],
            tokens_before=12,
            tokens_after=7,
            transforms_applied=["test_transform"],
            transforms_summary={"test_transform": 1},
            markers_inserted=[],
        )
        apply_pipeline = Mock(return_value=result)
        monkeypatch.setattr(
            proxy,
            "_no_ccr_pipeline",
            Mock(return_value=SimpleNamespace(apply=apply_pipeline)),
        )

        response = client.post(
            "/v1/compress",
            json={
                "messages": [
                    {"role": "system", "content": "preserve this"},
                    {"role": "user", "content": "compress this"},
                ],
                "model": "gpt-4",
                "biases": {"0": 2.0, "1": 0.5},
            },
        )

        assert response.status_code == 200
        assert apply_pipeline.call_args.kwargs["biases"] == {0: 2.0, 1: 0.5}

    def test_success_records_request_outcome(self, client, monkeypatch):
        """A completed compression should update request metrics."""
        proxy = client.app.state.proxy
        result = SimpleNamespace(
            messages=[{"role": "user", "content": "compressed"}],
            tokens_before=12,
            tokens_after=7,
            transforms_applied=["test_transform"],
            transforms_summary={"test_transform": 1},
            markers_inserted=[],
        )
        # The executor callable returns the 5-tuple contract of
        # _run_stateless/_run_session_turn:
        # (result, final_messages, tokens_before, tokens_after, session_info).
        run_compression = AsyncMock(
            return_value=(result, result.messages, result.tokens_before, result.tokens_after, None)
        )
        record_outcome = AsyncMock()
        monkeypatch.setattr(proxy, "_run_compression_in_executor", run_compression)
        monkeypatch.setattr(proxy, "_record_request_outcome", record_outcome)

        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "compress me"}],
                "model": "gpt-4",
            },
        )

        assert response.status_code == 200
        record_outcome.assert_awaited_once()
        outcome = record_outcome.await_args.args[0]
        assert outcome.provider == "compress"
        assert outcome.model == "gpt-4"
        assert outcome.original_tokens == 12
        assert outcome.optimized_tokens == 7
        assert outcome.tokens_saved == 5
        assert outcome.attempted_input_tokens == 12
        assert outcome.num_messages == 1
        assert outcome.transforms_applied == ("test_transform",)
        assert outcome.total_latency_ms >= 0

    def test_response_reports_embedded_ccr_hashes(self, client, monkeypatch):
        """The endpoint reports a CCR marker even when the transform omitted its registry."""
        proxy = client.app.state.proxy
        ccr_hash = "abc123def4567890abc123de"
        result = SimpleNamespace(
            messages=[
                {
                    "role": "tool",
                    "content": f"<<ccr:{ccr_hash} 10_rows_offloaded>>",
                }
            ],
            tokens_before=12,
            tokens_after=7,
            transforms_applied=["test_transform"],
            transforms_summary={"test_transform": 1},
            markers_inserted=["<headroom:tool_digest sha256=1234567890abcdef>"],
        )
        monkeypatch.setattr(
            proxy,
            "_run_compression_in_executor",
            # Same 5-tuple contract as _run_stateless (see above).
            AsyncMock(
                return_value=(
                    result,
                    result.messages,
                    result.tokens_before,
                    result.tokens_after,
                    None,
                )
            ),
        )
        monkeypatch.setattr(proxy, "_record_request_outcome", AsyncMock())

        response = client.post(
            "/v1/compress",
            json={"messages": [{"role": "user", "content": "compress me"}], "model": "gpt-4"},
        )

        assert response.status_code == 200
        assert response.json()["ccr_hashes"] == [ccr_hash]

    def test_compression_error_records_failed_request(self, client, monkeypatch):
        """A hard compression failure should increment failed metrics."""
        proxy = client.app.state.proxy
        run_compression = AsyncMock(side_effect=RuntimeError("compression broke"))
        record_failed = AsyncMock()
        monkeypatch.setattr(proxy, "_run_compression_in_executor", run_compression)
        monkeypatch.setattr(proxy.metrics, "record_failed", record_failed)

        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "compress me"}],
                "model": "gpt-4",
            },
        )

        assert response.status_code == 503
        assert response.json() == {
            "error": {
                "type": "compression_error",
                "message": "compression broke",
            }
        }
        record_failed.assert_awaited_once_with(provider="compress")

    def test_compression_timeout_records_fail_open_outcome(self, client, monkeypatch):
        """A timeout should count as a zero-savings completed request."""
        proxy = client.app.state.proxy
        run_compression = AsyncMock(side_effect=TimeoutError)
        record_outcome = AsyncMock()
        record_compression_failed = Mock()
        monkeypatch.setattr(proxy, "_run_compression_in_executor", run_compression)
        monkeypatch.setattr(proxy, "_record_request_outcome", record_outcome)
        monkeypatch.setattr(
            proxy.metrics,
            "record_compression_failed",
            record_compression_failed,
        )
        messages = [{"role": "user", "content": "compress me"}]

        response = client.post(
            "/v1/compress",
            json={"messages": messages, "model": "gpt-4"},
        )

        assert response.status_code == 200
        assert response.json()["messages"] == messages
        assert response.json()["skip_reason"] == "compression_timeout"
        record_compression_failed.assert_called_once_with("timeout")
        record_outcome.assert_awaited_once()
        outcome = record_outcome.await_args.args[0]
        assert outcome.provider == "compress"
        assert outcome.model == "gpt-4"
        assert outcome.original_tokens == 0
        assert outcome.optimized_tokens == 0
        assert outcome.tokens_saved == 0


class TestCompressEndpointLossyInlineMode:
    """config.mode="lossy_inline" must compress losslessly-then-lossily but emit
    NO CCR marker / retrieval round-trip, so the output is safe to forward
    straight to a provider (Kong-sidecar use case)."""

    def _big_tool_message(self):
        large_data = json.dumps(
            [
                {
                    "id": i,
                    "name": f"Item {i}",
                    "description": f"Detailed description for item {i}. "
                    f"Status active, created 2024-01-{(i % 28) + 1:02d}, "
                    f"category=electronics, price={i * 10.99:.2f}, stock={i * 5}.",
                    "tags": ["electronics", "sale", "featured", "new-arrival"],
                }
                for i in range(200)
            ]
        )
        return [
            {"role": "user", "content": "What items are available?"},
            {"role": "tool", "tool_call_id": "call_1", "content": large_data},
            {"role": "user", "content": "Summarize them."},
        ]

    @pytest.fixture
    def client(self):
        # disable_kompress keeps the real ONNX model out of the test: marker
        # suppression is exercised by SmartCrusher (pure-Python) regardless, and
        # the mode inherits enable_kompress from config so this stays fast.
        config = ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            disable_kompress=True,
        )
        app = create_app(config)
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as c:
            yield c

    def test_lossy_inline_emits_no_ccr_markers(self, client):
        messages = self._big_tool_message()
        response = client.post(
            "/v1/compress",
            json={"messages": messages, "model": "gpt-4", "config": {"mode": "lossy_inline"}},
        )
        assert response.status_code == 200
        data = response.json()

        # Core guarantee: no CCR markers anywhere — no store/retrieval needed.
        assert data["ccr_hashes"] == []
        blob = json.dumps(data["messages"])
        assert "<<ccr:" not in blob
        assert "Retrieve more: hash=" not in blob
        assert "Retrieve original: hash=" not in blob

        # Real compression happened. These guard against a fail-open-to-zero
        # (e.g. a content-detector hang tripping the executor timeout) passing
        # vacuously as 0 <= 0.
        assert data["tokens_before"] > 0
        assert data["tokens_saved"] > 0
        assert data["tokens_after"] < data["tokens_before"]
        assert data["tokens_saved"] == data["tokens_before"] - data["tokens_after"]

    def test_lossless_then_lossy_alias(self, client):
        """The spelled-out alias selects the same mode."""
        messages = self._big_tool_message()
        response = client.post(
            "/v1/compress",
            json={
                "messages": messages,
                "model": "gpt-4",
                "config": {"mode": "lossless_then_lossy"},
            },
        )
        assert response.status_code == 200
        assert response.json()["ccr_hashes"] == []


class TestCompressEndpointModeValidation:
    """``config.mode`` must be validated, not silently ignored.

    Before this, ``mode: "lossless"`` (a mode that does not exist) or any typo
    fell through to the default pipeline and the caller got a 200 describing a
    compression posture it never asked for.
    """

    @pytest.mark.parametrize(
        "bad_mode",
        ["lossless", "CCR", "ccr ", "no_ccr", "", 7, ["ccr"]],
    )
    def test_unknown_mode_returns_400(self, client, bad_mode):
        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "config": {"mode": bad_mode},
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["type"] == "invalid_request"
        message = data["error"]["message"]
        # The message must tell the caller what IS valid.
        for valid in ("ccr", "lossy_inline", "lossless_then_lossy"):
            assert valid in message

    @pytest.mark.parametrize(
        "config",
        [
            {},  # mode unset -> default marker-free pipeline
            {"mode": None},  # explicit null is the same as unset
            {"mode": "ccr"},
            {"mode": "lossy_inline"},
            {"mode": "lossless_then_lossy"},
        ],
        ids=["unset", "null", "ccr", "lossy_inline", "lossless_then_lossy"],
    )
    def test_valid_modes_return_200(self, client, config):
        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "config": config,
            },
        )
        assert response.status_code == 200
        assert isinstance(response.json()["messages"], list)


class TestCompressDefaultPipelineBuiltAtStartup:
    """The default (marker-free) /v1/compress pipeline must exist before the
    first request.

    It used to be built lazily, so a cold pod paid ContentRouter construction —
    and in-process ML model load — inside the bounded compression executor on
    its first real gateway request.
    """

    def test_default_pipeline_exists_before_any_request(self):
        config = ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
        # No TestClient / no request: only create_app().
        proxy = create_app(config).state.proxy

        cache = getattr(proxy, "_compress_pipeline_cache", None)
        assert cache, "default /v1/compress pipeline was not built at startup"
        assert "no_ccr" in cache
        # It must be a DERIVED pipeline, not the shared request pipeline.
        assert cache["no_ccr"] is not proxy.openai_pipeline

    def test_startup_warmup_covers_the_derived_router(self):
        """The eager compressor preload must walk the derived pipeline too."""
        config = ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
        proxy = create_app(config).state.proxy

        derived = proxy._compress_pipeline_cache["no_ccr"]
        derived_ids = {id(t) for t in derived.transforms}

        seen: list[int] = []
        for pipeline in (proxy.anthropic_pipeline, proxy.openai_pipeline):
            seen.extend(id(t) for t in pipeline.transforms)
        # Precondition: the derived router is NOT reachable via the request
        # pipelines, so dedup-by-id() cannot have covered it implicitly.
        assert derived_ids - set(seen)

        _status, transform_statuses = proxy._eager_preload_transforms()
        # Base router + derived router both report a status dict.
        assert len(transform_statuses) >= 2


class TestCompressContextLimitByModelFamily:
    """LiteLLM's guardrail forwards Anthropic model names through this
    OpenAI-shaped route; the context limit must come from the Anthropic
    provider for those, not the OpenAI provider's 128K default."""

    @staticmethod
    def _spy_providers(proxy, monkeypatch):
        anthropic_calls: list[str] = []
        openai_calls: list[str] = []

        def anthropic_limit(model):
            anthropic_calls.append(model)
            return 987_654

        def openai_limit(model):
            openai_calls.append(model)
            return 123_456

        monkeypatch.setattr(proxy.anthropic_provider, "get_context_limit", anthropic_limit)
        monkeypatch.setattr(proxy.openai_provider, "get_context_limit", openai_limit)
        return anthropic_calls, openai_calls

    @staticmethod
    def _spy_pipeline(proxy, monkeypatch):
        """Capture the kwargs handed to the default (marker-free) pipeline."""
        seen: dict = {}

        def fake_apply(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                messages=kwargs["messages"],
                tokens_before=10,
                tokens_after=10,
                transforms_applied=[],
                transforms_summary={},
                markers_inserted=[],
            )

        monkeypatch.setattr(proxy._compress_pipeline_cache["no_ccr"], "apply", fake_apply)
        return seen

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-5-20250929",
            "bedrock/anthropic.claude-3-5-sonnet",
            "anthropic/claude-opus-4",
            "CLAUDE-Sonnet-4-5",  # case-insensitive
        ],
    )
    def test_claude_models_use_anthropic_context_limit(self, client, monkeypatch, model):
        proxy = client.app.state.proxy
        anthropic_calls, openai_calls = self._spy_providers(proxy, monkeypatch)
        seen = self._spy_pipeline(proxy, monkeypatch)

        response = client.post(
            "/v1/compress",
            json={"messages": [{"role": "user", "content": "hello"}], "model": model},
        )

        assert response.status_code == 200
        assert anthropic_calls == [model]
        assert openai_calls == []
        assert seen["model_limit"] == 987_654

    def test_openai_models_still_use_openai_context_limit(self, client, monkeypatch):
        proxy = client.app.state.proxy
        anthropic_calls, openai_calls = self._spy_providers(proxy, monkeypatch)
        seen = self._spy_pipeline(proxy, monkeypatch)

        response = client.post(
            "/v1/compress",
            json={"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4o"},
        )

        assert response.status_code == 200
        assert openai_calls == ["gpt-4o"]
        assert anthropic_calls == []
        assert seen["model_limit"] == 123_456

    def test_token_budget_still_overrides_for_claude_models(self, client, monkeypatch):
        """token_budget precedence must survive the provider routing."""
        proxy = client.app.state.proxy
        anthropic_calls, openai_calls = self._spy_providers(proxy, monkeypatch)
        seen = self._spy_pipeline(proxy, monkeypatch)

        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "claude-sonnet-4-5-20250929",
                "token_budget": 4096,
            },
        )

        assert response.status_code == 200
        assert seen["model_limit"] == 4096
        # Neither provider is consulted when the caller pins a budget.
        assert anthropic_calls == []
        assert openai_calls == []


class TestCompressEndpointDoesNotBlockLoop:
    """/v1/compress must offload to the compression executor so a slow/large
    payload cannot freeze the single event loop (#718)."""

    async def test_compress_does_not_block_liveness(self, monkeypatch):
        import asyncio
        import threading
        from types import SimpleNamespace

        import httpx

        config = ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
        app = create_app(config)
        proxy = app.state.proxy

        entered = threading.Event()
        release = threading.Event()

        def blocking_apply(**kwargs):
            # Stand in for a large CPU-bound compression: blocks its worker until
            # released. If this ran inline on the loop (the bug), the loop would
            # be frozen and /livez below could not be served.
            entered.set()
            release.wait(timeout=10)
            return SimpleNamespace(
                messages=kwargs["messages"],
                tokens_before=10,
                tokens_after=5,
                transforms_applied=[],
                transforms_summary={},
                markers_inserted=[],
            )

        monkeypatch.setattr(proxy._ccr_pipeline(), "apply", blocking_apply)

        # /v1/compress is loopback-gated (#1227) — present as 127.0.0.1.
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            compress = asyncio.create_task(
                client.post(
                    "/v1/compress",
                    json={
                        "messages": [{"role": "user", "content": "hello world"}],
                        "model": "gpt-4",
                        # Every mode now runs a DERIVED pipeline so the tokenizer
                        # comes from the per-model registry; mode="ccr" is the
                        # marker-on one, patched above.
                        "config": {"mode": "ccr"},
                    },
                )
            )
            # Wait until the compression is actually in flight (running in the
            # executor thread), then prove the loop is still responsive.
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set(), "compression never started"

            livez = await asyncio.wait_for(client.get("/livez"), timeout=5)
            assert livez.status_code == 200
            assert livez.json()["alive"] is True
            # The compression is still blocked — /livez was served concurrently.
            assert not compress.done()

            release.set()
            resp = await asyncio.wait_for(compress, timeout=5)
            assert resp.status_code == 200
            assert resp.json()["tokens_saved"] == 5


class TestCompressEndpointFrozenMessageCount:
    """``config.frozen_message_count`` pins a prefix the provider has already cached.

    Callers that resend a growing conversation every turn (agent loops, the Strands
    plugin) need the leading messages to come back byte-for-byte identical. Without
    this the router compresses old messages harder as the conversation grows, their
    bytes change, and the provider's prompt cache misses from that point on — turning
    compression into a net cost. ``protect_recent`` guards the other end of the list
    and cannot express it.
    """

    @staticmethod
    def _conversation(turns: int) -> list[dict]:
        log = "\n".join(
            f"2026-07-31 12:00:{n:02d} INFO worker={n} req=r{n} took {n}ms" for n in range(60)
        )
        messages: list[dict] = []
        for i in range(turns):
            messages += [
                {"role": "user", "content": f"step {i}"},
                {"role": "assistant", "content": f"reading log {i}\n{log}"},
            ]
        return messages

    def test_pinned_prefix_is_returned_byte_for_byte(self, client):
        messages = self._conversation(12)
        response = client.post(
            "/v1/compress",
            json={
                "messages": messages,
                "model": "gpt-4",
                "config": {"compress_user_messages": True, "frozen_message_count": 8},
            },
        )
        assert response.status_code == 200
        assert response.json()["messages"][:8] == messages[:8]

    def test_the_unpinned_tail_is_still_compressed(self, client):
        messages = self._conversation(12)
        body = {"messages": messages, "model": "gpt-4", "config": {"compress_user_messages": True}}
        full = client.post("/v1/compress", json=body).json()
        pinned = client.post(
            "/v1/compress",
            json={**body, "config": {**body["config"], "frozen_message_count": 8}},
        ).json()

        # Pinning must not disable compression outright — only exempt the prefix.
        assert pinned["messages"][8:] != messages[8:], "tail was left uncompressed"
        assert pinned["tokens_after"] >= full["tokens_after"], "pinning should compress no harder"

    def test_a_pinned_prefix_does_not_drift_as_the_conversation_grows(self, client):
        """The regression this field exists to prevent."""
        short, long = self._conversation(6), self._conversation(24)
        config = {"compress_user_messages": True, "frozen_message_count": 12}

        a = client.post(
            "/v1/compress", json={"messages": short, "model": "gpt-4", "config": config}
        ).json()
        b = client.post(
            "/v1/compress", json={"messages": long, "model": "gpt-4", "config": config}
        ).json()

        assert a["messages"][:12] == b["messages"][:12], (
            "prefix was re-rendered as the conversation grew"
        )

    @pytest.mark.parametrize("value", ["8", -1, 3.5, True, [8], {"n": 8}])
    def test_invalid_values_return_400(self, client, value):
        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "config": {"frozen_message_count": value},
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["type"] == "invalid_request"
        assert "frozen_message_count" in data["error"]["message"]

    @pytest.mark.parametrize("value", [0, 1, 10_000], ids=["zero", "one", "beyond-the-list"])
    def test_valid_values_are_accepted(self, client, value):
        """0 means "pin nothing"; a count past the end simply pins everything."""
        response = client.post(
            "/v1/compress",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "config": {"frozen_message_count": value},
            },
        )
        assert response.status_code == 200

    def test_unset_is_unchanged_behaviour(self, client):
        messages = self._conversation(4)
        body = {"messages": messages, "model": "gpt-4", "config": {"compress_user_messages": True}}
        without = client.post("/v1/compress", json=body).json()
        explicit_zero = client.post(
            "/v1/compress", json={**body, "config": {**body["config"], "frozen_message_count": 0}}
        ).json()
        assert without["messages"] == explicit_zero["messages"]

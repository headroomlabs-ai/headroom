"""Regression tests for proxy hook integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.hooks import CompressionHooks
from headroom.proxy.savings_tracker import HEADROOM_SAVINGS_PATH_ENV_VAR
from headroom.proxy.server import ProxyConfig, create_app


def test_anthropic_hooks_do_not_break_extract_user_query_lookup(tmp_path, monkeypatch):
    """Hooks-enabled Anthropic requests should still reach the compression pipeline."""
    monkeypatch.setenv(
        HEADROOM_SAVINGS_PATH_ENV_VAR,
        str(tmp_path / "proxy_savings.json"),
    )

    config = ProxyConfig(
        cache_enabled=False,
        rate_limit_enabled=False,
        log_requests=False,
        hooks=CompressionHooks(),
    )

    app = create_app(config)

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        large_user_content = "hello " * 200

        # Mocked pipeline output. The proxy now recounts `optimized_tokens`
        # using its own tokenizer instead of trusting `result.tokens_after`
        # (issue #327 / Bug 3: cross-tokenizer comparison broke the
        # inflation guard and zeroed out compression on Anthropic).
        # The mocked content below tokenizes to a known value with the
        # proxy's EstimatingTokenCounter; the assertion below is computed
        # from that same tokenizer so the test stays robust to any future
        # tokenizer recalibration.
        compressed_messages = [{"role": "user", "content": "compressed"}]
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=SimpleNamespace(
                messages=compressed_messages,
                transforms_applied=["test_transform"],
                timing={},
                tokens_before=100,
                tokens_after=40,
                waste_signals=None,
            )
        )
        proxy._retry_request = AsyncMock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 40,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )
        )

        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": large_user_content}],
            },
        )

        assert response.status_code == 200
        assert proxy.anthropic_pipeline.apply.called

        # `x-headroom-tokens-after` reflects the proxy-side recount of the
        # pipeline's returned messages (NOT the mock's `tokens_after=40`).
        # See issue #327 Bug 3: prior behavior trusted pipeline tokens_after
        # which used a different tokenizer than `original_tokens`, breaking
        # the inflation guard.
        from headroom.tokenizers.registry import get_tokenizer

        expected_after = get_tokenizer("claude-sonnet-4-6").count_messages(compressed_messages)
        assert response.headers["x-headroom-tokens-after"] == str(expected_after)
        tokens_before = int(response.headers["x-headroom-tokens-before"])
        tokens_after = int(response.headers["x-headroom-tokens-after"])
        assert tokens_before > tokens_after, (
            f"compression should reduce tokens: before={tokens_before} after={tokens_after}"
        )
        assert int(response.headers["x-headroom-tokens-saved"]) > 0

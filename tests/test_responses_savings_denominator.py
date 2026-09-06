"""Recorded /v1/responses savings must form a coherent triple.

``tokens_saved`` on the Responses HTTP path includes tool schema/desc
compaction, which is measured against the ``tools`` array — content the
messages-only ``original_tokens`` count never included. Deriving
``optimized_tokens = max(0, original - saved)`` from that mismatched pair
clamped to 0 on schema-heavy turns and recorded savings rates above 100%
(surfaced in the desktop feed as e.g. "4,436 → 0 (208.4%)"). The fix folds
the tools schema into the pair, so ``original - optimized == saved`` and
the rate stays <= 100%.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx
from fastapi.testclient import TestClient

from headroom.proxy.loopback_guard import require_loopback
from headroom.proxy.server import ProxyConfig, create_app

_TOKENS_SAVED = 5_000


def _make_app():
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    app.dependency_overrides[require_loopback] = lambda: None
    return app


def _fake_upstream_response(url: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_test",
            "object": "response",
            "output": [],
            "usage": {"input_tokens": 12, "output_tokens": 3},
        },
        request=httpx.Request("POST", url),
    )


def test_schema_heavy_compression_keeps_savings_triple_coherent():
    app = _make_app()
    server = app.state.proxy

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):
        return _fake_upstream_response(url)

    server._retry_request = _fake_retry

    async def _fake_compress(payload, **kwargs):
        # Simulate schema compaction: savings far larger than the message
        # tokens (the "4,436 -> 0" shape), sourced mostly from the tools
        # array.
        compressed = dict(payload)
        compressed["tools"] = [{"type": "function", "name": "t", "parameters": {}}]
        return (
            compressed,
            True,
            _TOKENS_SAVED,
            ["openai:responses:tool_schema_compaction"],
            None,
            100_000,
            10_000,
            _TOKENS_SAVED,
            {},
        )

    server._compress_openai_responses_payload_in_executor = _fake_compress

    outcomes = []

    async def _capture_outcome(outcome):
        outcomes.append(outcome)

    server._record_request_outcome = _capture_outcome

    payload = {
        "model": "gpt-5-codex",
        "input": "list files",
        "tools": [
            {
                "type": "function",
                "name": "big_tool",
                # ~9k tokens of schema so original (messages + tools) can
                # absorb _TOKENS_SAVED without clamping.
                "description": "word " * 9_000,
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }

    with TestClient(app) as client:
        resp = client.post(
            "/v1/responses",
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    assert resp.status_code == 200

    assert outcomes, "no RequestOutcome recorded"
    o = outcomes[0]
    assert o.tokens_saved == _TOKENS_SAVED
    # Coherent triple: in - out == saved, out not clamped to zero, and the
    # savings rate (saved / original) never above 100%.
    assert o.original_tokens - o.optimized_tokens == o.tokens_saved
    assert o.optimized_tokens > 0
    assert o.tokens_saved <= o.original_tokens


def test_no_tools_payload_keeps_messages_only_pair():
    app = _make_app()
    server = app.state.proxy

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):
        return _fake_upstream_response(url)

    server._retry_request = _fake_retry

    saved = 5

    async def _fake_compress(payload, **kwargs):
        return (dict(payload), True, saved, ["openai:responses:trim"], None, 1_000, 900, saved, {})

    server._compress_openai_responses_payload_in_executor = _fake_compress

    outcomes = []

    async def _capture_outcome(outcome):
        outcomes.append(outcome)

    server._record_request_outcome = _capture_outcome

    payload = {"model": "gpt-5-codex", "input": "word " * 100}

    with TestClient(app) as client:
        resp = client.post(
            "/v1/responses",
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    assert resp.status_code == 200

    assert outcomes, "no RequestOutcome recorded"
    o = outcomes[0]
    assert o.tokens_saved == saved
    # No tools array: original_tokens must stay the messages-only count
    # (not widened), and the triple still holds.
    assert o.original_tokens - o.optimized_tokens == o.tokens_saved
    assert o.optimized_tokens > 0
    assert o.tokens_saved <= o.original_tokens

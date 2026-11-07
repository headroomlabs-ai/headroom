"""Handler-level wiring for ``--ccr-inline-resolve`` (issue #2509).

The pure-module tests in ``test_ccr_marker_resolution.py`` prove the
substitution logic. These tests prove the thing that actually broke: the
resolve call has to run on the response path *when the model never emitted a
``headroom_retrieve`` tool call at all*. That is the whole #2509 shape —
Headroom behind a LiteLLM guardrail hop with no tool-call turn — so any wiring
that sits behind a ``has_ccr_tool_calls`` gate is a no-op for its own use case.

Every response fixture below therefore has zero tool calls.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.cache.compression_store import (  # noqa: E402
    get_compression_store,
    reset_compression_store,
)
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

ORIGINAL = "the original uncompressed content"


@pytest.fixture(autouse=True)
def reset_store():
    reset_compression_store()
    yield
    reset_compression_store()


def _marker() -> str:
    hash_key = get_compression_store().store(
        original=ORIGINAL,
        compressed="[]",
        original_item_count=1,
        compressed_item_count=0,
    )
    return f"<<ccr:{hash_key},string,23.6KB>>"


def _config(*, inline_resolve: bool) -> ProxyConfig:
    # No backend -> the "Direct OpenAI API (no backend configured)" path.
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        ccr_resolve_markers_inline=inline_resolve,
    )


def _chat_response(marker: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"here it is: {marker}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _responses_response(marker: str) -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"here it is: {marker}"}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def _anthropic_response(marker: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": f"here it is: {marker}"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _run(config: ProxyConfig, path: str, body: dict, upstream: dict) -> httpx.Response:
    async def fake_retry(method, url, headers, req_body, *args, **kwargs):
        return httpx.Response(200, json=upstream, headers={"content-type": "application/json"})

    app = create_app(config)
    with TestClient(app) as client:
        client.app.state.proxy._retry_request = fake_retry
        return client.post(
            path,
            json=body,
            headers={"Authorization": "Bearer test-key", "x-api-key": "test-key"},
        )


@pytest.mark.parametrize("inline_resolve", [True, False])
def test_openai_chat_direct_path(inline_resolve):
    marker = _marker()
    resp = _run(
        _config(inline_resolve=inline_resolve),
        "/v1/chat/completions",
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        _chat_response(marker),
    )

    assert resp.status_code == 200, resp.text
    text = resp.json()["choices"][0]["message"]["content"]
    if inline_resolve:
        assert text == f"here it is: {ORIGINAL}"
    else:
        assert text == f"here it is: {marker}"


@pytest.mark.parametrize("inline_resolve", [True, False])
def test_openai_responses_path(inline_resolve):
    marker = _marker()
    resp = _run(
        _config(inline_resolve=inline_resolve),
        "/v1/responses",
        {"model": "gpt-4o", "input": "hi", "stream": False},
        _responses_response(marker),
    )

    assert resp.status_code == 200, resp.text
    text = resp.json()["output"][0]["content"][0]["text"]
    # Explicit inline resolution returns the original. The unconditional
    # client-bound invariant uses a bounded prose placeholder instead.
    expected = ORIGINAL if inline_resolve else "[compressed content: string,23.6KB]"
    assert text == f"here it is: {expected}"


@pytest.mark.parametrize("inline_resolve", [True, False])
def test_anthropic_messages_path(inline_resolve):
    marker = _marker()
    resp = _run(
        _config(inline_resolve=inline_resolve),
        "/v1/messages",
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
        _anthropic_response(marker),
    )

    assert resp.status_code == 200, resp.text
    text = resp.json()["content"][0]["text"]
    expected = ORIGINAL if inline_resolve else "[compressed content: string,23.6KB]"
    assert text == f"here it is: {expected}"

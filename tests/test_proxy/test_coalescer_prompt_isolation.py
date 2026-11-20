"""Prompt isolation through the production forwarding path.

The unit tests pin the coalescer's grouping. This test pins the property that
actually matters at the edge: two callers sending different prompts through
``/v1/messages`` with coalescing enabled must each receive their own answer.

Checking only that a shared response *id* propagates misses the defect — the
merged batch also reported a single id, and both callers got the second
prompt's answer.
"""

from __future__ import annotations

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from headroom.proxy import coalescer as coalescer_mod  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )


def _decode_body(body) -> dict:  # noqa: ANN001
    if isinstance(body, (bytes, bytearray)):
        return json.loads(body)
    if isinstance(body, str):
        return json.loads(body)
    return dict(body)


def _message(text: str, model: str) -> dict:
    return {
        "id": f"msg_{text}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }


@pytest.fixture
def coalescer():
    coalescer_mod.reset_coalescer()
    instance = coalescer_mod.get_coalescer()
    instance.window_sec = 0.3
    instance.max_batch = 8
    yield instance
    coalescer_mod.reset_coalescer()


@pytest.mark.asyncio
async def test_different_prompts_get_their_own_answers_through_messages_route(coalescer):
    app = create_app(_config())
    proxy = app.state.proxy
    dispatched: list[dict] = []

    async def _fake_retry(method, url, headers=None, body=None, stream=False, **kwargs):  # noqa: ANN001
        payload = _decode_body(body)
        prompt = payload["messages"][-1]["content"]
        dispatched.append(payload)
        # Deterministic upstream: it answers the final prompt of whatever
        # conversation it was handed, exactly like a real model would.
        text = prompt.split()[-2] if "only" in prompt else prompt
        text = "ALPHA" if "ALPHA" in prompt else ("BETA" if "BETA" in prompt else text)
        return httpx.Response(200, json=_message(text, payload.get("model", "test")))

    proxy._retry_request = _fake_retry

    headers = {
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-headroom-coalesce": "1",
    }

    async with AsyncClient(
        transport=ASGITransport(app=app, client=("127.0.0.1", 12345)),
        base_url="http://127.0.0.1",
    ) as client:

        async def ask(prompt: str):
            return await client.post(
                "/v1/messages",
                headers=headers,
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

        alpha_resp, beta_resp = await asyncio.gather(
            ask("Return ALPHA only"),
            ask("Return BETA only"),
        )

    assert alpha_resp.status_code == 200, alpha_resp.text
    assert beta_resp.status_code == 200, beta_resp.text

    alpha_text = alpha_resp.json()["content"][0]["text"]
    beta_text = beta_resp.json()["content"][0]["text"]
    assert alpha_text == "ALPHA"
    assert beta_text == "BETA"

    # Two independent sessions: two conversations forwarded, never merged.
    assert len(dispatched) == 2
    forwarded = [call["messages"] for call in dispatched]
    assert {"role": "user", "content": "Return ALPHA only"} in forwarded[0]
    assert {"role": "user", "content": "Return BETA only"} in forwarded[1]
    for conversation in forwarded:
        assert len(conversation) == 1, "prompts must not be concatenated"


@pytest.mark.asyncio
async def test_identical_prompts_still_coalesce_through_messages_route(coalescer):
    app = create_app(_config())
    proxy = app.state.proxy
    call_count = 0

    async def _fake_retry(method, url, headers=None, body=None, stream=False, **kwargs):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        payload = _decode_body(body)
        return httpx.Response(200, json=_message("SAME", payload.get("model", "test")))

    proxy._retry_request = _fake_retry

    headers = {
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-headroom-coalesce": "1",
    }

    async with AsyncClient(
        transport=ASGITransport(app=app, client=("127.0.0.1", 12345)),
        base_url="http://127.0.0.1",
    ) as client:

        async def ask():
            return await client.post(
                "/v1/messages",
                headers=headers,
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "Same question"}],
                },
            )

        first, second = await asyncio.gather(ask(), ask())

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert call_count == 1, "identical prompts should share one upstream call"
    assert first.json()["content"][0]["text"] == "SAME"
    assert second.json()["content"][0]["text"] == "SAME"

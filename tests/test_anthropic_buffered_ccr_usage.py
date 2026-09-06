"""A buffered-CCR turn is a billed call and its usage must reach accounting.

When a ``stream:true`` request carries the ``headroom_retrieve`` tool, the
Anthropic handler rewrites it to ``stream:false`` upstream so it can resolve
retrievals server-side, then re-synthesizes SSE for the client. That buffered
response carries the same ``usage`` block any non-stream reply does, so the
provider's cache-read / cache-write / output counts must land on the outcome.

If they don't, every cached token on the dominant Claude Code path is invisible:
``metrics.cache_by_provider`` only records a provider row when cache read or
write is non-zero, so the whole prefix-cache card (hit rate, savings, TTL mix)
is computed from whatever small fraction of traffic took another path, while
compression savings on the buffered turns still count in full.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.loopback_guard import require_loopback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

# Provider-reported usage for one warm turn: most of the prompt served from
# cache, a slice newly written, a little uncached, and a real completion.
_USAGE = {
    "input_tokens": 300,
    "output_tokens": 120,
    "cache_read_input_tokens": 46893,
    "cache_creation_input_tokens": 8197,
}

_RESPONSE = {
    "id": "msg_buffered",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "done"}],
    "stop_reason": "end_turn",
    "usage": _USAGE,
}

_RETRIEVE_TOOL = {
    "name": "headroom_retrieve",
    "description": "Retrieve compressed content",
    "input_schema": {"type": "object", "properties": {"ref": {"type": "string"}}},
}


def _app_and_outcomes(monkeypatch, **overrides):
    """App with a spy on the outcome record — where billed counts land."""
    kwargs: dict[str, Any] = {
        "optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
    }
    kwargs.update(overrides)
    app = create_app(ProxyConfig(**kwargs))
    app.dependency_overrides[require_loopback] = lambda: None
    outcomes: list[Any] = []

    async def _spy(_self, outcome, *a, **kw):
        outcomes.append(outcome)

    monkeypatch.setattr(type(app.state.proxy), "_record_request_outcome", _spy, raising=True)
    return app, outcomes


def _post(app, *, stream: bool, tools: list[dict] | None):
    body: dict[str, Any] = {
        "model": "claude-opus-5",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if stream:
        body["stream"] = True
    if tools is not None:
        body["tools"] = tools
    with TestClient(app) as client:
        return client.post("/v1/messages", json=body, headers={"x-api-key": "sk-ant-test"})


@respx.mock
def test_non_stream_turn_records_provider_usage(monkeypatch) -> None:
    """Control: the plain non-stream path already books the usage block."""
    app, outcomes = _app_and_outcomes(monkeypatch)
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=_RESPONSE)
    )

    r = _post(app, stream=False, tools=None)

    assert r.status_code == 200
    assert outcomes, "an outcome must be recorded"
    o = outcomes[-1]
    assert o.cache_read_tokens == 46893
    assert o.cache_write_tokens == 8197
    assert o.output_tokens == 120


@respx.mock
def test_buffered_ccr_turn_records_provider_usage(monkeypatch) -> None:
    """A stream:true + headroom_retrieve turn must book the same usage block."""
    app, outcomes = _app_and_outcomes(monkeypatch)
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=_RESPONSE)
    )

    r = _post(app, stream=True, tools=[_RETRIEVE_TOOL])

    assert r.status_code == 200
    # The handler must have buffered it: upstream saw stream:false.
    sent = route.calls.last.request
    assert b'"stream": false' in sent.content or b'"stream":false' in sent.content

    assert outcomes, "an outcome must be recorded"
    o = outcomes[-1]
    assert o.cache_read_tokens == 46893, (
        f"buffered-CCR turn dropped the provider cache-read count: {o.cache_read_tokens}"
    )
    assert o.cache_write_tokens == 8197, (
        f"buffered-CCR turn dropped the provider cache-write count: {o.cache_write_tokens}"
    )
    assert o.output_tokens == 120, (
        f"buffered-CCR turn dropped the provider output count: {o.output_tokens}"
    )


@respx.mock
def test_buffered_ccr_records_usage_with_compression_on(monkeypatch) -> None:
    """Same turn with the live token-mode pipeline running, as in production."""
    app, outcomes = _app_and_outcomes(monkeypatch, optimize=True, mode="token")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=_RESPONSE)
    )

    r = _post(app, stream=True, tools=[_RETRIEVE_TOOL])

    assert r.status_code == 200
    assert outcomes, "an outcome must be recorded"
    o = outcomes[-1]
    assert (o.cache_read_tokens, o.cache_write_tokens, o.output_tokens) == (46893, 8197, 120), (
        f"buffered-CCR turn dropped usage under compression: "
        f"cr={o.cache_read_tokens} cw={o.cache_write_tokens} out={o.output_tokens}"
    )

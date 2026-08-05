"""Integration RBP for the SemanticCache key fix.

Drives the real ``/v1/messages`` handler path with the cache enabled and a mocked
upstream, proving end to end that a second request with the same messages but a
different ``system`` prompt is NOT served the first request's cached response
(no cross-request contamination), while a repeat of the first request IS served
from cache. This is the deterministic stand-in for a live real-upstream e2e
(no API credits, fully reproducible) and covers what the cache-key unit tests
cannot: that the handler actually threads the response-shaping fields into the
cache get/set calls.

Before the fix the cache key omitted ``system``, so request B collided with
request A: it returned A's response and the upstream was never called.
"""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


def _make_cached_proxy_client() -> TestClient:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=True,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    return TestClient(create_app(config))


def _body(system: str) -> dict:
    return {
        "model": "claude-haiku-4-5",
        "max_tokens": 64,
        "system": system,
        "messages": [{"role": "user", "content": "Say hi."}],
        "stream": False,
    }


def _text(response: httpx.Response) -> str:
    return response.json()["content"][0]["text"]


def test_different_system_not_served_from_cache() -> None:
    calls = {"n": 0}

    with _make_cached_proxy_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            calls["n"] += 1
            system = body.get("system")
            sys_text = system if isinstance(system, str) else json.dumps(system)
            text = "Bonjour" if "French" in sys_text else "Hello"
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry
        headers = {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}

        # A: French system -> upstream call 1, cached under the French key.
        a = client.post("/v1/messages", headers=headers, json=_body("Answer only in French."))
        assert a.status_code == 200
        assert _text(a) == "Bonjour"
        assert calls["n"] == 1

        # B: English system, SAME messages -> must reach the upstream again, not
        # be served A's cached French response. Before the fix this returned
        # "Bonjour" with calls["n"] still 1 (the bug).
        b = client.post("/v1/messages", headers=headers, json=_body("Answer only in English."))
        assert b.status_code == 200
        assert _text(b) == "Hello"
        assert calls["n"] == 2

        # A again: French system -> served from cache, upstream NOT called.
        a2 = client.post("/v1/messages", headers=headers, json=_body("Answer only in French."))
        assert a2.status_code == 200
        assert _text(a2) == "Bonjour"
        assert calls["n"] == 2


def _body_thinking(thinking: dict) -> dict:
    return {
        "model": "claude-haiku-4-5",
        "max_tokens": 64,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Say hi."}],
        "thinking": thinking,
        "stream": False,
    }


def test_different_thinking_not_served_from_cache() -> None:
    """Same system + messages, different ``thinking`` config -> B must reach the
    upstream, not be served A's cached response. ``thinking`` is the field the
    #1473 review called out as still missing from the Anthropic key."""
    calls = {"n": 0}

    with _make_cached_proxy_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            calls["n"] += 1
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"resp-{calls['n']}"}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry
        headers = {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}
        enabled = {"type": "enabled", "budget_tokens": 2048}
        disabled = {"type": "disabled"}

        # A: thinking enabled -> upstream call 1, cached under A's key.
        a = client.post("/v1/messages", headers=headers, json=_body_thinking(enabled))
        assert a.status_code == 200
        assert _text(a) == "resp-1"
        assert calls["n"] == 1

        # B: thinking disabled, SAME messages -> must reach the upstream again.
        b = client.post("/v1/messages", headers=headers, json=_body_thinking(disabled))
        assert b.status_code == 200
        assert _text(b) == "resp-2"
        assert calls["n"] == 2

        # A again -> served from cache, upstream NOT called.
        a2 = client.post("/v1/messages", headers=headers, json=_body_thinking(enabled))
        assert a2.status_code == 200
        assert _text(a2) == "resp-1"
        assert calls["n"] == 2

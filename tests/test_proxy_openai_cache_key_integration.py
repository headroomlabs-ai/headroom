"""Integration RBP for the OpenAI handler's SemanticCache key threading.

Companion to ``test_proxy_semantic_cache_key_integration.py`` (Anthropic). Drives
the real ``/v1/chat/completions`` handler with the cache enabled and a stubbed
upstream, proving the OpenAI handler actually threads each newly-added
response-shaping field into the cache get/set calls: two requests with identical
``messages`` but a different ``response_format`` / ``tool_choice`` / ``seed`` must
NOT collide, while a repeat of the first IS served from cache.

A cache-key unit test cannot catch this — it exercises ``_compute_key`` directly.
The failure mode this guards is the handler's ``cache_key_fields`` snapshot
omitting a ``body.get(...)`` for a field: ``_compute_key`` would distinguish the
field fine, but the handler never passes it. Before the OpenAI snapshot widening
a request differing only in ``response_format`` collided and was served the
first request's response.
"""

from __future__ import annotations

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


def _body(**extra: object) -> dict:
    body: dict = {
        "model": "gpt-4o-mini",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hi."}],
        "stream": False,
    }
    body.update(extra)
    return body


def _content(response: httpx.Response) -> str:
    return response.json()["choices"][0]["message"]["content"]


@pytest.mark.parametrize(
    "field,a,b",
    [
        ("response_format", {"type": "json_object"}, {"type": "text"}),
        ("tool_choice", "auto", "none"),
        ("seed", 1, 2),
        ("reasoning_effort", "low", "high"),
    ],
)
def test_openai_differing_field_not_served_from_cache(field, a, b) -> None:
    """A and B share messages and differ only in ``field``; B must not be served
    A's cached response, and a repeat of A must hit the cache."""
    calls = {"n": 0}

    with _make_cached_proxy_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            calls["n"] += 1
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"resp-{calls['n']}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                },
            )

        proxy._retry_request = _fake_retry
        headers = {"authorization": "Bearer test-key"}

        # A: field=a -> upstream call 1, cached under A's key.
        ra = client.post("/v1/chat/completions", headers=headers, json=_body(**{field: a}))
        assert ra.status_code == 200
        assert _content(ra) == "resp-1"
        assert calls["n"] == 1

        # B: field=b, SAME messages -> must reach the upstream again, not be
        # served A's cached response. With the field missing from the key, B
        # collided with A and calls stayed 1 (the bug this guards).
        rb = client.post("/v1/chat/completions", headers=headers, json=_body(**{field: b}))
        assert rb.status_code == 200
        assert _content(rb) == "resp-2"
        assert calls["n"] == 2

        # A again -> served from cache, upstream NOT called.
        ra2 = client.post("/v1/chat/completions", headers=headers, json=_body(**{field: a}))
        assert ra2.status_code == 200
        assert _content(ra2) == "resp-1"
        assert calls["n"] == 2

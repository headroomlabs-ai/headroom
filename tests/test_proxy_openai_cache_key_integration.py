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


_AUDIO_ALLOY_WAV = {"voice": "alloy", "format": "wav"}


@pytest.mark.parametrize(
    "audio_b",
    [
        pytest.param({"voice": "echo", "format": "wav"}, id="voice-only"),
        pytest.param({"voice": "alloy", "format": "mp3"}, id="format-only"),
    ],
)
def test_openai_differing_audio_config_not_served_from_cache(audio_b) -> None:
    """Audio requests that differ only in ``audio.voice`` or ``audio.format``
    must not replay each other's audio (#4276); a repeat still hits."""
    calls: list[dict] = []

    with _make_cached_proxy_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            calls.append(body.get("audio"))
            return httpx.Response(
                200,
                json={
                    "id": f"chatcmpl_{len(calls)}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"resp-{len(calls)}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                },
            )

        proxy._retry_request = _fake_retry
        headers = {"authorization": "Bearer test-key"}

        def _audio_body(audio: dict) -> dict:
            return _body(modalities=["text", "audio"], audio=audio)

        ra = client.post(
            "/v1/chat/completions", headers=headers, json=_audio_body(_AUDIO_ALLOY_WAV)
        )
        assert _content(ra) == "resp-1"

        # Same messages and modalities, different audio config: must go upstream.
        rb = client.post("/v1/chat/completions", headers=headers, json=_audio_body(audio_b))
        assert rb.status_code == 200
        assert _content(rb) == "resp-2"
        assert calls == [_AUDIO_ALLOY_WAV, audio_b]

        # Repeating B is served from cache.
        rb2 = client.post("/v1/chat/completions", headers=headers, json=_audio_body(audio_b))
        assert _content(rb2) == "resp-2"
        assert len(calls) == 2


def test_openai_request_without_audio_keeps_its_cache_key(monkeypatch) -> None:
    """A request that sends no ``audio`` must not add an ``audio`` key field, so
    cache keys for text-only requests are the same as before #4276."""
    from headroom.proxy import semantic_cache

    seen: list[dict] = []
    real = semantic_cache.compute_semantic_cache_key

    def _spy(messages, model, *, partition, **key_fields):  # noqa: ANN001
        seen.append(key_fields)
        return real(messages, model, partition=partition, **key_fields)

    monkeypatch.setattr(semantic_cache, "compute_semantic_cache_key", _spy)

    with _make_cached_proxy_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "resp-1"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                },
            )

        proxy._retry_request = _fake_retry
        r = client.post(
            "/v1/chat/completions", headers={"authorization": "Bearer test-key"}, json=_body()
        )
        assert _content(r) == "resp-1"

    assert seen, "cache key was never computed"
    assert all("audio" not in fields for fields in seen)


def test_openai_different_custom_upstream_not_served_from_cache(monkeypatch) -> None:
    """Two OpenAI-compatible gateways must not share a response entry (#3346)."""
    calls: list[str] = []

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai._resolve_openai_upstream_base",
        lambda headers: headers.get("x-headroom-base-url"),
    )

    with _make_cached_proxy_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            calls.append(str(url))
            content = "gateway-A" if "gateway-a" in str(url) else "gateway-B"
            return httpx.Response(
                200,
                json={
                    "id": f"chatcmpl_{len(calls)}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                },
            )

        proxy._retry_request = _fake_retry
        headers = {"authorization": "Bearer test-key"}
        body = _body()

        first = client.post(
            "/v1/chat/completions",
            headers={**headers, "x-headroom-base-url": "https://gateway-a.example"},
            json=body,
        )
        second = client.post(
            "/v1/chat/completions",
            headers={**headers, "x-headroom-base-url": "https://gateway-b.example"},
            json=body,
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert _content(first) == "gateway-A"
    assert _content(second) == "gateway-B"
    assert calls == [
        "https://gateway-a.example/v1/chat/completions",
        "https://gateway-b.example/v1/chat/completions",
    ]

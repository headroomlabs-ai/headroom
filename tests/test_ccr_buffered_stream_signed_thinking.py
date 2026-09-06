"""Buffered-CCR streaming vs. byte-faithful passthrough (issue #2952).

The buffered-CCR path is the one place the Anthropic handler changes the
request *for its own benefit*: it flips ``stream`` to False so the reply comes
back as one JSON document it can inspect for ``headroom_retrieve`` calls, then
resynthesizes SSE for the client.

That only works if the flip reaches the wire. When conversation history carries
a signed ``thinking`` block, ``select_outbound_body`` forwards the client's
original bytes instead — ``"stream": true`` and all — so upstream streams, the
JSON parse fails, resynthesis is skipped, and the client is left with a 200 and
nothing it can read. These tests pin the three defenses: don't take the path,
survive the reply if we somehow do, and never cache a body in the wrong format.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.models import CacheEntry  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

RETRIEVE_TOOL = {
    "name": "headroom_retrieve",
    "description": "Retrieve original content",
    "input_schema": {"type": "object", "properties": {}},
}

SIGNED_THINKING_TURN = {
    "role": "assistant",
    "content": [
        {
            "type": "thinking",
            "thinking": "private reasoning",
            "signature": "sig-abc123",
        },
        {"type": "text", "text": "Answered."},
    ],
}

SSE_BODY = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1"}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def _config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=True,
        rate_limit_enabled=False,
        memory_enabled=False,
        ccr_event_streaming=False,
    )


@pytest.fixture
def ccr_marker() -> str:
    """A marker this proxy actually owns, so retrieval could really fire.

    The buffered path is only taken when the outgoing body carries a redeemable
    marker (#3071) — ``headroom_retrieve`` has nothing to expand otherwise. These
    tests are about what happens *on* that path, so they have to earn it.
    """
    from headroom.cache.backends import InMemoryBackend
    from headroom.cache.compression_store import get_compression_store, reset_compression_store

    reset_compression_store()
    store = get_compression_store(backend=InMemoryBackend())
    hash_key = store.store(
        "the original, uncompressed tool output",
        "<<ccr:placeholder>>",
        original_tokens=100,
        compressed_tokens=5,
        tool_name="Read",
    )
    try:
        yield hash_key
    finally:
        reset_compression_store()


def _body(*, with_thinking: bool, marker: str | None = None) -> dict:
    first = "hi" if marker is None else f"hi — earlier output is at <<ccr:{marker}>>"
    messages: list[dict] = [{"role": "user", "content": first}]
    if with_thinking:
        messages.append(SIGNED_THINKING_TURN)
        messages.append({"role": "user", "content": "continue"})
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 64,
        "stream": True,
        "tools": [RETRIEVE_TOOL],
        "messages": messages,
    }


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-key", "x-api-key": "test-key"}


@pytest.mark.parametrize(
    ("with_thinking", "relaxation_enabled", "expect_plain_streaming"),
    [
        # Locked (kill switch engaged): the flip cannot reach upstream, so
        # buffering would ask for a stream and then parse it as JSON, stranding
        # the client with an unreadable 200. This is #2952 exactly.
        (True, False, True),
        # Relaxed (default): no transform touched the thinking block, so the
        # flip DOES land and buffered retrieval becomes the coherent choice --
        # the outcome #2952 wanted before the blanket lock made it unreachable.
        (True, True, False),
        # No thinking block: unaffected in either regime.
        (False, True, False),
        (False, False, False),
    ],
)
def test_signed_thinking_history_skips_the_buffered_ccr_path(
    with_thinking: bool,
    relaxation_enabled: bool,
    expect_plain_streaming: bool,
    ccr_marker: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The buffered path is only chosen when the stream:false flip can land."""
    monkeypatch.setenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", "1" if relaxation_enabled else "0")
    calls: dict[str, object] = {}

    async def fake_stream_response(url, headers, body, *args, **kwargs):  # noqa: ANN001
        calls["stream_body"] = body
        return fastapi.responses.StreamingResponse(iter([SSE_BODY]), media_type="text/event-stream")

    async def fake_retry(method, url, headers, req_body, *args, **kwargs):  # noqa: ANN001
        calls["buffered_body"] = json.loads(json.dumps(req_body))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            headers={"content-type": "application/json"},
        )

    app = create_app(_config())
    with TestClient(app) as client:
        client.app.state.proxy._stream_response = fake_stream_response
        client.app.state.proxy._retry_request = fake_retry
        resp = client.post(
            "/v1/messages",
            json=_body(with_thinking=with_thinking, marker=ccr_marker),
            headers=_headers(),
        )

    assert resp.status_code == 200, resp.text
    if expect_plain_streaming:
        # Passthrough is locked in, so we must not pretend we can buffer.
        assert "stream_body" in calls, "expected the plain streaming path"
        assert "buffered_body" not in calls
        # The turn still leaves as a streaming request, matching the bytes
        # that passthrough will actually forward.
        assert calls["stream_body"]["stream"] is True
    else:
        assert "buffered_body" in calls, "expected the buffered CCR path"
        assert calls["buffered_body"]["stream"] is False


@pytest.mark.parametrize("upstream_delay", [0.0, 1.2], ids=["prompt", "past-keepalive"])
def test_buffered_ccr_relays_an_unexpected_sse_reply_and_does_not_cache_it(
    upstream_delay: float, ccr_marker: str
) -> None:
    """A 200 SSE reply on the buffered path reaches the client as a stream.

    The delay matters: ``_BufferedCCRResponse`` commits SSE response headers
    after a 1 s keepalive, and past that point it can only forward a result
    that exposes a ``body_iterator``. A plain ``Response`` there degrades to a
    bare ``event: error`` — which is what a real (multi-second) Anthropic turn
    hit in #2952.
    """

    async def fake_retry(method, url, headers, req_body, *args, **kwargs):  # noqa: ANN001
        if upstream_delay:
            await asyncio.sleep(upstream_delay)
        return httpx.Response(200, content=SSE_BODY, headers={"content-type": "text/event-stream"})

    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = fake_retry
        resp = client.post(
            "/v1/messages",
            json=_body(with_thinking=False, marker=ccr_marker),
            headers=_headers(),
        )

        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert b"message_start" in resp.content
        # Caching SSE bytes under a key with no `stream` component is what
        # served a stream to a buffered caller in the first place.
        assert proxy.cache is not None
        assert len(proxy.cache._cache) == 0, "an unparseable body must never be cached"


@pytest.mark.parametrize(
    ("marker_kind", "expect_buffered"),
    [
        ("owned", True),
        ("none", False),
        ("foreign", False),
    ],
)
def test_buffering_is_gated_on_a_redeemable_marker(
    marker_kind: str, expect_buffered: bool, ccr_marker: str
) -> None:
    """A resident ``headroom_retrieve`` is not on its own a reason to buffer (#3071).

    The tool is injected once and kept resident so the tools array stays
    byte-stable for the prompt cache. Buffering on its presence alone meant
    every later streaming turn of a sticky session lost incremental delivery —
    time-to-first-byte became the whole generation. Retrieval can only expand a
    marker that is in the outgoing body *and* redeemable now, so that is what
    the wire-format decision keys on.
    """
    marker = {
        "owned": ccr_marker,
        "none": None,
        # Correct shape, not ours: adopting it would send the model to a
        # retrieval that is guaranteed to miss (#2836).
        "foreign": "deadbeefcafe",
    }[marker_kind]

    calls: dict[str, object] = {}

    async def fake_stream_response(url, headers, body, *args, **kwargs):  # noqa: ANN001
        calls["stream_body"] = body
        return fastapi.responses.StreamingResponse(iter([SSE_BODY]), media_type="text/event-stream")

    async def fake_retry(method, url, headers, req_body, *args, **kwargs):  # noqa: ANN001
        calls["buffered_body"] = json.loads(json.dumps(req_body))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            headers={"content-type": "application/json"},
        )

    app = create_app(_config())
    with TestClient(app) as client:
        client.app.state.proxy._stream_response = fake_stream_response
        client.app.state.proxy._retry_request = fake_retry
        resp = client.post(
            "/v1/messages",
            json=_body(with_thinking=False, marker=marker),
            headers=_headers(),
        )

    assert resp.status_code == 200, resp.text
    if expect_buffered:
        assert "buffered_body" in calls, "a redeemable marker must still buffer"
        assert calls["buffered_body"]["stream"] is False
    else:
        assert "stream_body" in calls, "nothing to retrieve — the client must keep streaming"
        assert "buffered_body" not in calls
        assert calls["stream_body"]["stream"] is True


def test_cache_hit_never_replays_a_foreign_content_type() -> None:
    """A cache entry cannot hand a caller a wire format it did not ask for."""
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 64,
        "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload = json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "cached"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()

    async def fail_retry(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("upstream must not be called on a cache hit")

    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = fail_retry
        key = proxy.cache._compute_key(
            body["messages"],
            body["model"],
            system=None,
            tools=None,
            tool_choice=None,
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=64,
            stop=None,
            thinking=None,
            output_config=None,
        )
        proxy.cache._cache[key] = CacheEntry(
            response_body=payload,
            response_headers={"content-type": "text/event-stream"},
            created_at=datetime.now(),
            ttl_seconds=3600,
        )

        resp = client.post("/v1/messages", json=body, headers=_headers())

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["content"][0]["text"] == "cached"

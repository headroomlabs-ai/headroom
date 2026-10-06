"""Regression coverage for #3394: Gemini prefix freeze + compressed-prefix replay.

The Anthropic and OpenAI handlers keep a per-session PrefixCacheTracker and use
it two ways on every turn:

* a freeze floor (``frozen_message_count``) so the pipeline does not recompress
  already-forwarded history, and
* a ``finalize_turn`` replay so the previously forwarded COMPRESSED prefix goes
  out byte-identical - the provider cached the compressed bytes, so forwarding
  the raw originals for a frozen message would bust the prefix cache from the
  first changed byte.

The Gemini handler had neither: ``gemini.py`` resolved no tracker, passed no
``frozen_message_count`` on any of its three ``openai_pipeline.apply()`` call
sites, and never replayed the forwarded prefix - so every Gemini turn
recompressed the whole history and any compressor drift busted Gemini's
implicit prefix cache (whose hits the handler already reads back as
``cachedContentTokenCount``).

These tests drive multi-turn conversations through the real proxy with a
mocked Gemini upstream (MockTransport) and assert on the actual forwarded
bodies: a fix that only adds the kwarg (forwarding raw originals for the
frozen prefix) fails the byte-identity assertions.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

SESSION_ID = "gemini-prefix-freeze-test"
MODEL = "gemini-2.5-flash"
# Well above the default min_cached_tokens=1024 and above the local token
# estimate of the small fixture messages, so the freeze walk covers every
# forwarded message each turn.
PROVIDER_PROMPT_TOKENS = 10_000


def _config() -> ProxyConfig:
    return ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )


def _text(tag: str) -> str:
    return f"{tag} " + "lorem ipsum dolor sit amet " * 40


def _turn_body(n_turns: int) -> dict[str, Any]:
    """systemInstruction + n user/model exchanges + one trailing user turn."""
    contents: list[dict[str, Any]] = []
    for i in range(n_turns):
        contents.append({"role": "user", "parts": [{"text": _text(f"user-{i}")}]})
        contents.append({"role": "model", "parts": [{"text": _text(f"model-{i}")}]})
    contents.append({"role": "user", "parts": [{"text": _text(f"user-{n_turns}")}]})
    return {
        "systemInstruction": {"parts": [{"text": _text("system")}]},
        "contents": contents,
    }


def _fake_apply_factory(frozen_seen: list[int | None]):
    """Fake pipeline: marks only UNfrozen messages, like the real router.

    Records the raw ``frozen_message_count`` kwarg (None = never passed).
    """

    def _fake_apply(**kwargs: Any) -> SimpleNamespace:
        frozen_seen.append(kwargs.get("frozen_message_count"))
        frozen = kwargs.get("frozen_message_count") or 0
        out = [
            ({**m, "content": f"[z] {m['content']}"} if i >= frozen else m)
            for i, m in enumerate(kwargs["messages"])
        ]
        return SimpleNamespace(
            messages=out,
            transforms_applied=["fake:mark-unfrozen"],
            timing={},
            tokens_before=1000,
            tokens_after=900,
            waste_signals=None,
        )

    return _fake_apply


def _gemini_provider(sent: list[dict[str, Any]]):
    def _provider(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if ":countTokens" in str(request.url):
            return httpx.Response(200, json={"totalTokens": 123})
        usage = {
            "promptTokenCount": PROVIDER_PROMPT_TOKENS,
            "candidatesTokenCount": 5,
            "totalTokenCount": PROVIDER_PROMPT_TOKENS + 5,
            "cachedContentTokenCount": 0,
        }
        if ":streamGenerateContent" in str(request.url):
            chunks = (
                'data: {"candidates": [{"content": {"role": "model", '
                '"parts": [{"text": "ack"}]}}]}\n\n'
                + f"data: {json.dumps({'usageMetadata': usage})}\n\n"
            )
            return httpx.Response(
                200,
                content=chunks.encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "ack"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": usage,
            },
        )

    return _provider


def _install_fakes(proxy, sent: list[dict[str, Any]], frozen_seen: list[int | None]) -> None:
    proxy.openai_pipeline.apply = _fake_apply_factory(frozen_seen)
    proxy.http_client = httpx.AsyncClient(transport=httpx.MockTransport(_gemini_provider(sent)))


def _post(client: TestClient, body: dict[str, Any], *, stream: bool = False) -> None:
    url = f"/v1beta/models/{MODEL}:generateContent?key=test-key"
    if stream:
        url += "&alt=sse"
    resp = client.post(url, json=body, headers={"x-headroom-session-id": SESSION_ID})
    assert resp.status_code == 200, resp.text[:300]
    resp.read()


def test_gemini_buffered_replays_compressed_prefix_byte_identical() -> None:
    """Three buffered turns: turn N forwards turn N-1's prefix byte-identical.

    Fails before the fix on two independent grounds: the pipeline never sees
    ``frozen_message_count`` (None at every call) and the forwarded prefix
    bytes are the freshly re-marked forms rather than turn 1's forwarded ones.
    """
    sent: list[dict[str, Any]] = []
    frozen_seen: list[int | None] = []
    app = create_app(_config())
    with TestClient(app) as client:
        _install_fakes(client.app.state.proxy, sent, frozen_seen)
        for n in (0, 1, 2):
            _post(client, _turn_body(n))

    # Freeze floor reaches the pipeline: cold on turn 1, then the whole
    # forwarded history of the previous turn.
    assert frozen_seen == [0, 2, 4]

    # The compressed prefix is replayed byte-identical - not re-emitted raw.
    assert "[z]" in sent[0]["systemInstruction"]["parts"][0]["text"]
    assert sent[1]["systemInstruction"] == sent[0]["systemInstruction"]
    assert sent[2]["systemInstruction"] == sent[0]["systemInstruction"]
    assert sent[1]["contents"][:1] == sent[0]["contents"]
    assert sent[2]["contents"][:3] == sent[1]["contents"]

    # The new tail still gets compressed (only the frozen prefix is replayed).
    for payload in sent[1:]:
        assert "[z]" in payload["contents"][-1]["parts"][0]["text"]


def test_gemini_streaming_advances_tracker_and_replays_prefix() -> None:
    """A streaming turn must feed the tracker; the next turn freezes + replays.

    Before the fix the streaming branch never passed a prefix_tracker to
    ``_stream_response`` (and the finalizer could not read Gemini ``contents``
    anyway), so the tracker never advanced and no freeze ever engaged.
    """
    sent: list[dict[str, Any]] = []
    frozen_seen: list[int | None] = []
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        _install_fakes(proxy, sent, frozen_seen)
        _post(client, _turn_body(0), stream=True)

        tracker = proxy.session_tracker_store.peek(SESSION_ID)
        assert tracker is not None, "streaming turn never resolved a prefix tracker"
        assert tracker._turn_number == 1, (
            "streaming turn never called update_from_response - tracker stuck at turn 0"
        )
        assert tracker.get_frozen_message_count() == 2
        assert tracker.get_last_forwarded_messages(), (
            "tracker recorded no forwarded messages from the streaming turn"
        )

        _post(client, _turn_body(1), stream=True)

    assert frozen_seen == [0, 2]
    assert sent[1]["systemInstruction"] == sent[0]["systemInstruction"]
    assert sent[1]["contents"][:1] == sent[0]["contents"]


def test_gemini_count_tokens_passes_frozen_count() -> None:
    """countTokens compresses with the same freeze floor as generateContent."""
    sent: list[dict[str, Any]] = []
    frozen_seen: list[int | None] = []
    app = create_app(_config())
    with TestClient(app) as client:
        _install_fakes(client.app.state.proxy, sent, frozen_seen)
        _post(client, _turn_body(0))
        frozen_seen.clear()

        resp = client.post(
            f"/v1beta/models/{MODEL}:countTokens?key=test-key",
            json=_turn_body(1),
            headers={"x-headroom-session-id": SESSION_ID},
        )
        assert resp.status_code == 200, resp.text[:300]

    assert frozen_seen == [2]


def _post_cloudcode(client: TestClient, body: dict[str, Any]) -> None:
    resp = client.post(
        "/v1internal:streamGenerateContent",
        json={"model": MODEL, "request": body},
        headers={"x-headroom-session-id": SESSION_ID},
    )
    assert resp.status_code == 200, resp.text[:300]
    resp.read()


def test_cloudcode_streaming_advances_tracker_and_replays_prefix() -> None:
    """Cloud Code Assist (v1internal): same freeze + replay as the native path.

    Regression for the requested-changes review on #3394: the replay ran only
    after ``request_payload`` was already built (so the forwarded body never
    contained the replayed prefix), and ``_stream_response`` never received
    the tracker. Fails before this fix: the tracker stays at turn 0, the
    freeze floor never reaches the pipeline ([0, 0]), and turn 2's nested
    payload prefix is not turn 1's forwarded bytes.
    """
    sent: list[dict[str, Any]] = []
    frozen_seen: list[int | None] = []
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        _install_fakes(proxy, sent, frozen_seen)
        _post_cloudcode(client, _turn_body(0))

        tracker = proxy.session_tracker_store.peek(SESSION_ID)
        assert tracker is not None, "cloudcode stream never resolved a prefix tracker"
        assert tracker._turn_number == 1, (
            "cloudcode stream never called update_from_response - tracker stuck at turn 0"
        )
        assert tracker.get_frozen_message_count() == 2
        assert tracker.get_last_forwarded_messages(), (
            "tracker recorded no forwarded messages from the cloudcode stream"
        )

        _post_cloudcode(client, _turn_body(1))

    # Freeze floor reaches the pipeline on the cloudcode route too.
    assert frozen_seen == [0, 2]

    # The nested request payload carries the replayed prefix byte-identical.
    req0, req1 = sent[0]["request"], sent[1]["request"]
    assert "[z]" in req0["systemInstruction"]["parts"][0]["text"]
    assert req1["systemInstruction"] == req0["systemInstruction"]
    assert req1["contents"][:1] == req0["contents"]

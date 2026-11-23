"""WU2 (headroom-37g.2): matrix coverage for WU1's agy functionResponse leaf
compression that is NOT already covered by
``tests/test_agy_functionresponse_compression.py``.

Five gaps closed here:

1. Cross-turn cache stability: the SAME functionResponse entry, re-sent by agy
   unchanged across two turns (once as tail, once as history), must produce
   BYTE-IDENTICAL outbound compressed leaves -- this is what keeps Cloud Code
   Assist's server-side cached prefix stable.
2. Mixed functionResponse+text entry: the functionResponse leaf compresses and
   the outbound entry is the MUTATED one, not the pristine original; the
   co-located text is untouched.
3. No-double-count: ``tokens_saved`` reflects the functionResponse delta only
   -- the #819 ``waste_messages`` telemetry path never contributes to it.
4. #819 waste-signal non-regression: the ``include_function_responses=True``
   conversion feeding ``TransformPipeline.apply(waste_messages=...)`` still
   fires correctly alongside WU1's functionResponse compression.
5. Non-antigravity non-regression: a plain (non-antigravity) Gemini
   cloudcode/generateContent request is untouched by the functionResponse-leaf
   pass.

All upstream/network calls are stubbed via monkeypatch on
``HeadroomProxy._stream_response`` / ``openai_pipeline.apply``. Never contacts
the real 8787 proxy or any network destination.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.proxy.handlers.gemini import _FR_CCR_MARKER_PREFIX
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app
from headroom.tokenizers import get_tokenizer

# Mirrors tests/test_agy_functionresponse_compression.py -- large, single-line
# (no repeated lines, so lossless is a no-op), well above the marker floor.
BIG_LEAF = "search result row alpha beta gamma delta epsilon zeta eta " * 40

# Enough text to make CompressionDecision.should_compress True.
_REPEAT_UNIT = "The quick brown fox jumps over the lazy dog. " * 60

_MODEL = "gemini-3-flash-agent"
_SSE_PAYLOAD = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}\r\n\r\ndata: [DONE]\r\n\r\n'
)


def _make_sse() -> StreamingResponse:
    async def _body() -> Any:
        yield _SSE_PAYLOAD

    return StreamingResponse(_body(), status_code=200, media_type="text/event-stream")


def _fr_leaf(contents: list, entry: int, part: int = 0, key: str = "output") -> Any:
    return contents[entry]["parts"][part]["functionResponse"]["response"][key]


def _fr_entry(role: str = "user", leaf: Any = BIG_LEAF, name: str = "search") -> dict:
    return {
        "role": role,
        "parts": [{"functionResponse": {"name": name, "response": {"output": leaf}}}],
    }


class _FakeResult:
    """Stand-in for the compression pipeline result."""

    def __init__(self, messages: Any, tokens_before: int, tokens_after: int) -> None:
        self.messages = messages
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.transforms_applied: list[str] = ["noop"]


@pytest.fixture
def proxy() -> Any:
    with TestClient(create_app(ProxyConfig(optimize=True))) as client:
        yield client.app.state.proxy  # type: ignore[attr-defined]


@pytest.fixture
def tok() -> Any:
    return get_tokenizer(_MODEL)


@pytest.fixture
def ccr_store(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    reset_compression_store()
    store = get_compression_store()
    yield store
    reset_compression_store()


# ---------------------------------------------------------------------------
# Gap 1: cross-turn cache stability
# ---------------------------------------------------------------------------
def test_cross_turn_cache_bytes_identical_for_resent_entry(
    proxy: Any, tok: Any, ccr_store: Any
) -> None:
    # Turn N: the functionResponse entry is the newest turn (tail).
    turn_n = [
        {"role": "user", "parts": [{"text": "call search"}]},
        _fr_entry(role="user", name="search"),
    ]
    proxy._compress_agy_function_responses(turn_n, "ccr", tok, ccr_store)
    marker_turn_n = _fr_leaf(turn_n, 1)
    assert marker_turn_n.startswith(_FR_CCR_MARKER_PREFIX)

    # Turn N+1: agy re-sends the SAME entry with its ORIGINAL bytes, now
    # historical, plus a new tail turn.
    turn_n1 = [
        {"role": "user", "parts": [{"text": "call search"}]},
        _fr_entry(role="user", name="search"),  # identical original leaf, resent
        {"role": "model", "parts": [{"text": "reasoning about the result"}]},
        _fr_entry(role="user", name="search2", leaf=BIG_LEAF + "!"),  # new tail
    ]
    proxy._compress_agy_function_responses(turn_n1, "ccr", tok, ccr_store)
    marker_turn_n1 = _fr_leaf(turn_n1, 1)

    assert marker_turn_n1.startswith(_FR_CCR_MARKER_PREFIX)
    # BYTE-IDENTICAL across turns: the anti-cache-bust guarantee.
    assert marker_turn_n1 == marker_turn_n
    # Sanity: the new tail entry got its OWN, different marker.
    assert _fr_leaf(turn_n1, 3) != marker_turn_n


# ---------------------------------------------------------------------------
# Gap 2: mixed functionResponse + text entry
# ---------------------------------------------------------------------------
def test_mixed_text_and_functionresponse_entry_leaf_compressed_and_entry_mutated(
    proxy: Any, tok: Any, ccr_store: Any
) -> None:
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "here is context"},
                {"functionResponse": {"name": "search", "response": {"output": BIG_LEAF}}},
            ],
        }
    ]
    original = copy.deepcopy(contents[0])
    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)
    assert leaves == 1
    assert before > after > 0

    # functionResponse leaf compressed.
    assert _fr_leaf(contents, 0, part=1).startswith(_FR_CCR_MARKER_PREFIX)
    # Co-located text part untouched.
    assert contents[0]["parts"][0]["text"] == "here is context"
    # The outbound entry as a whole is the MUTATED one, not the pristine original.
    assert contents[0] != original
    assert contents[0]["parts"][1] != original["parts"][1]


# ---------------------------------------------------------------------------
# Gap 3: no-double-count against the #819 waste_messages path
# ---------------------------------------------------------------------------
def test_tokens_saved_reflects_fr_delta_only_no_double_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
    monkeypatch.setenv("HEADROOM_AGY_RETRIEVE_WIRED", "1")
    reset_compression_store()

    captured: dict[str, Any] = {}

    async def _fake_stream(
        proxy_self: Any,
        url: str,
        headers: dict,
        body: dict,
        provider: str,
        model: str,
        request_id: Any,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        *args: Any,
        **kwargs: Any,
    ) -> StreamingResponse:
        captured["tokens_saved"] = tokens_saved
        return _make_sse()

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    def _noop_apply(*, messages: Any, **kw: Any) -> _FakeResult:
        # Identical messages object -> text-side delta is exactly zero,
        # isolating the accounting to the functionResponse leaf pass.
        return _FakeResult(messages=messages, tokens_before=0, tokens_after=0)

    contents = [
        {"role": "user", "parts": [{"text": "call search"}]},
        _fr_entry(role="user", leaf=BIG_LEAF),
    ]
    body = {"model": _MODEL, "request": {"contents": copy.deepcopy(contents)}}

    with TestClient(create_app(ProxyConfig(optimize=True))) as client:
        proxy = client.app.state.proxy  # type: ignore[attr-defined]
        proxy.openai_pipeline.apply = _noop_apply  # type: ignore[method-assign]

        # Expected FR delta computed directly via the production method against
        # an INDEPENDENT copy (same tokenizer/store) -- proves the shipped
        # tokens_saved is exactly the functionResponse delta and nothing more
        # (the #819 waste_messages telemetry path contributes zero tokens).
        tok_ = get_tokenizer(_MODEL)
        store = get_compression_store()
        expected_contents = copy.deepcopy(contents)
        exp_before, exp_after, exp_leaves = proxy._compress_agy_function_responses(
            expected_contents, "ccr", tok_, store
        )
        assert exp_leaves == 1
        assert exp_before > exp_after > 0

        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={"User-Agent": "antigravity/1.0.5"},
            json=body,
        )

    assert response.status_code == 200
    assert captured["tokens_saved"] == exp_before - exp_after

    reset_compression_store()


# ---------------------------------------------------------------------------
# Gap 4: #819 waste-signal non-regression
# ---------------------------------------------------------------------------
def test_waste_signal_detection_path_intact_with_fr_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
    monkeypatch.setenv("HEADROOM_AGY_RETRIEVE_WIRED", "1")
    reset_compression_store()

    captured: dict[str, Any] = {}

    async def _fake_stream(
        proxy_self: Any,
        url: str,
        headers: dict,
        body: dict,
        provider: str,
        model: str,
        request_id: Any,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        *args: Any,
        **kwargs: Any,
    ) -> StreamingResponse:
        captured["body"] = body
        return _make_sse()

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    def _spy_apply(*, messages: Any, waste_messages: Any = None, **kw: Any) -> _FakeResult:
        captured["waste_messages"] = waste_messages
        return _FakeResult(messages=messages, tokens_before=0, tokens_after=0)

    # A payload with BOTH a large string leaf (WU1's compression target) and a
    # bulky array (drives #819 json-bloat waste-signal detection).
    tool_payload = {
        "output": BIG_LEAF,
        "rows": [{"id": i, "name": f"item_{i}"} for i in range(50)],
    }
    body = {
        "model": _MODEL,
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": _REPEAT_UNIT}]},
                {
                    "role": "user",
                    "parts": [
                        {"functionResponse": {"name": "fetch_data", "response": tool_payload}}
                    ],
                },
            ]
        },
    }

    with TestClient(create_app(ProxyConfig(optimize=True))) as client:
        proxy = client.app.state.proxy  # type: ignore[attr-defined]
        proxy.openai_pipeline.apply = _spy_apply  # type: ignore[method-assign]
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={"User-Agent": "antigravity/1.0.5"},
            json=body,
        )

    assert response.status_code == 200

    # #819: the tool-output payload reached waste-signal detection via the
    # include_function_responses=True conversion, unaffected by WU1.
    waste_msgs = captured.get("waste_messages")
    assert waste_msgs is not None
    tool_msgs = [m for m in waste_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert json.loads(tool_msgs[0]["content"]) == tool_payload

    # WU1's functionResponse compression also ran on the same request: the
    # large "output" leaf compressed; the small "rows" array untouched.
    contents = captured["body"]["request"]["contents"]
    fr_response = contents[1]["parts"][0]["functionResponse"]["response"]
    assert fr_response["output"].startswith(_FR_CCR_MARKER_PREFIX)
    assert fr_response["rows"] == tool_payload["rows"]

    reset_compression_store()


# ---------------------------------------------------------------------------
# Gap 5: non-antigravity non-regression
# ---------------------------------------------------------------------------
def test_non_antigravity_request_functionresponse_passthrough_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
    monkeypatch.setenv("HEADROOM_AGY_RETRIEVE_WIRED", "1")
    reset_compression_store()

    captured: dict[str, Any] = {}

    async def _fake_stream(
        proxy_self: Any,
        url: str,
        headers: dict,
        body: dict,
        provider: str,
        model: str,
        request_id: Any,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        *args: Any,
        **kwargs: Any,
    ) -> StreamingResponse:
        captured["body"] = body
        return _make_sse()

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    def _noop_apply(*, messages: Any, **kw: Any) -> _FakeResult:
        return _FakeResult(messages=messages, tokens_before=0, tokens_after=0)

    body = {
        # NOTE: model does not end in "-agent" and no antigravity User-Agent /
        # userAgent / requestType / project field is present -> is_antigravity
        # resolves to False (see _is_cloudcode_antigravity_request).
        "model": "gemini-3-pro",
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": _REPEAT_UNIT}]},
                _fr_entry(role="user"),
            ]
        },
    }

    with TestClient(create_app(ProxyConfig(optimize=True))) as client:
        proxy = client.app.state.proxy  # type: ignore[attr-defined]
        proxy.openai_pipeline.apply = _noop_apply  # type: ignore[method-assign]
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            # Deliberately NOT antigravity: default TestClient User-Agent.
            json=body,
        )

    assert response.status_code == 200
    contents = captured["body"]["request"]["contents"]
    fr_leaf_value = _fr_leaf(contents, 1)
    # Untouched: byte-identical to the original, no marker, no compaction.
    assert fr_leaf_value == BIG_LEAF
    assert _FR_CCR_MARKER_PREFIX not in fr_leaf_value

    reset_compression_store()


# ---------------------------------------------------------------------------
# Gap 6: historical prefix replay (#3394) + retrieval expansion contract
# ---------------------------------------------------------------------------
def test_historical_prefix_replay_and_fr_retrieval_expansion_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconcile #3394 text prefix replay with AGY historical functionResponse policy.

    - Frozen text messages are replayed byte-identical across turns by
      ``_gemini_replay_forwarded_prefix``.
    - Unretrieved historical ``functionResponse`` leaves stay byte-identical
      across turns via deterministic CCR markers.
    - When a later turn retrieves hash A, historical leaf A intentionally expands
      back to original (one-time prefix shift at that leaf to prevent retrieval
      thrash), while frozen text and unretrieved sibling leaf B stay byte-stable.
    """
    from headroom.cache.compression_store import default_ccr_hash

    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
    monkeypatch.setenv("HEADROOM_AGY_RETRIEVE_WIRED", "1")
    reset_compression_store()

    leaf_a = BIG_LEAF
    leaf_b = "sibling tool output row gamma delta epsilon zeta eta theta " * 40
    hash_a = default_ccr_hash(leaf_a)

    captured_bodies: list[dict[str, Any]] = []
    call_count = 0

    async def _fake_stream(
        proxy_self: Any,
        url: str,
        headers: dict,
        body: dict,
        provider: str,
        model: str,
        request_id: Any,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        tags: Any,
        optimization_latency: float,
        *,
        prefix_tracker: Any = None,
        original_messages: Any = None,
        **kwargs: Any,
    ) -> StreamingResponse:
        captured_bodies.append(copy.deepcopy(body))
        if prefix_tracker is not None and original_messages is not None:
            # Record forwarded text messages and confirm cached prefix like a real turn.
            forwarded_msgs, _ = proxy_self._gemini_contents_to_messages(
                body["request"]["contents"], None
            )
            prefix_tracker.update_from_response(
                cache_read_tokens=4000,
                cache_write_tokens=1000,
                messages=forwarded_msgs,
                original_messages=original_messages,
            )
        return _make_sse()

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    def _varying_text_apply(*, messages: Any, **kw: Any) -> _FakeResult:
        nonlocal call_count
        call_count += 1
        # Compress first user text differently per turn unless prefix replay freezes it.
        out = copy.deepcopy(messages)
        out[0]["content"] = f"compressed-turn-{call_count}"
        return _FakeResult(messages=out, tokens_before=1000, tokens_after=10)

    turn1_contents = [
        {"role": "user", "parts": [{"text": _REPEAT_UNIT}]},
        _fr_entry(role="user", leaf=leaf_a, name="read_a"),
        _fr_entry(role="user", leaf=leaf_b, name="read_b"),
    ]
    turn2_contents = copy.deepcopy(turn1_contents) + [
        {"role": "model", "parts": [{"text": "checking results"}]},
        {"role": "user", "parts": [{"text": "follow up without retrieve"}]},
    ]
    turn3_contents = copy.deepcopy(turn2_contents) + [
        {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "call_mcp_tool",
                        "args": {"tool": "headroom_retrieve", "arguments": {"hash": hash_a}},
                    }
                }
            ],
        },
        {"role": "user", "parts": [{"text": "continue after retrieve"}]},
    ]

    with TestClient(create_app(ProxyConfig(optimize=True))) as client:
        proxy = client.app.state.proxy  # type: ignore[attr-defined]
        proxy.openai_pipeline.apply = _varying_text_apply  # type: ignore[method-assign]

        for turn_contents in (turn1_contents, turn2_contents, turn3_contents):
            response = client.post(
                "/v1internal:streamGenerateContent",
                params={"alt": "sse"},
                headers={"User-Agent": "antigravity/1.0.5"},
                json={"model": _MODEL, "request": {"contents": copy.deepcopy(turn_contents)}},
            )
            assert response.status_code == 200

    t1, t2, t3 = (b["request"]["contents"] for b in captured_bodies)

    # Turn 1: text compressed to turn-1 form; both FR leaves compressed to markers.
    assert t1[0]["parts"][0]["text"] == "compressed-turn-1"
    marker_a = _fr_leaf(t1, 1)
    marker_b = _fr_leaf(t1, 2)
    assert marker_a.startswith(_FR_CCR_MARKER_PREFIX)
    assert marker_b.startswith(_FR_CCR_MARKER_PREFIX)

    # Turn 2 (no retrieve): frozen text replayed byte-identical; both FR markers byte-identical.
    assert t2[0]["parts"][0]["text"] == "compressed-turn-1"
    assert _fr_leaf(t2, 1) == marker_a
    assert _fr_leaf(t2, 2) == marker_b

    # Turn 3 (after retrieve of hash_a): frozen text STILL replayed byte-identical and
    # unretrieved sibling leaf_b stays marker-identical, while retrieved leaf_a
    # intentionally expands back to original.
    assert t3[0]["parts"][0]["text"] == "compressed-turn-1"
    assert _fr_leaf(t3, 1) == leaf_a
    assert _fr_leaf(t3, 2) == marker_b

    reset_compression_store()

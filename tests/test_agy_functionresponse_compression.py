"""WU1 (headroom-37g.1): uniform deterministic recoverable compression of agy
functionResponse leaves in ``handle_google_cloudcode_stream``.

Scope (proves WU1's DoD):
- determinism / idempotency: same leaf -> identical bytes; f(f(x)) == f(x).
- CCR byte-recovery: retrieve(hash) returns the ORIGINAL leaf bytes.
- delivery-on-revert: text pipeline reverts, but a large functionResponse leaf
  still ships compressed in ``request_payload["contents"]`` and tokens_saved > 0.
- uniform: a historical (non-tail) functionResponse entry is ALSO compressed.
- pairing/shape preserved; functionCall untouched; non-string leaf skipped;
  multi-functionResponse-part entry all compressed.
- retrieve-gating: mode=ccr without HEADROOM_AGY_RETRIEVE_WIRED -> no
  unrecoverable marker (lossless / no-op).

All upstream/network calls are stubbed via monkeypatch on
``HeadroomProxy._stream_response`` / ``openai_pipeline.apply``. Never contacts
the real 8787 proxy or any network destination.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.tool_injection import is_headroom_retrieve_name
from headroom.parser import CCR_RETRIEVAL_MARKER_RE
from headroom.proxy.handlers.gemini import (
    _FR_CCR_MARKER_PREFIX,
    _resolve_agy_fr_mode,
)
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app
from headroom.tokenizers import get_tokenizer

# A large, single-line string leaf: no repeated lines (so lossless is a no-op),
# well above the marker-derived floor (~2x a ~20-token marker).
BIG_LEAF = "search result row alpha beta gamma delta epsilon zeta eta " * 40

# Enough text to make CompressionDecision.should_compress True (mirrors the
# shared fixture in test_proxy_agy_compression.py).
_REPEAT_UNIT = "The quick brown fox jumps over the lazy dog. " * 60

_MODEL = "gemini-3-flash-agent"
_SSE_PAYLOAD = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}\r\n\r\ndata: [DONE]\r\n\r\n'
)


def _make_sse() -> StreamingResponse:
    async def _body() -> Any:
        yield _SSE_PAYLOAD

    return StreamingResponse(_body(), status_code=200, media_type="text/event-stream")


def _hash_of(marker: str) -> str:
    assert marker.startswith(_FR_CCR_MARKER_PREFIX), marker
    # split: single-hash marker now, in the ``Retrieve more: hash=<h>]`` form
    # the parser regex keys on.
    return marker.split("hash=", 1)[1].rstrip("]")


def _fr_leaf(contents: list, entry: int, part: int = 0, key: str = "output") -> Any:
    return contents[entry]["parts"][part]["functionResponse"]["response"][key]


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
    # Force an in-memory backend and a clean store for hermetic byte-recovery.
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    reset_compression_store()
    store = get_compression_store()
    yield store
    reset_compression_store()


def _fr_entry(role: str = "user", leaf: Any = BIG_LEAF, name: str = "search") -> dict:
    return {
        "role": role,
        "parts": [{"functionResponse": {"name": name, "response": {"output": leaf}}}],
    }


# ---------------------------------------------------------------------------
# Determinism + idempotency
# ---------------------------------------------------------------------------
def test_ccr_deterministic_and_idempotent(proxy: Any, tok: Any, ccr_store: Any) -> None:
    c1 = [_fr_entry()]
    c2 = [_fr_entry()]

    b1, a1, l1 = proxy._compress_agy_function_responses(c1, "ccr", tok, ccr_store)
    b2, a2, l2 = proxy._compress_agy_function_responses(c2, "ccr", tok, ccr_store)

    assert l1 == 1 and l2 == 1
    # Deterministic: identical original -> identical marker bytes.
    assert _fr_leaf(c1, 0) == _fr_leaf(c2, 0)
    assert (b1, a1) == (b2, a2)

    # Idempotent: f(f(x)) == f(x) -- re-running is a no-op, bytes stable.
    stable = _fr_leaf(c1, 0)
    b3, a3, l3 = proxy._compress_agy_function_responses(c1, "ccr", tok, ccr_store)
    assert l3 == 0
    assert _fr_leaf(c1, 0) == stable


# ---------------------------------------------------------------------------
# CCR byte-recovery
# ---------------------------------------------------------------------------
def test_ccr_byte_recovery(proxy: Any, tok: Any, ccr_store: Any) -> None:
    contents = [_fr_entry()]
    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)
    assert leaves == 1 and before > after > 0

    marker = _fr_leaf(contents, 0)
    entry = ccr_store.retrieve(_hash_of(marker))
    assert entry is not None
    assert entry.original_content == BIG_LEAF


# ---------------------------------------------------------------------------
# Self-describing marker: names the retrieval tool so a model that needs the
# compressed detail knows how to expand it (WU4 observed 0 retrieve calls
# because the old marker named no tool).
# ---------------------------------------------------------------------------
def test_marker_names_headroom_retrieve_tool(proxy: Any, tok: Any, ccr_store: Any) -> None:
    c1 = [_fr_entry()]
    c2 = [_fr_entry()]
    proxy._compress_agy_function_responses(c1, "ccr", tok, ccr_store)
    proxy._compress_agy_function_responses(c2, "ccr", tok, ccr_store)
    marker = _fr_leaf(c1, 0)

    # Names the tool + gives a one-line call-to-expand instruction.
    assert "headroom_retrieve" in marker
    # Store-lookup path substring intact: parser.CCR_RETRIEVAL_MARKER_RE and
    # the CCR retrieval path both key on this exact substring.
    assert "Retrieve more: hash=" in marker
    assert CCR_RETRIEVAL_MARKER_RE.search(marker) is not None
    # Deterministic: identical original -> identical marker bytes, twice over.
    assert marker == _fr_leaf(c2, 0)
    # Round-trips via the store using the trailing canonical hash.
    entry = ccr_store.retrieve(_hash_of(marker))
    assert entry is not None
    assert entry.original_content == BIG_LEAF


# ---------------------------------------------------------------------------
# Floor + non-string leaves
# ---------------------------------------------------------------------------
def test_sub_floor_and_non_string_leaves_skipped(proxy: Any, tok: Any, ccr_store: Any) -> None:
    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "search",
                        "response": {
                            "small": "hi there",  # below floor
                            "count": 12345,  # non-string scalar
                            "ok": True,  # non-string scalar
                            "nothing": None,  # non-string scalar
                            "big": BIG_LEAF,  # compressed
                        },
                    }
                }
            ],
        }
    ]
    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)
    assert leaves == 1
    resp = contents[0]["parts"][0]["functionResponse"]["response"]
    assert resp["small"] == "hi there"
    assert resp["count"] == 12345
    assert resp["ok"] is True
    assert resp["nothing"] is None
    assert resp["big"].startswith(_FR_CCR_MARKER_PREFIX)


# ---------------------------------------------------------------------------
# Pairing / shape: functionCall untouched
# ---------------------------------------------------------------------------
def test_functioncall_untouched_pairing_preserved(proxy: Any, tok: Any, ccr_store: Any) -> None:
    contents = [
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "search", "args": {"query": BIG_LEAF}}}],
        },
        _fr_entry(role="user"),
    ]
    original_call = copy.deepcopy(contents[0])
    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)
    assert leaves == 1
    # functionCall entry byte-identical (never touched).
    assert contents[0] == original_call
    # functionResponse leaf compressed.
    assert _fr_leaf(contents, 1).startswith(_FR_CCR_MARKER_PREFIX)


# ---------------------------------------------------------------------------
# Multiple functionResponse parts in one entry: all compressed
# ---------------------------------------------------------------------------
def test_multi_functionresponse_parts_all_compressed(proxy: Any, tok: Any, ccr_store: Any) -> None:
    contents = [
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "a", "response": {"output": BIG_LEAF}}},
                {"functionResponse": {"name": "b", "response": {"output": BIG_LEAF + "!"}}},
            ],
        }
    ]
    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)
    assert leaves == 2
    for part in contents[0]["parts"]:
        assert part["functionResponse"]["response"]["output"].startswith(_FR_CCR_MARKER_PREFIX)


# ---------------------------------------------------------------------------
# Retrieve-gating: ccr without wired -> lossless (no unrecoverable marker)
# ---------------------------------------------------------------------------
def test_mode_downgrades_to_lossless_when_retrieve_not_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HEADROOM_AGY_RETRIEVE_WIRED", raising=False)
    monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
    assert _resolve_agy_fr_mode() == "lossless"

    monkeypatch.setenv("HEADROOM_AGY_RETRIEVE_WIRED", "1")
    assert _resolve_agy_fr_mode() == "ccr"

    monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "lossless")
    assert _resolve_agy_fr_mode() == "lossless"


def test_lossless_mode_emits_no_unrecoverable_marker(proxy: Any, tok: Any) -> None:
    contents = [_fr_entry()]
    # lossless never needs a store; a single-line leaf is a no-op.
    before, after, leaves = proxy._compress_agy_function_responses(contents, "lossless", tok, None)
    leaf = _fr_leaf(contents, 0)
    assert "Retrieve more: hash=" not in leaf
    assert leaf == BIG_LEAF  # unchanged no-op, still recoverable (byte-identical)


# ---------------------------------------------------------------------------
# Uniform: historical (non-tail) functionResponse entry also compressed
# ---------------------------------------------------------------------------
def test_uniform_historical_and_tail_compressed(proxy: Any, tok: Any, ccr_store: Any) -> None:
    contents = [
        _fr_entry(role="user", name="hist"),  # historical (index 0)
        {"role": "model", "parts": [{"text": "some reasoning"}]},
        _fr_entry(role="user", name="tail"),  # tail (index 2)
    ]
    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)
    assert leaves == 2
    assert _fr_leaf(contents, 0).startswith(_FR_CCR_MARKER_PREFIX)
    assert _fr_leaf(contents, 2).startswith(_FR_CCR_MARKER_PREFIX)


# ---------------------------------------------------------------------------
# Anti-self-defeating-loop: headroom_retrieve's OWN output must never be
# re-compressed back into the marker it was expanded from (headroom-37g.8).
# ---------------------------------------------------------------------------
def test_is_headroom_retrieve_name_matching() -> None:
    # Bare name and MCP-prefixed / custom-prefixed variants match.
    assert is_headroom_retrieve_name("headroom_retrieve") is True
    assert is_headroom_retrieve_name("mcp__headroom__headroom_retrieve") is True
    assert is_headroom_retrieve_name("custom__headroom_retrieve") is True
    # Unrelated / near-miss names must NOT match.
    assert is_headroom_retrieve_name("read_file") is False
    assert is_headroom_retrieve_name("my_headroom_retrieve_helper") is False
    # No double-underscore boundary -- must NOT match (single "x" prefix).
    assert is_headroom_retrieve_name("xheadroom_retrieve") is False
    assert is_headroom_retrieve_name(None) is False
    assert is_headroom_retrieve_name("") is False
    # untrusted JSON: non-str name must return False, never raise
    assert is_headroom_retrieve_name(123) is False
    assert is_headroom_retrieve_name({"headroom_retrieve": 1}) is False
    assert is_headroom_retrieve_name(["headroom_retrieve"]) is False


def test_headroom_retrieve_output_exempted_from_recompression(
    proxy: Any, tok: Any, ccr_store: Any
) -> None:
    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "mcp__headroom__headroom_retrieve",
                        "response": {"output": BIG_LEAF},
                    }
                },
                {
                    "functionResponse": {
                        "name": "read_file",
                        "response": {"output": BIG_LEAF + "!"},
                    }
                },
            ],
        }
    ]
    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)
    # Only the normal tool's leaf is compressed; the retrieve leaf is skipped.
    assert leaves == 1
    retrieve_leaf = _fr_leaf(contents, 0, part=0)
    normal_leaf = _fr_leaf(contents, 0, part=1)
    assert retrieve_leaf == BIG_LEAF
    assert not retrieve_leaf.startswith(_FR_CCR_MARKER_PREFIX)
    assert normal_leaf.startswith(_FR_CCR_MARKER_PREFIX)


def test_headroom_retrieve_bare_and_suffixed_names_both_exempted(
    proxy: Any, tok: Any, ccr_store: Any
) -> None:
    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "headroom_retrieve",
                        "response": {"output": BIG_LEAF},
                    }
                },
                {
                    "functionResponse": {
                        "name": "toolgroup__headroom_retrieve",
                        "response": {"output": BIG_LEAF + "!"},
                    }
                },
            ],
        }
    ]
    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)
    assert leaves == 0
    assert _fr_leaf(contents, 0, part=0) == BIG_LEAF
    assert _fr_leaf(contents, 0, part=1) == BIG_LEAF + "!"


# ---------------------------------------------------------------------------
# Integration: delivery + accounting even when the text pipeline REVERTS
# ---------------------------------------------------------------------------
def test_delivery_on_text_revert_ships_and_counts(
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
        captured["tokens_saved"] = tokens_saved
        return _make_sse()

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    # Force the text pipeline to REVERT: report inflation (after > before).
    def _inflating_apply(**kw: Any) -> _FakeResult:
        return _FakeResult(
            messages=[{"role": "user", "content": "x"}], tokens_before=5, tokens_after=99999
        )

    body = {
        "model": _MODEL,
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": _REPEAT_UNIT}]},
                _fr_entry(role="user"),
            ]
        },
    }

    with TestClient(create_app(ProxyConfig(optimize=True))) as client:
        proxy = client.app.state.proxy  # type: ignore[attr-defined]
        proxy.openai_pipeline.apply = _inflating_apply  # type: ignore[method-assign]
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={"User-Agent": "antigravity/1.0.5"},
            json=body,
        )

    assert response.status_code == 200
    contents = captured["body"]["request"]["contents"]
    marker = _fr_leaf(contents, 1)
    # Compressed leaf SHIPPED despite the text-pipeline revert.
    assert marker.startswith(_FR_CCR_MARKER_PREFIX)
    # Text entry preserved (revert kept original text).
    assert contents[0]["parts"][0]["text"] == _REPEAT_UNIT
    # Saving counted even though the text pipeline reverted (704->718 case).
    assert captured["tokens_saved"] > 0
    # Byte-recoverable.
    entry = get_compression_store().retrieve(_hash_of(marker))
    assert entry is not None and entry.original_content == BIG_LEAF

    reset_compression_store()

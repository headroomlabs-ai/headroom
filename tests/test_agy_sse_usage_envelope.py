"""agy / Cloud Code Assist SSE usage: unwrap the response envelope (headroom-sit).

Cloud Code Assist wraps streaming chunks in a ``response`` envelope
(``{"response": {"usageMetadata": {...}}}``), mirroring the request-side wrap
(gemini.py ``body.get("request")``). Both gemini SSE usage parsers read
``usageMetadata`` at the top level only, so agy's ``candidatesTokenCount``
(output tokens) never parsed and every turn fell back to a bytes//40 estimate
(PR #1044 symptom (b): "Could not parse output_tokens from SSE, estimating ...").
Native-Gemini (top-level ``usageMetadata``) must keep working.
"""

import json

from headroom.proxy.server import HeadroomProxy


def _proxy() -> HeadroomProxy:
    # The gemini branch of both parsers is pure JSON parsing — no proxy
    # dependencies are touched, so a bare instance is sufficient.
    return object.__new__(HeadroomProxy)


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


# -- _parse_sse_usage_from_buffer (buffered path, primary agy streaming path) --


def test_buffer_gemini_unwraps_cloudcode_response_envelope():
    chunk = {
        "response": {
            "usageMetadata": {
                "promptTokenCount": 1234,
                "candidatesTokenCount": 567,
                "cachedContentTokenCount": 89,
            }
        }
    }
    state = {"sse_buffer": bytearray(_sse(chunk))}
    usage = _proxy()._parse_sse_usage_from_buffer(state, "gemini")
    assert usage == {
        "input_tokens": 1234,
        "output_tokens": 567,
        "cache_read_input_tokens": 89,
    }


def test_buffer_gemini_native_top_level_still_parses():
    chunk = {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20}}
    state = {"sse_buffer": bytearray(_sse(chunk))}
    usage = _proxy()._parse_sse_usage_from_buffer(state, "gemini")
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 20


# -- _parse_sse_usage (raw-chunk path) --


def test_chunk_gemini_unwraps_cloudcode_response_envelope():
    chunk = {"response": {"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7}}}
    usage = _proxy()._parse_sse_usage(_sse(chunk), "gemini")
    assert usage is not None
    assert usage["output_tokens"] == 7
    assert usage["input_tokens"] == 5


def test_chunk_gemini_native_top_level_still_parses():
    chunk = {"usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4}}
    usage = _proxy()._parse_sse_usage(_sse(chunk), "gemini")
    assert usage is not None
    assert usage["output_tokens"] == 4


# -- _gemini_usage_meta helper: guard malformed upstream (never crash) --


def test_gemini_usage_meta_guards_non_dict_metadata():
    m = HeadroomProxy._gemini_usage_meta
    # A truthy non-dict usageMetadata must NOT reach .get() and crash the parser.
    assert m({"usageMetadata": "garbage"}) is None
    assert m({"usageMetadata": [1, 2]}) is None
    assert m({"usageMetadata": 42}) is None
    assert m({"response": {"usageMetadata": "x"}}) is None
    assert m({"response": "notadict"}) is None
    assert m({}) is None
    # Well-formed top-level and enveloped both resolve to the inner dict.
    assert m({"usageMetadata": {"candidatesTokenCount": 7}}) == {"candidatesTokenCount": 7}
    assert m({"response": {"usageMetadata": {"candidatesTokenCount": 7}}}) == {
        "candidatesTokenCount": 7
    }
    # Empty-but-present dict passes through; callers skip it on falsy (no zero-overwrite).
    assert m({"usageMetadata": {}}) == {}

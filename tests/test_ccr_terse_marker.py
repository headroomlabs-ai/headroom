"""Terse CCR marker ``[ccr:<id>]`` + variable-width id detection.

With HEADROOM_CCR_TERSE_MARKER on, producers collapse the ~15-token verbose
boilerplate to ``[ccr:<id>]`` (~4 tokens) and the retrieval instruction moves
into the injected tool description. Detection (parser, compression_units,
tool_injection) also learns variable-width hex ids (2-24) — which is what makes
the adaptive short-label feature actually retrievable end-to-end.
"""

from __future__ import annotations

import json

import pytest

from headroom.cache.backends import InMemoryBackend
from headroom.cache.compression_store import (
    CompressionStore,
    get_compression_store,
    reset_compression_store,
)
from headroom.cache.label_allocator import DEFAULT_MIN_WIDTH
from headroom.ccr.marker import (
    CCR_HASH_FULL_WIDTH,
    CCR_TERSE_MARKER_RE,
    is_ccr_hex_id,
    terse_marker,
    terse_markers_enabled,
)
from headroom.ccr.response_handler import CCRResponseHandler
from headroom.ccr.tool_injection import CCR_TOOL_NAME, CCRToolInjector, parse_tool_call
from headroom.parser import CCR_RETRIEVAL_MARKER_RE
from headroom.transforms.compression_units import _CCR_MARKER_RE


def test_terse_marker_format():
    assert terse_marker("f2") == "[ccr:f2]"


def test_terse_markers_enabled_env(monkeypatch):
    monkeypatch.delenv("HEADROOM_CCR_TERSE_MARKER", raising=False)
    assert terse_markers_enabled() is False
    monkeypatch.setenv("HEADROOM_CCR_TERSE_MARKER", "on")
    assert terse_markers_enabled() is True


def test_terse_regex_extracts_id():
    m = CCR_TERSE_MARKER_RE.search("tool output [ccr:a3f] trailing")
    assert m is not None
    assert m.group(1) == "a3f"


def test_both_detection_regexes_match_terse():
    text = "some output\n[ccr:a3f]"
    assert CCR_RETRIEVAL_MARKER_RE.search(text)  # parser (reversibility guard, etc.)
    assert _CCR_MARKER_RE.search(text)  # compression_units


def test_tool_injection_detects_terse_marker():
    inj = CCRToolInjector()
    ids = inj.scan_for_markers([{"role": "tool", "content": "result body\n[ccr:a3f]"}])
    assert "a3f" in ids
    assert inj.has_compressed_content


def test_tool_injection_detects_short_hash_marker():
    # Short-label markers (hash=f2) must be detected — not only the legacy 24-hex,
    # or the retrieve tool never gets injected and retrieval breaks end-to-end.
    inj = CCRToolInjector()
    ids = inj.scan_for_markers(
        [{"role": "tool", "content": "[40 items compressed to 8. Retrieve more: hash=f2]"}]
    )
    assert "f2" in ids


def test_tool_injection_still_detects_full_hash_marker():
    inj = CCRToolInjector()
    full = "a" * 24
    ids = inj.scan_for_markers(
        [{"role": "tool", "content": f"[40 items compressed to 8. Retrieve more: hash={full}]"}]
    )
    assert full in ids


def test_terse_id_resolves_via_store_end_to_end():
    # store short label -> producer emits terse marker -> tool_injection extracts
    # the id -> store.retrieve resolves it. The whole point, wired.
    store = CompressionStore(backend=InMemoryBackend(), short_labels=True)
    content = "big tool output " * 40
    label = store.store(content, "compressed")
    marker = terse_marker(label)
    inj = CCRToolInjector()
    ids = inj.scan_for_markers([{"role": "tool", "content": f"body {marker}"}])
    assert label in ids  # the id the model would pass to headroom_retrieve
    entry = store.retrieve(label)
    assert entry is not None and entry.original_content == content


# --------------------------------------------------------------------------- #
# Regression: short adaptive labels must survive the RESPONSE-HANDLER parse path
# (parse_tool_call), not just the marker scan. The scan told the model it could
# retrieve `[ccr:f2]`; parse_tool_call then rejected hash="f2" as malformed (its
# width check was hardcoded to 12/24), so CCRResponseHandler classified the call
# as non-CCR and retrieval never ran. These lock the emit<->parse symmetry.
# --------------------------------------------------------------------------- #


def _call(provider: str, hash_key: str) -> dict:
    """A headroom_retrieve tool call in the given provider's shape."""
    if provider == "openai":
        return {"function": {"name": CCR_TOOL_NAME, "arguments": json.dumps({"hash": hash_key})}}
    return {"name": CCR_TOOL_NAME, "input": {"hash": hash_key}}


def _response(provider: str, hash_key: str) -> dict:
    """A full model response whose only tool call is headroom_retrieve(hash=...)."""
    if provider == "openai":
        return {
            "choices": [
                {"message": {"role": "assistant", "tool_calls": [_call(provider, hash_key)]}}
            ]
        }
    return {
        "content": [
            {"type": "text", "text": "retrieving"},
            {"type": "tool_use", "id": "tu_1", **_call(provider, hash_key)},
        ]
    }


@pytest.mark.parametrize("width", range(DEFAULT_MIN_WIDTH, CCR_HASH_FULL_WIDTH + 1))
def test_is_ccr_hex_id_accepts_every_valid_width(width):
    assert is_ccr_hex_id("a" * width)


@pytest.mark.parametrize(
    "bad",
    ["", "a", "a" * (CCR_HASH_FULL_WIDTH + 1), "zz", "g2", "ab!", 42, None, b"f2"],
)
def test_is_ccr_hex_id_rejects_invalid(bad):
    assert not is_ccr_hex_id(bad)


def test_is_ccr_hex_id_case_insensitive():
    assert is_ccr_hex_id("AB") and is_ccr_hex_id("F2")


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("label", ["f2", "abc", "a" * 12, "a" * 24])
def test_parse_tool_call_accepts_short_and_full_labels(provider, label):
    assert parse_tool_call(_call(provider, label), provider) == label


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("bad", ["a", "a" * 25, "zz", "g1"])
def test_parse_tool_call_rejects_malformed_ids(provider, bad):
    assert parse_tool_call(_call(provider, bad), provider) is None


def test_parse_tool_call_ignores_non_ccr_tool():
    assert parse_tool_call({"name": "grep", "input": {"hash": "f2"}}, "anthropic") is None


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_short_label_reaches_response_handler_as_ccr_call(provider):
    # The exact bug: a short-label retrieve call must be classified as CCR (so it
    # dispatches to retrieval), not shunted into `other_calls`.
    handler = CCRResponseHandler()
    ccr_calls, other = handler._parse_ccr_tool_calls(_response(provider, "f2"), provider)
    assert [c.hash_key for c in ccr_calls] == ["f2"]
    assert other == []


@pytest.fixture
def short_label_global_store(monkeypatch):
    """Global CCR store with short labels on + in-memory backend; reset around the test."""
    monkeypatch.setenv("HEADROOM_CCR_SHORT_LABELS", "1")
    reset_compression_store()
    store = get_compression_store(backend=InMemoryBackend())
    try:
        yield store
    finally:
        reset_compression_store()


def test_store_issued_short_labels_all_parse(short_label_global_store):
    # emit<->parse symmetry: every id the store hands out is one parse_tool_call accepts.
    for i in range(30):
        label = short_label_global_store.store(f"distinct block {i} " * 12, "compressed")
        for provider in ("anthropic", "openai"):
            assert parse_tool_call(_call(provider, label), provider) == label, (label, provider)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_short_label_full_retrieval_end_to_end(short_label_global_store, provider):
    # store short label -> model calls headroom_retrieve(hash=<short>) -> handler
    # classifies it as CCR -> _execute_retrieval returns the ORIGINAL content.
    content = "the original tool output that was compressed away " * 20
    label = short_label_global_store.store(content, "compressed")
    assert len(label) < CCR_HASH_FULL_WIDTH  # genuinely a short label, not the full hash

    handler = CCRResponseHandler()
    ccr_calls, other = handler._parse_ccr_tool_calls(_response(provider, label), provider)
    assert len(ccr_calls) == 1 and other == []

    result = handler._execute_retrieval(ccr_calls[0])
    assert result.success
    assert content in result.content

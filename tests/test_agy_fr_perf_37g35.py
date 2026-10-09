"""headroom-37g.35: instrumentation proving the per-leaf double-work is gone.

Byte-parity alone does not prove the redundant work was eliminated -- a
regression that silently reintroduces a duplicate ``count_text`` or
``default_ccr_hash`` call would still pass every existing behavioral test.
These tests spy on the real call counts instead.

Item A: ``_compress_fr_leaf`` used to recompute (a) the marker's own token
cost (identical to the per-request floor calculation) and (b) the leaf's
own token count (identical to what ``_walk_fr_compress`` already computed),
and ``store.store()`` used to re-derive SHA-256(leaf)[:24] internally even
though ``_walk_fr_compress`` already computed it for the retrieve-hash
exemption check. All three are now computed exactly once and threaded
through.

Item D: ``_collect_retrieved_hashes`` used to serialize a functionCall's
``args`` via ``json.dumps`` just to substring-search it for
``"headroom_retrieve"``. That is replaced with a direct recursive scan
(``_args_mention_retrieve``) -- ``json.dumps`` must not be called at all.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from headroom.cache.compression_store import (
    default_ccr_hash,
    get_compression_store,
    reset_compression_store,
)
from headroom.tokenizers import get_tokenizer
from headroom.transforms import agy_fr_compressor
from headroom.transforms.agy_fr_compressor import (
    _FR_CCR_MARKER_TEMPLATE,
    _collect_retrieved_hashes,
    compress_function_response_leaves,
)

_MODEL = "gemini-3-flash-agent"

# Large, single-line, non-repeating-line leaf: well above the marker-derived
# floor (~2x a ~20-token marker), so it is compressed exactly once.
_COMPRESSIBLE_LEAF = "search result row alpha beta gamma delta epsilon zeta eta " * 40

_MARKER_PLACEHOLDER = _FR_CCR_MARKER_TEMPLATE.format(hash="0" * 24)


@pytest.fixture
def tok() -> Any:
    return get_tokenizer(_MODEL)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    reset_compression_store()
    s = get_compression_store()
    yield s
    reset_compression_store()


def _contents_one_compressible_leaf() -> list[dict]:
    return [
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "search",
                        "response": {"output": _COMPRESSIBLE_LEAF},
                    }
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Item A: count_text call count.
# ---------------------------------------------------------------------------
def test_count_text_called_exactly_once_per_leaf_plus_one_per_request(tok: Any, store: Any) -> None:
    """Pin the exact ``count_text`` call count for ONE compressed leaf.

    Expected calls (3 total, not the pre-fix 5):
      1. ``_fr_marker_tokens_and_floor`` -- ONE per-request call on the
         placeholder marker (``hash="0"*24``), shared for both the floor
         and ``marker_body_tokens`` (no longer recomputed inside
         ``_compress_fr_leaf``).
      2. ``_walk_fr_compress`` -- ONE call on the original leaf text
         (``leaf_tokens``), no longer recomputed a second time as
         ``original_tokens`` inside ``store.store()``.
      3. ``_walk_fr_compress`` -- ONE call on the resulting marker text
         (real hash digits, not the placeholder) to compute the actual
         ``stats["after"]`` token count. This one is NOT eliminated by
         37g.35 -- it counts a different string than call 1 (real hash
         vs. placeholder hash) and is required for accurate before/after
         stats.
    """
    with patch.object(tok, "count_text", wraps=tok.count_text) as spy:
        before, after, leaves = compress_function_response_leaves(
            _contents_one_compressible_leaf(), "ccr", tok, store
        )

    assert leaves == 1
    assert before > after > 0

    calls = [call.args[0] for call in spy.call_args_list]
    assert len(calls) == 3, f"expected exactly 3 count_text calls, got {len(calls)}: {calls}"

    # The placeholder marker is counted exactly once (the old code counted
    # it a second time inside _compress_fr_leaf -- that recompute is gone).
    assert calls.count(_MARKER_PLACEHOLDER) == 1
    # The original leaf text is counted exactly once (the old code counted
    # it a second time as store.store()'s `original_tokens` arg -- gone).
    assert calls.count(_COMPRESSIBLE_LEAF) == 1
    # The remaining call is the actual (real-hash) marker text, required
    # for stats and NOT part of the eliminated double-work.
    remaining = [c for c in calls if c not in (_MARKER_PLACEHOLDER, _COMPRESSIBLE_LEAF)]
    assert len(remaining) == 1
    assert remaining[0].startswith("[functionResponse compressed.")


# ---------------------------------------------------------------------------
# Item A: default_ccr_hash call count.
# ---------------------------------------------------------------------------
def test_default_ccr_hash_called_exactly_once_per_leaf(tok: Any, store: Any) -> None:
    """``default_ccr_hash`` must be called exactly once for one compressed
    leaf: once in ``_walk_fr_compress`` for the exemption check, and that
    SAME value is threaded into ``store.store(..., explicit_hash=...)``.

    Scope note: this spy patches the compressor module's ``default_ccr_hash``
    binding, so it proves the compressor computes the hash once (not twice).
    That the store then SKIPS its own internal recompute when ``explicit_hash``
    is passed is a separate fact, verified by inspection of
    ``compression_store.store`` (the ``explicit_hash is not None`` branch skips
    ``default_ccr_hash(original)``), not asserted by this spy.
    """
    with patch.object(agy_fr_compressor, "default_ccr_hash", wraps=default_ccr_hash) as hash_spy:
        before, after, leaves = compress_function_response_leaves(
            _contents_one_compressible_leaf(), "ccr", tok, store
        )

    assert leaves == 1
    assert hash_spy.call_count == 1
    assert hash_spy.call_args.args[0] == _COMPRESSIBLE_LEAF


# ---------------------------------------------------------------------------
# Item A: explicit_hash produces byte-identical store keys / markers.
# ---------------------------------------------------------------------------
def test_explicit_hash_matches_implicit_default_hash(tok: Any, store: Any) -> None:
    """The threaded ``explicit_hash`` must yield the SAME store key as the
    old implicit default (``default_ccr_hash(original)``) -- otherwise
    ``/v1/retrieve/{hash}`` would 404 for previously-cached content."""
    contents = _contents_one_compressible_leaf()
    compress_function_response_leaves(contents, "ccr", tok, store)

    marker = contents[0]["parts"][0]["functionResponse"]["response"]["output"]
    expected_hash = default_ccr_hash(_COMPRESSIBLE_LEAF)
    assert marker.endswith(f"hash={expected_hash}]")

    entry = store.retrieve(expected_hash)
    assert entry is not None
    assert entry.original_content == _COMPRESSIBLE_LEAF


# ---------------------------------------------------------------------------
# Item D: json.dumps must not be called by _collect_retrieved_hashes.
# ---------------------------------------------------------------------------
def _mcp_retrieve_call_entry(hash_value: str) -> dict:
    """Generic MCP dispatch shape: a ``call_mcp_tool`` functionCall whose
    args reference ``headroom_retrieve`` (as a VALUE, not a key) and carry
    the target hash -- the case the old ``json.dumps(args)`` substring scan
    was covering."""
    return {
        "role": "model",
        "parts": [
            {
                "functionCall": {
                    "name": "call_mcp_tool",
                    "args": {"tool": "headroom_retrieve", "arguments": {"hash": hash_value}},
                }
            }
        ],
    }


def test_collect_retrieved_hashes_never_calls_json_dumps() -> None:
    h = default_ccr_hash(_COMPRESSIBLE_LEAF)
    contents = [_mcp_retrieve_call_entry(h)]

    with patch("json.dumps", side_effect=AssertionError("json.dumps must not be called")) as spy:
        hashes = _collect_retrieved_hashes(contents)

    assert spy.call_count == 0
    assert h in hashes


def test_collect_retrieved_hashes_still_finds_bare_retrieve_call() -> None:
    """Sanity check the replacement scan is not merely absent-of-crash --
    it must still find hashes via the bare ``headroom_retrieve`` name path
    (which never touched ``json.dumps`` even before this change)."""
    h = default_ccr_hash(_COMPRESSIBLE_LEAF)
    contents = [
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "headroom_retrieve", "args": {"hash": h}}}],
        }
    ]

    with patch("json.dumps", side_effect=AssertionError("json.dumps must not be called")):
        hashes = _collect_retrieved_hashes(contents)

    assert h in hashes

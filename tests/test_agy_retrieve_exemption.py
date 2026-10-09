"""WU2-A (headroom-37g.17): agy cold-original retrieve-hash exemption.

agy resends full history with ORIGINAL tool outputs every turn (it never
rewrites local history to hold headroom's markers). The agy FR compressor
(``_compress_agy_function_responses``) therefore re-compresses the resent
cold original into the SAME marker every turn -- but a model that already
retrieved that hash via ``headroom_retrieve`` this turn should not be forced
to re-retrieve it again (observed 236x thrash without this exemption).

At parity with the Rust path (``live_zone.rs:2362-2384``, which exempts by
call_id), this exemption keys on the retrieved HASH itself -- agy has no
call_id: a functionResponse string leaf is exempt from compression iff a
``headroom_retrieve`` call for its default CCR hash appears ANYWHERE in the
same request's ``contents`` (any entry, historical or tail).

Scope:
1. cold-original leaf exempt when call_mcp_tool-shaped functionCall args
   reference headroom_retrieve + the leaf's hash.
2. same, via a bare ``headroom_retrieve`` functionCall.
3. ordering-independent: retrieve call in a LATER contents[] entry than the
   leaf still exempts it (proves the pre-scan runs before compression).
4. case-insensitive: an uppercased hash in args still exempts.
5. no over-exemption: a leaf whose hash is NOT retrieved is compressed.
6. convergence: running the compressor twice leaves the exempt leaf stable.
7. multi-leaf: only the leaf matching a retrieved hash is exempt; sibling
   leaves are still compressed.
8. ``default_ccr_hash`` is the single source of truth shared with
   ``CompressionStore.store``'s default hash, and matches ``_FR_CCR_HASH_LEN``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from headroom.cache.compression_store import (
    default_ccr_hash,
    get_compression_store,
    reset_compression_store,
)
from headroom.proxy.handlers.gemini import _FR_CCR_HASH_LEN, _FR_CCR_MARKER_PREFIX
from headroom.proxy.server import ProxyConfig, create_app
from headroom.tokenizers import get_tokenizer

_MODEL = "gemini-3-flash-agent"

# Distinct large, single-line strings (no repeated lines, so lossless would be
# a no-op) -- well above the marker-derived compression floor.
_LEAF_A = "search result row alpha beta gamma delta epsilon zeta eta " * 40
_LEAF_B = "search result row omega psi chi phi upsilon tau sigma rho " * 40
_LEAF_C = "search result row kappa iota theta eta zeta epsilon delta " * 40


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


def _fr_entry(leaf: Any, role: str = "user", name: str = "search") -> dict:
    return {
        "role": role,
        "parts": [{"functionResponse": {"name": name, "response": {"output": leaf}}}],
    }


def _fr_leaf(contents: list, entry: int, part: int = 0, key: str = "output") -> Any:
    return contents[entry]["parts"][part]["functionResponse"]["response"][key]


def _mcp_retrieve_call_entry(hash_value: str, role: str = "model") -> dict:
    """Generic MCP dispatch shape: a ``call_mcp_tool`` functionCall whose args
    reference ``headroom_retrieve`` and carry the target hash."""
    return {
        "role": role,
        "parts": [
            {
                "functionCall": {
                    "name": "call_mcp_tool",
                    "args": {"tool": "headroom_retrieve", "arguments": {"hash": hash_value}},
                }
            }
        ],
    }


def _bare_retrieve_call_entry(hash_value: str, role: str = "model") -> dict:
    return {
        "role": role,
        "parts": [{"functionCall": {"name": "headroom_retrieve", "args": {"hash": hash_value}}}],
    }


# ---------------------------------------------------------------------------
# 1. call_mcp_tool-shaped retrieve call exempts the matching cold leaf.
# ---------------------------------------------------------------------------
def test_mcp_dispatch_retrieve_call_exempts_matching_leaf(
    proxy: Any, tok: Any, ccr_store: Any
) -> None:
    h = default_ccr_hash(_LEAF_A)
    contents = [_mcp_retrieve_call_entry(h), _fr_entry(_LEAF_A)]

    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)

    assert leaves == 0
    assert _fr_leaf(contents, 1) == _LEAF_A
    assert not _fr_leaf(contents, 1).startswith(_FR_CCR_MARKER_PREFIX)


# ---------------------------------------------------------------------------
# 2. bare headroom_retrieve functionCall exempts the matching cold leaf.
# ---------------------------------------------------------------------------
def test_bare_retrieve_call_exempts_matching_leaf(proxy: Any, tok: Any, ccr_store: Any) -> None:
    h = default_ccr_hash(_LEAF_A)
    contents = [_bare_retrieve_call_entry(h), _fr_entry(_LEAF_A)]

    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)

    assert leaves == 0
    assert _fr_leaf(contents, 1) == _LEAF_A


# ---------------------------------------------------------------------------
# 3. Ordering: retrieve call AFTER the leaf still exempts it (pre-scan).
# ---------------------------------------------------------------------------
def test_retrieve_call_after_leaf_still_exempts(proxy: Any, tok: Any, ccr_store: Any) -> None:
    h = default_ccr_hash(_LEAF_A)
    contents = [_fr_entry(_LEAF_A), _mcp_retrieve_call_entry(h)]

    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)

    assert leaves == 0
    assert _fr_leaf(contents, 0) == _LEAF_A


# ---------------------------------------------------------------------------
# 4. Case-insensitivity: an uppercased hash in args still exempts.
# ---------------------------------------------------------------------------
def test_uppercased_hash_in_args_still_exempts(proxy: Any, tok: Any, ccr_store: Any) -> None:
    h = default_ccr_hash(_LEAF_A).upper()
    contents = [_mcp_retrieve_call_entry(h), _fr_entry(_LEAF_A)]

    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)

    assert leaves == 0
    assert _fr_leaf(contents, 1) == _LEAF_A


# ---------------------------------------------------------------------------
# 5. No over-exemption: a leaf whose hash is NOT retrieved is compressed.
# ---------------------------------------------------------------------------
def test_unrelated_retrieve_hash_does_not_exempt(proxy: Any, tok: Any, ccr_store: Any) -> None:
    unrelated_hash = default_ccr_hash("something else entirely")
    contents = [_mcp_retrieve_call_entry(unrelated_hash), _fr_entry(_LEAF_A)]

    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)

    assert leaves == 1
    assert _fr_leaf(contents, 1).startswith(_FR_CCR_MARKER_PREFIX)


# ---------------------------------------------------------------------------
# 6. Convergence: f(f(x)) == f(x) for the exempt leaf.
# ---------------------------------------------------------------------------
def test_convergence_exempt_leaf_stable_across_runs(proxy: Any, tok: Any, ccr_store: Any) -> None:
    h = default_ccr_hash(_LEAF_A)
    contents = [_mcp_retrieve_call_entry(h), _fr_entry(_LEAF_A)]

    proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)
    first = _fr_leaf(contents, 1)
    assert first == _LEAF_A

    proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)
    second = _fr_leaf(contents, 1)
    assert second == first == _LEAF_A


# ---------------------------------------------------------------------------
# 7. Multi-leaf: only the leaf matching a retrieved hash is exempt.
# ---------------------------------------------------------------------------
def test_multi_leaf_only_matching_hash_exempt(proxy: Any, tok: Any, ccr_store: Any) -> None:
    h_b = default_ccr_hash(_LEAF_B)
    contents = [
        _mcp_retrieve_call_entry(h_b),
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "search",
                        "response": {
                            "leaf_a": _LEAF_A,
                            "nested": {"leaf_b": _LEAF_B, "list": [_LEAF_C]},
                        },
                    }
                }
            ],
        },
    ]

    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, ccr_store)

    response = contents[1]["parts"][0]["functionResponse"]["response"]
    assert leaves == 2  # leaf_a and leaf_c compressed; leaf_b exempt
    assert response["leaf_a"].startswith(_FR_CCR_MARKER_PREFIX)
    assert response["nested"]["list"][0].startswith(_FR_CCR_MARKER_PREFIX)
    assert response["nested"]["leaf_b"] == _LEAF_B


# ---------------------------------------------------------------------------
# 8. default_ccr_hash is the single source of truth shared with the store.
# ---------------------------------------------------------------------------
def test_default_ccr_hash_matches_store_and_marker_length(ccr_store: Any) -> None:
    store_hash = ccr_store.store(
        _LEAF_A,
        "compressed-placeholder",
        original_tokens=1,
        compressed_tokens=1,
        tool_name="x",
    )
    assert default_ccr_hash(_LEAF_A) == store_hash
    assert len(default_ccr_hash(_LEAF_A)) == _FR_CCR_HASH_LEN

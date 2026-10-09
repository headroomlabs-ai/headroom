"""headroom-vb5: deterministic proof that the CCR retrieve MECHANISM

resolves a ccr-compressed functionResponse marker's hash back to the
byte-identical original leaf.

Scope: this module proves the RECOVERY MECHANISM only --
1. WU1's ccr compressor (``HeadroomProxy._compress_agy_function_responses``)
   really does replace a functionResponse leaf with a self-describing
   ``headroom_retrieve`` marker (single-hash, ``Retrieve more: hash=...``
   form), and the original bytes are gone from the shipped payload.
2. ``CompressionStore.retrieve(hash)`` resolves that hash back to the
   byte-identical original via TWO independent paths that mirror the real
   ``headroom mcp serve`` child:
   a) PRIMARY -- two ``SQLiteBackend`` handles opened on the SAME on-disk
      ccr_store.db file (the child's local-resolution path, since proxy and
      child share one sqlite file).
   b) SECONDARY -- the ``POST /v1/retrieve`` HTTP endpoint that
      ``_retrieve_via_proxy`` (headroom/ccr/mcp_server.py) falls back to when
      local resolution misses (memory backend / workspace mismatch / sqlite
      init failure).
3. A bogus hash never produces a false recovery.

OUT OF SCOPE (explicitly, so the finding is owned and not dropped): whether
the MODEL actually chooses to *emit* a ``headroom_retrieve`` tool call when
it sees a marker in context ("0 retrieve calls" in the live trial) is MODEL
BEHAVIOR, not a mechanism defect. That is owned by headroom-y4q.

Fully deterministic and hermetic: no live agy, no Cloud Code Assist network,
no ``:8787`` proxy. Only ``fastapi.testclient.TestClient`` (in-process ASGI)
and in-process ``CompressionStore``/``SQLiteBackend`` handles. An autouse
fixture isolates the process-global CCR store to a per-test tmp SQLite file
so no test ever touches the real ``~/.headroom`` workspace db.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from headroom.cache.backends import InMemoryBackend, SQLiteBackend
from headroom.cache.compression_store import (
    CompressionStore,
    get_compression_store,
    reset_compression_store,
)
from headroom.proxy.handlers.gemini import _FR_CCR_MARKER_PREFIX
from headroom.proxy.server import ProxyConfig, create_app
from headroom.tokenizers import get_tokenizer

_MODEL = "gemini-3-flash-agent"

# A unique needle so we can prove it is (a) absent from the shipped marker
# bytes and (b) present, byte-identically, in whatever retrieve() returns.
NEEDLE = "UNIQUE-NEEDLE-c9f3a7d1-92be-4e6a-8c31-roundtrip-marker"

# Large, single-line leaf (no repeated lines, so lossless compaction would be
# a no-op) well above the marker-derived compression floor -- mirrors the
# BIG_LEAF fixture in tests/test_agy_functionresponse_compression.py.
BIG_LEAF = "search result row alpha beta gamma delta epsilon zeta eta " * 40 + NEEDLE


def _fr_entry(role: str = "user", leaf: Any = BIG_LEAF, name: str = "search") -> dict:
    """Build a historical (non-tail) functionResponse contents[] entry."""
    return {
        "role": role,
        "parts": [{"functionResponse": {"name": name, "response": {"output": leaf}}}],
    }


def _fr_leaf(contents: list, entry: int = 0, part: int = 0, key: str = "output") -> Any:
    return contents[entry]["parts"][part]["functionResponse"]["response"][key]


def _hash_of(marker: str) -> str:
    """Extract the hash from a ``headroom_retrieve`` marker.

    Mirrors ``_hash_of`` in test_agy_functionresponse_compression.py:
    single-hash marker, in the ``Retrieve more: hash=<h>]`` form that
    ``parser.CCR_RETRIEVAL_MARKER_RE`` keys on.
    """
    assert marker.startswith(_FR_CCR_MARKER_PREFIX), marker
    return marker.split("hash=", 1)[1].rstrip("]")


@pytest.fixture(autouse=True)
def _isolate_global_compression_store(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Isolate the process-global CCR store for every test in this module.

    ``create_app()`` lazily calls ``get_compression_store()`` at startup
    (memory-tracker registration), which would otherwise bind the global
    singleton to the real ``workspace_dir()/ccr_store.db``. Point it at a
    per-test tmp file instead and reset the singleton around the test so no
    test touches real on-disk state or leaks into another test.
    """
    db_path = tmp_path_factory.mktemp("ccr") / "global_ccr.db"
    monkeypatch.setenv("HEADROOM_CCR_SQLITE_PATH", str(db_path))
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    reset_compression_store()
    yield
    reset_compression_store()


@pytest.fixture
def proxy() -> Any:
    """A HeadroomProxy instance exposing ``_compress_agy_function_responses``."""
    with TestClient(create_app(ProxyConfig(optimize=True))) as client:
        yield client.app.state.proxy  # type: ignore[attr-defined]


@pytest.fixture
def tok() -> Any:
    return get_tokenizer(_MODEL)


def test_marker_replaces_needle_in_shipped_bytes(proxy: Any, tok: Any) -> None:
    """WU1's ccr compressor ships a marker in place of the leaf; the NEEDLE
    must NOT be literally present anywhere in the shipped contents[] bytes."""
    store = CompressionStore(backend=InMemoryBackend())
    contents = [_fr_entry()]

    before, after, leaves = proxy._compress_agy_function_responses(contents, "ccr", tok, store)

    assert leaves == 1
    assert before > after
    marker = _fr_leaf(contents)
    assert marker.startswith(_FR_CCR_MARKER_PREFIX)
    assert marker != BIG_LEAF
    shipped_bytes = json.dumps(contents)
    assert NEEDLE not in shipped_bytes


def test_shared_sqlite_two_handles_resolve_byte_identical(
    proxy: Any, tok: Any, tmp_path: Path
) -> None:
    """PRIMARY: two independent SQLiteBackend handles opened on the SAME
    ccr_store.db file -- the real local-resolution path the ``headroom mcp
    serve`` child uses, since proxy and child open the same sqlite file in
    one interpreter. NOT a claim of OS-level cross-process resolution."""
    db_path = tmp_path / "shared_ccr.db"
    store_a = CompressionStore(backend=SQLiteBackend(db_path))
    contents = [_fr_entry()]

    proxy._compress_agy_function_responses(contents, "ccr", tok, store_a)
    marker = _fr_leaf(contents)
    hash_key = _hash_of(marker)

    # SECOND handle, independently opened on the SAME sqlite file.
    store_b = CompressionStore(backend=SQLiteBackend(db_path))
    entry = store_b.retrieve(hash_key)

    assert entry is not None
    assert entry.original_content == BIG_LEAF  # byte-identical
    assert NEEDLE in entry.original_content


def test_http_retrieve_fallback_byte_identical(proxy: Any, tok: Any) -> None:
    """SECONDARY: HTTP fallback via POST /v1/retrieve -- the path
    ``_retrieve_via_proxy`` (headroom/ccr/mcp_server.py) uses when local
    resolution misses. We force an empty LOCAL store (a fresh in-memory
    handle that never saw this hash, e.g. memory backend / workspace
    mismatch) and resolve via the proxy's HTTP endpoint instead, which is
    backed by the same process-global store the compressor wrote to."""
    contents = [_fr_entry()]
    proxy._compress_agy_function_responses(contents, "ccr", tok, get_compression_store())
    marker = _fr_leaf(contents)
    hash_key = _hash_of(marker)

    # Local store miss: a fresh, unrelated in-memory store never populated
    # with this hash. This is the condition that forces the HTTP fallback.
    empty_local_store = CompressionStore(backend=InMemoryBackend())
    assert empty_local_store.retrieve(hash_key) is None

    app = create_app(ProxyConfig(optimize=True))
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        response = client.post("/v1/retrieve", json={"hash": hash_key})

    assert response.status_code == 200
    body = response.json()
    assert body["original_content"] == BIG_LEAF  # byte-identical
    assert NEEDLE in body["original_content"]


def test_bogus_hash_returns_no_recovery(proxy: Any, tok: Any) -> None:
    """A bogus hash (well-formed hex, never stored) must not resolve --
    neither locally nor via the HTTP endpoint. No false recovery."""
    # Populate the store with something so the store is non-empty, then ask
    # for a hash that was never returned by store().
    contents = [_fr_entry()]
    proxy._compress_agy_function_responses(contents, "ccr", tok, get_compression_store())
    real_hash = _hash_of(_fr_leaf(contents))
    bogus_hash = "0" * len(real_hash)
    assert bogus_hash != real_hash

    assert get_compression_store().retrieve(bogus_hash) is None

    app = create_app(ProxyConfig(optimize=True))
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        response = client.post("/v1/retrieve", json={"hash": bogus_hash})

    assert response.status_code == 404

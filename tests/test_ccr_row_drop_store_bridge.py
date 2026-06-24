"""Issue #389: SmartCrusher row-drop CCR hash → Python compression_store bridge.

The row-drop and opaque-blob paths in the Rust SmartCrusher emit
``<<ccr:HASH ...>>`` markers and stash the original payload in the
Rust process-local CCR store. The Python proxy's ``/v1/retrieve``
endpoint queries the Python ``compression_store`` (not the Rust one),
so without a bridge every retrieve call for a Rust-emitted marker
returns 404.

These tests pin the bridge:

1. Unit: lossy crush of a 200-item array populates the Python
   compression_store keyed by the same hash that's in the marker, with
   the canonical original retrievable via ``store.retrieve(hash)``.

2. Integration: ``/v1/compress`` followed by ``/v1/retrieve/{hash}``
   on the same hash returns 200 with the original content. This is
   the failing case from the issue's reproducer.

3. Edge: opaque-blob markers (the ``<<ccr:HASH,KIND,SIZE>>`` shape)
   also bridge — the document walker emits these for long strings.

If these regress, ``/v1/retrieve`` silently 404s for every
SmartCrusher-emitted marker even though the data is held in the Rust
store. The LLM follows the marker, gets nothing, and the proxy's CCR
contract is broken at the bridge.
"""

from __future__ import annotations

import json

import pytest


def _build_extension() -> None:
    try:
        from headroom._core import SmartCrusher  # noqa: F401
    except ImportError:
        pytest.skip(
            "headroom._core not built — run `bash scripts/build_rust_extension.sh`",
            allow_module_level=True,
        )


_build_extension()


# Skip if fastapi not available — needed for integration tests but not unit.
try:
    import fastapi  # noqa: F401

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


# ─── Unit tests: shim populates Python store ─────────────────────────────


def test_lossy_crush_populates_python_compression_store() -> None:
    """The cornerstone unit test: trigger a row-drop via the Python
    SmartCrusher shim, then verify the Python compression_store has
    an entry keyed by the marker's hash with the canonical original.
    """
    from headroom.cache.compression_store import (
        get_compression_store,
        reset_compression_store,
    )
    from headroom.config import CCRConfig
    from headroom.config import SmartCrusherConfig as PyConfig
    from headroom.transforms.smart_crusher import SmartCrusher

    reset_compression_store()
    try:
        crusher = SmartCrusher(PyConfig(), ccr_config=CCRConfig(), with_compaction=False)

        # 60 items is well above adaptive_k → lossy path fires.
        original = [{"id": i, "status": "ok", "tag": "alpha"} for i in range(60)]
        original_json = json.dumps(original)

        result = crusher.crush_array_json(original_json)

        # Sanity: lossy path actually fired.
        assert result["ccr_hash"] is not None, (
            f"expected lossy drop, got strategy={result['strategy_info']!r}"
        )
        ccr_hash = result["ccr_hash"]
        # The marker text embeds the same hash.
        assert ccr_hash in result["dropped_summary"], (
            f"marker {result['dropped_summary']!r} should embed hash {ccr_hash}"
        )

        # The bridge: Python compression_store now holds an entry keyed
        # by the Rust-emitted hash.
        store = get_compression_store()
        entry = store.retrieve(ccr_hash)
        assert entry is not None, (
            f"compression_store has no entry for hash {ccr_hash!r}; "
            f"the bridge dropped the row-drop hash on the floor"
        )

        # The entry's original content is the canonical-JSON
        # serialization of the original array — same bytes the Rust
        # store has under the same hash.
        rust_canonical = crusher.ccr_get(ccr_hash)
        assert rust_canonical is not None, "Rust store lost the entry"
        assert entry.original_content == rust_canonical, (
            "Python store's canonical bytes diverged from Rust store"
        )

        # Round-trip: parse the canonical and compare to input.
        retrieved = json.loads(entry.original_content)
        assert retrieved == original
    finally:
        reset_compression_store()


def test_smart_crush_content_populates_python_store() -> None:
    """Same bridge but driven through the runtime path the proxy
    actually uses: `_smart_crush_content` (not `crush_array_json`).
    This is what `apply()` invokes per-message."""
    from headroom.cache.compression_store import (
        get_compression_store,
        reset_compression_store,
    )
    from headroom.config import CCRConfig
    from headroom.config import SmartCrusherConfig as PyConfig
    from headroom.transforms.smart_crusher import SmartCrusher

    reset_compression_store()
    try:
        crusher = SmartCrusher(PyConfig(), ccr_config=CCRConfig(), with_compaction=False)

        # Mix of "ok" + occasional "error" → variance the lossy path
        # can latch onto. A purely-uniform array gets a `skip:unique_
        # entities_no_signal` strategy and never row-drops.
        original = [
            {
                "id": i,
                "level": "error" if i % 30 == 0 else "info",
                "msg": f"line {i}",
            }
            for i in range(80)
        ]
        content = json.dumps(original)

        crushed, was_modified, info = crusher._smart_crush_content(content)
        assert was_modified, f"expected lossy modification, got info={info!r}"
        assert "<<ccr:" in crushed, (
            f"expected CCR marker in output (info={info!r}): {crushed[:200]!r}"
        )

        # Pull the hash from the rendered output via JSON parse.
        # The output is a JSON array with a `_ccr_dropped` sentinel.
        parsed = json.loads(crushed)
        assert isinstance(parsed, list), f"expected JSON array, got {type(parsed).__name__}"
        sentinel = parsed[-1]
        assert isinstance(sentinel, dict) and "_ccr_dropped" in sentinel, (
            f"expected _ccr_dropped sentinel as last element, got {sentinel!r}"
        )
        marker_text = sentinel["_ccr_dropped"]
        assert marker_text.startswith("<<ccr:") and "rows_offloaded>>" in marker_text

        # Extract hash by structural slice (no regex).
        ccr_hash = marker_text[len("<<ccr:") :].split(" ", 1)[0]
        assert all(c in "0123456789abcdef" for c in ccr_hash), f"hash should be hex: {ccr_hash!r}"

        # The Python store now has the entry under this hash.
        store = get_compression_store()
        entry = store.retrieve(ccr_hash)
        assert entry is not None, (
            f"_smart_crush_content didn't bridge hash {ccr_hash!r} to "
            f"Python store; /v1/retrieve would 404"
        )

        # Original is recoverable byte-for-byte from the bridged entry.
        retrieved = json.loads(entry.original_content)
        assert retrieved == original
    finally:
        reset_compression_store()


def test_passthrough_does_not_populate_store() -> None:
    """Below adaptive_k → no row drop → no Python store write.
    Pins that we don't accidentally store on every crush call."""
    from headroom.cache.compression_store import (
        get_compression_store,
        reset_compression_store,
    )
    from headroom.config import CCRConfig
    from headroom.config import SmartCrusherConfig as PyConfig
    from headroom.transforms.smart_crusher import SmartCrusher

    reset_compression_store()
    try:
        crusher = SmartCrusher(PyConfig(), ccr_config=CCRConfig(), with_compaction=False)

        # 3 items: well below the threshold; no compression happens.
        small = json.dumps([{"id": i} for i in range(3)])
        crusher._smart_crush_content(small)

        store = get_compression_store()
        stats = store.get_stats()
        assert stats["entry_count"] == 0, (
            f"passthrough crush should not write to compression_store, "
            f"got {stats['entry_count']} entries"
        )
    finally:
        reset_compression_store()


def test_marker_disabled_skips_python_store() -> None:
    """`ccr_config.enabled=False` flips off marker emission AND store
    writes on the Rust side. The bridge should have nothing to do."""
    from headroom.cache.compression_store import (
        get_compression_store,
        reset_compression_store,
    )
    from headroom.config import CCRConfig
    from headroom.config import SmartCrusherConfig as PyConfig
    from headroom.transforms.smart_crusher import SmartCrusher

    reset_compression_store()
    try:
        # Markers disabled → Rust skips both marker emission and store write.
        crusher = SmartCrusher(
            PyConfig(),
            ccr_config=CCRConfig(enabled=False),
            with_compaction=False,
        )

        original = [{"id": i, "status": "ok"} for i in range(60)]
        crushed, was_modified, _info = crusher._smart_crush_content(json.dumps(original))

        # Compression still happens (rows still drop), but no marker.
        assert was_modified
        assert "<<ccr:" not in crushed, (
            f"expected no marker when ccr_config.enabled=False, got: {crushed[:200]!r}"
        )

        # And the Python store stays empty — bridge had nothing to mirror.
        store = get_compression_store()
        assert store.get_stats()["entry_count"] == 0
    finally:
        reset_compression_store()


def test_distinct_payloads_get_distinct_python_store_entries() -> None:
    """Two unrelated payloads → two row drops → two entries under
    distinct hashes in the Python store; both retrievable independently."""
    from headroom.cache.compression_store import (
        get_compression_store,
        reset_compression_store,
    )
    from headroom.config import CCRConfig
    from headroom.config import SmartCrusherConfig as PyConfig
    from headroom.transforms.smart_crusher import SmartCrusher

    reset_compression_store()
    try:
        crusher = SmartCrusher(PyConfig(), ccr_config=CCRConfig(), with_compaction=False)

        a = [{"id": i, "tag": "alpha"} for i in range(50)]
        b = [{"id": i, "tag": "beta"} for i in range(50)]

        ra = crusher.crush_array_json(json.dumps(a))
        rb = crusher.crush_array_json(json.dumps(b))

        ha, hb = ra["ccr_hash"], rb["ccr_hash"]
        assert ha and hb and ha != hb

        store = get_compression_store()
        ea = store.retrieve(ha)
        eb = store.retrieve(hb)
        assert ea is not None and eb is not None
        assert json.loads(ea.original_content) == a
        assert json.loads(eb.original_content) == b
    finally:
        reset_compression_store()


# ─── compression_store: explicit_hash parameter ───────────────────────────


def test_compression_store_explicit_hash_round_trips() -> None:
    """The new `explicit_hash` parameter on `store.store()` keys the
    entry by the caller-supplied hash instead of MD5(original)[:24]."""
    from headroom.cache.compression_store import (
        get_compression_store,
        reset_compression_store,
    )

    reset_compression_store()
    try:
        store = get_compression_store()
        # SmartCrusher emits 12-char SHA-256 hashes — much shorter than
        # the default MD5[:24].
        explicit = "abc123def456"
        returned = store.store(
            original='[{"id":1}]',
            compressed="<<placeholder>>",
            explicit_hash=explicit,
        )
        assert returned == explicit
        entry = store.retrieve(explicit)
        assert entry is not None
        assert entry.original_content == '[{"id":1}]'
    finally:
        reset_compression_store()


def test_compression_store_explicit_hash_rejects_non_hex() -> None:
    """Non-hex `explicit_hash` raises ValueError. No silent fallback
    to MD5 — that would re-introduce the marker/store mismatch."""
    from headroom.cache.compression_store import (
        get_compression_store,
        reset_compression_store,
    )

    reset_compression_store()
    try:
        store = get_compression_store()
        with pytest.raises(ValueError, match="hex string"):
            store.store(
                original="x",
                compressed="y",
                explicit_hash="NOT_HEX!@#",
            )
        with pytest.raises(ValueError, match="hex string"):
            store.store(
                original="x",
                compressed="y",
                explicit_hash="",
            )
    finally:
        reset_compression_store()


# ─── Integration test: /v1/compress → /v1/retrieve via FastAPI ──────────


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_v1_compress_then_v1_retrieve_resolves_marker_hash() -> None:
    """End-to-end issue #389 reproducer:
    1. POST a 200-item tool message to /v1/compress.
    2. Parse the `<<ccr:HASH N_rows_offloaded>>` marker out of the
       compressed messages.
    3. GET /v1/retrieve/{hash} — must return 200 with the original.

    Before the fix this returns 404 because the Rust crusher's marker
    points at a hash the Python compression_store never received.
    """
    from fastapi.testclient import TestClient

    from headroom.cache.compression_store import reset_compression_store
    from headroom.proxy.server import ProxyConfig, create_app

    reset_compression_store()
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    # Build a payload similar to the issue's reproducer — 200 items
    # with enough variation to trigger the lossy path. The Rust
    # crusher's adaptive_k will keep ~15 and drop the rest.
    #
    # The blob is unique-per-item and long relative to the key names so
    # the lossless Table/CSV path (which wins by stripping repeated keys
    # when it saves >= lossless_min_savings_ratio) cannot clear the bar —
    # this test exists to exercise the LOSSY row-drop path and its
    # Rust -> Python CCR store bridge.
    items = [
        {
            "id": i,
            "score": 0.99 if i % 30 == 0 else 0.6,
            "msg": f"Result {i:03d}{' error' if i % 30 == 0 else ' ok'}",
            "blob": f"payload-{i:04d}-" + "".join(chr(97 + (i * 7 + j) % 26) for j in range(240)),
        }
        for i in range(200)
    ]
    req = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Get items"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(items)},
        ],
    }

    try:
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
            resp = client.post("/v1/compress", json=req)
            assert resp.status_code == 200, resp.text
            body = resp.json()

            # The compressed messages should embed at least one CCR marker.
            messages_blob = json.dumps(body["messages"])
            assert "<<ccr:" in messages_blob, (
                f"no CCR marker after /v1/compress on a 200-item array; "
                f"compression didn't fire as expected: "
                f"transforms={body.get('transforms_applied')}"
            )

            # Pull the hash with a substring scan (no regex).
            start = messages_blob.find("<<ccr:") + len("<<ccr:")
            end = start
            while end < len(messages_blob) and messages_blob[end] in ("0123456789abcdef"):
                end += 1
            ccr_hash = messages_blob[start:end]
            assert ccr_hash, "couldn't extract hash from marker"

            # The retrieve stats endpoint should now show ≥ 1 entry.
            stats_resp = client.get("/v1/retrieve/stats")
            assert stats_resp.status_code == 200
            stats = stats_resp.json()
            assert stats["store"]["entry_count"] >= 1, (
                f"compression_store empty after /v1/compress; stats={stats!r}"
            )

            # The actual /v1/retrieve call from the issue:
            retrieve_resp = client.post("/v1/retrieve", json={"hash": ccr_hash})
            assert retrieve_resp.status_code == 200, (
                f"/v1/retrieve for marker hash {ccr_hash!r} returned "
                f"{retrieve_resp.status_code} ({retrieve_resp.text}). "
                f"The Rust→Python store bridge dropped the entry."
            )
            retrieve_body = retrieve_resp.json()
            assert retrieve_body["hash"] == ccr_hash
            # The retrieved content should parse back to a JSON array
            # of the original items.
            retrieved_items = json.loads(retrieve_body["original_content"])
            assert isinstance(retrieved_items, list)
            # The original was 200 items. The Rust hash is over the
            # canonical-JSON form of the parsed input — should round-trip.
            assert len(retrieved_items) == 200, (
                f"expected 200 items in retrieved content, got {len(retrieved_items)}"
            )
            # Spot-check the first item.
            assert retrieved_items[0]["id"] == 0

            # And the GET shape (used by some clients) returns the same.
            get_resp = client.get(f"/v1/retrieve/{ccr_hash}")
            assert get_resp.status_code == 200
            get_body = get_resp.json()
            assert get_body["hash"] == ccr_hash
    finally:
        reset_compression_store()


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_v1_retrieve_unknown_hash_still_404() -> None:
    """Sanity: unknown hashes still return 404 (the bridge doesn't
    accidentally make the store too permissive)."""
    from fastapi.testclient import TestClient

    from headroom.cache.compression_store import reset_compression_store
    from headroom.proxy.server import ProxyConfig, create_app

    reset_compression_store()
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    try:
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
            resp = client.post("/v1/retrieve", json={"hash": "deadbeef0000"})
            assert resp.status_code == 404
    finally:
        reset_compression_store()

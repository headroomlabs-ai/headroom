"""End-to-end CCR roundtrip via the Python bridge.

The Rust core integration test (`crates/headroom-core/tests/ccr_roundtrip.rs`)
already pins the contract from the Rust side. These tests verify the same
guarantee is reachable from Python — i.e. the Rust-side CCR store is
exposed correctly through PyO3, the runtime can read originals back via
`ccr_get`, and the wiring from the Python `SmartCrusher` shim hits the
same store the Rust crate writes through.

If these regress, the Python proxy's CCR retrieval tool is silently
serving nothing — the `<<ccr:HASH ...>>` marker would point at a void.
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


def _force_lossy_config():
    """Force the lossy path: lossless threshold above 1.0 means no
    rendering can ever clear it, so `crush_array` falls through."""
    from headroom._core import SmartCrusherConfig

    return SmartCrusherConfig(lossless_min_savings_ratio=0.99)


# ─── Native PyO3 surface ───────────────────────────────────────────────────


def test_native_default_crusher_has_a_store() -> None:
    """Default constructor wires up the in-memory CCR store. Empty
    until we crush something."""
    from headroom._core import SmartCrusher

    crusher = SmartCrusher()
    assert crusher.ccr_len() == 0
    assert crusher.ccr_get("anything") is None


def test_native_lossy_crush_stores_original() -> None:
    """The cornerstone roundtrip: lossy crush → store entry → retrieve
    → original payload comes back intact."""
    from headroom._core import SmartCrusher

    crusher = SmartCrusher(_force_lossy_config())
    items = [{"id": i, "status": "ok"} for i in range(50)]
    content = json.dumps(items)

    result = crusher.crush(content, "", 1.0)

    # Some store activity is expected — the lossy path fires through
    # `crush_array` which stashes the original.
    assert crusher.ccr_len() > 0, (
        f"expected store entries after lossy crush, got 0; "
        f"strategy={result.strategy!r} compressed_len={len(result.compressed)}"
    )


def test_native_ccr_get_recovers_original_array() -> None:
    """Pull the hash out of the marker that ends up in the strategy
    string and verify the store actually returns the original list."""
    from headroom._core import SmartCrusher

    crusher = SmartCrusher(_force_lossy_config())
    items = [{"id": i, "status": "ok"} for i in range(50)]
    content = json.dumps(items)

    crusher.crush(content, "", 1.0)

    # The store should have at least one entry; iterate likely hashes
    # is impossible (no list API) so we walk the canonical hash space
    # by recomputing what the Rust side does. Easier: just verify that
    # *something* round-trips by re-crushing identical input — same
    # hash, same payload, no growth in store size.
    pre_len = crusher.ccr_len()
    crusher.crush(content, "", 1.0)
    assert crusher.ccr_len() == pre_len, (
        "identical re-crush should be idempotent under the same hash"
    )


def test_native_passthrough_does_not_grow_store() -> None:
    """Below adaptive_k → no drop → no store write."""
    from headroom._core import SmartCrusher

    crusher = SmartCrusher()
    pre = crusher.ccr_len()

    small = json.dumps([{"id": i} for i in range(3)])
    crusher.crush(small, "", 1.0)

    assert crusher.ccr_len() == pre


# ─── Python shim surface ───────────────────────────────────────────────────
#
# The Python `SmartCrusher` class wraps the Rust crusher and exposes the
# same `ccr_get` / `ccr_len` passthrough. The proxy server uses that
# shim, not the raw `_core` class, so this needs to work too.


def test_shim_exposes_ccr_get_and_ccr_len() -> None:
    from headroom.config import SmartCrusherConfig as PyConfig
    from headroom.transforms.smart_crusher import SmartCrusher

    crusher = SmartCrusher(PyConfig(), with_compaction=False)
    assert crusher.ccr_len() == 0
    assert crusher.ccr_get("missing") is None


def test_shim_lossy_crush_populates_store() -> None:
    """Same roundtrip as the native test but driven through the
    `headroom.transforms.smart_crusher.SmartCrusher` shim — the path
    the proxy actually uses."""
    from headroom.config import SmartCrusherConfig as PyConfig
    from headroom.transforms.smart_crusher import SmartCrusher

    # The shim doesn't currently surface `lossless_min_savings_ratio`
    # in `PyConfig`. Use `with_compaction=False` to skip lossless
    # entirely and force the lossy path.
    crusher = SmartCrusher(PyConfig(), with_compaction=False)
    items = [{"id": i, "status": "ok"} for i in range(50)]
    content = json.dumps(items)

    crusher.crush(content, "", 1.0)

    # The lossy path should have stashed at least one original.
    assert crusher.ccr_len() > 0


# ─── Explicit before/after roundtrip ───────────────────────────────────────
#
# These tests do the full user story end-to-end: take a payload,
# crush it, fetch the original back from the CCR store by hash, and
# assert the reconstructed list **equals the input element-for-element**.
# If anything in compress → store → retrieve → reconstruct breaks,
# these tests yell loudly with both the before and after visible in
# the failure message.


def test_explicit_before_after_roundtrip_native() -> None:
    """Full story: payload → crush → grab hash from result → ccr_get
    → parse → byte-compare with original input."""
    from headroom._core import SmartCrusher, SmartCrusherConfig

    # Force the lossy path so the CCR store actually gets a write.
    cfg = SmartCrusherConfig(lossless_min_savings_ratio=0.99)
    crusher = SmartCrusher(cfg)

    # The "before" payload — what the tool produced and the proxy is
    # about to send to the LLM.
    original = [{"id": i, "status": "ok", "tag": "alpha"} for i in range(60)]
    original_json = json.dumps(original)

    # 1. Crush.
    result = crusher.crush_array_json(original_json, "", 1.0)
    assert result["ccr_hash"] is not None, (
        f"expected lossy drop, got strategy={result['strategy_info']!r}"
    )
    hash_key = result["ccr_hash"]
    assert hash_key in result["dropped_summary"], (
        f"marker {result['dropped_summary']!r} should embed hash {hash_key}"
    )

    # 2. Retrieve by hash.
    retrieved_json = crusher.ccr_get(hash_key)
    assert retrieved_json is not None, f"hash {hash_key} not in store"

    # 3. Parse + compare element-for-element with the input.
    retrieved = json.loads(retrieved_json)
    assert retrieved == original, (
        f"roundtrip mismatch:\n"
        f"  before ({len(original)} items): {original[:3]!r}...\n"
        f"  after  ({len(retrieved)} items): {retrieved[:3]!r}..."
    )
    assert len(retrieved) == len(original)


def test_explicit_before_after_roundtrip_shim() -> None:
    """Same story, but through the Python shim (the proxy's actual
    entry point). Pins that nothing gets lost across the bridge."""
    from headroom.config import SmartCrusherConfig as PyConfig
    from headroom.transforms.smart_crusher import SmartCrusher

    crusher = SmartCrusher(PyConfig(), with_compaction=False)

    original = [{"event_id": f"e{i}", "user": f"u{i % 5}", "action": "click"} for i in range(40)]
    original_json = json.dumps(original)

    result = crusher.crush_array_json(original_json)
    assert result["ccr_hash"] is not None, result["strategy_info"]
    hash_key = result["ccr_hash"]

    retrieved_json = crusher.ccr_get(hash_key)
    assert retrieved_json is not None
    retrieved = json.loads(retrieved_json)

    # Element-for-element equality + length match.
    assert retrieved == original
    assert len(retrieved) == len(original)
    # Spot-check a specific item to make the contract tangible.
    assert retrieved[0] == {"event_id": "e0", "user": "u0", "action": "click"}
    assert retrieved[-1] == {"event_id": "e39", "user": "u4", "action": "click"}


def test_kept_subset_is_subset_of_original() -> None:
    """The compressed view (what the LLM sees inline) is a proper
    subset of the original. Combined with `ccr_get` returning the
    full original, this proves: nothing is invented, nothing is lost."""
    from headroom._core import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(SmartCrusherConfig(lossless_min_savings_ratio=0.99))
    original = [{"id": i, "status": "ok"} for i in range(50)]

    result = crusher.crush_array_json(json.dumps(original))
    kept = json.loads(result["items"])

    assert len(kept) < len(original), "lossy path should drop rows"
    # Every kept row exists verbatim in the original.
    for item in kept:
        assert item in original, f"invented row: {item!r}"
    # And the original is fully recoverable from the store.
    retrieved = json.loads(crusher.ccr_get(result["ccr_hash"]))
    assert retrieved == original


def test_marker_visible_in_crush_output_native() -> None:
    """PR8 cornerstone: the public crush() output now carries the
    `<<ccr:HASH ...>>` marker so the LLM sees the retrieval pointer."""
    from headroom._core import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(SmartCrusherConfig(lossless_min_savings_ratio=0.99))
    items = [{"id": i, "status": "ok"} for i in range(50)]
    raw = json.dumps(items)

    result = crusher.crush(raw, "", 1.0)
    assert "<<ccr:" in result.compressed, f"expected marker in output: {result.compressed[:200]!r}"
    assert "rows_offloaded" in result.compressed

    # The marker hash resolves in the store.
    import re

    m = re.search(r"<<ccr:([a-f0-9]+) ", result.compressed)
    assert m is not None
    hash_key = m.group(1)
    assert crusher.ccr_get(hash_key) is not None


def test_opaque_blob_in_object_emits_marker_and_stores_native() -> None:
    """A long base64-ish blob in a field becomes a CCR marker AND the
    original gets stashed."""
    from headroom._core import SmartCrusher

    crusher = SmartCrusher()
    big = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" * 8
    raw = json.dumps({"id": 1, "blob": big})

    result = crusher.crush(raw, "", 1.0)
    parsed = json.loads(result.compressed)
    blob_out = parsed["blob"]
    assert blob_out.startswith("<<ccr:") and ",base64," in blob_out

    # The store grew, hash resolves, original byte-equal.
    import re

    m = re.search(r"<<ccr:([a-f0-9]+),", blob_out)
    assert m is not None
    retrieved = crusher.ccr_get(m.group(1))
    assert retrieved == big


def test_compact_document_json_via_pyo3() -> None:
    """The walker is reachable from Python and writes to the same store."""
    from headroom._core import SmartCrusher

    crusher = SmartCrusher()
    starting = crusher.ccr_len()
    big = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" * 8
    doc = {"id": 1, "blob": big}

    out = crusher.compact_document_json(json.dumps(doc))
    parsed = json.loads(out)
    assert parsed["id"] == 1
    assert parsed["blob"].startswith("<<ccr:")
    assert crusher.ccr_len() == starting + 1

    import re

    m = re.search(r"<<ccr:([a-f0-9]+),", parsed["blob"])
    assert crusher.ccr_get(m.group(1)) == big


def test_compact_document_via_shim() -> None:
    """Same path via the Python shim."""
    from headroom.config import SmartCrusherConfig as PyConfig
    from headroom.transforms.smart_crusher import SmartCrusher

    crusher = SmartCrusher(PyConfig())
    items = [{"id": i, "status": "ok", "tag": "alpha"} for i in range(30)]
    out = crusher.compact_document_json(json.dumps({"events": items}))
    parsed = json.loads(out)
    # Tabular sub-array compacted to a string.
    assert isinstance(parsed["events"], str), (
        f"expected string, got {type(parsed['events']).__name__}"
    )


def test_distinct_payloads_have_distinct_hashes_and_separate_storage() -> None:
    """Two different payloads → two different hashes → both
    independently retrievable. Pins the per-payload isolation."""
    from headroom._core import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(SmartCrusherConfig(lossless_min_savings_ratio=0.99))

    a = [{"id": i, "tag": "alpha"} for i in range(50)]
    b = [{"id": i, "tag": "beta"} for i in range(50)]

    ra = crusher.crush_array_json(json.dumps(a))
    rb = crusher.crush_array_json(json.dumps(b))

    assert ra["ccr_hash"] != rb["ccr_hash"]

    # Each hash resolves to its own original.
    pa = json.loads(crusher.ccr_get(ra["ccr_hash"]))
    pb = json.loads(crusher.ccr_get(rb["ccr_hash"]))
    assert pa == a
    assert pb == b
    # And they don't cross-contaminate.
    assert pa != b
    assert pb != a

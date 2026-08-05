"""TOIN re-attachment regression tests for the Rust-backed SmartCrusher.

When SmartCrusher's Python implementation was retired in Stage 3c.1b,
its inline `toin.record_compression()` call was lost. The Rust port
doesn't know about TOIN, and `ContentRouter._record_to_toin` skips
SmartCrusher on the assumption SmartCrusher records its own. The net
result was a silent regression: JSON-array compressions — the
highest-traffic strategy — stopped feeding the learning loop.

The shim now bridges the gap. These tests assert that:

1. Calling `SmartCrusher.crush(...)` on a JSON array of dicts produces a
   `record_compression` event in TOIN when the input is large enough to
   actually compress.
2. Pass-through inputs (no modification) do NOT record — TOIN only
   learns from real compression events.
3. The recorded `tool_signature.structure_hash` is computed over the
   parsed items, so two compressions of structurally-similar inputs
   land on the same pattern.
4. Calling `_smart_crush_content(...)` (the legacy `apply()` path) also
   records.
5. TOIN failures are non-fatal — compression completes even if the
   telemetry import or call raises.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from headroom.telemetry.toin import TOINConfig, get_toin, reset_toin
from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig


@pytest.fixture
def fresh_toin():
    reset_toin()
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = str(Path(tmpdir) / "toin.json")
        toin = get_toin(
            TOINConfig(
                storage_path=storage,
                auto_save_interval=0,
            )
        )
        yield toin
        reset_toin()


def _bigger_array(n: int = 60) -> str:
    """JSON array of `n` dicts, large enough to trigger crushing.

    Items use a low-uniqueness shape (`{"status": "ok", "tag": "x"}`)
    so the analyzer recommends compaction or row drops. We need at
    least 200 tokens (`min_tokens_to_crush` default) to enter the
    crusher; 60 items keeps us well above that.
    """
    items = [{"status": "ok", "tag": "x", "n": i} for i in range(n)]
    import json as _json

    return _json.dumps(items)


# ─── crush() path (router-style) ───────────────────────────────────────


def test_crush_records_to_toin_on_modification(fresh_toin):
    crusher = SmartCrusher(SmartCrusherConfig())
    payload = _bigger_array(60)
    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())

    result = crusher.crush(payload, query="test query", bias=1.0)

    if not result.was_modified:
        pytest.skip("payload didn't trigger compression — bump the size")
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    assert post > pre, "TOIN should have recorded a compression event"


def test_crush_does_not_record_on_passthrough(fresh_toin):
    """A small input the analyzer doesn't compress should produce no
    TOIN recording, even if the Rust port flipped `was_modified=True`
    from JSON whitespace re-canonicalization. The strategy stays
    `passthrough` in that case and we filter on it."""
    crusher = SmartCrusher(SmartCrusherConfig())
    payload = '[{"id": 1}]'  # Single item — below min_items_to_analyze.
    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())

    result = crusher.crush(payload, query="", bias=1.0)
    assert result.strategy == "passthrough"
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    assert post == pre, "no recording when strategy is passthrough"


def test_crush_signature_groups_similar_inputs(fresh_toin):
    """Two structurally-similar payloads should record under the same
    tool signature so TOIN can aggregate the pattern across calls."""
    crusher = SmartCrusher(SmartCrusherConfig())

    payload_a = _bigger_array(60)
    payload_b = _bigger_array(80)

    crusher.crush(payload_a, query="", bias=1.0)
    crusher.crush(payload_b, query="", bias=1.0)

    if not fresh_toin._patterns:
        pytest.skip("neither payload compressed — bump the size")
    # Both share field shape {status, tag, n} → same structure hash →
    # one pattern with at least 2 recordings.
    pattern_counts = {h: p.total_compressions for h, p in fresh_toin._patterns.items()}
    assert max(pattern_counts.values()) >= 2, (
        f"expected the same pattern to be recorded twice, got {pattern_counts}"
    )


# ─── _smart_crush_content() path (legacy apply()) ──────────────────────


def test_smart_crush_content_records_to_toin(fresh_toin):
    crusher = SmartCrusher(SmartCrusherConfig())
    payload = _bigger_array(60)
    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())

    crushed, was_modified, _info = crusher._smart_crush_content(
        payload, query_context="user query", tool_name="get_records", bias=1.0
    )

    if not was_modified:
        pytest.skip("payload didn't trigger compression")
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    assert post > pre


# ─── Failure modes ─────────────────────────────────────────────────────


def test_toin_failure_does_not_break_compression(fresh_toin):
    """If TOIN's `record_compression` raises, compression still
    completes and returns a valid CrushResult. Telemetry is best-
    effort; the request must not fail."""
    crusher = SmartCrusher(SmartCrusherConfig())
    payload = _bigger_array(60)

    def _boom(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("simulated TOIN backend down")

    with patch.object(fresh_toin, "record_compression", side_effect=_boom):
        result = crusher.crush(payload, query="", bias=1.0)
        assert isinstance(result.compressed, str)
        # Crusher itself should still report a result regardless of
        # whether the underlying analysis chose to modify.


def test_non_json_input_does_not_record(fresh_toin):
    """Non-JSON input shouldn't blow up — it just doesn't record.
    The Rust crusher returns `was_modified=False` for non-arrays;
    even if it didn't, the helper guards against `json.loads`
    failure."""
    crusher = SmartCrusher(SmartCrusherConfig())
    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())

    result = crusher.crush("not json at all", query="", bias=1.0)
    assert result.was_modified is False  # passthrough
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    assert post == pre


# ─── CCR marker knob ───────────────────────────────────────────────────


def test_ccr_inject_marker_false_suppresses_markers_in_output(fresh_toin):
    """`inject_retrieval_marker=False` is honored end-to-end now. The
    Rust crusher's `enable_ccr_marker` flips off and the lossy path
    skips both the `<<ccr:HASH>>` marker text and the CCR store write.
    Compression itself still happens — rows still drop — just without
    a retrieval pointer in the prompt."""
    from headroom.config import CCRConfig

    crusher = SmartCrusher(
        SmartCrusherConfig(),
        ccr_config=CCRConfig(enabled=True, inject_retrieval_marker=False),
    )
    payload = _bigger_array(60)
    result = crusher.crush(payload, query="", bias=1.0)

    if result.strategy == "passthrough":
        pytest.skip("payload didn't trigger compression — bump the size")

    assert "<<ccr:" not in result.compressed, f"expected no marker, got: {result.compressed!r}"
    assert "_ccr_dropped" not in result.compressed


def test_ccr_inject_marker_false_suppresses_opaque_blob_markers(fresh_toin):
    """#1091: `inject_retrieval_marker=False` must also suppress the
    *opaque-blob* CCR markers, not just the row-drop path.

    A long string cell (> opaque_min_bytes) used to be substituted with a
    `<<ccr:HASH,string,KB>>` marker unconditionally — so no config produced
    guaranteed-lossless output. This test pins both directions: with markers
    ON the opaque blob IS replaced by a marker (proving the input genuinely
    triggers the opaque path), and with markers OFF the blob survives verbatim
    with no marker."""
    import json

    from headroom.config import CCRConfig

    # Distinct >256-byte string cells trigger the opaque-blob path.
    payload = json.dumps(
        [{"id": i, "name": f"row{i}", "blob": f"sentinel{i}_" + "x" * 400} for i in range(60)]
    )

    on = SmartCrusher(
        SmartCrusherConfig(),
        ccr_config=CCRConfig(enabled=True, inject_retrieval_marker=True),
    ).crush(payload, query="", bias=1.0)
    # Sanity: the input really does exercise the opaque-blob path.
    assert "<<ccr:" in on.compressed, "input should trigger an opaque marker when markers are ON"

    off = SmartCrusher(
        SmartCrusherConfig(),
        ccr_config=CCRConfig(enabled=True, inject_retrieval_marker=False),
    ).crush(payload, query="", bias=1.0)
    assert "<<ccr:" not in off.compressed, f"expected no opaque marker, got: {off.compressed!r}"
    # The original blob content must survive verbatim (guaranteed-lossless).
    assert "sentinel5_" in off.compressed


def test_ccr_inject_marker_true_emits_markers_when_lossy(fresh_toin):
    """The opt-in case keeps marker emission on. If the lossy path
    runs (which it should for a sufficiently big crushable payload),
    the `<<ccr:HASH>>` marker appears in the compressed output."""
    from headroom.config import CCRConfig

    crusher = SmartCrusher(
        SmartCrusherConfig(),
        ccr_config=CCRConfig(enabled=True, inject_retrieval_marker=True),
    )
    payload = _bigger_array(60)
    result = crusher.crush(payload, query="", bias=1.0)

    if result.strategy == "passthrough":
        pytest.skip("payload didn't trigger compression")
    # If lossless won, marker won't appear (no row drops). If lossy
    # ran on these uniform `{status, tag, n}` records, we expect rows
    # to drop and the marker to fire.
    if "lossy" in result.strategy or "row" in result.strategy.lower():
        assert "<<ccr:" in result.compressed


def test_ccr_enabled_false_suppresses_markers_in_output(fresh_toin):
    """`CCRConfig.enabled=False` is the master kill-switch and must
    behave the same as `inject_retrieval_marker=False`: no marker text,
    no sentinel key, no CCR store write. Both flags collapse to the
    Rust-side `enable_ccr_marker=False` gate; storing a payload under
    `enabled=False` would be a surprise side effect the user
    explicitly opted out of."""
    from headroom.config import CCRConfig

    crusher = SmartCrusher(
        SmartCrusherConfig(),
        # Note: inject_retrieval_marker stays True — we want to prove
        # `enabled=False` alone is enough to suppress.
        ccr_config=CCRConfig(enabled=False, inject_retrieval_marker=True),
    )
    payload = _bigger_array(60)
    result = crusher.crush(payload, query="", bias=1.0)

    if result.strategy == "passthrough":
        pytest.skip("payload didn't trigger compression — bump the size")

    assert "<<ccr:" not in result.compressed, f"expected no marker, got: {result.compressed!r}"
    assert "_ccr_dropped" not in result.compressed


# ─── Custom scorer / relevance_config override ─────────────────────────


def test_custom_scorer_arg_raises_not_implemented():
    """The Rust port doesn't support custom scorers yet. Silently
    dropping a user-supplied scorer would be a textbook silent
    fallback (the user's scoring logic gets ignored, compression
    looks fine but is wrong). Fail loud instead."""

    class FakeScorer:
        pass

    with pytest.raises(NotImplementedError, match="relevance_config.*scorer"):
        SmartCrusher(SmartCrusherConfig(), scorer=FakeScorer())


def test_custom_relevance_config_arg_raises_not_implemented():
    """Same fail-loud contract for `relevance_config`."""
    with pytest.raises(NotImplementedError, match="relevance_config.*scorer"):
        SmartCrusher(SmartCrusherConfig(), relevance_config={"alpha": 0.7})


def test_default_construction_still_works():
    """Sanity: the audit fail-loud only triggers when the user passes
    one of the unsupported args. Default `SmartCrusher()` still
    constructs fine."""
    SmartCrusher(SmartCrusherConfig())  # no raise

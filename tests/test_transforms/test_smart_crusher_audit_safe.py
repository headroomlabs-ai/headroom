"""Audit-safe mode for SmartCrusher.crush_array_json (#1705).

`crush_array_json`'s row selection is purely statistical (variance,
anomaly, position) — it has no concept of "this row is a rare,
audit-relevant record that must stay visible in the prompt." A
compliance-significant row can be sampled out, or replaced by an
opaque `<<ccr:...>>` retrieval marker, exactly like any other row.

Audit-safe mode (`audit_safe=True` + `protected_patterns`) bolts
protection onto the existing Rust-backed compression without touching
the Rust selection logic: scan rows for pattern matches before
compression, then guarantee matched rows survive the compressed
output verbatim afterward — never dropped, never marker-only.

These tests exercise:
- Default (audit_safe=False) — zero behavior change, even if
  `protected_patterns` happens to be set.
- The splice-back mechanism directly (deterministic, doesn't depend on
  Rust's row-selection outcome for a given input).
- End-to-end through `crush_array_json` with a real lossy compression.
- The fail-closed / warn-and-ship-best-effort fork when verification
  still finds a shortfall after splicing.
- Loud failure on an invalid regex in `protected_patterns`.
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


def test_audit_safe_disabled_by_default_no_behavior_change() -> None:
    """`protected_patterns` set but `audit_safe` left at its False
    default → identical output to a crusher with no audit-safe config
    at all. The flag gates the whole feature, not just the presence of
    patterns."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    items = [{"id": i, "status": "ok"} for i in range(50)]
    items_json = json.dumps(items)

    baseline = SmartCrusher(SmartCrusherConfig(), with_compaction=False)
    configured_but_off = SmartCrusher(
        SmartCrusherConfig(protected_patterns=["ok"]), with_compaction=False
    )

    r1 = baseline.crush_array_json(items_json)
    r2 = configured_but_off.crush_array_json(items_json)

    assert r1["items"] == r2["items"]
    assert r1["ccr_hash"] == r2["ccr_hash"]
    assert r1["strategy_info"] == r2["strategy_info"]


def test_audit_safe_with_no_patterns_is_a_no_op() -> None:
    """`audit_safe=True` but `protected_patterns` empty/None → nothing
    to protect, output unchanged from the unguarded crusher."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    items = [{"id": i, "status": "ok"} for i in range(50)]
    items_json = json.dumps(items)

    baseline = SmartCrusher(SmartCrusherConfig(), with_compaction=False)
    audit_safe_no_patterns = SmartCrusher(
        SmartCrusherConfig(audit_safe=True, protected_patterns=None),
        with_compaction=False,
    )

    r1 = baseline.crush_array_json(items_json)
    r2 = audit_safe_no_patterns.crush_array_json(items_json)

    assert r1["items"] == r2["items"]
    assert r1["ccr_hash"] == r2["ccr_hash"]


def test_splice_back_restores_a_dropped_protected_row() -> None:
    """Direct unit test of `_apply_audit_safe_protection`: given a
    `result["items"]` that's missing a protected row (as if the Rust
    row-drop path had sampled it out), the method appends it back and
    reports no loss.

    Deterministic — doesn't depend on Rust's actual sampling decision
    for a given input, only on the splice/verify logic this PR adds.
    """
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(
        SmartCrusherConfig(audit_safe=True, protected_patterns=["AUDIT_FLAG"]),
        with_compaction=False,
    )

    protected_row = {"id": 7, "note": "AUDIT_FLAG: rare compliance event"}
    kept_without_protected = [{"id": i, "status": "ok"} for i in range(5)]
    fake_result = {
        "items": json.dumps(kept_without_protected),
        "ccr_hash": "deadbeefcafe",
        "dropped_summary": "<<ccr:deadbeefcafe 10_rows_offloaded>>",
        "strategy_info": "smart_sample",
        "compacted": None,
        "compaction_kind": None,
    }

    out = crusher._apply_audit_safe_protection(
        [protected_row], json.dumps(kept_without_protected + [protected_row]), fake_result
    )

    kept = json.loads(out["items"])
    assert protected_row in kept
    assert len(kept) == len(kept_without_protected) + 1
    # Everything else about the result (ccr_hash, marker) is untouched —
    # splicing doesn't erase the CCR pointer for the rows that really
    # were dropped, it only guarantees the protected one is inline too.
    assert out["ccr_hash"] == "deadbeefcafe"


def test_audit_safe_preserves_protected_rows_end_to_end() -> None:
    """Full path through `crush_array_json`: a real lossy compression
    runs, and every protected row is present in the output afterward,
    regardless of what the statistical row-selection decided on its
    own."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    background = [{"id": i, "status": "ok"} for i in range(60)]
    protected_rows = [
        {"id": 25, "status": "ok", "note": "AUDIT_FLAG: rare compliance event A"},
        {"id": 35, "status": "ok", "note": "AUDIT_FLAG: rare compliance event B"},
    ]
    items = (
        background[:25]
        + [protected_rows[0]]
        + background[25:35]
        + [protected_rows[1]]
        + background[35:]
    )
    items_json = json.dumps(items)

    crusher = SmartCrusher(
        SmartCrusherConfig(audit_safe=True, protected_patterns=["AUDIT_FLAG"]),
        with_compaction=False,
    )
    result = crusher.crush_array_json(items_json)
    kept = json.loads(result["items"])

    for row in protected_rows:
        assert row in kept, f"protected row missing from compressed output: {row!r}"
    # The array as a whole still compressed — audit-safe protection
    # isn't a blanket opt-out of compression, only a guarantee for the
    # specific protected rows.
    assert len(kept) < len(items)


def test_audit_safe_fails_closed_when_verification_still_finds_loss(monkeypatch) -> None:
    """Defensive path: if the post-splice verification still finds a
    shortfall (simulated here — normal splicing always succeeds, so we
    force the mismatch by making `_canon` non-idempotent), and
    `fail_closed_on_protected_loss=True` (the default), the whole
    array is returned unmodified instead of shipping a result with
    fewer protected-row matches than the input had."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(
        SmartCrusherConfig(audit_safe=True, protected_patterns=["AUDIT_FLAG"]),
        with_compaction=False,
    )

    # Force every `_canon` call to return a fresh, never-repeating
    # value so the Counter-based matching in both the splice and the
    # verify phase can never line up — modeling an internal
    # inconsistency the verification step exists to catch.
    counter = iter(range(10_000))
    monkeypatch.setattr(
        SmartCrusher, "_canon", staticmethod(lambda item: f"unique-{next(counter)}")
    )

    protected_row = {"id": 1, "note": "AUDIT_FLAG"}
    items_json = json.dumps([protected_row])
    fake_result = {
        "items": json.dumps([]),
        "ccr_hash": "aaaa",
        "dropped_summary": "<<ccr:aaaa 1_rows_offloaded>>",
        "strategy_info": "smart_sample",
        "compacted": None,
        "compaction_kind": None,
    }

    out = crusher._apply_audit_safe_protection([protected_row], items_json, fake_result)

    assert out["items"] == items_json
    assert out["ccr_hash"] is None
    assert out["strategy_info"] == "audit_safe:fail_closed"


def test_audit_safe_ships_best_effort_when_fail_closed_disabled(monkeypatch) -> None:
    """Same forced-mismatch scenario, but `fail_closed_on_protected_loss
    =False` — ship the spliced best-effort result (with a logged
    warning) instead of refusing to compress."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(
        SmartCrusherConfig(
            audit_safe=True,
            protected_patterns=["AUDIT_FLAG"],
            fail_closed_on_protected_loss=False,
        ),
        with_compaction=False,
    )

    counter = iter(range(10_000))
    monkeypatch.setattr(
        SmartCrusher, "_canon", staticmethod(lambda item: f"unique-{next(counter)}")
    )

    protected_row = {"id": 1, "note": "AUDIT_FLAG"}
    items_json = json.dumps([protected_row])
    fake_result = {
        "items": json.dumps([]),
        "ccr_hash": "aaaa",
        "dropped_summary": "<<ccr:aaaa 1_rows_offloaded>>",
        "strategy_info": "smart_sample",
        "compacted": None,
        "compaction_kind": None,
    }

    out = crusher._apply_audit_safe_protection([protected_row], items_json, fake_result)

    # Best-effort: the splice phase still ran and appended the
    # protected row (splicing itself doesn't depend on `_canon` being
    # idempotent across calls — only the *verification* mismatch is
    # forced), so the row is present even though the strategy wasn't
    # replaced with the fail-closed sentinel.
    assert out["strategy_info"] != "audit_safe:fail_closed"
    kept = json.loads(out["items"])
    assert protected_row in kept


def test_invalid_protected_pattern_raises() -> None:
    """A regex that fails to compile is a caller bug and must raise
    loudly at construction time — silently treating it as "nothing
    protected" would defeat the point of audit-safe mode."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    with pytest.raises(ValueError, match="invalid protected_patterns regex"):
        SmartCrusher(
            SmartCrusherConfig(audit_safe=True, protected_patterns=["("]),
            with_compaction=False,
        )


# ─── Production path: _smart_crush_content / apply() ───────────────────────
#
# `crush_array_json` is a convenience API used by tests and the CCR
# retrieval flow. The path `apply()` actually calls for every compressed
# tool/tool_result message is `_smart_crush_content`. Audit-safe mode has
# to hold on that path too, or it would only ever protect a code path
# real traffic never exercises.


def test_smart_crush_content_preserves_protected_rows_end_to_end() -> None:
    """Real lossy compression via `_smart_crush_content` (the method
    `apply()` calls) still surfaces every protected row afterward."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    background = [{"id": i, "status": "ok"} for i in range(60)]
    protected_rows = [
        {"id": 25, "status": "ok", "note": "AUDIT_FLAG: rare compliance event A"},
        {"id": 35, "status": "ok", "note": "AUDIT_FLAG: rare compliance event B"},
    ]
    items = (
        background[:25]
        + [protected_rows[0]]
        + background[25:35]
        + [protected_rows[1]]
        + background[35:]
    )
    content = json.dumps(items)

    crusher = SmartCrusher(
        SmartCrusherConfig(audit_safe=True, protected_patterns=["AUDIT_FLAG"]),
        with_compaction=False,
    )
    crushed, was_modified, info = crusher._smart_crush_content(content)

    assert was_modified
    kept = json.loads(crushed)
    for row in protected_rows:
        assert row in kept, f"protected row missing from _smart_crush_content output: {row!r}"


def test_apply_preserves_protected_rows_in_tool_message() -> None:
    """Full `Transform.apply()` path: a tool message with a large JSON
    array containing protected rows gets compressed, and the resulting
    message content (before the digest marker) still contains every
    protected row. This is the exact code path the real proxy runs for
    every tool output — proof audit-safe mode isn't test-only plumbing."""
    from headroom import OpenAIProvider, Tokenizer
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    background = [{"id": i, "status": "ok"} for i in range(60)]
    protected_rows = [
        {"id": 25, "status": "ok", "note": "AUDIT_FLAG: rare compliance event A"},
        {"id": 35, "status": "ok", "note": "AUDIT_FLAG: rare compliance event B"},
    ]
    items = (
        background[:25]
        + [protected_rows[0]]
        + background[25:35]
        + [protected_rows[1]]
        + background[35:]
    )
    content = json.dumps(items)

    crusher = SmartCrusher(
        SmartCrusherConfig(
            audit_safe=True, protected_patterns=["AUDIT_FLAG"], min_tokens_to_crush=10
        ),
        with_compaction=False,
    )
    messages = [
        {"role": "user", "content": "check the results"},
        {"role": "assistant", "tool_calls": [{"id": "t1", "function": {"name": "query"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": content},
    ]

    provider = OpenAIProvider()
    tokenizer = Tokenizer(provider.get_token_counter("gpt-4o"), "gpt-4o")
    result = crusher.apply(messages, tokenizer)

    tool_message = result.messages[-1]
    assert tool_message["role"] == "tool"
    # Content is `<crushed>\n<digest_marker>` — strip the marker line.
    crushed_body = tool_message["content"].rsplit("\n", 1)[0]
    kept = json.loads(crushed_body)
    for row in protected_rows:
        assert row in kept, f"protected row missing from apply() output: {row!r}"
    assert len(kept) < len(items)


def test_content_protection_falls_back_to_pattern_count_for_non_array_output() -> None:
    """Direct unit test of `_apply_audit_safe_protection_to_content`'s
    non-list branch: when `crushed` isn't a JSON array (e.g. a
    lossless CSV/table render, or an opaque marker string), there's no
    row structure to splice into, so verification counts protected
    pattern matches in the raw text instead. A drop in match count
    still fails closed."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(
        SmartCrusherConfig(audit_safe=True, protected_patterns=["AUDIT_FLAG"]),
        with_compaction=False,
    )

    protected_row = {"id": 1, "note": "AUDIT_FLAG"}
    original_content = json.dumps([protected_row, {"id": 2, "note": "fine"}])
    # Simulate a lossless render that dropped the marker text entirely.
    crushed_without_marker = "id,note\n2,fine"

    out_text, was_modified, info = crusher._apply_audit_safe_protection_to_content(
        [protected_row], original_content, crushed_without_marker, True, "lossless:table"
    )

    assert out_text == original_content
    assert was_modified is False
    assert info == "audit_safe:fail_closed"


def test_content_protection_no_op_when_pattern_count_preserved() -> None:
    """Non-list `crushed` output that still contains every protected
    pattern occurrence is left untouched — the fallback only fires on
    an actual count decrease."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(
        SmartCrusherConfig(audit_safe=True, protected_patterns=["AUDIT_FLAG"]),
        with_compaction=False,
    )

    protected_row = {"id": 1, "note": "AUDIT_FLAG"}
    original_content = json.dumps([protected_row])
    crushed_with_marker = "id,note\n1,AUDIT_FLAG"

    out_text, was_modified, info = crusher._apply_audit_safe_protection_to_content(
        [protected_row], original_content, crushed_with_marker, True, "lossless:table"
    )

    assert out_text == crushed_with_marker
    assert was_modified is True
    assert info == "lossless:table"

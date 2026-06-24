"""Tests for :class:`headroom.proxy.image_compression_decision.ImageCompressionDecision`.

Image compression today is gated at two sites (``openai.py:1203`` +
``anthropic.py:868``) by inline conjunctions. Both already check
``_bypass`` (the drift problem CompressionDecision fixed is NOT
present here), but consolidating into a value type still pays off:

* test-lockable contract (no future site can forget bypass)
* ``apply_to_tags()`` surfaces ``image_skip_reason`` in
  ``RequestOutcome.tags`` — dashboards can slice image-skipped
  traffic by cause (same observability surface as
  ``passthrough_reason`` and ``memory_skip_reason``)
* Rust-portable shape mirrors :class:`CompressionDecision` exactly

Precedence (highest first):
  1. ``bypass_header``           — user opt-out
  2. ``image_optimize_disabled`` — operator config off
  3. ``no_messages``             — nothing to inspect
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

from headroom.proxy.image_compression_decision import ImageCompressionDecision

# ── Helpers ───────────────────────────────────────────────────────────


def _config(*, image_optimize: bool = True) -> Any:
    """Minimal stand-in for ``HeadroomConfig`` — only the field the
    decision reads."""
    return SimpleNamespace(image_optimize=image_optimize)


def _msgs(n: int = 1) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"hi-{i}"} for i in range(n)]


# ── Value-type contract ───────────────────────────────────────────────


def test_decision_is_frozen() -> None:
    d = ImageCompressionDecision.decide(headers={}, config=_config(), messages=_msgs())
    try:
        d.should_compress = False  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("ImageCompressionDecision must be frozen")


def test_decision_is_value_equal() -> None:
    a = ImageCompressionDecision.decide(headers={}, config=_config(), messages=_msgs())
    b = ImageCompressionDecision.decide(headers={}, config=_config(), messages=_msgs())
    assert a == b


# ── Precedence ────────────────────────────────────────────────────────


def test_compresses_when_every_gate_open() -> None:
    d = ImageCompressionDecision.decide(
        headers={}, config=_config(image_optimize=True), messages=_msgs()
    )
    assert d.should_compress is True
    assert d.passthrough_reason is None


def test_bypass_header_wins() -> None:
    """Bypass is the user's explicit "don't touch my bytes" signal —
    image compression mutates bytes (tile-aligns / re-encodes), so
    bypass must skip it. Mirror of CompressionDecision."""
    d = ImageCompressionDecision.decide(
        headers={"x-headroom-bypass": "true"},
        config=_config(image_optimize=True),
        messages=_msgs(),
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "bypass_header"


def test_passthrough_mode_header_also_triggers_bypass_skip() -> None:
    """``x-headroom-mode: passthrough`` alt spelling — mirrors
    ``_headroom_bypass_enabled`` semantics across all proxy gates."""
    d = ImageCompressionDecision.decide(
        headers={"x-headroom-mode": "passthrough"},
        config=_config(image_optimize=True),
        messages=_msgs(),
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "bypass_header"


def test_image_optimize_disabled_is_skip() -> None:
    """Operator config — ``config.image_optimize = False``. Distinct
    reason from ``compression_disabled`` (text compression's gate);
    operators can enable text + disable image independently."""
    d = ImageCompressionDecision.decide(
        headers={}, config=_config(image_optimize=False), messages=_msgs()
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "image_optimize_disabled"


def test_no_messages_is_skip() -> None:
    """Empty or missing messages — nothing to look at. Same shape as
    CompressionDecision's no_messages reason."""
    d = ImageCompressionDecision.decide(headers={}, config=_config(), messages=[])
    assert d.should_compress is False
    assert d.passthrough_reason == "no_messages"


def test_messages_none_is_skip() -> None:
    """``messages=None`` is treated identically to empty list."""
    d = ImageCompressionDecision.decide(headers={}, config=_config(), messages=None)
    assert d.should_compress is False
    assert d.passthrough_reason == "no_messages"


# ── Precedence ordering ──────────────────────────────────────────────


def test_bypass_beats_image_optimize_disabled() -> None:
    """User signal beats operator signal — bypass is the more
    informative dashboard slice."""
    d = ImageCompressionDecision.decide(
        headers={"x-headroom-bypass": "true"},
        config=_config(image_optimize=False),
        messages=_msgs(),
    )
    assert d.passthrough_reason == "bypass_header"


def test_bypass_beats_no_messages() -> None:
    """Bypass+no-messages surfaces bypass — user opted out, the
    empty body is incidental."""
    d = ImageCompressionDecision.decide(
        headers={"x-headroom-bypass": "true"},
        config=_config(),
        messages=[],
    )
    assert d.passthrough_reason == "bypass_header"


def test_image_optimize_disabled_beats_no_messages() -> None:
    """When config is off AND messages empty, surface the operator
    decision — more meaningful for dashboards."""
    d = ImageCompressionDecision.decide(
        headers={}, config=_config(image_optimize=False), messages=[]
    )
    assert d.passthrough_reason == "image_optimize_disabled"


# ── Observability fields ─────────────────────────────────────────────


def test_observability_booleans_populated_when_compressing() -> None:
    d = ImageCompressionDecision.decide(
        headers={}, config=_config(image_optimize=True), messages=_msgs(2)
    )
    assert d.bypass_header_set is False
    assert d.image_optimize_enabled is True
    assert d.has_messages is True


def test_observability_booleans_populated_when_passthrough() -> None:
    d = ImageCompressionDecision.decide(
        headers={"x-headroom-bypass": "true"},
        config=_config(image_optimize=True),
        messages=_msgs(),
    )
    assert d.bypass_header_set is True
    assert d.image_optimize_enabled is True
    assert d.has_messages is True


# ── apply_to_tags ────────────────────────────────────────────────────


def test_apply_to_tags_stamps_reason_when_passthrough() -> None:
    d = ImageCompressionDecision.decide(
        headers={"x-headroom-bypass": "true"}, config=_config(), messages=_msgs()
    )
    tags: dict[str, str] = {}
    d.apply_to_tags(tags)
    assert tags == {"image_skip_reason": "bypass_header"}


def test_apply_to_tags_is_a_noop_when_compressing() -> None:
    d = ImageCompressionDecision.decide(headers={}, config=_config(), messages=_msgs())
    tags: dict[str, str] = {"client": "codex"}
    d.apply_to_tags(tags)
    assert tags == {"client": "codex"}


def test_apply_to_tags_preserves_pre_existing_entries() -> None:
    """Image skip reason coexists with other slicing tags (client,
    passthrough_reason, memory_skip_reason) — they all live in the
    same RequestOutcome.tags dict."""
    d = ImageCompressionDecision.decide(
        headers={}, config=_config(image_optimize=False), messages=_msgs()
    )
    tags: dict[str, str] = {
        "client": "claude-code",
        "passthrough_reason": "compression_disabled",
        "memory_skip_reason": "no_user_id",
    }
    d.apply_to_tags(tags)
    assert tags["client"] == "claude-code"
    assert tags["passthrough_reason"] == "compression_disabled"
    assert tags["memory_skip_reason"] == "no_user_id"
    assert tags["image_skip_reason"] == "image_optimize_disabled"


def test_apply_to_tags_for_every_skip_reason() -> None:
    """Every reason name round-trips cleanly into the tag dict."""
    cases: dict[str, dict[str, Any]] = {
        "bypass_header": {
            "headers": {"x-headroom-bypass": "true"},
            "config": _config(),
            "messages": _msgs(),
        },
        "image_optimize_disabled": {
            "headers": {},
            "config": _config(image_optimize=False),
            "messages": _msgs(),
        },
        "no_messages": {
            "headers": {},
            "config": _config(),
            "messages": [],
        },
    }
    for expected_reason, kwargs in cases.items():
        d = ImageCompressionDecision.decide(**kwargs)
        tags: dict[str, str] = {}
        d.apply_to_tags(tags)
        assert tags.get("image_skip_reason") == expected_reason, expected_reason

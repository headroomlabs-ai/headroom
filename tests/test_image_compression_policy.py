"""Tests for pure image-compression decision policy helpers."""

from __future__ import annotations

from headroom.proxy.image_compression_policy import (
    apply_image_skip_reason,
    decide_image_compression,
)


def test_decide_image_compression_allows_happy_path() -> None:
    decision = decide_image_compression(
        headers={},
        image_optimize_enabled=True,
        has_messages=True,
    )

    assert decision.should_compress is True
    assert decision.passthrough_reason is None


def test_decide_image_compression_uses_canonical_precedence() -> None:
    decision = decide_image_compression(
        headers={"x-headroom-bypass": "true"},
        image_optimize_enabled=False,
        has_messages=False,
    )

    assert decision.should_compress is False
    assert decision.passthrough_reason == "bypass_header"
    assert decision.bypass_header_set is True
    assert decision.image_optimize_enabled is False
    assert decision.has_messages is False


def test_decide_image_compression_reports_config_and_message_reasons() -> None:
    disabled = decide_image_compression(
        headers={},
        image_optimize_enabled=False,
        has_messages=True,
    )
    empty = decide_image_compression(
        headers={},
        image_optimize_enabled=True,
        has_messages=False,
    )

    assert disabled.passthrough_reason == "image_optimize_disabled"
    assert empty.passthrough_reason == "no_messages"


def test_apply_image_skip_reason_stamps_only_when_skipping() -> None:
    tags: dict[str, str] = {}
    apply_image_skip_reason(tags, None)
    assert tags == {}

    apply_image_skip_reason(tags, "no_messages")
    assert tags == {"image_skip_reason": "no_messages"}

"""Pure image-compression decision policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from headroom.proxy.helpers import _headroom_bypass_enabled


@dataclass(frozen=True)
class ImageCompressionPolicyResult:
    """Raw image-compression gate result before wrapping in public value types."""

    should_compress: bool
    passthrough_reason: str | None
    bypass_header_set: bool
    image_optimize_enabled: bool
    has_messages: bool


def decide_image_compression(
    *,
    headers: Any,
    image_optimize_enabled: bool,
    has_messages: bool,
) -> ImageCompressionPolicyResult:
    """Compute the canonical image-compression gate."""
    bypass = _headroom_bypass_enabled(headers)

    if bypass:
        reason: str | None = "bypass_header"
        should = False
    elif not image_optimize_enabled:
        reason = "image_optimize_disabled"
        should = False
    elif not has_messages:
        reason = "no_messages"
        should = False
    else:
        reason = None
        should = True

    return ImageCompressionPolicyResult(
        should_compress=should,
        passthrough_reason=reason,
        bypass_header_set=bypass,
        image_optimize_enabled=image_optimize_enabled,
        has_messages=has_messages,
    )


def apply_image_skip_reason(tags: dict[str, str], passthrough_reason: str | None) -> None:
    """Stamp image skip reason into tags when present."""
    if passthrough_reason is not None:
        tags["image_skip_reason"] = passthrough_reason

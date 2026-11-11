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
    image_selected: bool
    has_messages: bool


def image_compressor_selected(compressors: Any) -> bool:
    """Whether a ``--compressor`` selection leaves the image compressor on.

    Mirrors how the proxy maps a selection onto the router's ``enable_*``
    flags: no selection (``None``, empty, or only blank names) keeps every built-in,
    ``"*"`` selects all, and otherwise ``image`` must be named.
    """
    if not isinstance(compressors, (set, frozenset, list, tuple)) or not compressors:
        return True
    selected = {name.strip() for name in compressors if name.strip()}
    return not selected or "*" in selected or "image" in selected


def decide_image_compression(
    *,
    headers: Any,
    image_optimize_enabled: bool,
    has_messages: bool,
    image_selected: bool = True,
) -> ImageCompressionPolicyResult:
    """Compute the canonical image-compression gate."""
    bypass = _headroom_bypass_enabled(headers)

    if bypass:
        reason: str | None = "bypass_header"
        should = False
    elif not image_optimize_enabled:
        reason = "image_optimize_disabled"
        should = False
    elif not image_selected:
        reason = "image_not_selected"
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
        image_selected=image_selected,
        has_messages=has_messages,
    )


def apply_image_skip_reason(tags: dict[str, str], passthrough_reason: str | None) -> None:
    """Stamp image skip reason into tags when present."""
    if passthrough_reason is not None:
        tags["image_skip_reason"] = passthrough_reason

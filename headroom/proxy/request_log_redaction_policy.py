"""Pure request-log redaction policy for image-bearing payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Phase G PR-G3 - base64 redaction threshold (P4-45).
#
# Anthropic image blocks carry base64-encoded JPEGs/PNGs in
# ``source.data``; OpenAI's vision shape carries them in
# ``image_url.url`` as a ``data:image/...;base64,<payload>`` URL.
# The threshold gates "real image payload" against short base64
# strings (which can appear in arguments, signatures, etc.).
IMAGE_BASE64_REDACT_THRESHOLD_BYTES = 1024

# Phase G PR-G3 - replacement-marker format. Operators can grep the
# JSONL for ``<image:base64-redacted`` to count the redactions; the
# byte count keeps cost attribution honest even after redaction.
# M5: ``bytes=`` is the UTF-8 byte length, not the character count.
IMAGE_BASE64_REPLACEMENT_TEMPLATE = "<image:base64-redacted bytes={n}>"

# M2: JSON field names that carry image payloads in either the
# Anthropic or OpenAI shapes. Strings reached via one of these key
# names (at any depth) are eligible for the redaction heuristic.
# Anything OUTSIDE these paths is left untouched even if it looks
# base64-shaped - encrypted blobs, signed tokens, minified JSON,
# tool outputs all live elsewhere and stay verbatim.
IMAGE_BEARING_FIELD_NAMES: frozenset[str] = frozenset(
    {
        # Anthropic image-block shape: ``{"type":"image","source":{"type":"base64","data":"..."}}``.
        "data",
        # OpenAI vision shape: ``{"type":"image_url","image_url":{"url":"data:image/..."}}``.
        "url",
        # OpenAI Responses input_image: ``{"type":"input_image","image_url":"..."}``
        # - string-valued directly under the key (not nested).
        "image_url",
        # Some SDKs put the URL under ``image`` directly. Tolerated.
        "image",
        # Anthropic vision blocks sometimes wrap under ``source.data``;
        # ``source`` is a container, not a string field, so it doesn't
        # need to be in this set, but the data string itself is keyed
        # by ``data`` (already above).
    }
)

# M2: explicit data-URL MIME prefix. A string starting with this
# prefix is always treated as an image payload, regardless of where
# it lives in the JSON - operators occasionally embed data URLs in
# arbitrary fields and we want those redacted to keep logs small.
_DATA_IMAGE_URL_PREFIX = "data:image/"


@dataclass(frozen=True)
class RedactionResult:
    """A redacted value and the number of replacements made."""

    value: Any
    redactions: int


def is_base64_image_payload(value: object) -> bool:
    """Return True if ``value`` is an over-threshold image data URL.

    Per M2 remediation the prior bare-base64 density heuristic over-fired on
    non-image content. This helper only recognizes explicit image data URLs;
    image-bearing JSON-path eligibility is handled by the recursive policy.
    """
    if not isinstance(value, str):
        return False
    if len(value) < IMAGE_BASE64_REDACT_THRESHOLD_BYTES:
        return False
    return value.startswith(_DATA_IMAGE_URL_PREFIX)


def redact_image_base64_value(payload: Any) -> RedactionResult:
    """Return ``payload`` with over-threshold image strings redacted."""
    return _redact_value(payload, in_image_path=False)


def _redact_value(value: Any, *, in_image_path: bool) -> RedactionResult:
    if isinstance(value, str):
        should_redact = is_base64_image_payload(value) or (
            in_image_path and len(value) >= IMAGE_BASE64_REDACT_THRESHOLD_BYTES
        )
        if not should_redact:
            return RedactionResult(value=value, redactions=0)

        byte_len = len(value.encode("utf-8"))
        return RedactionResult(
            value=IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=byte_len),
            redactions=1,
        )

    if isinstance(value, Mapping):
        redactions = 0
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            item_result = _redact_value(
                item,
                in_image_path=(key in IMAGE_BEARING_FIELD_NAMES),
            )
            redacted[key] = item_result.value
            redactions += item_result.redactions
        return RedactionResult(value=redacted, redactions=redactions)

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        redactions = 0
        redacted_items: list[Any] = []
        for item in value:
            item_result = _redact_value(item, in_image_path=in_image_path)
            redacted_items.append(item_result.value)
            redactions += item_result.redactions
        return RedactionResult(value=redacted_items, redactions=redactions)

    return RedactionResult(value=value, redactions=0)

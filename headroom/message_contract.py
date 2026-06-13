"""Message-shape helpers for provider-specific metadata.

Headroom routes messages through several provider adapters and compression
stages. Those stages often rebuild a message dict from ``role`` and
``content``; that is exactly where provider-specific fields such as
DeepSeek's ``reasoning_content`` can disappear. Keep the allowlist and the
copy operation centralized so adapters do not need one-off field handling.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Provider-specific message-level fields that must survive message
# reconstruction. These are intentionally message metadata, not content blocks.
PRESERVED_MESSAGE_FIELDS: tuple[str, ...] = (
    "reasoning_content",
    "redacted_reasoning_content",
    "reasoning_details",
    "provider_data",
)


def preserve_message_fields(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    fields: Iterable[str] = PRESERVED_MESSAGE_FIELDS,
) -> dict[str, Any]:
    """Copy known provider-specific message fields from ``source`` to ``target``.

    The target is mutated and returned for ergonomic use at reconstruction
    sites:

    ``converted.append(preserve_message_fields(msg, {"role": role, "content": text}))``
    """

    for field in fields:
        if field in source:
            target[field] = source[field]
    return target

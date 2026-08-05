"""Byte-faithful outbound request body forwarding policy.

This module owns the small algebra used by Python proxy forwarders to decide
which bytes leave Headroom:

* unmutated body with original bytes -> byte-for-byte passthrough
* mutated body or missing original bytes -> canonical JSON bytes
* explicit rollback mode -> legacy httpx-style JSON bytes
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from headroom.proxy import python_forwarder_mode_policy

_PYTHON_FORWARDER_MODE_ENV = python_forwarder_mode_policy.PYTHON_FORWARDER_MODE_ENV

PythonForwarderMode = python_forwarder_mode_policy.PythonForwarderMode
OutboundBodySource = Literal["passthrough", "canonical", "legacy"]

_PYTHON_FORWARDER_MODE_DEFAULT = python_forwarder_mode_policy.PYTHON_FORWARDER_MODE_DEFAULT


@dataclass(frozen=True, slots=True)
class OutboundBody:
    """Concrete outbound body bytes plus their provenance."""

    content: bytes
    source: OutboundBodySource


def get_python_forwarder_mode() -> PythonForwarderMode:
    """Return the active Python-forwarder mode.

    Read at request time. Unknown values raise loudly per the no-silent-
    fallback build constraint. The ``legacy_json_kwarg`` value is an
    explicit operator opt-in for emergency rollback, not a fallback.
    """
    return python_forwarder_mode_policy.resolve_python_forwarder_mode(
        os.environ.get(_PYTHON_FORWARDER_MODE_ENV)
    )


def serialize_body_canonical(body: dict[str, Any]) -> bytes:
    """Re-serialize a request body deterministically with cache-stable formatting."""
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class BodyMutationTracker:
    """Records whether a request body was mutated and why."""

    __slots__ = ("_mutated", "_reasons")

    def __init__(self) -> None:
        self._mutated: bool = False
        self._reasons: list[str] = []

    def mark_mutated(self, reason: str) -> None:
        """Mark the body as mutated and record the stable aggregation reason."""
        if not reason:
            raise ValueError("BodyMutationTracker.mark_mutated: reason must be non-empty")
        self._mutated = True
        if reason not in self._reasons:
            self._reasons.append(reason)

    @property
    def mutated(self) -> bool:
        return self._mutated

    @property
    def reasons(self) -> list[str]:
        return list(self._reasons)


def select_outbound_body(
    *,
    body: dict[str, Any],
    original_body_bytes: bytes | None,
    body_mutated: bool,
    forwarder_mode: PythonForwarderMode | None = None,
) -> OutboundBody:
    """Select the exact bytes to forward upstream."""
    mode = forwarder_mode if forwarder_mode is not None else get_python_forwarder_mode()
    if mode == "legacy_json_kwarg":
        content = json.dumps(body, separators=(", ", ": "), ensure_ascii=True).encode("utf-8")
        return OutboundBody(content=content, source="legacy")

    if body_mutated or original_body_bytes is None:
        return OutboundBody(content=serialize_body_canonical(body), source="canonical")
    return OutboundBody(content=original_body_bytes, source="passthrough")


def prepare_outbound_body_bytes(
    *,
    body: dict[str, Any],
    original_body_bytes: bytes | None,
    body_mutated: bool,
    forwarder_mode: PythonForwarderMode | None = None,
) -> tuple[bytes, OutboundBodySource]:
    """Compatibility tuple wrapper around :func:`select_outbound_body`."""
    outbound = select_outbound_body(
        body=body,
        original_body_bytes=original_body_bytes,
        body_mutated=body_mutated,
        forwarder_mode=forwarder_mode,
    )
    return outbound.content, outbound.source

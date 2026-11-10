"""Stable attribution identifiers for the G4 -> G3 -> G2 -> G1 drill-down.

Per ``data-model.md`` Entity 5, an attribution chain is the deterministic
string

    ``{tier}:{request_path}:{stage}:{transform}:{function}``

where the leaf slots may be empty for coarse-grained tiers. Two attribution
records with identical inputs MUST resolve to the same string; the function
below is the canonical implementation.

The chain is rendered as ``[G4][G3][G2][G1]`` in Markdown summaries so a
maintainer can click one link from a G4 report to the matching G1 artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .schema import Attribution

# Maximum number of bytes for a single attribution segment (defensive cap).
MAX_SEGMENT_LEN = 96


_EMPTY_SLOT = "-"


def _sanitize(seg: str | None) -> str:
    """Replace characters that would break a Markdown link or a JSON path.

    Allowed characters are ``[A-Za-z0-9_./-]`` (everything else collapses to
    ``-``). A trailing ``-`` introduced by sanitization is trimmed.

    Returns ``""`` if ``seg`` is falsy so :func:`build_id` can distinguish a
    *missing* leaf from a present-but-empty one.
    """

    if not seg:
        return ""
    out: list[str] = []
    for ch in seg:
        if ch.isalnum() or ch in "_./-":
            out.append(ch)
        else:
            out.append("-")
    s = "".join(out).rstrip("-")
    return s[:MAX_SEGMENT_LEN]


def _normalize(slot: str | None) -> str:
    """Return the canonical slot value, mapping empty -> ``_EMPTY_SLOT``."""

    return _sanitize(slot) if slot else _EMPTY_SLOT


def build_id(
    tier: str,
    request_path: str,
    stage: str | None = None,
    transform: str | None = None,
    function: str | None = None,
    *,
    parents: Sequence[str] = (),
) -> str:
    """Return the canonical attribution string.

    Empty leaf segments are kept as ``-`` so the chain length is stable
    regardless of tier granularity. Example::

        build_id("G2", "/v1/messages", "compression", "smart_crusher", "apply")
        # "G2:-v1-messages:compression:smart_crusher:apply"

        build_id("G4", "/v1/messages")
        # "G4:-v1-messages:-:-:-"
    """

    parts = [_normalize(p) for p in (tier, request_path, stage, transform, function)]
    # Force exactly 5 slots by padding with ``-``.
    while len(parts) < 5:
        parts.append(_EMPTY_SLOT)
    # Parent IDs are *already* canonical stable IDs; sanitize only the path
    # separators between them so the chain survives the same rules as the
    # head. We keep the inner ``:`` intact.
    parent_chain = "/".join(_sanitize_id(p) for p in parents)
    head = ":".join(parts)
    return f"{head}@{parent_chain}" if parent_chain else head


def _sanitize_id(stable_id: str) -> str:
    """Sanitize an existing canonical stable ID (preserve ``:`` and ``-``).

    Stable IDs are joined by ``/`` to form a parent chain; we must keep the
    5-slot ``a:b:c:d:e`` shape readable. The only disallowed character is
    ``@`` (which would prematurely terminate the parent chain).
    """

    return stable_id.replace("@", "-")


def attribution_from(
    tier: str,
    request_path: str,
    stage: str | None = None,
    transform: str | None = None,
    function: str | None = None,
    *,
    parents: Sequence[str] = (),
) -> Attribution:
    """Materialize an :class:`Attribution` from the canonical 5-slot tuple."""

    stable_id = build_id(
        tier,
        request_path,
        stage,
        transform,
        function,
        parents=parents,
    )
    return Attribution(
        stable_id=stable_id,
        tier=tier,
        request_path=request_path,
        stage=stage,
        transform=transform,
        function=function,
        parents=tuple(parents),
    )


# ---------------------------------------------------------------------------
# Determinism checks (used by tests + the comparator identity gate).
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AttributionDiff:
    """Two attribution records that should normally be equal."""

    a: str
    b: str
    differs: bool
    diff_reason: str = ""


def diff_ids(a: str, b: str) -> AttributionDiff:
    """Return a deterministic ``AttributionDiff`` for two stable IDs.

    The comparator identity gate rejects a baseline when the diff includes
    anything beyond a matching parent chain length.
    """

    if a == b:
        return AttributionDiff(a=a, b=b, differs=False)
    # The @parents suffix is the only allowed variation.
    a_head = a.split("@", 1)[0]
    b_head = b.split("@", 1)[0]
    if a_head != b_head:
        return AttributionDiff(a=a, b=b, differs=True, diff_reason="head_mismatch")
    return AttributionDiff(a=a, b=b, differs=True, diff_reason="parents_changed")


def display_form(stable_id: str) -> str:
    """Render ``stable_id`` for terminal output.

    ``G4:/v1/messages:-:-:-`` becomes ``[G4] /v1/messages`` for compactness.
    The terminal drop-in is the most readable form; release notes use the
    full stable ID.
    """

    head = stable_id.split("@", 1)[0]
    parts = head.split(":")
    tier = parts[0]
    request_path = parts[1] if len(parts) > 1 else ""
    return f"[{tier}] {request_path}".rstrip() if request_path else f"[{tier}]"

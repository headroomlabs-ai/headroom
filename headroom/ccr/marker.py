"""Central definition of the terse CCR retrieval marker (``[ccr:<id>]``).

The verbose marker — ``[N items compressed to M. Retrieve more: hash=<id>]`` —
spends ~15 tokens of fixed boilerplate on every compressed block just to say
"you can retrieve this." When ``HEADROOM_CCR_TERSE_MARKER`` is set, producers
emit the terse ``[ccr:<id>]`` form instead (~4 tokens) and the retrieval
instructions move, once, into the injected tool description (see
``ccr/tool_injection.py``). Off by default while retrieval reliability with the
terse form is validated against a live model.

``<id>`` is whatever the store returns — a short adaptive label when
``HEADROOM_CCR_SHORT_LABELS`` is on, else the full 24-hex hash. The two knobs are
orthogonal and compound: short-label shrinks the id, terse-marker shrinks the
boilerplate around it.
"""

from __future__ import annotations

import os
import re
from typing import TypeGuard

from headroom.cache.label_allocator import DEFAULT_MIN_WIDTH

# Terse retrieval marker: ``[ccr:<id>]``. The id is any run of non-bracket,
# non-space characters, so the form stays encoding-agnostic (hex today,
# base32/wordlist later) and the capture group yields the id to retrieve.
CCR_TERSE_MARKER_RE = re.compile(r"\[ccr:([^\]\s]+)\]")

# --- Canonical CCR content-id acceptance (single source of truth) ------------
#
# A CCR id is a prefix of the content hash: from the adaptive labeler's minimum
# width (``HEADROOM_CCR_SHORT_LABELS`` issues prefixes this short) up to the full
# ``SHA-256(original)[:24]`` the store emits by default. The marker SCAN (what the
# model is told it may retrieve) and the tool-call PARSE (what retrieval will
# accept) MUST agree on this range: when they drifted, the model saw ``[ccr:f2]``
# but ``headroom_retrieve(hash="f2")`` was rejected as malformed and retrieval
# never ran. Everything that validates an id derives its bounds from here.
CCR_HASH_FULL_WIDTH = 24  # SHA-256(original)[:24]; see cache/compression_store.py
CCR_HEX_ID_INNER = rf"[a-f0-9]{{{DEFAULT_MIN_WIDTH},{CCR_HASH_FULL_WIDTH}}}"
CCR_HEX_ID_RE = re.compile(rf"\A{CCR_HEX_ID_INNER}\Z")


def is_ccr_hex_id(value: object) -> TypeGuard[str]:
    """True iff ``value`` is a well-formed CCR hex id (adaptive short label .. full hash).

    Case-insensitive on the hex — the store emits lowercase, but a model may echo
    the id in either case. A ``TypeGuard`` so callers narrow the id to ``str``.
    """
    return isinstance(value, str) and CCR_HEX_ID_RE.fullmatch(value.lower()) is not None


def terse_markers_enabled() -> bool:
    """True when producers should emit ``[ccr:<id>]`` instead of the verbose marker."""
    return os.environ.get("HEADROOM_CCR_TERSE_MARKER", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def terse_marker(label: str) -> str:
    """The terse retrieval marker naming a stored content ``label``."""
    return f"[ccr:{label}]"

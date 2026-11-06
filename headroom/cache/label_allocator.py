"""Adaptive, token-efficient labels for CCR retrieval markers.

A CCR marker (``… Retrieve more: hash={hash}]``) currently embeds a fixed
24-hex content hash — ~10-14 tokens of high-entropy text paid on *every*
compressed block, even though a session rarely stashes more than a few hundred
items (a handful of ID bits). This allocator hands each stashed content hash the
SHORTEST prefix of itself that is unique among the labels issued so far, growing
width only as collisions actually demand it.

Two invariants make the labels safe to embed in a prompt-cached prefix and let
them converge on the current fixed-width strategy rather than replacing it:

* **Earliest-wins.** A collision always widens the NEWCOMER; an already-issued
  label is never re-assigned or lengthened. So replaying the same sequence of
  :meth:`label_for` calls reproduces every earlier label byte-for-byte — the
  assignment is prefix-monotonic, exactly like the turn ordinals in
  ``cross_turn_dedup``, so the upstream prompt cache never busts.
* **Content-derived.** The label is a prefix of the canonical content hash, so
  identical content dedups to one label, retrieval can verify integrity, and the
  label converges to the full hash (today's strategy) as a *ceiling* — never
  longer, and only that long if the population ever demands it.

Idempotent: :meth:`label_for` returns the same label for a repeated hash.
Thread-safe within one process. Multi-worker deployments need the issued set in
a shared backend (assignment reads the live population); that is a follow-up —
until then the allocator is per-process and used with the in-memory backend.
"""

from __future__ import annotations

import threading

# Start every label at this width. 2 hex chars (8 bits, 256 slots) tokenizes to
# ~1 token yet gives enough headroom that most sessions never widen a label — a
# strictly better token/collision trade than starting at 1 (same token cost,
# 16x fewer first-round collisions).
DEFAULT_MIN_WIDTH = 2


class CcrLabelAllocator:
    """Assigns adaptive, cache-safe short labels (prefixes of a content hash).

    See the module docstring for the earliest-wins / content-derived invariants.
    """

    def __init__(self, min_width: int = DEFAULT_MIN_WIDTH) -> None:
        if min_width < 1:
            raise ValueError(f"min_width must be >= 1, got {min_width}")
        self._min_width = min_width
        self._by_label: dict[str, str] = {}  # issued label -> full content hash
        self._by_hash: dict[str, str] = {}  # full content hash -> issued label
        self._lock = threading.Lock()

    def label_for(self, full_hash: str) -> str:
        """Return the (stable) short label for ``full_hash``.

        Idempotent: the same hash always yields the same label. A new hash gets
        the shortest prefix of itself not already issued, starting at
        ``min_width`` and widening on collision; if every prefix is taken (only
        possible against hashes that share a very long prefix) it falls back to
        the full hash — the ceiling, never longer.
        """
        if not full_hash:
            raise ValueError("full_hash must be a non-empty string")
        with self._lock:
            existing = self._by_hash.get(full_hash)
            if existing is not None:
                return existing

            label = full_hash  # ceiling fallback: reveal the whole hash
            width = min(self._min_width, len(full_hash))
            while width <= len(full_hash):
                candidate = full_hash[:width]
                if candidate not in self._by_label:
                    label = candidate
                    break
                width += 1

            self._by_label[label] = full_hash
            self._by_hash[full_hash] = label
            return label

    def resolve(self, label: str) -> str | None:
        """Full content hash for an issued ``label``, or ``None`` if unknown.

        Callers fall back to treating an unresolved key as a full hash directly,
        so pre-existing 24-hex markers (and other producers) keep resolving.
        """
        with self._lock:
            return self._by_label.get(label)

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_label)

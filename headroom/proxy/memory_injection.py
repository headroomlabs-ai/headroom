"""``MemoryInjectionBudget``: uniform token/entry cap on retrieved memory.

Pre-this-PR Headroom had NO token cap on injected memory. Top-K=10
candidates × ~400 tokens each = up to ~4000 tokens injected per
request. None of Letta/Mem0/Cognee/Supermemory ship a token-uncapped
injection path on the hot wire.

This budget is applied at the formatting boundary in
``memory_handler.search_and_format_context`` (after the backend
returns candidates, before the formatted block is appended to the
request). One value type → consistent enforcement across all five
sites.

The budget bounds are configurable; the defaults are conservative
(1024 tokens / 10 entries / 0.3 similarity floor) so a missing config
can't accidentally restore the unbounded behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Rough char-per-token heuristic for budget enforcement. Used only to
# bound the OUTPUT block (the formatted injection text) — INPUT
# fidelity is preserved by MemoryQuery (no truncation).
#
# 4 chars/token is the standard heuristic for English text. We use it
# for the cap; the actual token count at upstream is decided by the
# upstream provider's tokenizer.
_CHARS_PER_TOKEN_HEURISTIC = 4


@dataclass(frozen=True)
class MemoryInjectionBudget:
    """Frozen budget applied at the injection boundary.

    Three independent dials:

    * ``max_tokens`` — total bytes (heuristically converted) in the
      formatted injection block. Default 1024 tokens (~4KB).
    * ``max_entries`` — cap on the number of memory entries included.
      Default 10 (matches backend top_k).
    * ``min_similarity`` — floor on cosine similarity; entries below
      are dropped. Default 0.3 (matches backend default).

    Operators tune via constructor args. Defaults are hard-coded so a
    misconfigured caller can't accidentally restore unbounded
    behaviour (the pre-this-PR state).
    """

    max_tokens: int = 1024
    max_entries: int = 10
    min_similarity: float = 0.3

    def apply_to_text(self, text: str) -> str:
        """Bound a formatted injection block by ``max_tokens``.

        Truncation prefers line boundaries (memory entries are
        line-delimited so the dashboard renders intact bullet points
        rather than mid-word cuts). Empty input → empty output.

        This caps the OUTPUT block. The INPUT (the query going to the
        embedder) is NOT truncated — that's MemoryQuery's contract.
        """
        if not text:
            return text
        char_budget = self.max_tokens * _CHARS_PER_TOKEN_HEURISTIC
        if len(text) <= char_budget:
            return text
        # Truncate at the last newline at or before the budget so the
        # final included line is complete.
        cut = text.rfind("\n", 0, char_budget)
        if cut <= 0:
            # No newline within budget — fall back to hard cut.
            return text[:char_budget]
        return text[: cut + 1]

    def apply_to_entries(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Cap a list of ranked memory candidates by entry count + min similarity.

        Preserves order (assumes upstream has already ranked by
        score). The budget does NOT re-rank — it only filters/caps.

        Filtering precedence:
          1. Drop entries with ``score < min_similarity``
          2. Cap to ``max_entries``

        Entries are dict-shaped (the backend returns dicts with at
        minimum ``content`` and ``score`` keys).
        """
        filtered = [e for e in entries if float(e.get("score", 0.0)) >= self.min_similarity]
        return filtered[: self.max_entries]

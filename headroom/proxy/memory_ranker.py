"""``MemoryRanker``: pluggable re-ranker for memory candidates.

Pre-this-PR Headroom ranked memory candidates by pure cosine
similarity. Every other memory system we surveyed
(Letta / Mem0 / Cognee / Supermemory) re-ranks beyond cosine — recency,
source weight, access count, and decay are table-stakes for not
returning 6-month-old "winners" when fresh signal exists.

This module ships the first ranker — :class:`RecencyBoostRanker` —
plus the :class:`MemoryRanker` protocol that future rankers
(source-weight, access-count) plug into.

The ranker is **pure**: ``rank(candidates) -> ranked_candidates``,
no I/O, no state, no mutation of inputs. Same Rust-port shape as
``CompressionDecision`` and ``MemoryDecision``.

Performance: O(N) over candidates where N = top_k. One ``math.exp()``
per candidate. Sub-microsecond per request — no embedding compute,
no network, no disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from headroom.proxy.memory_rank_policy import (
    boost_memory_score,
    memory_recency_factor,
    parse_memory_created_at,
)

# Use ``timezone.utc`` (always available) instead of ``datetime.UTC``
# (Python 3.11+) so this module imports cleanly on older interpreters.
_UTC = timezone.utc


@dataclass(frozen=True)
class MemoryCandidate:
    """Immutable retrieval candidate as it flows through the ranker.

    The shape is the **proxy-side internal contract** — the backend's
    return type (typically ``MemoryResult`` with nested
    ``MemoryResult.memory.created_at``) is adapted into this flatter
    shape at the ranker boundary so the ranker stays backend-agnostic.

    ``score`` is the cosine similarity as returned by the backend.
    Rankers MAY mutate ``score`` by returning a new candidate with an
    updated score (frozen dataclass means they cannot mutate in place).
    """

    content: str
    score: float
    created_at: datetime | None = None
    source: str | None = None  # e.g. "memory_save" | "traffic_learner" | "inline"
    related_entities: tuple[str, ...] = field(default_factory=tuple)
    # Backend memory ID. Empty string when not preserved (test fixtures,
    # legacy callers). Rendered as ``[id]`` in the auto-tail block so the
    # model can pass it to memory_update / memory_delete directly.
    id: str = ""

    @classmethod
    def from_backend_result(cls, result: object) -> MemoryCandidate:
        """Adapter from backend ``MemoryResult`` shape to ``MemoryCandidate``.

        The backend returns objects with ``.score``, ``.memory.content``,
        ``.memory.id``, and (optionally) ``.memory.created_at`` (str ISO
        timestamp) + ``.related_entities``. This adapter flattens that to
        the ranker's expected shape and parses the timestamp to
        ``datetime``.

        Missing / unparseable timestamps → ``None`` (recency-neutral).
        Missing IDs → ``""`` (rendered as ``[?]`` in the auto-tail block).
        """
        score = float(getattr(result, "score", 0.0))
        memory = getattr(result, "memory", None)
        content = str(getattr(memory, "content", "")) if memory is not None else ""
        memory_id = str(getattr(memory, "id", "") or "") if memory is not None else ""
        raw_dt = getattr(memory, "created_at", None) if memory is not None else None
        created_at = parse_memory_created_at(raw_dt)
        raw_related = getattr(result, "related_entities", None) or ()
        related = tuple(str(x) for x in raw_related)
        source_meta = getattr(memory, "metadata", None) or {}
        source = source_meta.get("source") if isinstance(source_meta, dict) else None
        return cls(
            content=content,
            score=score,
            created_at=created_at,
            source=source,
            related_entities=related,
            id=memory_id,
        )


def _parse_created_at(value: object) -> datetime | None:
    """Best-effort parse of a timestamp into a UTC-aware datetime.

    Accepts ``datetime`` (returned as-is, UTC-normalized) or ISO-8601
    string (with or without trailing ``Z``). Anything else → ``None``
    so the ranker treats the candidate as recency-neutral.
    """
    return parse_memory_created_at(value)


class MemoryRanker(Protocol):
    """Re-ranks retrieval candidates. Pure function.

    Implementations MUST:
    * Not mutate the input list or its elements
    * Be deterministic (same input → same output) for prefix-cache
      stability across consecutive turns
    * Be backend-agnostic — work with any
      :class:`MemoryCandidate`, regardless of which backend produced it
    """

    def rank(self, candidates: list[MemoryCandidate]) -> list[MemoryCandidate]: ...


@dataclass(frozen=True)
class RecencyBoostRanker:
    """Re-ranker applying an exponential recency decay to cosine scores.

    Final score: ``cosine × exp(-age_days / decay_days)``.

    At ``decay_days=30``:
    * age = 0 days  → factor 1.000
    * age = 15 days → factor 0.607
    * age = 30 days → factor 0.368
    * age = 60 days → factor 0.135
    * age = 90 days → factor 0.050

    Tuned so a fresh memory with weak cosine doesn't dominate (factor
    decays gradually), but a 6-month-old strong-cosine can't dominate
    either (factor approaches zero). Operators tune ``decay_days`` for
    their codebase's rate of change — 7 days for rapidly-evolving
    repos, 90 days for stable archival.

    Backwards-compat: candidates with ``created_at=None`` get factor
    1.0 — treated as recency-neutral. Lets a backend during a
    migration return some rows with timestamps and some without
    without breaking the ranker.

    Defensive: negative ages (clock skew on future-timestamped rows)
    are clamped to factor 1.0 — a clock-skewed candidate cannot
    outrank a real fresh one with the same cosine.
    """

    decay_days: float = 30.0

    def rank(self, candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
        """Re-rank by ``score × recency_factor``. Pure; input unchanged.

        Sorts descending by boosted score. Ties broken by input order
        (Python's sort is stable) — deterministic output for prefix-
        cache stability across turns.
        """
        if not candidates:
            return []

        now = datetime.now(_UTC)
        boosted: list[tuple[int, MemoryCandidate, float]] = []
        for idx, c in enumerate(candidates):
            new_score = boost_memory_score(
                score=c.score,
                now=now,
                created_at=c.created_at,
                decay_days=self.decay_days,
            )
            boosted.append((idx, c, new_score))

        # Sort descending by boosted score; stable on ties via the
        # captured idx — same input order preserved on ties for
        # deterministic output across turns.
        boosted.sort(key=lambda triple: (-triple[2], triple[0]))

        # Return new candidates with the boosted score so downstream
        # consumers (logging, budget filter) see the post-boost number.
        return [
            MemoryCandidate(
                content=c.content,
                score=new_score,
                created_at=c.created_at,
                source=c.source,
                related_entities=c.related_entities,
                id=c.id,
            )
            for _, c, new_score in boosted
        ]

    def _recency_factor(self, now: datetime, created_at: datetime | None) -> float:
        """Compute the recency multiplier for a single candidate.

        ``None`` timestamp → 1.0 (recency-neutral, backwards-compat).
        Future timestamps → 1.0 (clock-skew defence).
        Otherwise: ``exp(-age_days / decay_days)``.
        """
        return memory_recency_factor(
            now=now,
            created_at=created_at,
            decay_days=self.decay_days,
        )

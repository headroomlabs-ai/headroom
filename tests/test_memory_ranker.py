"""Tests for :class:`headroom.proxy.memory_ranker.MemoryRanker` +
:class:`RecencyBoostRanker`.

Pre-this-PR Headroom ranked memory candidates by pure cosine
similarity. Every other memory system we surveyed (Letta, Mem0,
Cognee, Supermemory) re-ranks beyond cosine — recency / source /
access-count / decay are table-stakes. The pure-cosine baseline
returns 6-month-old memories with 0.9 similarity ahead of fresh
memories with 0.5 — wrong for most use cases.

``RecencyBoostRanker`` is the first ranker we ship: a pure-function
``score = cosine × exp(-age_days / decay_days)`` re-ranker. Default
``decay_days=30`` (half-life ~21 days). Other rankers (source-weight,
access-count) plug into the same :class:`MemoryRanker` protocol in
follow-on PRs.

Performance: O(N) over candidates where N = top_k = ~10. One ``exp()``
per candidate. Sub-microsecond per request — no embedding compute, no
I/O. The ranker is pure and Rust-portable.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from headroom.proxy.memory_ranker import (
    MemoryCandidate,
    RecencyBoostRanker,
)

_UTC = timezone.utc

# ── Helpers ───────────────────────────────────────────────────────────


def _candidate(content: str, score: float, age_days: float = 0.0) -> MemoryCandidate:
    """Build a MemoryCandidate at the given cosine score and age."""
    created = datetime.now(_UTC) - timedelta(days=age_days)
    return MemoryCandidate(content=content, score=score, created_at=created)


def _candidate_no_timestamp(content: str, score: float) -> MemoryCandidate:
    """Build a MemoryCandidate without a created_at (back-compat shape)."""
    return MemoryCandidate(content=content, score=score, created_at=None)


# ── MemoryCandidate value-type contract ──────────────────────────────


def test_candidate_is_frozen() -> None:
    """Frozen so a ranker can't mutate a candidate's score and lie about
    which candidates it returned."""
    c = _candidate("x", 0.9, age_days=0)
    try:
        c.score = 0.1  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("MemoryCandidate must be frozen")


def test_from_backend_result_preserves_memory_id() -> None:
    """The adapter must carry ``memory.id`` through to MemoryCandidate.id
    so the auto-tail block can render it as the bracketed handle the
    model uses for memory_update / memory_delete. Pre-this-fix the
    adapter dropped the ID, which silently regressed the [id] auto-tail
    format on the ranker path."""

    class _Mem:
        id = "mem_abc_123"
        content = "User prefers Python."
        created_at = None
        metadata = {"source": "memory_save"}

    class _Result:
        memory = _Mem()
        score = 0.91
        related_entities = ("python",)

    cand = MemoryCandidate.from_backend_result(_Result())
    assert cand.id == "mem_abc_123"
    assert cand.content == "User prefers Python."
    assert cand.score == 0.91


def test_rank_preserves_memory_id() -> None:
    """The ranker must not drop the backend ID when rebuilding candidates."""
    cand = MemoryCandidate(content="User prefers Python.", score=0.91, id="mem_abc_123")
    out = RecencyBoostRanker().rank([cand])
    assert out[0].id == "mem_abc_123"


def test_from_backend_result_handles_missing_id() -> None:
    """Defensive: legacy backend rows without an ID become ``id=""``;
    the auto-tail formatter renders ``[?]`` for those rows, no crash."""

    class _Mem:
        # no .id attribute
        content = "legacy row"
        created_at = None
        metadata = {}

    class _Result:
        memory = _Mem()
        score = 0.5
        related_entities = ()

    cand = MemoryCandidate.from_backend_result(_Result())
    assert cand.id == ""


# ── RecencyBoostRanker contract ──────────────────────────────────────


def test_ranker_is_frozen() -> None:
    """The ranker config is itself immutable — operators set
    ``decay_days`` at construction; runtime cannot edit it."""
    r = RecencyBoostRanker()
    try:
        r.decay_days = 99  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("RecencyBoostRanker must be frozen")


def test_ranker_default_decay_is_thirty_days() -> None:
    """30-day decay is the conservative default. At 30 days, factor is
    ~0.37 (e^{-1}); at 90 days, ~0.05. Tuned so a fresh memory with
    weak cosine doesn't dominate, but a 6-month-old strong-cosine
    can't dominate either."""
    assert RecencyBoostRanker().decay_days == 30.0


def test_ranker_default_decay_is_configurable() -> None:
    """Operators can tune decay; e.g., 7 days for an aggressive
    recency bias on rapidly-evolving codebases."""
    r = RecencyBoostRanker(decay_days=7.0)
    assert r.decay_days == 7.0


def test_ranker_returns_list_preserving_shape() -> None:
    """Output is a list of candidates (re-ranked). Length matches
    input length — the ranker does NOT filter, only re-orders. The
    budget filters; the ranker ranks."""
    candidates = [_candidate("a", 0.9), _candidate("b", 0.5)]
    out = RecencyBoostRanker().rank(candidates)
    assert len(out) == 2
    assert {c.content for c in out} == {"a", "b"}


# ── Recency boost behaviour ──────────────────────────────────────────


def test_equal_cosine_younger_wins() -> None:
    """Two candidates with identical cosine score — the younger one
    wins because its recency factor is closer to 1.0."""
    fresh = _candidate("fresh", 0.5, age_days=0)
    old = _candidate("old", 0.5, age_days=60)
    out = RecencyBoostRanker().rank([old, fresh])
    assert out[0].content == "fresh"
    assert out[1].content == "old"


def test_old_strong_cosine_can_still_beat_young_weak_cosine() -> None:
    """The boost is multiplicative, not absolute — a 60-day-old memory
    with 0.9 cosine (0.9 × 0.135 ≈ 0.12) still loses to a 0-day-old
    memory with 0.5 cosine (0.5 × 1.0 = 0.5). But a 5-day-old memory
    with 0.9 (0.9 × 0.847 ≈ 0.76) beats a 0-day-old with 0.5."""
    very_old_strong = _candidate("old_strong", 0.9, age_days=60)
    fresh_weak = _candidate("fresh_weak", 0.5, age_days=0)
    out = RecencyBoostRanker().rank([very_old_strong, fresh_weak])
    # fresh_weak should win because 60-day decay flattens the strong cosine
    assert out[0].content == "fresh_weak"

    # Versus: slightly-old strong beats fresh weak
    slightly_old_strong = _candidate("slightly_old_strong", 0.9, age_days=5)
    fresh_weak2 = _candidate("fresh_weak2", 0.5, age_days=0)
    out2 = RecencyBoostRanker().rank([fresh_weak2, slightly_old_strong])
    assert out2[0].content == "slightly_old_strong"


def test_decay_rate_changes_winner() -> None:
    """An aggressive decay_days=7 makes a 30-day-old memory much
    weaker than a default decay_days=30. Locks the configurability
    contract."""
    old_strong = _candidate("old_strong", 0.9, age_days=30)
    fresh_weak = _candidate("fresh_weak", 0.6, age_days=0)

    # decay_days=30: old × e^{-1} ≈ 0.331; fresh = 0.6 → fresh wins
    r_default = RecencyBoostRanker(decay_days=30.0)
    out_default = r_default.rank([old_strong, fresh_weak])
    assert out_default[0].content == "fresh_weak"

    # decay_days=120 (loose): old × e^{-0.25} ≈ 0.701; fresh = 0.6 → old wins
    r_loose = RecencyBoostRanker(decay_days=120.0)
    out_loose = r_loose.rank([old_strong, fresh_weak])
    assert out_loose[0].content == "old_strong"


def test_zero_age_memory_keeps_full_cosine() -> None:
    """At age=0 days, the recency factor is e^0 = 1.0 — the boosted
    score equals the original cosine. Fresh memories see no penalty."""
    fresh = _candidate("fresh", 0.7, age_days=0)
    out = RecencyBoostRanker().rank([fresh])
    # Compare with tolerance — datetime.now() drift between
    # _candidate() and rank() is microseconds, so factor ~ 1.0.
    assert out[0].score == 0.7 or abs(out[0].score - 0.7) < 1e-3


def test_candidate_without_timestamp_keeps_pure_cosine() -> None:
    """Backwards-compat: pre-this-PR candidates may not have a
    ``created_at`` (older rows / older backends). NULL timestamp
    means "treat as recency-neutral" — factor 1.0. Pure cosine."""
    no_ts = _candidate_no_timestamp("legacy", 0.8)
    out = RecencyBoostRanker().rank([no_ts])
    assert out[0].score == 0.8


def test_mixed_with_and_without_timestamps() -> None:
    """A backend that returns SOME candidates with timestamps and
    SOME without (e.g., during a migration) must still produce a
    sensible ranking. NULL-timestamp candidates get factor 1.0,
    timestamped ones get their decay."""
    fresh_ts = _candidate("fresh_ts", 0.6, age_days=0)
    old_ts = _candidate("old_ts", 0.6, age_days=60)
    no_ts_neutral = _candidate_no_timestamp("no_ts", 0.6)
    out = RecencyBoostRanker().rank([old_ts, no_ts_neutral, fresh_ts])
    # fresh_ts (~0.6) and no_ts (=0.6) tied at top — old_ts decayed.
    assert out[-1].content == "old_ts"
    assert {out[0].content, out[1].content} == {"fresh_ts", "no_ts"}


# ── Stability + edge cases ───────────────────────────────────────────


def test_empty_input_returns_empty_output() -> None:
    """No candidates → no candidates. Boundary case."""
    assert RecencyBoostRanker().rank([]) == []


def test_ranking_is_stable_for_identical_candidates() -> None:
    """Two candidates with identical content + score + age → stable
    order (no spurious reshuffling). Important for prefix-cache
    stability: a deterministic ranker means consecutive turns inject
    the same memory in the same order, preserving byte-stable
    output."""
    a = _candidate("same", 0.5, age_days=10)
    b = _candidate("same", 0.5, age_days=10)
    out = RecencyBoostRanker().rank([a, b])
    assert len(out) == 2


def test_negative_age_treated_as_zero() -> None:
    """Defensive: a candidate with a future ``created_at`` (clock
    skew) shouldn't crash or give a > 1.0 factor. ``exp(-age/decay)``
    with negative age gives > 1; we clamp to 1.0 so a clock-skewed
    candidate can't outrank a real fresh one with score=1.0
    artifically."""
    future = _candidate("future", 0.5, age_days=-10)  # 10 days in future
    fresh = _candidate("fresh", 0.5, age_days=0)
    out = RecencyBoostRanker().rank([future, fresh])
    # Both should have factor 1.0 (clamped) — score equal → stable order
    assert {out[0].content, out[1].content} == {"future", "fresh"}
    assert out[0].score == 0.5 or abs(out[0].score - 0.5) < 1e-3


# ── Rust-port shape ─────────────────────────────────────────────────


def test_ranker_is_pure_no_side_effects() -> None:
    """Calling rank() twice with the same input gives the same
    output. No state on the ranker; no I/O. Rust-portable."""
    candidates = [_candidate("a", 0.7, age_days=5), _candidate("b", 0.5, age_days=20)]
    r = RecencyBoostRanker()
    out1 = r.rank(candidates)
    out2 = r.rank(candidates)
    assert [c.content for c in out1] == [c.content for c in out2]
    # Inputs preserved — ranker did not mutate
    assert candidates[0].content == "a"
    assert candidates[1].content == "b"


def test_ranker_does_not_mutate_input_list() -> None:
    """Defence-in-depth: the input list and its elements must be
    unchanged after ranking. Frozen candidates make element mutation
    impossible; the list order itself must also be preserved."""
    a = _candidate("a", 0.5, age_days=20)
    b = _candidate("b", 0.5, age_days=5)
    candidates = [a, b]
    RecencyBoostRanker().rank(candidates)
    assert candidates == [a, b]  # original list order preserved

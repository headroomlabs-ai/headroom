"""Unit tests for the prompt-conditioned relevance split (Stage B core).

Uses a deterministic fake scorer -- no embedding model / network needed -- so
these run fast and pin the segmentation + partition logic, not the ML model.
"""

from __future__ import annotations

from headroom.relevance.base import RelevanceScore, RelevanceScorer
from headroom.transforms.relevance_split import (
    adaptive_threshold,
    build_relevance_query,
    plan_relevance_split,
    segment,
)


class KeywordScorer(RelevanceScorer):
    """Score = fraction of query terms present in the item. No model."""

    def score(self, item: str, context: str) -> RelevanceScore:
        terms = context.lower().split()
        if not terms:
            return RelevanceScore(score=0.0)
        hits = sum(1 for t in terms if t in item.lower())
        return RelevanceScore(score=hits / len(terms))

    def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
        return [self.score(it, context) for it in items]


def test_segment_partition_is_lossless():
    text = "a\nb\n\n  cont\nc\n"
    assert "".join(segment(text)) == text


def test_segment_windows_dense_stream_losslessly():
    text = "".join(f"line{i}\n" for i in range(20))
    segs = segment(text, window=5)
    assert "".join(segs) == text
    assert len(segs) > 1  # dense blank-free stream got windowed


def test_segment_keeps_indented_continuation_attached():
    # window=1 forces splitting, but indented continuation lines must stay
    # with their head line (stack-trace / pretty-JSON safety).
    text = "ERROR boom\n  File a.py line 1\n  File b.py line 2\nnext record\n"
    segs = segment(text, window=1)
    assert "".join(segs) == text
    for s in segs:
        assert not s.startswith((" ", "\t"))  # every segment starts at a head line


def test_split_keeps_relevant_drops_irrelevant():
    content = (
        "the oauth token refresh failed here\n"
        "\n"
        "unrelated debug noise about widgets\n"
        "\n"
        "another oauth token line\n"
    )
    runs = plan_relevance_split(content, "oauth token", KeywordScorer(), threshold=0.5)
    kept = "".join(t for k, t in runs if k)
    dropped = "".join(t for k, t in runs if not k)
    assert "oauth token" in kept
    assert "widgets" in dropped
    # partition stays lossless regardless of keep/drop labels
    assert "".join(t for _, t in runs) == content


def test_empty_query_yields_no_split():
    assert plan_relevance_split("x\ny\n", "", KeywordScorer(), threshold=0.5) == [(True, "x\ny\n")]


def test_single_record_yields_no_split():
    assert plan_relevance_split("solo", "anything", KeywordScorer(), threshold=0.5) == [
        (True, "solo")
    ]


def test_build_query_composes_prompt_and_tool_args():
    q = build_relevance_query("I need entities", "Bash", "grep -rn 'class .*Entity' src/")
    assert "entities" in q
    assert "grep" in q
    assert "Entity" in q


def test_build_query_handles_missing_pieces():
    assert build_relevance_query("", "", "") == ""
    assert build_relevance_query("just a prompt") == "just a prompt"


# --- Adaptive threshold (Otsu) --------------------------------------------------


def test_adaptive_threshold_splits_at_the_natural_gap():
    # Bimodal: cut lands in the valley between the high and low clusters, so the
    # high cluster is kept and the low one dropped -- not at a fixed constant.
    t = adaptive_threshold([0.92, 0.88, 0.12, 0.05], floor=0.25)
    assert 0.12 < t < 0.88


def test_adaptive_threshold_is_floored():
    # A mostly-irrelevant output: the natural break is low, but the floor keeps
    # us from retaining absolute junk verbatim.
    assert adaptive_threshold([0.30, 0.28, 0.05, 0.03], floor=0.25) == 0.25


def test_adaptive_threshold_all_equal_uses_floor():
    assert adaptive_threshold([0.4, 0.4, 0.4], floor=0.25) == 0.25


def test_adaptive_threshold_moves_with_distribution():
    # High-scoring output → higher cut than a low-scoring one: the bar adapts.
    high = adaptive_threshold([0.95, 0.9, 0.6, 0.55], floor=0.1)
    low = adaptive_threshold([0.4, 0.35, 0.08, 0.05], floor=0.1)
    assert high > low


# --- Router integration (real _apply_strategy_to_content path) -----------------
# Fake scorer + stubbed Kompress tail → deterministic and offline (no model).

from headroom.config import RelevanceScorerConfig  # noqa: E402
from headroom.transforms.content_router import (  # noqa: E402
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)

_SEARCH = (
    "src/auth.py:12:oauth token refresh\n"
    "src/auth.py:13:validate oauth token here\n"
    "\n"
    "src/widget.py:5:render the widget layout\n"
    "src/widget.py:6:widget styling code\n"
)


def _router(split_on: bool, *, lossless: bool = True) -> ContentRouter:
    cfg = ContentRouterConfig(
        lossless=lossless,
        relevance_split=split_on,
        relevance=RelevanceScorerConfig(tier="bm25", relevance_threshold=0.5),
    )
    r = ContentRouter(cfg)
    # Inject deterministic scorer + Kompress-tail stub (no model / network).
    r._relevance_scorer = KeywordScorer()
    r._relevance_scorer_tried = True
    r._try_ml_compressor = lambda text, ctx, question=None: ("[TAIL]", 1)  # type: ignore[assignment]
    return r


def test_router_lossless_mode_folds_only_no_drop():
    # Lossless-only mode NEVER layers a lossy drop on top of the byte-exact fold:
    # the fold is the whole answer (marker-free, fully recoverable). The relevance
    # split — which lossy-drops the low-value tail — only rides on top in lossy/CCR
    # mode (see test_router_relevance_split_fires_in_ccr_mode). So here the
    # irrelevant "widget" records must be PRESERVED, not silently dropped, and the
    # Kompress tail stub must never run.
    r = _router(split_on=True)  # lossless mode
    out, _, chain = r._apply_strategy_to_content(_SEARCH, CompressionStrategy.SEARCH, "oauth token")
    assert chain == ["lossless_search"]
    assert "oauth token" in out  # relevant records kept
    assert "widget" in out  # irrelevant tail ALSO kept — no silent drop in lossless mode
    assert "[TAIL]" not in out  # the lossy Kompress stub never fired


def test_router_relevance_split_fires_in_ccr_mode():
    # lossless=False → CCR mode. Same split, unprefixed label. The DROP tail's
    # retrieval marker is emitted by Kompress when ccr_inject_marker is on (see
    # #1721); the _try_ml_compressor stub stands in for it here. Proves the
    # split is mode-agnostic, not lossless-only.
    r = _router(split_on=True, lossless=False)
    out, _, chain = r._apply_strategy_to_content(_SEARCH, CompressionStrategy.SEARCH, "oauth token")
    assert chain == ["search", "relevance_split"]
    assert "oauth token" in out
    assert "[TAIL]" in out


def test_router_diff_stays_pure_lossless():
    r = _router(split_on=True)
    diff = "diff --git a/x b/x\nindex 111..222 100644\n@@ -1 +1 @@\n-old widget\n+new oauth token\n"
    _, _, chain = r._apply_strategy_to_content(diff, CompressionStrategy.DIFF, "oauth token")
    assert "relevance_split" not in chain  # Kompressing hunks would break apply
    assert chain == ["lossless_diff"]


def test_router_split_can_be_disabled():
    r = _router(split_on=False)
    _, _, chain = r._apply_strategy_to_content(_SEARCH, CompressionStrategy.SEARCH, "oauth token")
    assert "relevance_split" not in chain


def test_relevance_split_on_by_default_and_non_blocking(monkeypatch):
    from headroom.relevance.bm25 import BM25Scorer

    r = ContentRouter(ContentRouterConfig())
    assert r.config.relevance_split is True
    # Stub the background warm-up so this is deterministic: with a warm HF cache
    # the prewarm thread could otherwise swap in the hybrid scorer before we
    # read it. We assert the *synchronous* hot path serves BM25 without loading
    # the embedding model on the request thread (the swap happens later, in the
    # background thread — proven separately).
    monkeypatch.setattr(r, "_start_relevance_prewarm", lambda tier: None)
    assert isinstance(r._get_relevance_scorer(), BM25Scorer)


def test_split_respects_max_records_cap():
    content = "".join(f"rec {i} widget\n\n" for i in range(10))  # 10 blank-sep records
    runs = plan_relevance_split(content, "widget", KeywordScorer(), threshold=0.5, max_records=3)
    assert runs == [(True, content)]  # over the cap → no split, caller falls back

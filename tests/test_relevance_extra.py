from __future__ import annotations

import builtins
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import headroom.relevance as relevance_mod
from headroom.relevance import (
    BM25Scorer,
    EmbeddingScorer,
    HybridScorer,
    create_scorer,
    embedding,
    hybrid,
)
from headroom.relevance.base import RelevanceScore, RelevanceScorer, default_batch_score


@dataclass
class DummyRelevanceScorer(RelevanceScorer):
    def score(self, item: str, context: str) -> RelevanceScore:
        return RelevanceScore(score=0.4, reason=f"{item}:{context}")

    def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
        return [RelevanceScore(score=0.2, reason=context) for _ in items]


def test_base_default_batch_and_abstract_methods() -> None:
    scorer = DummyRelevanceScorer()
    batch = default_batch_score(scorer, ["a", "b"], "ctx")
    assert [item.reason for item in batch] == ["a:ctx", "b:ctx"]

    assert RelevanceScorer.score(scorer, "a", "ctx") is None
    assert RelevanceScorer.score_batch(scorer, ["a"], "ctx") is None
    assert RelevanceScorer.is_available() is True


def test_create_scorer_embedding_unavailable_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        relevance_mod.EmbeddingScorer, "is_available", classmethod(lambda cls: False)
    )
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        create_scorer("embedding")


def test_bm25_internal_paths_and_non_normalized_mode() -> None:
    scorer = BM25Scorer(normalize_score=False)
    assert scorer._tokenize("") == []
    assert scorer._compute_idf("x", doc_count=1, doc_freq=0) == 0.0
    assert scorer._compute_idf("x", doc_count=1, doc_freq=1) > 0
    assert scorer._bm25_score([], ["a"]) == (0.0, [])
    assert scorer._bm25_score(["a"], []) == (0.0, [])

    no_match = scorer.score("hello world", "missing")
    assert no_match.reason == "BM25: no term matches"

    one_match = scorer.score("find alice", "alice")
    assert one_match.reason == "BM25: matched 'alice'"
    assert one_match.score > 0

    many_match = scorer.score("alpha beta gamma delta", "alpha beta gamma delta")
    assert many_match.reason.startswith("BM25: matched 4 terms")

    batch = scorer.score_batch(["alpha", "alpha beta"], "alpha beta")
    assert [item.reason for item in batch] == ["BM25: 1 terms", "BM25: 2 terms"]


def test_embedding_numpy_and_model_error_paths(monkeypatch) -> None:
    embedding._numpy = None
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="numpy is required"):
        embedding._get_numpy()

    monkeypatch.setattr(builtins, "__import__", real_import)
    fake_np = SimpleNamespace(
        linalg=SimpleNamespace(norm=lambda value: 0 if value == [0, 0] else 1),
        dot=lambda a, b: -1,
    )
    monkeypatch.setattr(embedding, "_numpy", fake_np)
    assert embedding._cosine_similarity([0, 0], [1, 0]) == 0.0
    assert embedding._cosine_similarity([1, 0], [0, 1]) == 0.0

    monkeypatch.setattr(EmbeddingScorer, "is_available", classmethod(lambda cls: False))
    with pytest.raises(RuntimeError, match="requires fastembed"):
        EmbeddingScorer()._get_model()


def test_embedding_score_empty_and_batch_shortcuts() -> None:
    scorer = EmbeddingScorer()
    assert scorer.score("", "ctx").reason == "Embedding: empty input"
    assert scorer.score("item", "").reason == "Embedding: empty input"
    assert scorer.score_batch([], "ctx") == []
    assert scorer.score_batch(["item"], "")[0].reason == "Embedding: empty context"


def test_embedding_score_and_batch_with_fake_model(monkeypatch) -> None:
    scorer = EmbeddingScorer()
    monkeypatch.setattr(
        scorer,
        "_encode",
        lambda texts: (
            [[1.0, 0.0], [0.5, 0.5]] if len(texts) == 2 else [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        ),
    )
    monkeypatch.setattr(
        embedding, "_cosine_similarity", lambda a, b: 0.75 if a == [1.0, 0.0] else 0.25
    )

    single = scorer.score("item", "ctx")
    assert single.score == 0.75
    assert single.reason == "Embedding: semantic similarity 0.75"

    batch = scorer.score_batch(["first", "second"], "ctx")
    assert [item.score for item in batch] == [0.75, 0.25]
    assert [item.reason for item in batch] == ["Embedding: 0.75", "Embedding: 0.25"]


def test_hybrid_constructor_alpha_variants_and_single_score_paths(monkeypatch) -> None:
    bm25_result = RelevanceScore(score=0.1, reason="bm25", matched_terms=["term"])
    emb_result = RelevanceScore(score=0.9, reason="emb", matched_terms=[])

    class FakeBM25:
        def score(self, item: str, context: str) -> RelevanceScore:
            return bm25_result

        def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
            return [bm25_result for _ in items]

    class FakeEmbedding:
        def score(self, item: str, context: str) -> RelevanceScore:
            return emb_result

        def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
            return [emb_result for _ in items]

    scorer = HybridScorer(
        alpha=0.4, adaptive=True, bm25_scorer=FakeBM25(), embedding_scorer=FakeEmbedding()
    )
    assert scorer.has_embedding_support() is True
    assert scorer._compute_alpha("find id 1234") == 0.65
    assert scorer._compute_alpha("find host api.example.com") == 0.6
    assert scorer._compute_alpha("find email test@example.com") == 0.6

    single = scorer.score("item", "show me errors")
    assert single.score == pytest.approx(0.58)
    assert "Hybrid (α=0.40): BM25=0.10, Semantic=0.90" == single.reason

    batch = scorer.score_batch(["a", "b"], "show me errors")
    assert len(batch) == 2
    assert batch[0].reason == "Hybrid (α=0.40): BM25=0.10, Emb=0.90"


def test_hybrid_fallback_and_empty_batch(monkeypatch) -> None:
    scorer = HybridScorer(bm25_scorer=BM25Scorer())
    scorer._embedding_available = False
    scorer.embedding = None

    empty = scorer.score_batch([], "ctx")
    assert empty == []

    boosted = scorer.score('{"id":"123","name":"alice"}', "alice")
    assert boosted.score >= 0.3
    assert "BM25 only, boosted" in boosted.reason

    boosted_batch = scorer.score_batch(['{"id":"123"}', '{"id":"456"}'], "123 456")
    assert all("BM25 only, boosted" in item.reason for item in boosted_batch)


def test_hybrid_auto_fallback_when_embeddings_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(hybrid.EmbeddingScorer, "is_available", classmethod(lambda cls: False))
    scorer = HybridScorer()
    assert scorer.has_embedding_support() is False

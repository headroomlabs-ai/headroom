from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from headroom.evals import metrics


def test_normalize_tokenize_and_exact_match() -> None:
    assert metrics.normalize_text("  Hello,\nWORLD  ") == "hello, world"
    assert metrics.tokenize("Hello, world! API_v2") == ["hello", "world", "api_v2"]
    assert metrics.compute_exact_match(" Hello World ", "hello\nworld") is True
    assert metrics.compute_exact_match("hello", "world") is False


def test_f1_bleu_and_rouge_cover_edge_cases() -> None:
    assert metrics.compute_f1("", "value") == 0.0
    assert metrics.compute_f1("alpha beta", "gamma delta") == 0.0
    assert metrics.compute_f1("alpha beta gamma", "alpha gamma") == pytest.approx(0.8)

    assert metrics.compute_bleu("", "value") == 0.0
    assert metrics.compute_bleu("one", "one") == pytest.approx(1.0)
    assert metrics.compute_bleu("alpha beta", "gamma delta") == 0.0
    assert metrics.compute_bleu("alpha beta", "alpha beta gamma", max_n=4) == pytest.approx(1.0)

    assert metrics.compute_rouge_l("", "value") == 0.0
    assert metrics.compute_rouge_l("alpha beta", "gamma delta") == 0.0
    assert metrics.compute_rouge_l("alpha beta gamma", "alpha gamma") == pytest.approx(0.8)


def test_compute_semantic_similarity_and_zero_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_numpy = SimpleNamespace(
        dot=lambda a, b: sum(x * y for x, y in zip(a, b)),
        linalg=SimpleNamespace(norm=lambda a: math.sqrt(sum(x * x for x in a))),
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    class FakeModel:
        def __init__(self, embeddings: list[list[float]]) -> None:
            self.embeddings = embeddings

        def encode(self, values: list[str]) -> list[list[float]]:
            assert values == ["first", "second"]
            return self.embeddings

    monkeypatch.setattr(
        "headroom.models.ml_models.MLModelRegistry.get_sentence_transformer",
        lambda model_name=None: FakeModel([[1.0, 0.0], [1.0, 0.0]]),
    )
    assert metrics.compute_semantic_similarity("first", "second") == 1.0

    monkeypatch.setattr(
        "headroom.models.ml_models.MLModelRegistry.get_sentence_transformer",
        lambda model_name=None: FakeModel([[0.0, 0.0], [1.0, 0.0]]),
    )
    assert metrics.compute_semantic_similarity("first", "second") == 0.0


def test_compute_answer_equivalence_uses_multiple_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "compute_semantic_similarity", lambda a, b: 0.2)
    exact = metrics.compute_answer_equivalence("Answer", "answer", ground_truth="missing")
    assert exact["exact_match"] is True
    assert exact["equivalent"] is True
    assert exact["ground_truth_in_a"] is False
    assert exact["ground_truth_in_b"] is False

    monkeypatch.setattr(metrics, "compute_semantic_similarity", lambda a, b: 0.1)
    high_f1 = metrics.compute_answer_equivalence(
        "alpha beta gamma",
        "alpha gamma",
        semantic_threshold=0.95,
        f1_threshold=0.75,
    )
    assert high_f1["equivalent"] is True
    assert high_f1["semantic_similarity"] == 0.1

    monkeypatch.setattr(metrics, "compute_semantic_similarity", lambda a, b: 0.95)
    semantic = metrics.compute_answer_equivalence(
        "completely different",
        "nothing in common",
        semantic_threshold=0.9,
        f1_threshold=0.99,
    )
    assert semantic["equivalent"] is True
    assert semantic["semantic_similarity"] == 0.95

    def raise_import_error(a: str, b: str) -> float:
        raise ImportError("missing dependency")

    monkeypatch.setattr(metrics, "compute_semantic_similarity", raise_import_error)
    ground_truth = metrics.compute_answer_equivalence(
        "The capital is Paris.",
        "Paris is definitely the capital city.",
        ground_truth="paris",
        semantic_threshold=0.99,
        f1_threshold=0.99,
    )
    assert ground_truth["semantic_similarity"] is None
    assert ground_truth["ground_truth_in_a"] is True
    assert ground_truth["ground_truth_in_b"] is True
    assert ground_truth["equivalent"] is True

    not_equivalent = metrics.compute_answer_equivalence(
        "alpha beta",
        "gamma delta",
        ground_truth="omega",
        semantic_threshold=0.99,
        f1_threshold=0.99,
    )
    assert not_equivalent["equivalent"] is False


def test_information_recall_reports_preserved_and_missing_facts() -> None:
    result = metrics.compute_information_recall(
        "Alice likes pizza and Bob likes ramen.",
        "Alice likes pizza.",
        ["Alice", "Bob", "ramen", "Carol"],
    )
    assert result == {
        "total_probes": 4,
        "facts_in_original": 3,
        "facts_preserved": 1,
        "facts_lost": ["Bob", "ramen"],
        "recall": pytest.approx(1 / 3),
    }

    empty_original = metrics.compute_information_recall("No facts here", "Still none", ["Alice"])
    assert empty_original["facts_in_original"] == 0
    assert empty_original["recall"] == 1.0


def test_tool_schema_compaction_integrity() -> None:
    """Property names that collide with DROP_KEYS must survive schema compaction.

    Runs the full CompressionOnlyRunner.evaluate_tool_schema_compaction() path
    against the built-in cases and asserts zero failures.  This is zero-cost
    (no API calls) and safe for CI smoke runs.
    """
    from headroom.evals.runners.compression_only import CompressionOnlyRunner

    runner = CompressionOnlyRunner()
    result = runner.evaluate_tool_schema_compaction()

    assert result.passed, (
        f"Tool schema compaction integrity failures "
        f"({result.failed_cases}/{result.total_cases}):\n" + "\n".join(result.errors)
    )
    assert result.total_cases == 4, f"Expected 4 built-in cases, got {result.total_cases}"
    # Annotations were stripped so byte count must have shrunk.
    assert result.total_tokens_saved > 0, "Expected at least some annotation tokens to be stripped"

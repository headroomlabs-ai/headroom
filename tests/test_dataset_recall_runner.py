"""Hermetic unit tests for CompressionOnlyRunner.evaluate_dataset_recall.

Exercises the dataset-recall plumbing with synthetic JSON-array contexts (which
route through SmartCrusher / Rust — no model, no network) so it runs in the
standard [dev] shard. The weekly job drives the same method with real prose
datasets (HotpotQA), which is intentionally not exercised here.
"""

from __future__ import annotations

import json

from headroom.evals.core import EvalCase, EvalSuite
from headroom.evals.runners.compression_only import CompressionOnlyRunner


def _array_context_with(answer: str) -> str:
    """A JSON-array tool output whose error row embeds ``answer`` (a kept row)."""
    rows = [{"seq": i, "level": "INFO", "status": "ok", "msg": f"heartbeat {i}"} for i in range(30)]
    rows[14] = {"seq": 14, "level": "ERROR", "status": "failed", "msg": answer}
    return json.dumps(rows)


def _suite() -> EvalSuite:
    answer = "PaymentService NullPointerException at charge line 88"
    return EvalSuite(
        name="synthetic",
        cases=[
            # Probeable: answer is in an error row -> retained -> recall 1.0.
            EvalCase(
                id="probeable",
                context=_array_context_with(answer),
                query="what failed?",
                ground_truth=answer,
            ),
            # Skipped: trivial yes/no answer.
            EvalCase(
                id="trivial",
                context=_array_context_with(answer),
                query="did it fail?",
                ground_truth="yes",
            ),
            # Skipped: answer not present in the context at all.
            EvalCase(
                id="absent",
                context=_array_context_with(answer),
                query="?",
                ground_truth="totally-absent-token-xyz",
            ),
        ],
    )


def test_dataset_recall_counts_only_probeable_cases() -> None:
    result = CompressionOnlyRunner().evaluate_dataset_recall(_suite())
    # Only the "probeable" case is measurable; trivial + absent are skipped.
    assert result.total_cases == 1
    assert result.passed_cases == 1
    assert result.accuracy_rate == 1.0
    assert result.benchmark == "dataset_recall:synthetic"


def test_dataset_recall_empty_suite_is_safe() -> None:
    result = CompressionOnlyRunner().evaluate_dataset_recall(EvalSuite(name="empty", cases=[]))
    assert result.total_cases == 0
    assert result.accuracy_rate == 0.0
    assert result.errors == []


def test_warm_kompress_model_returns_false_when_unavailable(monkeypatch) -> None:
    # Guard path: no Kompress backend -> no download attempt, returns False.
    import headroom.transforms.kompress_compressor as kc

    monkeypatch.setattr(kc, "is_kompress_available", lambda: False)
    assert kc.warm_kompress_model() is False

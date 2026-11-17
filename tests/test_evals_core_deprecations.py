"""Deprecated EvalSuite / CompressionEvaluator methods still work and warn."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from headroom.evals.core import CompressionEvaluator, EvalCase, EvalSuite


def _case(case_id: str) -> EvalCase:
    return EvalCase(id=case_id, context="The answer is 42.", query="What is it?", ground_truth="42")


def test_deprecated_methods_warn_and_still_work(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        suite = EvalSuite(name="s")  # supported paths stay silent
        EvalSuite(name="t", cases=[_case("c0")]).to_jsonl(tmp_path / "suite.jsonl")

    with pytest.warns(DeprecationWarning, match="EvalSuite.add_case is deprecated"):
        suite.add_case(_case("c1"))
    assert [c.id for c in suite] == ["c1"]

    suite.cases.append(_case("c2"))
    evaluator = CompressionEvaluator(llm_fn=lambda context, query: "It is 42.")
    with pytest.warns(
        DeprecationWarning, match="CompressionEvaluator.evaluate_suite is deprecated"
    ):
        result = evaluator.evaluate_suite(suite)
    assert result.total_cases == 2
    assert result.passed_cases == 2
    assert [r.case_id for r in result.results] == ["c1", "c2"]

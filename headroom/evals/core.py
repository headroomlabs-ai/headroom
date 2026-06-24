"""Core evaluation infrastructure for Headroom.

This module provides the foundation for proving that compression
preserves LLM accuracy through rigorous A/B testing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig


class EvalMode(Enum):
    """Evaluation mode."""

    BEFORE_AFTER = "before_after"  # Compare original vs compressed
    GROUND_TRUTH = "ground_truth"  # Compare against known answer
    RETRIEVAL = "retrieval"  # Test information retrieval
    AGENTIC = "agentic"  # Test agent task completion


@dataclass
class EvalCase:
    """A single evaluation case.

    Attributes:
        id: Unique identifier for this case
        context: The original context (tool output, document, etc.)
        query: The question or task to perform
        ground_truth: Optional known correct answer
        metadata: Additional metadata (source, category, etc.)
    """

    id: str
    context: str
    query: str
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> EvalCase:
        return cls(
            id=d["id"],
            context=d["context"],
            query=d["query"],
            ground_truth=d.get("ground_truth"),
            metadata=d.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "context": self.context,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


@dataclass
class EvalResult:
    """Result of a single evaluation.

    Captures both the original and compressed responses,
    along with metrics comparing them.
    """

    case_id: str
    mode: EvalMode

    # Context stats
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float

    # Responses
    response_original: str
    response_compressed: str

    # Metrics
    exact_match: bool
    f1_score: float
    semantic_similarity: float | None = None
    contains_ground_truth: bool | None = None

    # Timing
    latency_original_ms: float = 0.0
    latency_compressed_ms: float = 0.0

    # Overall verdict
    accuracy_preserved: bool = True

    # Metadata
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "mode": self.mode.value,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "response_original": self.response_original[:500],  # Truncate for storage
            "response_compressed": self.response_compressed[:500],
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "semantic_similarity": self.semantic_similarity,
            "contains_ground_truth": self.contains_ground_truth,
            "latency_original_ms": self.latency_original_ms,
            "latency_compressed_ms": self.latency_compressed_ms,
            "accuracy_preserved": self.accuracy_preserved,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class EvalSuiteResult:
    """Aggregated results from an evaluation suite."""

    suite_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int

    # Aggregate metrics
    avg_compression_ratio: float
    avg_f1_score: float
    avg_semantic_similarity: float | None
    accuracy_preservation_rate: float

    # Token savings
    total_original_tokens: int
    total_compressed_tokens: int
    total_tokens_saved: int

    # Individual results
    results: list[EvalResult] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "avg_compression_ratio": self.avg_compression_ratio,
            "avg_f1_score": self.avg_f1_score,
            "avg_semantic_similarity": self.avg_semantic_similarity,
            "accuracy_preservation_rate": self.accuracy_preservation_rate,
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "total_tokens_saved": self.total_tokens_saved,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== {self.suite_name} ===",
            f"Cases: {self.passed_cases}/{self.total_cases} passed ({self.accuracy_preservation_rate:.1%})",
            f"Compression: {self.avg_compression_ratio:.1%} average",
            f"F1 Score: {self.avg_f1_score:.3f}",
        ]
        if self.avg_semantic_similarity is not None:
            lines.append(f"Semantic Similarity: {self.avg_semantic_similarity:.3f}")
        lines.append(
            f"Tokens: {self.total_original_tokens:,} → {self.total_compressed_tokens:,} "
            f"({self.total_tokens_saved:,} saved)"
        )
        return "\n".join(lines)


class EvalSuite:
    """A collection of evaluation cases."""

    def __init__(self, name: str, cases: list[EvalCase] | None = None):
        self.name = name
        self.cases: list[EvalCase] = cases or []

    def add_case(self, case: EvalCase) -> None:
        self.cases.append(case)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    @classmethod
    def from_jsonl(cls, path: Path | str, name: str | None = None) -> EvalSuite:
        """Load suite from JSONL file."""
        path = Path(path)
        cases = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    cases.append(EvalCase.from_dict(json.loads(line)))
        return cls(name=name or path.stem, cases=cases)

    def to_jsonl(self, path: Path | str) -> None:
        """Save suite to JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for case in self.cases:
                f.write(json.dumps(case.to_dict()) + "\n")


class CompressionEvaluator:
    """Main evaluator for compression accuracy.

    This is the core class that runs A/B comparisons between
    original and compressed contexts.
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str], str] | None = None,
        crusher_config: SmartCrusherConfig | None = None,
        semantic_similarity_fn: Callable[[str, str], float] | None = None,
    ):
        """Initialize evaluator.

        Args:
            llm_fn: Function that takes (context, query) and returns response.
                    If None, uses a mock for testing.
            crusher_config: Configuration for SmartCrusher.
            semantic_similarity_fn: Optional function to compute semantic similarity.
        """
        self.llm_fn = llm_fn or self._mock_llm
        self.crusher = SmartCrusher(config=crusher_config or SmartCrusherConfig())
        self.semantic_similarity_fn = semantic_similarity_fn

    def _mock_llm(self, context: str, query: str) -> str:
        """Mock LLM for testing without API calls."""
        # Return a response that includes key terms from context
        words = context.split()[:50]
        return f"Based on the context, {' '.join(words[:20])}..."

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token)."""
        return len(text) // 4

    def evaluate_case(
        self,
        case: EvalCase,
        mode: EvalMode = EvalMode.BEFORE_AFTER,
    ) -> EvalResult:
        """Evaluate a single case.

        Runs the query against both original and compressed context,
        then compares the responses.
        """
        from headroom.evals.metrics import (
            compute_exact_match,
            compute_f1,
        )

        original_tokens = self._estimate_tokens(case.context)

        # Compress the context
        try:
            compressed_result = self.crusher.crush(case.context)
            compressed_context = compressed_result.compressed
            compressed_tokens = self._estimate_tokens(compressed_context)
        except Exception:
            # If compression fails, use original
            compressed_context = case.context
            compressed_tokens = original_tokens

        compression_ratio = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0

        # Run LLM with original context
        start = time.time()
        try:
            response_original = self.llm_fn(case.context, case.query)
        except Exception as e:
            response_original = f"ERROR: {e}"
        latency_original = (time.time() - start) * 1000

        # Run LLM with compressed context
        start = time.time()
        try:
            response_compressed = self.llm_fn(compressed_context, case.query)
        except Exception as e:
            response_compressed = f"ERROR: {e}"
        latency_compressed = (time.time() - start) * 1000

        # Compute metrics
        exact_match = compute_exact_match(response_original, response_compressed)
        f1_score = compute_f1(response_original, response_compressed)

        # Semantic similarity if available
        semantic_sim = None
        if self.semantic_similarity_fn:
            try:
                semantic_sim = self.semantic_similarity_fn(response_original, response_compressed)
            except Exception:
                pass

        # Check ground truth if available
        contains_ground_truth = None
        if case.ground_truth:
            gt_lower = case.ground_truth.lower()
            contains_ground_truth = (
                gt_lower in response_compressed.lower()
                or compute_f1(response_compressed, case.ground_truth) > 0.5
            )

        # Determine if accuracy is preserved
        # Criteria: F1 > 0.8 OR semantic similarity > 0.9 OR contains ground truth
        accuracy_preserved = (
            f1_score > 0.8
            or (semantic_sim is not None and semantic_sim > 0.9)
            or (contains_ground_truth is True)
        )

        return EvalResult(
            case_id=case.id,
            mode=mode,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            response_original=response_original,
            response_compressed=response_compressed,
            exact_match=exact_match,
            f1_score=f1_score,
            semantic_similarity=semantic_sim,
            contains_ground_truth=contains_ground_truth,
            latency_original_ms=latency_original,
            latency_compressed_ms=latency_compressed,
            accuracy_preserved=accuracy_preserved,
        )

    def evaluate_suite(
        self,
        suite: EvalSuite,
        mode: EvalMode = EvalMode.BEFORE_AFTER,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EvalSuiteResult:
        """Evaluate an entire suite of cases.

        Args:
            suite: The evaluation suite to run
            mode: Evaluation mode
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Aggregated results with statistics
        """
        start_time = time.time()
        results: list[EvalResult] = []

        for i, case in enumerate(suite):
            result = self.evaluate_case(case, mode)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(suite))

        # Aggregate metrics
        passed = sum(1 for r in results if r.accuracy_preserved)
        failed = len(results) - passed

        total_original = sum(r.original_tokens for r in results)
        total_compressed = sum(r.compressed_tokens for r in results)

        avg_compression = sum(r.compression_ratio for r in results) / len(results) if results else 0
        avg_f1 = sum(r.f1_score for r in results) / len(results) if results else 0

        semantic_sims = [
            r.semantic_similarity for r in results if r.semantic_similarity is not None
        ]
        avg_semantic = sum(semantic_sims) / len(semantic_sims) if semantic_sims else None

        return EvalSuiteResult(
            suite_name=suite.name,
            total_cases=len(results),
            passed_cases=passed,
            failed_cases=failed,
            avg_compression_ratio=avg_compression,
            avg_f1_score=avg_f1,
            avg_semantic_similarity=avg_semantic,
            accuracy_preservation_rate=passed / len(results) if results else 0,
            total_original_tokens=total_original,
            total_compressed_tokens=total_compressed,
            total_tokens_saved=total_original - total_compressed,
            results=results,
            duration_seconds=time.time() - start_time,
        )

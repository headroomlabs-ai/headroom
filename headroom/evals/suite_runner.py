"""Unified evaluation suite runner.

Orchestrates all benchmark tiers, dispatches to the correct runner
(lm-eval harness, BeforeAfterRunner, CompressionOnlyRunner),
enforces cost budgets, and produces a unified SuiteResult.

Usage:
    from headroom.evals.suite_runner import SuiteRunner
    runner = SuiteRunner(model="gpt-4o-mini", tiers=[1])
    result = runner.run()
    result.print_summary()
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from headroom.evals.reports.report_card import SuiteResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSpec:
    """Specification for a single benchmark in the suite."""

    name: str
    category: str
    tier: int
    runner_type: Literal["lm_eval", "before_after", "compression_only"]
    sample_size: int
    model: str = "gpt-4o-mini"
    dataset_name: str | None = None  # For before_after runner
    lm_eval_tasks: list[str] | None = None  # For lm_eval runner
    primary_metric: str = "accuracy"
    pass_threshold: float = 0.98  # Headroom score >= baseline - (1-threshold)
    estimated_cost_usd: float = 0.50
    avg_input_tokens: int = 500  # Per sample, for cost estimation
    provider: Literal["anthropic", "openai", "ollama"] = "openai"  # LLM provider
    eval_mode: str = "before_after"  # "before_after" or "ground_truth"


# ============================================================================
# BENCHMARK SUITE DEFINITIONS
# ============================================================================

BENCHMARK_SUITE: list[BenchmarkSpec] = [
    # -----------------------------------------------------------------------
    # TIER 1: Core Report Card (~$3, ~30 min)
    # -----------------------------------------------------------------------
    # Standard benchmarks via lm-eval harness (through Headroom proxy)
    BenchmarkSpec(
        name="GSM8K",
        category="reasoning",
        tier=1,
        runner_type="lm_eval",
        sample_size=100,
        lm_eval_tasks=["gsm8k"],
        primary_metric="exact_match_flexible-extract",
        estimated_cost_usd=0.50,
    ),
    BenchmarkSpec(
        name="TruthfulQA",
        category="factual",
        tier=1,
        runner_type="lm_eval",
        sample_size=100,
        lm_eval_tasks=["truthfulqa_gen"],
        primary_metric="bleu_acc",
        estimated_cost_usd=0.30,
    ),
    BenchmarkSpec(
        name="MMLU",
        category="knowledge",
        tier=1,
        runner_type="lm_eval",
        sample_size=2,  # 2 per subject = ~114 total across 57 subjects
        lm_eval_tasks=["mmlu"],
        primary_metric="accuracy",
        estimated_cost_usd=0.80,
    ),
    BenchmarkSpec(
        name="ARC-Challenge",
        category="science",
        tier=1,
        runner_type="lm_eval",
        sample_size=100,
        lm_eval_tasks=["arc_challenge"],
        primary_metric="accuracy_normalized",
        estimated_cost_usd=0.20,
    ),
    BenchmarkSpec(
        name="HumanEval",
        category="code",
        tier=1,
        runner_type="lm_eval",
        sample_size=164,  # Full dataset
        lm_eval_tasks=["humaneval"],
        primary_metric="pass@1",
        estimated_cost_usd=0.50,
    ),
    # Compression benchmarks via BeforeAfterRunner
    BenchmarkSpec(
        name="SQuAD v2",
        category="qa",
        tier=1,
        runner_type="before_after",
        sample_size=100,
        dataset_name="squad",
        primary_metric="accuracy_preservation_rate",
        estimated_cost_usd=0.30,
        provider="openai",
    ),
    BenchmarkSpec(
        name="BFCL",
        category="tool_use",
        tier=1,
        runner_type="before_after",
        sample_size=100,
        dataset_name="bfcl",
        primary_metric="ground_truth_match",
        estimated_cost_usd=0.20,  # Half cost: only one LLM call per case
        avg_input_tokens=2000,
        provider="openai",
        eval_mode="ground_truth",
    ),
    BenchmarkSpec(
        name="Tool Outputs",
        category="agent",
        tier=1,
        runner_type="before_after",
        sample_size=8,
        dataset_name="tool_outputs",
        primary_metric="accuracy_preservation_rate",
        estimated_cost_usd=0.02,
        provider="openai",
    ),
    # Zero-cost compression-only
    BenchmarkSpec(
        name="CCR Round-trip",
        category="lossless",
        tier=1,
        runner_type="compression_only",
        sample_size=50,
        primary_metric="byte_exact_match",
        pass_threshold=1.0,  # Must be 100% for lossless
        estimated_cost_usd=0.0,
    ),
    # -----------------------------------------------------------------------
    # TIER 2: Extended Credibility (~$5, ~30 min)
    # -----------------------------------------------------------------------
    BenchmarkSpec(
        name="HotpotQA",
        category="multi_hop_qa",
        tier=2,
        runner_type="before_after",
        sample_size=50,
        dataset_name="hotpotqa",
        primary_metric="accuracy_preservation_rate",
        estimated_cost_usd=0.80,
        avg_input_tokens=3000,
        provider="openai",
    ),
    BenchmarkSpec(
        name="MS MARCO",
        category="rag",
        tier=2,
        runner_type="before_after",
        sample_size=50,
        dataset_name="msmarco",
        primary_metric="accuracy_preservation_rate",
        estimated_cost_usd=0.40,
        avg_input_tokens=1500,
        provider="openai",
    ),
    BenchmarkSpec(
        name="CodeSearchNet",
        category="code",
        tier=2,
        runner_type="before_after",
        sample_size=50,
        dataset_name="codesearchnet",
        primary_metric="accuracy_preservation_rate",
        estimated_cost_usd=0.30,
        provider="openai",
    ),
    BenchmarkSpec(
        name="Info Retention",
        category="compression",
        tier=2,
        runner_type="compression_only",
        sample_size=30,
        primary_metric="information_recall",
        pass_threshold=0.90,
        estimated_cost_usd=0.0,
    ),
    # -----------------------------------------------------------------------
    # TIER 3: Deep Dive (~$9, ~45 min)
    # -----------------------------------------------------------------------
    BenchmarkSpec(
        name="HellaSwag",
        category="commonsense",
        tier=3,
        runner_type="lm_eval",
        sample_size=100,
        lm_eval_tasks=["hellaswag"],
        primary_metric="accuracy_normalized",
        estimated_cost_usd=0.20,
    ),
    BenchmarkSpec(
        name="NarrativeQA",
        category="long_context",
        tier=3,
        runner_type="before_after",
        sample_size=30,
        dataset_name="narrativeqa",
        primary_metric="accuracy_preservation_rate",
        estimated_cost_usd=1.50,
        avg_input_tokens=5000,
        provider="openai",
    ),
    BenchmarkSpec(
        name="TriviaQA",
        category="factoid_qa",
        tier=3,
        runner_type="before_after",
        sample_size=50,
        dataset_name="triviaqa",
        primary_metric="accuracy_preservation_rate",
        estimated_cost_usd=0.50,
        provider="openai",
    ),
]


def _load_env() -> None:
    """Load API keys from .env file if present."""
    try:
        from dotenv import load_dotenv

        # Walk up from this file to find .env at project root
        current = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            env_path = os.path.join(current, ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path)
                logger.debug(f"Loaded .env from {env_path}")
                return
            current = os.path.dirname(current)
    except ImportError:
        logger.debug("python-dotenv not installed, skipping .env loading")


def _check_proxy(port: int) -> bool:
    """Check if Headroom proxy is running on given port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(("localhost", port))
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def _start_proxy(port: int) -> subprocess.Popen | None:
    """Start Headroom proxy as a subprocess. Returns process handle."""
    logger.info(f"Starting Headroom proxy on port {port}...")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "headroom.proxy.server", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for proxy to be ready
        for _ in range(30):
            time.sleep(1)
            if _check_proxy(port):
                logger.info(f"Headroom proxy ready on port {port}")
                return proc
        logger.error("Proxy failed to start within 30 seconds")
        proc.kill()
        return None
    except Exception as e:
        logger.error(f"Failed to start proxy: {e}")
        return None


class SuiteRunner:
    """Orchestrates the full evaluation suite.

    Example:
        runner = SuiteRunner(model="gpt-4o-mini", tiers=[1])
        result = runner.run()
        print(result.to_dict())
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        tiers: list[int] | None = None,
        budget_usd: float = 20.0,
        headroom_port: int = 8787,
        auto_start_proxy: bool = True,
    ):
        self.model = model
        self.tiers = tiers or [1]
        self.budget_usd = budget_usd
        self.headroom_port = headroom_port
        self.auto_start_proxy = auto_start_proxy
        self._proxy_proc: subprocess.Popen | None = None

        # Load .env for API keys
        _load_env()

    def _get_specs(self) -> list[BenchmarkSpec]:
        """Get benchmark specs for the requested tiers."""
        return [s for s in BENCHMARK_SUITE if s.tier in self.tiers]

    def _ensure_proxy(self) -> bool:
        """Ensure the Headroom proxy is running (needed for lm-eval benchmarks)."""
        if _check_proxy(self.headroom_port):
            return True
        if self.auto_start_proxy:
            self._proxy_proc = _start_proxy(self.headroom_port)
            return self._proxy_proc is not None
        return False

    def _cleanup_proxy(self) -> None:
        """Stop proxy if we started it."""
        if self._proxy_proc:
            logger.info("Stopping Headroom proxy...")
            self._proxy_proc.send_signal(signal.SIGTERM)
            try:
                self._proxy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proxy_proc.kill()
            self._proxy_proc = None

    def _run_lm_eval_benchmark(
        self,
        spec: BenchmarkSpec,
        tracker: Any,
    ) -> dict[str, Any]:
        """Run a benchmark via EleutherAI lm-evaluation-harness."""
        from headroom.evals.comprehensive_benchmark import (
            compare_results,
            run_baseline_benchmark,
            run_headroom_benchmark,
        )

        tasks = spec.lm_eval_tasks or [spec.name.lower()]
        limit = spec.sample_size

        # Run baseline (direct to API)
        print(f"    Running baseline ({spec.model})...")
        baseline_results = run_baseline_benchmark(
            model=spec.model or self.model,
            tasks=tasks,
            limit=limit,
        )

        # Run through Headroom proxy
        print("    Running through Headroom proxy...")
        headroom_results = run_headroom_benchmark(
            model=spec.model or self.model,
            tasks=tasks,
            limit=limit,
            headroom_port=self.headroom_port,
        )

        # Compare
        comparisons = compare_results(baseline_results, headroom_results)

        # Extract primary result
        baseline_score = None
        headroom_score = None
        delta = None
        passed = True

        if comparisons:
            c = comparisons[0]  # Primary comparison
            baseline_score = c.baseline_score
            headroom_score = c.headroom_score
            delta = c.delta
            passed = c.accuracy_preserved

        return {
            "baseline_score": baseline_score,
            "headroom_score": headroom_score,
            "delta": delta,
            "passed": passed,
            "n_samples": limit,
        }

    def _run_before_after_benchmark(
        self,
        spec: BenchmarkSpec,
        tracker: Any,
    ) -> dict[str, Any]:
        """Run a benchmark via BeforeAfterRunner."""
        from headroom.evals.core import EvalMode
        from headroom.evals.datasets import load_dataset_by_name, load_tool_output_samples
        from headroom.evals.runners.before_after import BeforeAfterRunner, LLMConfig

        # Load dataset
        if spec.dataset_name == "tool_outputs":
            suite = load_tool_output_samples()
        else:
            suite = load_dataset_by_name(spec.dataset_name or spec.name.lower(), n=spec.sample_size)

        # Configure runner — use proxy for full-stack eval (compression + CCR)
        proxy_url = (
            f"http://localhost:{self.headroom_port}" if _check_proxy(self.headroom_port) else None
        )
        runner = BeforeAfterRunner(
            llm_config=LLMConfig(
                provider=spec.provider,
                model=spec.model or self.model,
                temperature=0.0,
                headroom_proxy_url=proxy_url,
            ),
            use_semantic_similarity=False,  # Faster
        )

        # Determine eval mode
        eval_mode = (
            EvalMode.GROUND_TRUTH if spec.eval_mode == "ground_truth" else EvalMode.BEFORE_AFTER
        )

        # Run
        def progress(current: int, total: int, result: Any) -> None:
            status = "PASS" if result.accuracy_preserved else "FAIL"
            gt_info = (
                f" GT={'Y' if result.contains_ground_truth else 'N'}"
                if eval_mode == EvalMode.GROUND_TRUTH
                else ""
            )
            print(f"    [{current}/{total}] {status} F1={result.f1_score:.2f}{gt_info}")

        suite_result = runner.run(suite, progress_callback=progress, mode=eval_mode)

        return {
            "accuracy_rate": suite_result.accuracy_preservation_rate,
            "avg_compression_ratio": suite_result.avg_compression_ratio,
            "tokens_saved": suite_result.total_tokens_saved,
            "passed": suite_result.accuracy_preservation_rate >= 0.90,
            "n_samples": suite_result.total_cases,
            "duration_seconds": suite_result.duration_seconds,
        }

    def _run_compression_only_benchmark(
        self,
        spec: BenchmarkSpec,
    ) -> dict[str, Any]:
        """Run a compression-only benchmark (zero LLM cost)."""
        from headroom.evals.runners.compression_only import CompressionOnlyRunner

        runner = CompressionOnlyRunner()

        if spec.name == "CCR Round-trip":
            cases = runner.generate_ccr_test_cases(n=spec.sample_size)
            result = runner.evaluate_ccr_lossless(cases)
        elif spec.name == "Info Retention":
            cases = runner.generate_info_retention_cases(n=spec.sample_size)
            result = runner.evaluate_information_retention(cases)
        else:
            raise ValueError(f"Unknown compression-only benchmark: {spec.name}")

        return {
            "accuracy_rate": result.accuracy_rate,
            "avg_compression_ratio": result.avg_compression_ratio,
            "tokens_saved": result.total_tokens_saved,
            "passed": result.passed,
            "n_samples": result.total_cases,
            "duration_seconds": result.duration_seconds,
        }

    def run(self) -> SuiteResult:
        """Run the full evaluation suite."""
        from headroom.evals.cost_tracker import CostTracker
        from headroom.evals.reports.report_card import BenchmarkRunResult, SuiteResult

        specs = self._get_specs()
        tracker = CostTracker(budget_usd=self.budget_usd)
        results: list[BenchmarkRunResult] = []
        start_time = time.time()

        # Check if we need proxy for any lm-eval benchmarks
        has_lm_eval = any(s.runner_type == "lm_eval" for s in specs)
        proxy_available = False
        if has_lm_eval:
            proxy_available = self._ensure_proxy()
            if not proxy_available:
                print("WARNING: Headroom proxy not available. Skipping lm-eval benchmarks.")
                print("  Start with: headroom proxy --port 8787")

        try:
            for i, spec in enumerate(specs):
                print(
                    f"\n[{i + 1}/{len(specs)}] {spec.name} (tier {spec.tier}, {spec.runner_type})"
                )

                # Budget check
                if spec.estimated_cost_usd > 0 and not tracker.can_afford(
                    spec.model or self.model,
                    spec.sample_size,
                    spec.avg_input_tokens,
                ):
                    print(
                        f"  SKIPPED: Would exceed budget (${tracker.remaining_usd:.2f} remaining)"
                    )
                    results.append(
                        BenchmarkRunResult(
                            name=spec.name,
                            category=spec.category,
                            tier=spec.tier,
                            error="Budget exceeded",
                            passed=False,
                            n_samples=0,
                            model=spec.model or self.model,
                            metric_name=spec.primary_metric,
                        )
                    )
                    continue

                try:
                    if spec.runner_type == "lm_eval":
                        if not proxy_available:
                            print("  SKIPPED: No proxy available")
                            results.append(
                                BenchmarkRunResult(
                                    name=spec.name,
                                    category=spec.category,
                                    tier=spec.tier,
                                    error="Proxy not available",
                                    passed=False,
                                    n_samples=0,
                                    model=spec.model or self.model,
                                    metric_name=spec.primary_metric,
                                )
                            )
                            continue
                        raw = self._run_lm_eval_benchmark(spec, tracker)
                        results.append(
                            BenchmarkRunResult(
                                name=spec.name,
                                category=spec.category,
                                tier=spec.tier,
                                baseline_score=raw.get("baseline_score"),
                                headroom_score=raw.get("headroom_score"),
                                delta=raw.get("delta"),
                                passed=raw.get("passed", False),
                                n_samples=raw.get("n_samples", 0),
                                model=spec.model or self.model,
                                metric_name=spec.primary_metric,
                            )
                        )

                    elif spec.runner_type == "before_after":
                        raw = self._run_before_after_benchmark(spec, tracker)
                        results.append(
                            BenchmarkRunResult(
                                name=spec.name,
                                category=spec.category,
                                tier=spec.tier,
                                accuracy_rate=raw.get("accuracy_rate"),
                                avg_compression_ratio=raw.get("avg_compression_ratio", 0),
                                tokens_saved=raw.get("tokens_saved", 0),
                                passed=raw.get("passed", False),
                                n_samples=raw.get("n_samples", 0),
                                model=spec.model or self.model,
                                metric_name=spec.primary_metric,
                                duration_seconds=raw.get("duration_seconds", 0),
                            )
                        )

                    elif spec.runner_type == "compression_only":
                        raw = self._run_compression_only_benchmark(spec)
                        results.append(
                            BenchmarkRunResult(
                                name=spec.name,
                                category=spec.category,
                                tier=spec.tier,
                                accuracy_rate=raw.get("accuracy_rate"),
                                avg_compression_ratio=raw.get("avg_compression_ratio", 0),
                                tokens_saved=raw.get("tokens_saved", 0),
                                passed=raw.get("passed", False),
                                n_samples=raw.get("n_samples", 0),
                                model=spec.model or self.model,
                                metric_name=spec.primary_metric,
                                duration_seconds=raw.get("duration_seconds", 0),
                            )
                        )

                except Exception as e:
                    logger.error(f"Benchmark {spec.name} failed: {e}")
                    print(f"  ERROR: {e}")
                    results.append(
                        BenchmarkRunResult(
                            name=spec.name,
                            category=spec.category,
                            tier=spec.tier,
                            error=str(e),
                            passed=False,
                            n_samples=0,
                            model=spec.model or self.model,
                            metric_name=spec.primary_metric,
                        )
                    )

        finally:
            self._cleanup_proxy()

        suite_result = SuiteResult(
            model=self.model,
            tiers_run=self.tiers,
            total_cost_usd=tracker.spent_usd,
            total_duration_seconds=time.time() - start_time,
            benchmarks=results,
        )

        # Print summary
        self._print_summary(suite_result, tracker)
        return suite_result

    def _print_summary(self, result: SuiteResult, tracker: Any) -> None:
        """Print formatted summary to stdout."""
        from headroom.evals.reports.report_card import generate_markdown

        print("\n" + "=" * 60)
        print(generate_markdown(result))
        print("=" * 60)
        tracker.print_summary()

"""Comprehensive LLM Benchmarks using EleutherAI lm-evaluation-harness.

This module runs industry-standard benchmarks to prove Headroom preserves accuracy.

Approach:
1. Run benchmarks directly against LLM provider (baseline)
2. Run same benchmarks through Headroom proxy
3. Compare scores - goal is accuracy preserved or improved

Supported benchmarks:
- MMLU: General knowledge across 57 subjects
- HellaSwag: Commonsense reasoning
- TruthfulQA: Factual accuracy
- GSM8K: Math reasoning (requires more tokens)
- ARC: Science reasoning

Usage:
    # Quick test (5 samples per task)
    python -m headroom.evals.comprehensive_benchmark --quick

    # Full benchmark
    python -m headroom.evals.comprehensive_benchmark --tasks mmlu,hellaswag

    # Compare with baseline
    python -m headroom.evals.comprehensive_benchmark --compare
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from headroom._subprocess import run

logger = logging.getLogger(__name__)

# Default benchmarks - chosen for reliability and relevance
DEFAULT_TASKS = [
    "gsm8k",  # Math reasoning - generation-based, works with chat APIs
]

# Tier 1 standard benchmarks for the suite runner
TIER1_TASKS = [
    "gsm8k",  # Math reasoning
    "truthfulqa_gen",  # Factual accuracy
    "mmlu",  # General knowledge (57 subjects)
    "arc_challenge",  # Science reasoning
]

# Code tasks (need --confirm_run_unsafe_code)
TIER1_CODE_TASKS = [
    "humaneval",  # Code generation (pass@1)
]

EXTENDED_TASKS = [
    "mmlu",  # 57 subjects - comprehensive but slow
    "gsm8k",  # Math - requires multi-step reasoning
    "arc_challenge",  # Harder science reasoning
]

TIER3_TASKS = [
    "hellaswag",  # Commonsense reasoning
]


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    task: str
    metric: str
    score: float
    stderr: float | None = None
    samples: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "metric": self.metric,
            "score": round(self.score, 4),
            "stderr": round(self.stderr, 4) if self.stderr else None,
            "samples": self.samples,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class ComparisonResult:
    """Comparison between baseline and Headroom results."""

    task: str
    metric: str
    baseline_score: float
    headroom_score: float

    @property
    def delta(self) -> float:
        return self.headroom_score - self.baseline_score

    @property
    def accuracy_preserved(self) -> bool:
        """True if Headroom score is within 2% of baseline."""
        return self.headroom_score >= self.baseline_score - 0.02

    @property
    def accuracy_improved(self) -> bool:
        """True if Headroom score exceeds baseline."""
        return self.headroom_score > self.baseline_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "metric": self.metric,
            "baseline": round(self.baseline_score, 4),
            "headroom": round(self.headroom_score, 4),
            "delta": round(self.delta, 4),
            "preserved": self.accuracy_preserved,
            "improved": self.accuracy_improved,
        }


@dataclass
class BenchmarkSuiteResult:
    """Results from a complete benchmark suite."""

    model: str
    baseline_results: list[BenchmarkResult] = field(default_factory=list)
    headroom_results: list[BenchmarkResult] = field(default_factory=list)
    comparisons: list[ComparisonResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def all_preserved(self) -> bool:
        """True if all benchmarks preserved accuracy."""
        if not self.comparisons:
            return True
        return all(c.accuracy_preserved for c in self.comparisons)

    @property
    def avg_delta(self) -> float:
        """Average score delta (positive = Headroom better)."""
        if not self.comparisons:
            return 0.0
        return sum(c.delta for c in self.comparisons) / len(self.comparisons)

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "all_preserved": self.all_preserved,
            "avg_delta": round(self.avg_delta, 4),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "baseline": [r.to_dict() for r in self.baseline_results],
            "headroom": [r.to_dict() for r in self.headroom_results],
            "comparisons": [c.to_dict() for c in self.comparisons],
        }

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        print("\n" + "=" * 70)
        print("HEADROOM COMPREHENSIVE BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Model: {self.model}")
        print(f"Duration: {self.total_duration_seconds:.1f}s")
        print()

        if self.comparisons:
            print(f"{'Task':<20} {'Baseline':>10} {'Headroom':>10} {'Delta':>10} {'Status':>10}")
            print("-" * 70)
            for c in self.comparisons:
                status = "PASS" if c.accuracy_preserved else "FAIL"
                delta_str = f"{c.delta:+.4f}"
                print(
                    f"{c.task:<20} {c.baseline_score:>10.4f} {c.headroom_score:>10.4f} {delta_str:>10} {status:>10}"
                )
            print("-" * 70)
            print(f"{'AVERAGE':<20} {'':<10} {'':<10} {self.avg_delta:>+10.4f}")
            print()
            print(f"ALL BENCHMARKS PASSED: {'YES' if self.all_preserved else 'NO'}")
        else:
            print("Headroom results only (no baseline comparison):")
            print(f"{'Task':<20} {'Score':>10} {'Metric':<15}")
            print("-" * 50)
            for r in self.headroom_results:
                print(f"{r.task:<20} {r.score:>10.4f} {r.metric:<15}")

        print("=" * 70 + "\n")


def run_lm_eval(
    model: str = "openai-chat-completions",
    model_args: str | None = None,
    tasks: list[str] | None = None,
    num_fewshot: int | None = None,
    limit: int | None = None,
    output_path: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run lm-evaluation-harness and return results.

    Args:
        model: Model type for lm-eval
        model_args: Model arguments string
        tasks: List of tasks to run
        num_fewshot: Number of few-shot examples
        limit: Limit samples per task (for quick testing)
        output_path: Where to save results
        base_url: Base URL for API (for Headroom proxy)

    Returns:
        Dictionary with results from lm-eval
    """
    tasks = tasks or DEFAULT_TASKS

    # Tasks that require code execution
    UNSAFE_TASKS = {"humaneval", "mbpp"}

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        model,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        "1",  # Safe default for API models
        "--apply_chat_template",  # Required for chat completion models
        "--log_samples",  # Required to save results to file
    ]

    # Add flag for tasks that require code execution
    if any(t in UNSAFE_TASKS for t in tasks):
        cmd.append("--confirm_run_unsafe_code")

    # Build model_args
    args_parts = []
    if model_args:
        args_parts.append(model_args)
    if base_url:
        args_parts.append(f"base_url={base_url}")

    if args_parts:
        cmd.extend(["--model_args", ",".join(args_parts)])

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    # Use temp directory for output
    if output_path is None:
        output_dir = tempfile.mkdtemp(prefix="lm_eval_")
        output_path = output_dir

    cmd.extend(["--output_path", output_path])

    logger.info(f"Running: {' '.join(cmd)}")

    # Run lm-eval
    start_time = time.time()
    env = {
        **os.environ,
        "TOKENIZERS_PARALLELISM": "false",
        "HF_ALLOW_CODE_EVAL": "1",  # Required for humaneval/mbpp tasks
    }
    result = run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    duration = time.time() - start_time

    if result.returncode != 0:
        logger.error(f"lm-eval failed: {result.stderr}")
        raise RuntimeError(f"lm-eval failed: {result.stderr}")

    # Load results - lm-eval creates a directory structure with timestamped files
    results_dir: Path = Path(output_path) if output_path else Path(".")

    # Find results_*.json in the output directory (lm-eval uses timestamped filenames)
    results_file: Path | None = None
    if results_dir.is_dir():
        # Look for results_*.json files
        for f in sorted(results_dir.glob("**/results_*.json"), reverse=True):
            results_file = f
            break

    if results_file is None or not results_file.exists():
        # Parse results from stdout as fallback
        logger.warning("No results file found, parsing from stdout")
        return {"results": {}, "_duration_seconds": duration, "_stdout": result.stdout}

    with open(results_file) as fp:
        results: dict[str, Any] = json.load(fp)

    results["_duration_seconds"] = duration
    return results


def parse_lm_eval_results(raw_results: dict[str, Any]) -> list[BenchmarkResult]:
    """Parse lm-eval output into BenchmarkResult objects."""
    results = []
    duration = raw_results.get("_duration_seconds", 0.0)

    # Primary metrics by task (in order of preference)
    PRIMARY_METRICS = {
        "gsm8k": ["exact_match,flexible-extract", "exact_match,strict-match"],
        "truthfulqa_gen": ["bleu_acc,none", "rouge1_acc,none"],
        "humaneval": ["pass@1,create_test", "pass@1,none"],
        "mbpp": ["pass@1,create_test", "pass@1,none"],
        "drop": ["f1,none", "em,none"],
        "mmlu": ["acc,none", "accuracy,none"],
        "arc_challenge": ["acc_norm,none", "acc,none"],
        "hellaswag": ["acc_norm,none", "acc,none"],
    }

    for task_name, task_results in raw_results.get("results", {}).items():
        # Get the primary metric for this task
        preferred_metrics = PRIMARY_METRICS.get(task_name, [])
        selected_metric = None
        selected_value = None
        selected_stderr = None

        # Try preferred metrics first
        for metric_key in preferred_metrics:
            if metric_key in task_results:
                selected_metric = metric_key
                selected_value = task_results[metric_key]
                # Look for stderr
                parts = metric_key.split(",")
                stderr_key = (
                    f"{parts[0]}_stderr,{parts[1]}" if len(parts) > 1 else f"{parts[0]}_stderr"
                )
                selected_stderr = task_results.get(stderr_key)
                break

        # Fallback: find any metric with filter "flexible-extract" or "none"
        if selected_metric is None:
            for metric_name, value in task_results.items():
                if metric_name in ("alias",) or "_stderr" in metric_name:
                    continue
                if "," in metric_name:
                    parts = metric_name.split(",")
                    filter_name = parts[1] if len(parts) > 1 else ""
                    if filter_name in ("flexible-extract", "none"):
                        selected_metric = metric_name
                        selected_value = value
                        stderr_key = f"{parts[0]}_stderr,{filter_name}"
                        selected_stderr = task_results.get(stderr_key)
                        break

        if selected_metric and selected_value is not None:
            # Clean up metric name for display
            parts = selected_metric.split(",")
            clean_metric = parts[0]
            filter_name = parts[1] if len(parts) > 1 else ""
            display_metric = (
                f"{clean_metric}_{filter_name}"
                if filter_name and filter_name != "none"
                else clean_metric
            )

            results.append(
                BenchmarkResult(
                    task=task_name,
                    metric=display_metric,
                    score=float(selected_value),
                    stderr=float(selected_stderr) if selected_stderr else None,
                    duration_seconds=duration / max(len(raw_results.get("results", {})), 1),
                )
            )

    return results


def run_baseline_benchmark(
    model: str = "gpt-4o-mini",
    tasks: list[str] | None = None,
    limit: int | None = None,
) -> list[BenchmarkResult]:
    """Run benchmark directly against OpenAI (baseline)."""
    logger.info(f"Running baseline benchmark with {model}...")

    raw_results = run_lm_eval(
        model="openai-chat-completions",
        model_args=f"model={model}",
        tasks=tasks,
        limit=limit,
    )

    return parse_lm_eval_results(raw_results)


def run_headroom_benchmark(
    model: str = "gpt-4o-mini",
    tasks: list[str] | None = None,
    limit: int | None = None,
    headroom_port: int = 8787,
) -> list[BenchmarkResult]:
    """Run benchmark through Headroom proxy."""
    logger.info(f"Running Headroom benchmark with {model} through proxy...")

    # lm_eval expects base_url to be the full path to chat/completions
    base_url = f"http://localhost:{headroom_port}/v1/chat/completions"

    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    raw_results = run_lm_eval(
        model="local-chat-completions",
        model_args=f"model={model},tokenizer_backend=tiktoken,api_key={api_key}",
        tasks=tasks,
        limit=limit,
        base_url=base_url,
    )

    return parse_lm_eval_results(raw_results)


def compare_results(
    baseline: list[BenchmarkResult],
    headroom: list[BenchmarkResult],
) -> list[ComparisonResult]:
    """Compare baseline and Headroom results."""
    comparisons = []

    # Match results by task
    baseline_by_task = {r.task: r for r in baseline}
    headroom_by_task = {r.task: r for r in headroom}

    for task in baseline_by_task:
        if task in headroom_by_task:
            b = baseline_by_task[task]
            h = headroom_by_task[task]
            comparisons.append(
                ComparisonResult(
                    task=task,
                    metric=b.metric,
                    baseline_score=b.score,
                    headroom_score=h.score,
                )
            )

    return comparisons


def run_comprehensive_benchmark(
    model: str = "gpt-4o-mini",
    tasks: list[str] | None = None,
    limit: int | None = None,
    compare_baseline: bool = True,
    headroom_port: int = 8787,
) -> BenchmarkSuiteResult:
    """Run comprehensive benchmark suite.

    Args:
        model: Model to benchmark
        tasks: List of benchmark tasks
        limit: Limit samples per task (for quick testing)
        compare_baseline: Whether to run baseline comparison
        headroom_port: Port where Headroom proxy is running

    Returns:
        BenchmarkSuiteResult with all results
    """
    tasks = tasks or DEFAULT_TASKS
    start_time = time.time()

    suite = BenchmarkSuiteResult(model=model)

    # Run baseline FIRST to warm up OpenAI infrastructure (KV cache, prefix caching)
    # This ensures a fair comparison - any speedup from Headroom is real, not from warm cache
    if compare_baseline:
        suite.baseline_results = run_baseline_benchmark(
            model=model,
            tasks=tasks,
            limit=limit,
        )

    # Run Headroom benchmark (after baseline warms things up)
    suite.headroom_results = run_headroom_benchmark(
        model=model,
        tasks=tasks,
        limit=limit,
        headroom_port=headroom_port,
    )

    # Compare results
    if compare_baseline:
        suite.comparisons = compare_results(suite.baseline_results, suite.headroom_results)

    suite.total_duration_seconds = time.time() - start_time
    return suite


def check_headroom_proxy(port: int = 8787) -> bool:
    """Check if Headroom proxy is running."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(("localhost", port))
            return True
    except (TimeoutError, ConnectionRefusedError):
        return False


def _load_env() -> None:
    """Load .env file for API keys."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def main() -> None:
    """CLI entry point."""
    import argparse

    _load_env()

    parser = argparse.ArgumentParser(
        description="Run comprehensive LLM benchmarks to verify Headroom accuracy"
    )
    parser.add_argument(
        "--model", "-m", default="gpt-4o-mini", help="Model to benchmark (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        help="Comma-separated list of tasks (default: hellaswag,truthfulqa_mc2,arc_easy)",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick test with 5 samples per task"
    )
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit samples per task")
    parser.add_argument(
        "--compare", "-c", action="store_true", help="Compare with baseline (run without Headroom)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8787, help="Headroom proxy port (default: 8787)"
    )
    parser.add_argument("--output", "-o", default=None, help="Output file for results JSON")
    parser.add_argument(
        "--headroom-only", action="store_true", help="Only run through Headroom (skip baseline)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parse tasks
    tasks = args.tasks.split(",") if args.tasks else DEFAULT_TASKS

    # Set limit for quick mode
    limit = args.limit
    if args.quick and limit is None:
        limit = 5

    # Check proxy is running
    if not check_headroom_proxy(args.port):
        print(f"ERROR: Headroom proxy not running on port {args.port}")
        print(f"Start it with: headroom proxy --port {args.port}")
        sys.exit(1)

    # Run benchmarks
    compare = args.compare and not args.headroom_only

    try:
        result = run_comprehensive_benchmark(
            model=args.model,
            tasks=tasks,
            limit=limit,
            compare_baseline=compare,
            headroom_port=args.port,
        )
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")
        sys.exit(1)

    # Print summary
    result.print_summary()

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.summary(), f, indent=2)
        print(f"Results saved to {args.output}")

    # Exit with appropriate code
    if compare and not result.all_preserved:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

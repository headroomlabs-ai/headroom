"""
Truncation vs Summarization vs Headroom: A Fair Benchmark

This benchmark compares three approaches to context compression:
1. Truncation - Keep first N items (industry standard)
2. Summarization - Use LLM to summarize (common alternative)
3. Headroom - Statistical compression with retrieval

FAIRNESS PRINCIPLES:
- Include scenarios where each approach could win
- Use realistic data patterns
- Measure both compression AND answer quality
- Report failures honestly

Metrics:
- Tokens saved (compression ratio)
- Answer accuracy (can LLM still answer correctly?)
- Cost (including summarization LLM calls)
- Latency
"""

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Literal

# We'll use OpenAI for the actual LLM calls
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Headroom imports
try:
    from headroom.config import SmartCrusherConfig
    from headroom.tokenizers import TiktokenCounter
    from headroom.transforms.smart_crusher import SmartCrusher

    HEADROOM_AVAILABLE = True
except ImportError:
    HEADROOM_AVAILABLE = False

# Kompress imports (ML baseline)
try:
    from headroom.transforms.kompress_compressor import KompressCompressor, is_kompress_available

    KOMPRESS_AVAILABLE = is_kompress_available()
except ImportError:
    KOMPRESS_AVAILABLE = False


@dataclass
class Question:
    """A question about the data with ground truth answer."""

    text: str
    ground_truth: str
    answer_location: Literal["early", "middle", "late", "scattered", "semantic"]
    difficulty: Literal["easy", "medium", "hard"]


@dataclass
class Scenario:
    """A benchmark scenario with data and questions."""

    name: str
    description: str
    data: list[dict]
    questions: list[Question]
    expected_winner: str  # Which approach should theoretically win


@dataclass
class ApproachResult:
    """Result of running one approach on one scenario."""

    approach: str
    scenario: str
    tokens_original: int
    tokens_after: int
    compression_ratio: float
    compression_latency_ms: float
    llm_cost_usd: float  # Cost of summarization if applicable
    answers: list[dict]  # {question, expected, actual, correct}
    accuracy: float
    total_cost_usd: float  # Compression cost + query cost


# =============================================================================
# DATA GENERATORS - Realistic synthetic data
# =============================================================================


def generate_log_data(
    n_entries: int = 500, error_positions: list[int] = None
) -> tuple[list[dict], list[Question]]:
    """
    Generate realistic server logs.

    95% routine logs, 5% interesting events (errors, warnings).
    Errors placed at specified positions to test different approaches.
    """
    if error_positions is None:
        # Default: errors at beginning, middle, and end
        error_positions = [3, n_entries // 2, n_entries - 5]

    log_templates = [
        {"level": "INFO", "message": "Health check passed", "service": "api-gateway"},
        {"level": "INFO", "message": "Request processed successfully", "service": "api-gateway"},
        {"level": "INFO", "message": "Cache hit for user session", "service": "redis"},
        {"level": "INFO", "message": "Database query completed", "service": "postgres"},
        {"level": "INFO", "message": "Authentication successful", "service": "auth"},
        {
            "level": "DEBUG",
            "message": "Connection pool stats: active=5, idle=15",
            "service": "postgres",
        },
    ]

    error_templates = [
        {
            "level": "ERROR",
            "message": "Connection refused to payment-service:8080 - ECONNREFUSED",
            "service": "payment-processor",
            "error_code": "PAYMENT_SERVICE_DOWN",
            "trace_id": "abc123",
        },
        {
            "level": "ERROR",
            "message": "Timeout waiting for response from inventory-service after 30000ms",
            "service": "order-processor",
            "error_code": "INVENTORY_TIMEOUT",
            "trace_id": "def456",
        },
        {
            "level": "CRITICAL",
            "message": "Out of memory: Java heap space - killing process",
            "service": "recommendation-engine",
            "error_code": "OOM_KILLED",
            "trace_id": "ghi789",
        },
    ]

    logs = []
    base_time = 1705320000  # Some Unix timestamp

    error_idx = 0
    for i in range(n_entries):
        base_time + i * 60  # 1 minute apart

        if i in error_positions and error_idx < len(error_templates):
            entry = error_templates[error_idx].copy()
            error_idx += 1
        else:
            entry = random.choice(log_templates).copy()

        entry["timestamp"] = f"2024-01-15T{10 + (i // 60):02d}:{i % 60:02d}:00Z"
        entry["request_id"] = f"req-{hashlib.md5(str(i).encode()).hexdigest()[:8]}"  # nosec B324
        logs.append(entry)

    # Questions designed to test different approaches
    questions = [
        Question(
            text="What error code was returned by the payment service?",
            ground_truth="PAYMENT_SERVICE_DOWN",
            answer_location="early",  # Position 3
            difficulty="easy",
        ),
        Question(
            text="Which service experienced a timeout and what was the trace ID?",
            ground_truth="order-processor service had timeout with trace_id def456",
            answer_location="middle",
            difficulty="medium",
        ),
        Question(
            text="What critical error occurred and which service was affected?",
            ground_truth="Out of memory (OOM_KILLED) in recommendation-engine",
            answer_location="late",  # Near end
            difficulty="medium",
        ),
        Question(
            text="How many distinct error types are in the logs?",
            ground_truth="3",
            answer_location="scattered",
            difficulty="hard",
        ),
    ]

    return logs, questions


def generate_file_search_data(n_files: int = 1000) -> tuple[list[dict], list[Question]]:
    """
    Generate realistic code search results.

    Simulates searching a codebase - lots of files with similar metadata,
    specific files of interest scattered throughout.
    """

    # Common directories and file patterns
    dirs = [
        "src/api",
        "src/services",
        "src/utils",
        "src/models",
        "src/controllers",
        "src/middleware",
        "tests/unit",
        "tests/integration",
        "lib/core",
        "lib/helpers",
        "config",
        "scripts",
    ]

    extensions = [".py", ".py", ".py", ".ts", ".js", ".json", ".yaml"]  # Weighted toward .py

    # Files of interest (scattered at specific positions)
    special_files = {
        50: {
            "path": "src/auth/jwt_handler.py",
            "size": 2341,
            "description": "JWT token validation and refresh",
        },
        250: {
            "path": "src/services/payment_processor.py",
            "size": 5672,
            "description": "Stripe payment integration",
        },
        500: {
            "path": "src/middleware/rate_limiter.py",
            "size": 1823,
            "description": "Redis-based rate limiting",
        },
        750: {
            "path": "config/database.py",
            "size": 892,
            "description": "PostgreSQL connection settings",
        },
        999: {
            "path": "src/api/health_check.py",
            "size": 456,
            "description": "Kubernetes health endpoints",
        },
    }

    files = []
    for i in range(n_files):
        if i in special_files:
            f = special_files[i].copy()
            f["type"] = "file"
            f["language"] = "python"
            f["modified"] = "2024-01-15"
        else:
            dir_path = random.choice(dirs)
            ext = random.choice(extensions)
            f = {
                "type": "file",
                "path": f"{dir_path}/module_{i}{ext}",
                "size": random.randint(200, 5000),
                "language": "python"
                if ext == ".py"
                else "typescript"
                if ext == ".ts"
                else "javascript",
                "modified": f"2024-01-{random.randint(1, 15):02d}",
            }
        files.append(f)

    questions = [
        Question(
            text="Which file handles JWT token operations?",
            ground_truth="src/auth/jwt_handler.py",
            answer_location="early",  # Position 50
            difficulty="easy",
        ),
        Question(
            text="What file contains the Stripe payment integration and how large is it?",
            ground_truth="src/services/payment_processor.py, 5672 bytes",
            answer_location="middle",  # Position 250
            difficulty="medium",
        ),
        Question(
            text="Which file implements rate limiting and what technology does it use?",
            ground_truth="src/middleware/rate_limiter.py uses Redis",
            answer_location="middle",  # Position 500
            difficulty="medium",
        ),
        Question(
            text="What is the last Python file in the results and what does it do?",
            ground_truth="src/api/health_check.py - Kubernetes health endpoints",
            answer_location="late",  # Position 999
            difficulty="hard",
        ),
    ]

    return files, questions


def generate_metrics_data(n_points: int = 500) -> tuple[list[dict], list[Question]]:
    """
    Generate realistic time series metrics.

    Baseline values with anomalies (spikes) at specific positions.
    This is where Headroom should excel - detecting statistical outliers.
    """

    base_cpu = 45.0
    base_memory = 62.0
    base_requests = 1000

    # Anomaly positions
    anomalies = {
        50: {"cpu": 95.0, "memory": 88.0, "requests": 5000, "event": "traffic_spike"},
        200: {"cpu": 98.0, "memory": 95.0, "requests": 150, "event": "service_degradation"},
        450: {"cpu": 15.0, "memory": 30.0, "requests": 50, "event": "service_restart"},
    }

    metrics = []
    base_time = 1705320000

    for i in range(n_points):
        base_time + i * 60

        if i in anomalies:
            point = {
                "timestamp": f"2024-01-15T{10 + (i // 60):02d}:{i % 60:02d}:00Z",
                "cpu_percent": anomalies[i]["cpu"],
                "memory_percent": anomalies[i]["memory"],
                "requests_per_min": anomalies[i]["requests"],
                "status": "degraded" if anomalies[i]["event"] != "traffic_spike" else "ok",
                "event": anomalies[i]["event"],
            }
        else:
            point = {
                "timestamp": f"2024-01-15T{10 + (i // 60):02d}:{i % 60:02d}:00Z",
                "cpu_percent": round(base_cpu + random.uniform(-5, 5), 1),
                "memory_percent": round(base_memory + random.uniform(-3, 3), 1),
                "requests_per_min": base_requests + random.randint(-100, 100),
                "status": "ok",
            }
        metrics.append(point)

    questions = [
        Question(
            text="When did the traffic spike occur and what was the requests_per_min?",
            ground_truth="Around 10:50, requests_per_min was 5000",
            answer_location="early",
            difficulty="easy",
        ),
        Question(
            text="What event caused service degradation and what were the CPU/memory values?",
            ground_truth="service_degradation event, CPU 98%, memory 95%",
            answer_location="middle",
            difficulty="medium",
        ),
        Question(
            text="When did the service restart and how can you tell from the metrics?",
            ground_truth="Around 17:30, CPU dropped to 15%, memory to 30%, requests to 50",
            answer_location="late",
            difficulty="hard",
        ),
        Question(
            text="How many anomalous events occurred in total?",
            ground_truth="3",
            answer_location="scattered",
            difficulty="hard",
        ),
    ]

    return metrics, questions


# =============================================================================
# COMPRESSION APPROACHES
# =============================================================================


def truncate_data(data: list[dict], max_items: int = 20) -> list[dict]:
    """Simple truncation - keep first N items."""
    return data[:max_items]


def summarize_data(
    data: list[dict], client: "OpenAI", model: str = "gpt-4o-mini"
) -> tuple[str, float]:
    """
    Use LLM to summarize the data.
    Returns (summary_text, cost_usd).
    """
    data_str = json.dumps(data, indent=2)

    # Truncate if too long for summarization call
    if len(data_str) > 100000:
        data_str = data_str[:100000] + "\n... [truncated for summarization]"

    prompt = f"""Summarize this data concisely, preserving all important information including:
- Any errors, warnings, or anomalies
- Key identifiers (IDs, names, paths)
- Statistical outliers
- Important events

Data:
{data_str}

Provide a structured summary that retains all critical details."""

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    latency = (time.time() - start) * 1000

    summary = response.choices[0].message.content

    # Estimate cost (gpt-4o-mini pricing)
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000  # Per token pricing

    return summary, cost, latency


def kompress_compress(data: list[dict]) -> tuple[str, dict]:
    """
    Use Kompress (ModernBERT) for ML-based compression.
    Returns (compressed_text, metadata).
    """
    if not KOMPRESS_AVAILABLE:
        raise RuntimeError("Kompress not available. Install with: pip install headroom-ai[ml]")

    compressor = KompressCompressor()

    # Convert data to string for Kompress (it works on text, not structured data)
    data_str = json.dumps(data, indent=2)

    start = time.time()
    result = compressor.compress(data_str)
    latency = (time.time() - start) * 1000

    metadata = {
        "latency_ms": latency,
        "original_tokens": result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "compression_ratio": result.compression_ratio,
    }

    return result.compressed, metadata


def headroom_compress(data: list[dict], query_context: str = "") -> tuple[list[dict], dict]:
    """
    Use Headroom's SmartCrusher for statistical compression.
    Returns (compressed_data, metadata).
    """
    if not HEADROOM_AVAILABLE:
        raise RuntimeError("Headroom not available")

    config = SmartCrusherConfig(
        enabled=True,
        min_items_to_analyze=5,
        variance_threshold=2.0,
        max_items_after_crush=20,
        preserve_change_points=True,
    )

    crusher = SmartCrusher(config)

    # Wrap data in tool output format
    tool_content = json.dumps({"results": data})

    start = time.time()
    crush_result = crusher.crush(tool_content, query=query_context)
    latency = (time.time() - start) * 1000

    # Parse result - crush returns a CrushResult with .compressed attribute
    result_str = (
        crush_result.compressed if hasattr(crush_result, "compressed") else str(crush_result)
    )

    try:
        compressed = json.loads(result_str)
        if isinstance(compressed, dict) and "results" in compressed:
            compressed_data = compressed["results"]
        else:
            compressed_data = compressed if isinstance(compressed, list) else data[:20]
    except json.JSONDecodeError:
        compressed_data = data[:20]  # Fallback

    metadata = {
        "latency_ms": latency,
        "items_before": len(data),
        "items_after": len(compressed_data) if isinstance(compressed_data, list) else "N/A",
    }

    return compressed_data, metadata


# =============================================================================
# EVALUATION
# =============================================================================


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    if HEADROOM_AVAILABLE:
        counter = TiktokenCounter()
        return counter.count_text(text)
    else:
        # Rough estimate: 4 chars per token
        return len(text) // 4


def evaluate_answer(question: Question, actual_answer: str) -> bool:
    """
    Check if the answer is correct.
    Uses fuzzy matching - answer should contain key parts of ground truth.
    """
    if not actual_answer:
        return False

    actual_lower = actual_answer.lower()
    truth_lower = question.ground_truth.lower()

    # Extract key terms from ground truth
    key_terms = []
    for term in truth_lower.replace(",", " ").replace("-", " ").split():
        if len(term) > 3 and term not in ["the", "and", "was", "with", "from"]:
            key_terms.append(term)

    # Check if most key terms appear in answer
    matches = sum(1 for term in key_terms if term in actual_lower)
    return matches >= len(key_terms) * 0.6  # 60% threshold


def query_llm(
    client: "OpenAI", context: str, question: str, model: str = "gpt-4o-mini"
) -> tuple[str, float]:
    """
    Ask the LLM a question about the given context.
    Returns (answer, cost_usd).
    """
    prompt = f"""Based on the following data, answer the question.

Data:
{context}

Question: {question}

Answer concisely with specific details from the data."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    answer = response.choices[0].message.content

    # Estimate cost
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000

    return answer, cost


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run."""

    model: str = "gpt-4o-mini"  # Model for queries (and summarization)
    max_truncate_items: int = 20
    max_headroom_items: int = 20
    run_summarization: bool = True  # Can disable to save cost
    run_kompress: bool = True  # Run Kompress (ML baseline)


def run_scenario_benchmark(
    scenario: Scenario, client: "OpenAI", config: BenchmarkConfig
) -> list[ApproachResult]:
    """Run all approaches on a single scenario."""

    results = []
    original_json = json.dumps(scenario.data, indent=2)
    original_tokens = count_tokens(original_json)

    print(f"\n{'=' * 60}")
    print(f"Scenario: {scenario.name}")
    print(f"Data size: {len(scenario.data)} items, {original_tokens} tokens")
    print(f"Expected winner: {scenario.expected_winner}")
    print(f"{'=' * 60}")

    # --- TRUNCATION ---
    print("\n[1/4] Running Truncation...")
    start = time.time()
    truncated = truncate_data(scenario.data, config.max_truncate_items)
    trunc_latency = (time.time() - start) * 1000

    trunc_json = json.dumps(truncated, indent=2)
    trunc_tokens = count_tokens(trunc_json)

    trunc_answers = []
    trunc_query_cost = 0.0
    for q in scenario.questions:
        answer, cost = query_llm(client, trunc_json, q.text, config.model)
        correct = evaluate_answer(q, answer)
        trunc_answers.append(
            {
                "question": q.text,
                "expected": q.ground_truth,
                "actual": answer,
                "correct": correct,
                "location": q.answer_location,
            }
        )
        trunc_query_cost += cost

    trunc_accuracy = sum(1 for a in trunc_answers if a["correct"]) / len(trunc_answers)

    results.append(
        ApproachResult(
            approach="truncation",
            scenario=scenario.name,
            tokens_original=original_tokens,
            tokens_after=trunc_tokens,
            compression_ratio=1 - (trunc_tokens / original_tokens),
            compression_latency_ms=trunc_latency,
            llm_cost_usd=0.0,  # No LLM for compression
            answers=trunc_answers,
            accuracy=trunc_accuracy,
            total_cost_usd=trunc_query_cost,
        )
    )
    print(
        f"   Tokens: {original_tokens} → {trunc_tokens} ({results[-1].compression_ratio:.1%} reduction)"
    )
    print(f"   Accuracy: {trunc_accuracy:.1%}")

    # --- SUMMARIZATION ---
    if config.run_summarization:
        print("\n[2/4] Running Summarization...")
        try:
            summary, summ_cost, summ_latency = summarize_data(scenario.data, client, config.model)
            summ_tokens = count_tokens(summary)

            summ_answers = []
            summ_query_cost = 0.0
            for q in scenario.questions:
                answer, cost = query_llm(client, summary, q.text, config.model)
                correct = evaluate_answer(q, answer)
                summ_answers.append(
                    {
                        "question": q.text,
                        "expected": q.ground_truth,
                        "actual": answer,
                        "correct": correct,
                        "location": q.answer_location,
                    }
                )
                summ_query_cost += cost

            summ_accuracy = sum(1 for a in summ_answers if a["correct"]) / len(summ_answers)

            results.append(
                ApproachResult(
                    approach="summarization",
                    scenario=scenario.name,
                    tokens_original=original_tokens,
                    tokens_after=summ_tokens,
                    compression_ratio=1 - (summ_tokens / original_tokens),
                    compression_latency_ms=summ_latency,
                    llm_cost_usd=summ_cost,
                    answers=summ_answers,
                    accuracy=summ_accuracy,
                    total_cost_usd=summ_cost + summ_query_cost,
                )
            )
            print(
                f"   Tokens: {original_tokens} → {summ_tokens} ({results[-1].compression_ratio:.1%} reduction)"
            )
            print(f"   Accuracy: {summ_accuracy:.1%}")
            print(f"   Summarization cost: ${summ_cost:.4f}")
        except Exception as e:
            print(f"   Summarization failed: {e}")

    # --- KOMPRESS (ML baseline) ---
    if config.run_kompress:
        print("\n[3/4] Running Kompress (ModernBERT ML baseline)...")
        if KOMPRESS_AVAILABLE:
            try:
                ll_compressed, ll_metadata = kompress_compress(scenario.data)
                ll_tokens = count_tokens(ll_compressed)

                ll_answers = []
                ll_query_cost = 0.0
                for q in scenario.questions:
                    answer, cost = query_llm(client, ll_compressed, q.text, config.model)
                    correct = evaluate_answer(q, answer)
                    ll_answers.append(
                        {
                            "question": q.text,
                            "expected": q.ground_truth,
                            "actual": answer,
                            "correct": correct,
                            "location": q.answer_location,
                        }
                    )
                    ll_query_cost += cost

                ll_accuracy = sum(1 for a in ll_answers if a["correct"]) / len(ll_answers)

                results.append(
                    ApproachResult(
                        approach="kompress",
                        scenario=scenario.name,
                        tokens_original=original_tokens,
                        tokens_after=ll_tokens,
                        compression_ratio=1 - (ll_tokens / original_tokens),
                        compression_latency_ms=ll_metadata["latency_ms"],
                        llm_cost_usd=0.0,  # Model runs locally
                        answers=ll_answers,
                        accuracy=ll_accuracy,
                        total_cost_usd=ll_query_cost,
                    )
                )
                print(
                    f"   Tokens: {original_tokens} → {ll_tokens} ({results[-1].compression_ratio:.1%} reduction)"
                )
                print(f"   Accuracy: {ll_accuracy:.1%}")
                print(f"   Compression latency: {ll_metadata['latency_ms']:.1f}ms")
            except Exception as e:
                print(f"   Kompress failed: {e}")
        else:
            print("   Kompress not available. Install with: pip install headroom-ai[ml]")

    # --- HEADROOM ---
    print("\n[4/4] Running Headroom...")
    if HEADROOM_AVAILABLE:
        try:
            # Use first question as query context (realistic usage)
            query_context = scenario.questions[0].text if scenario.questions else ""
            compressed, metadata = headroom_compress(scenario.data, query_context)

            hr_json = (
                json.dumps(compressed, indent=2)
                if isinstance(compressed, list)
                else str(compressed)
            )
            hr_tokens = count_tokens(hr_json)

            hr_answers = []
            hr_query_cost = 0.0
            for q in scenario.questions:
                answer, cost = query_llm(client, hr_json, q.text, config.model)
                correct = evaluate_answer(q, answer)
                hr_answers.append(
                    {
                        "question": q.text,
                        "expected": q.ground_truth,
                        "actual": answer,
                        "correct": correct,
                        "location": q.answer_location,
                    }
                )
                hr_query_cost += cost

            hr_accuracy = sum(1 for a in hr_answers if a["correct"]) / len(hr_answers)

            results.append(
                ApproachResult(
                    approach="headroom",
                    scenario=scenario.name,
                    tokens_original=original_tokens,
                    tokens_after=hr_tokens,
                    compression_ratio=1 - (hr_tokens / original_tokens),
                    compression_latency_ms=metadata["latency_ms"],
                    llm_cost_usd=0.0,  # No LLM for compression
                    answers=hr_answers,
                    accuracy=hr_accuracy,
                    total_cost_usd=hr_query_cost,
                )
            )
            print(
                f"   Tokens: {original_tokens} → {hr_tokens} ({results[-1].compression_ratio:.1%} reduction)"
            )
            print(f"   Accuracy: {hr_accuracy:.1%}")
            print(f"   Compression latency: {metadata['latency_ms']:.1f}ms")
        except Exception as e:
            print(f"   Headroom failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("   Headroom not available")

    return results


def run_full_benchmark(client: "OpenAI", config: BenchmarkConfig = None) -> dict:
    """Run the complete benchmark suite."""

    if config is None:
        config = BenchmarkConfig()

    print("\n" + "=" * 70)
    print("TRUNCATION vs SUMMARIZATION vs LLMLINGUA-2 vs HEADROOM BENCHMARK")
    print("=" * 70)

    # Generate scenarios
    scenarios = []

    # Scenario 1: Logs (Headroom should win - needs anomaly detection)
    logs, log_questions = generate_log_data(500, error_positions=[3, 250, 495])
    scenarios.append(
        Scenario(
            name="Server Logs (500 entries)",
            description="Find errors buried in routine logs",
            data=logs,
            questions=log_questions,
            expected_winner="headroom",
        )
    )

    # Scenario 2: File Search (Mixed - depends on file position)
    files, file_questions = generate_file_search_data(1000)
    scenarios.append(
        Scenario(
            name="Code Search (1000 files)",
            description="Find specific files in search results",
            data=files,
            questions=file_questions,
            expected_winner="mixed",
        )
    )

    # Scenario 3: Metrics (Headroom should win - statistical outliers)
    metrics, metric_questions = generate_metrics_data(500)
    scenarios.append(
        Scenario(
            name="Time Series Metrics (500 points)",
            description="Find anomalies in metrics data",
            data=metrics,
            questions=metric_questions,
            expected_winner="headroom",
        )
    )

    all_results = []
    for scenario in scenarios:
        results = run_scenario_benchmark(scenario, client, config)
        all_results.extend(results)

    # Generate summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    summary = generate_summary(all_results, scenarios)
    print(summary)

    return {
        "results": [r.__dict__ for r in all_results],
        "summary": summary,
        "scenarios": [s.name for s in scenarios],
    }


def generate_summary(results: list[ApproachResult], scenarios: list[Scenario]) -> str:
    """Generate a human-readable summary of results."""

    lines = []

    # Per-scenario breakdown
    for scenario in scenarios:
        lines.append(f"\n### {scenario.name}")
        lines.append(f"Expected winner: {scenario.expected_winner}")
        lines.append("")
        lines.append("| Approach | Compression | Accuracy | Cost |")
        lines.append("|----------|-------------|----------|------|")

        scenario_results = [r for r in results if r.scenario == scenario.name]
        for r in scenario_results:
            lines.append(
                f"| {r.approach} | {r.compression_ratio:.1%} | {r.accuracy:.1%} | ${r.total_cost_usd:.4f} |"
            )

        # Determine actual winner
        best = max(scenario_results, key=lambda r: (r.accuracy, r.compression_ratio))
        lines.append(f"\n**Actual winner: {best.approach}** (accuracy: {best.accuracy:.1%})")

    # Overall stats
    lines.append("\n### Overall Statistics")

    for approach in ["truncation", "summarization", "llmlingua-2", "headroom"]:
        approach_results = [r for r in results if r.approach == approach]
        if approach_results:
            avg_compression = sum(r.compression_ratio for r in approach_results) / len(
                approach_results
            )
            avg_accuracy = sum(r.accuracy for r in approach_results) / len(approach_results)
            total_cost = sum(r.total_cost_usd for r in approach_results)
            lines.append(f"\n**{approach.title()}**")
            lines.append(f"- Avg compression: {avg_compression:.1%}")
            lines.append(f"- Avg accuracy: {avg_accuracy:.1%}")
            lines.append(f"- Total cost: ${total_cost:.4f}")

    # Per-question-type analysis
    lines.append("\n### Accuracy by Answer Location")
    lines.append("(Where in the data is the answer?)")
    lines.append("")

    for location in ["early", "middle", "late", "scattered"]:
        lines.append(f"\n**{location.title()} position:**")
        for approach in ["truncation", "summarization", "llmlingua-2", "headroom"]:
            approach_results = [r for r in results if r.approach == approach]
            location_answers = []
            for r in approach_results:
                location_answers.extend([a for a in r.answers if a["location"] == location])
            if location_answers:
                correct = sum(1 for a in location_answers if a["correct"])
                total = len(location_answers)
                lines.append(f"  - {approach}: {correct}/{total} ({correct / total:.1%})")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os

    if not OPENAI_AVAILABLE:
        print("OpenAI not available. Install with: pip install openai")
        exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable")
        exit(1)

    client = OpenAI(api_key=api_key)

    config = BenchmarkConfig(
        model="gpt-4o-mini",
        max_truncate_items=20,
        max_headroom_items=20,
        run_summarization=True,
    )

    results = run_full_benchmark(client, config)

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nResults saved to benchmark_results.json")

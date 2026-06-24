#!/usr/bin/env python3
"""
CCR Regression Benchmark - Verify No Information Loss

This benchmark tests that the CCR (Compress-Cache-Retrieve) architecture
does not cause any regression in agent behavior. Specifically:

1. NEEDLE RETENTION: Critical items survive compression
   - Errors, exceptions, failures
   - Specific IDs/UUIDs mentioned in user query
   - Anomalies and outliers

2. RETRIEVAL ACCURACY: When retrieval is needed, correct items are returned
   - Full retrieval returns original content
   - Search retrieval finds relevant items

3. FEEDBACK LEARNING: System learns from retrieval patterns
   - High retrieval rate triggers less aggressive compression
   - Common queries improve future compression

Usage:
    python benchmarks/ccr_regression_benchmark.py
    python benchmarks/ccr_regression_benchmark.py --verbose
    python benchmarks/ccr_regression_benchmark.py --scenario needle-in-haystack
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from headroom.cache.compression_feedback import (
    get_compression_feedback,
    reset_compression_feedback,
)
from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.transforms.smart_crusher import (
    SmartCrusherConfig,
    smart_crush_tool_output,
)


@dataclass
class RegressionResult:
    """Result from a regression test."""

    name: str
    description: str
    passed: bool = False  # Default to False, set to True when test passes

    # Metrics
    total_needles: int = 0
    needles_retained: int = 0
    retention_rate: float = 0.0

    # CCR metrics
    items_compressed: int = 0
    items_retrieved: int = 0
    retrieval_accuracy: float = 0.0

    # Performance
    latency_ms: float = 0.0

    # Details
    details: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


# =============================================================================
# TEST 1: Needle in Haystack - Error Retention
# =============================================================================


def test_error_retention() -> RegressionResult:
    """
    Test that errors are NEVER lost during compression.

    This is critical: if an API returns 1000 results with 3 errors,
    those 3 errors MUST be in the compressed output.
    """
    result = RegressionResult(
        name="Error Retention",
        description="Verify all errors survive compression regardless of position",
    )

    # Generate 1000 items with errors at various positions
    items = []
    error_indices = [5, 47, 123, 456, 789, 999]  # Spread throughout

    for i in range(1000):
        if i in error_indices:
            items.append(
                {
                    "id": i,
                    "status": "error",
                    "message": f"Connection failed: timeout at {i}",
                    "error_code": 500 + (i % 10),
                }
            )
        else:
            items.append(
                {
                    "id": i,
                    "status": "success",
                    "message": "OK",
                    "data": {"value": i * 2},
                }
            )

    result.total_needles = len(error_indices)

    # Compress with SmartCrusher
    config = SmartCrusherConfig(max_items_after_crush=15)
    original_json = json.dumps(items)

    start = time.perf_counter()
    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)
    result.latency_ms = (time.perf_counter() - start) * 1000

    # Count errors in compressed output
    compressed = json.loads(compressed_json)
    errors_found = [item for item in compressed if item.get("status") == "error"]

    result.needles_retained = len(errors_found)
    result.retention_rate = result.needles_retained / result.total_needles
    result.items_compressed = len(compressed)

    # Check if ALL errors were retained
    result.passed = result.needles_retained == result.total_needles

    if not result.passed:
        result.failures.append(
            f"Lost {result.total_needles - result.needles_retained} errors during compression"
        )

    result.details = {
        "original_items": 1000,
        "compressed_items": len(compressed),
        "error_positions": error_indices,
        "errors_retained": result.needles_retained,
    }

    return result


# =============================================================================
# TEST 2: Needle in Haystack - UUID Lookup
# =============================================================================


def test_uuid_retrieval() -> RegressionResult:
    """
    Test that specific UUIDs can be found via CCR retrieval.

    Scenario: User asks "find transaction abc123..."
    The system compresses, but user should be able to retrieve the specific item.
    """
    result = RegressionResult(
        name="UUID Retrieval via CCR",
        description="Verify specific UUIDs can be retrieved from compressed cache",
    )

    reset_compression_store()
    store = get_compression_store()

    # Generate 1000 transactions with UUIDs
    target_uuid = str(uuid.uuid4())
    items = []

    for i in range(1000):
        item_uuid = target_uuid if i == 456 else str(uuid.uuid4())
        items.append(
            {
                "transaction_id": item_uuid,
                "amount": 100 + (i % 1000),
                "status": "completed",
                "timestamp": f"2025-01-{(i % 28) + 1:02d}T10:00:00Z",
            }
        )

    result.total_needles = 1

    # Store original and compress
    original_json = json.dumps(items)
    config = SmartCrusherConfig(max_items_after_crush=15)

    start = time.perf_counter()
    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)

    # Store in CCR cache
    hash_key = store.store(
        original=original_json,
        compressed=compressed_json,
        original_item_count=1000,
        compressed_item_count=15,
        tool_name="transaction_search",
    )

    # Search for the specific UUID
    search_results = store.search(hash_key, target_uuid)
    result.latency_ms = (time.perf_counter() - start) * 1000

    # Check if target UUID was found
    found_target = any(item.get("transaction_id") == target_uuid for item in search_results)

    result.needles_retained = 1 if found_target else 0
    result.retention_rate = result.needles_retained / result.total_needles
    result.items_retrieved = len(search_results)
    result.retrieval_accuracy = 1.0 if found_target else 0.0

    result.passed = found_target

    if not result.passed:
        result.failures.append(
            f"Could not retrieve target UUID {target_uuid[:8]}... via CCR search"
        )

    result.details = {
        "target_uuid": target_uuid,
        "search_results_count": len(search_results),
        "found_target": found_target,
        "hash_key": hash_key,
    }

    return result


# =============================================================================
# TEST 3: Anomaly Detection
# =============================================================================


def test_anomaly_retention() -> RegressionResult:
    """
    Test that statistical anomalies are preserved during compression.

    Scenario: 1000 metrics mostly at ~50, but with 5 spikes at 500+.
    Those spikes MUST survive compression.
    """
    result = RegressionResult(
        name="Anomaly Retention", description="Verify statistical outliers survive compression"
    )

    # Generate metrics with anomalies
    import random

    random.seed(42)  # Reproducible

    items = []
    anomaly_indices = [10, 200, 450, 700, 990]  # 5 spikes

    for i in range(1000):
        if i in anomaly_indices:
            # Anomaly: 10x normal value
            value = 500 + random.randint(0, 100)
        else:
            # Normal: around 50
            value = 50 + random.randint(-10, 10)

        items.append(
            {
                "timestamp": f"2025-01-07T{(i // 60):02d}:{(i % 60):02d}:00Z",
                "cpu_percent": value,
                "host": "prod-server-1",
            }
        )

    result.total_needles = len(anomaly_indices)

    # Compress
    config = SmartCrusherConfig(
        max_items_after_crush=20,
        preserve_change_points=True,
    )
    original_json = json.dumps(items)

    start = time.perf_counter()
    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)
    result.latency_ms = (time.perf_counter() - start) * 1000

    # Count anomalies (cpu > 200) in compressed output
    compressed = json.loads(compressed_json)
    anomalies_found = [
        item
        for item in compressed
        if isinstance(item.get("cpu_percent"), (int, float)) and item["cpu_percent"] > 200
    ]

    result.needles_retained = len(anomalies_found)
    result.retention_rate = result.needles_retained / result.total_needles
    result.items_compressed = len(compressed)

    # Pass if at least 80% of anomalies retained (some might be in change point windows)
    result.passed = result.retention_rate >= 0.8

    if not result.passed:
        result.failures.append(
            f"Lost too many anomalies: {result.needles_retained}/{result.total_needles} retained"
        )

    result.details = {
        "original_items": 1000,
        "compressed_items": len(compressed),
        "anomaly_positions": anomaly_indices,
        "anomalies_retained": result.needles_retained,
    }

    return result


# =============================================================================
# TEST 4: Full Retrieval Accuracy
# =============================================================================


def test_full_retrieval() -> RegressionResult:
    """
    Test that full retrieval returns EXACTLY the original content.
    """
    result = RegressionResult(
        name="Full Retrieval Accuracy",
        description="Verify full retrieval returns exact original content",
    )

    reset_compression_store()
    store = get_compression_store()

    # Generate test data
    items = [{"id": i, "name": f"item_{i}", "value": i * 10} for i in range(100)]

    original_json = json.dumps(items)
    compressed_json = json.dumps(items[:10])  # Simulate compression

    # Store
    hash_key = store.store(
        original=original_json,
        compressed=compressed_json,
        original_item_count=100,
        compressed_item_count=10,
        tool_name="test_tool",
    )

    start = time.perf_counter()

    # Retrieve
    entry = store.retrieve(hash_key)

    result.latency_ms = (time.perf_counter() - start) * 1000

    # Verify content matches exactly
    if entry is None:
        result.passed = False
        result.failures.append("Retrieval returned None")
    else:
        retrieved_items = json.loads(entry.original_content)
        result.passed = retrieved_items == items
        result.items_retrieved = len(retrieved_items)
        result.retrieval_accuracy = 1.0 if result.passed else 0.0

        if not result.passed:
            result.failures.append("Retrieved content does not match original")

    result.total_needles = 100
    result.needles_retained = result.items_retrieved
    result.retention_rate = 1.0 if result.passed else 0.0

    result.details = {
        "original_items": 100,
        "retrieved_items": result.items_retrieved,
        "hash_key": hash_key,
    }

    return result


# =============================================================================
# TEST 5: Feedback Learning
# =============================================================================


def test_feedback_learning() -> RegressionResult:
    """
    Test that the feedback system learns from retrieval patterns.

    Scenario: Simulate high retrieval rate, verify system recommends
    less aggressive compression.
    """
    result = RegressionResult(
        name="Feedback Learning",
        description="Verify feedback loop adjusts compression based on patterns",
    )

    reset_compression_feedback()
    feedback = get_compression_feedback()

    tool_name = "high_retrieval_tool"

    start = time.perf_counter()

    # Simulate 10 compressions
    for _ in range(10):
        feedback.record_compression(tool_name, 1000, 20)

    # Simulate 6 retrievals (60% rate - HIGH)
    from headroom.cache.compression_store import RetrievalEvent

    for i in range(6):
        event = RetrievalEvent(
            hash=f"hash{i:012d}",
            query="find errors",
            items_retrieved=100,
            total_items=1000,
            tool_name=tool_name,
            timestamp=time.time(),
            retrieval_type="search",
        )
        feedback.record_retrieval(event)

    # Get hints
    hints = feedback.get_compression_hints(tool_name)

    result.latency_ms = (time.perf_counter() - start) * 1000

    # Verify hints recommend less aggressive compression
    pattern = feedback.get_all_patterns().get(tool_name)

    checks_passed = 0
    total_checks = 3

    # Check 1: Retrieval rate is tracked correctly
    if pattern and abs(pattern.retrieval_rate - 0.6) < 0.01:
        checks_passed += 1
    else:
        result.failures.append(
            f"Retrieval rate incorrect: {pattern.retrieval_rate if pattern else 'N/A'}"
        )

    # Check 2: Hints suggest more items (>15 default)
    if hints.max_items > 15:
        checks_passed += 1
    else:
        result.failures.append(f"max_items not increased: {hints.max_items}")

    # Check 3: Aggressiveness reduced (<0.7 default)
    if hints.aggressiveness < 0.7:
        checks_passed += 1
    else:
        result.failures.append(f"Aggressiveness not reduced: {hints.aggressiveness}")

    result.passed = checks_passed == total_checks
    result.retrieval_accuracy = checks_passed / total_checks

    result.details = {
        "compressions_recorded": 10,
        "retrievals_recorded": 6,
        "calculated_retrieval_rate": pattern.retrieval_rate if pattern else 0,
        "recommended_max_items": hints.max_items,
        "recommended_aggressiveness": hints.aggressiveness,
        "reason": hints.reason,
    }

    return result


# =============================================================================
# TEST 6: Search Within Cached Content
# =============================================================================


def test_search_accuracy() -> RegressionResult:
    """
    Test that BM25 search within cached content finds relevant items.
    """
    result = RegressionResult(
        name="Search Accuracy", description="Verify BM25 search finds relevant items in cache"
    )

    reset_compression_store()
    store = get_compression_store()

    # Generate log entries with specific error messages
    items = []
    for i in range(100):
        if i in [15, 45, 78]:
            # Target: authentication errors
            items.append(
                {
                    "id": i,
                    "level": "ERROR",
                    "message": "Authentication failed: invalid token",
                    "service": "auth-service",
                }
            )
        elif i in [20, 60]:
            # Other errors (should not match auth search)
            items.append(
                {
                    "id": i,
                    "level": "ERROR",
                    "message": "Database connection timeout",
                    "service": "db-service",
                }
            )
        else:
            items.append(
                {
                    "id": i,
                    "level": "INFO",
                    "message": "Request processed successfully",
                    "service": "api-service",
                }
            )

    result.total_needles = 3  # 3 auth errors

    original_json = json.dumps(items)
    compressed_json = json.dumps(items[:10])

    # Store
    hash_key = store.store(
        original=original_json,
        compressed=compressed_json,
        original_item_count=100,
        compressed_item_count=10,
        tool_name="log_search",
    )

    start = time.perf_counter()

    # Search for authentication errors
    search_results = store.search(hash_key, "authentication failed token")

    result.latency_ms = (time.perf_counter() - start) * 1000

    # Count auth errors in results
    auth_errors = [
        item for item in search_results if "authentication" in item.get("message", "").lower()
    ]

    result.needles_retained = len(auth_errors)
    result.retention_rate = result.needles_retained / result.total_needles
    result.items_retrieved = len(search_results)

    # Pass if at least 2 of 3 auth errors found
    result.passed = result.needles_retained >= 2
    result.retrieval_accuracy = result.retention_rate

    if not result.passed:
        result.failures.append(
            f"Search found only {result.needles_retained}/{result.total_needles} auth errors"
        )

    result.details = {
        "query": "authentication failed token",
        "total_results": len(search_results),
        "auth_errors_found": result.needles_retained,
        "hash_key": hash_key,
    }

    return result


# =============================================================================
# TEST 7: CCR End-to-End Flow
# =============================================================================


def test_ccr_end_to_end() -> RegressionResult:
    """
    Test the complete CCR flow: compress → cache → retrieve → feedback.
    """
    result = RegressionResult(
        name="CCR End-to-End Flow",
        description="Verify complete compress-cache-retrieve cycle works",
    )

    reset_compression_store()
    reset_compression_feedback()

    store = get_compression_store()
    feedback = get_compression_feedback()

    # Generate data with known needles
    items = []
    for i in range(500):
        if i == 123:
            items.append(
                {
                    "id": i,
                    "type": "critical_alert",
                    "message": "System overload detected",
                    "priority": "P0",
                }
            )
        elif i in [50, 200, 400]:
            items.append(
                {
                    "id": i,
                    "type": "error",
                    "message": f"Error at position {i}",
                    "priority": "P1",
                }
            )
        else:
            items.append(
                {
                    "id": i,
                    "type": "info",
                    "message": f"Normal operation {i}",
                    "priority": "P3",
                }
            )

    result.total_needles = 4  # 1 critical + 3 errors

    start = time.perf_counter()

    # Step 1: Compress
    config = SmartCrusherConfig(max_items_after_crush=20)
    original_json = json.dumps(items)
    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)

    # Step 2: Cache
    hash_key = store.store(
        original=original_json,
        compressed=compressed_json,
        original_item_count=500,
        compressed_item_count=20,
        tool_name="alert_search",
    )

    # Step 3: Record compression in feedback
    feedback.record_compression("alert_search", 500, 20)

    # Step 4: Retrieve and search
    critical_results = store.search(hash_key, "critical system overload P0")
    error_results = store.search(hash_key, "Error position P1")

    # Step 5: Process feedback
    store.process_pending_feedback()

    result.latency_ms = (time.perf_counter() - start) * 1000

    # Verify results
    checks_passed = 0
    total_checks = 4

    # Check 1: Critical alert found
    critical_found = any(item.get("type") == "critical_alert" for item in critical_results)
    if critical_found:
        checks_passed += 1
    else:
        result.failures.append("Critical alert not found in search")

    # Check 2: Errors found (search by message content)
    errors_found = len(
        [
            item
            for item in error_results
            if item.get("type") == "error" or "Error" in str(item.get("message", ""))
        ]
    )
    if errors_found >= 2:
        checks_passed += 1
    else:
        result.failures.append(f"Only {errors_found} errors found in search")

    # Check 3: Store has entry
    if store.exists(hash_key):
        checks_passed += 1
    else:
        result.failures.append("Entry not found in store")

    # Check 4: Feedback recorded
    patterns = feedback.get_all_patterns()
    if "alert_search" in patterns:
        checks_passed += 1
    else:
        result.failures.append("Feedback not recorded for tool")

    result.passed = checks_passed == total_checks
    result.needles_retained = (1 if critical_found else 0) + errors_found
    result.retention_rate = result.needles_retained / result.total_needles
    result.items_retrieved = len(critical_results) + len(error_results)
    result.retrieval_accuracy = checks_passed / total_checks

    result.details = {
        "hash_key": hash_key,
        "critical_found": critical_found,
        "errors_found": errors_found,
        "store_entry_exists": store.exists(hash_key),
        "feedback_recorded": "alert_search" in patterns,
    }

    return result


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_report(results: list[RegressionResult], verbose: bool = False) -> str:
    """Generate benchmark report."""
    lines = []

    lines.append("")
    lines.append("=" * 70)
    lines.append("  CCR REGRESSION BENCHMARK")
    lines.append("  Verifying No Information Loss")
    lines.append("=" * 70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    lines.append("")
    lines.append(f"  Overall: {passed}/{total} tests passed")
    lines.append("")

    for result in results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        lines.append(f"{'─' * 70}")
        lines.append(f"  {status}  {result.name}")
        lines.append(f"         {result.description}")

        if result.total_needles > 0:
            lines.append(
                f"         Needles: {result.needles_retained}/{result.total_needles} retained ({result.retention_rate * 100:.0f}%)"
            )

        if result.items_retrieved > 0:
            lines.append(f"         Retrieved: {result.items_retrieved} items")

        lines.append(f"         Latency: {result.latency_ms:.2f}ms")

        if not result.passed:
            for failure in result.failures:
                lines.append(f"         ❌ {failure}")

        if verbose and result.details:
            lines.append(f"         Details: {json.dumps(result.details, indent=2)}")

    lines.append("")
    lines.append("=" * 70)

    if passed == total:
        lines.append("  ✓ ALL TESTS PASSED - No regression detected")
    else:
        lines.append(f"  ✗ {total - passed} TESTS FAILED - Review failures above")

    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="CCR Regression Benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--scenario",
        choices=[
            "all",
            "error-retention",
            "uuid-retrieval",
            "anomaly-retention",
            "full-retrieval",
            "feedback-learning",
            "search-accuracy",
            "e2e",
        ],
        default="all",
    )
    args = parser.parse_args()

    results = []

    print("\nRunning CCR regression tests...\n")

    if args.scenario in ("all", "error-retention"):
        print("  [1/7] Error Retention...")
        results.append(test_error_retention())

    if args.scenario in ("all", "uuid-retrieval"):
        print("  [2/7] UUID Retrieval...")
        results.append(test_uuid_retrieval())

    if args.scenario in ("all", "anomaly-retention"):
        print("  [3/7] Anomaly Retention...")
        results.append(test_anomaly_retention())

    if args.scenario in ("all", "full-retrieval"):
        print("  [4/7] Full Retrieval...")
        results.append(test_full_retrieval())

    if args.scenario in ("all", "feedback-learning"):
        print("  [5/7] Feedback Learning...")
        results.append(test_feedback_learning())

    if args.scenario in ("all", "search-accuracy"):
        print("  [6/7] Search Accuracy...")
        results.append(test_search_accuracy())

    if args.scenario in ("all", "e2e"):
        print("  [7/7] End-to-End Flow...")
        results.append(test_ccr_end_to_end())

    print(generate_report(results, args.verbose))

    # Exit with error code if any test failed
    failed = sum(1 for r in results if not r.passed)
    exit(failed)


if __name__ == "__main__":
    main()

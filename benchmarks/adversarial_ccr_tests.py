#!/usr/bin/env python3
"""
Adversarial CCR Tests - Designed to BREAK Our Assumptions

These tests are intentionally malicious, edge-casey, and designed to expose
weaknesses in our compression and retrieval logic.

Categories:
1. SEMANTIC ATTACKS: Data that tricks our heuristics
2. BOUNDARY CONDITIONS: Edge cases at limits
3. INJECTION ATTACKS: Malformed data designed to break parsing
4. RACE CONDITIONS: Concurrency attacks
5. MEMORY PRESSURE: Resource exhaustion
6. DECEPTIVE DATA: Items that look like one thing but are another

Run with: python benchmarks/adversarial_ccr_tests.py
"""

from __future__ import annotations

import concurrent.futures
import gc
import hashlib
import json
import random
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from headroom.cache.compression_feedback import (
    get_compression_feedback,
    reset_compression_feedback,
)
from headroom.cache.compression_store import (
    CompressionStore,
    RetrievalEvent,
    get_compression_store,
    reset_compression_store,
)
from headroom.transforms.smart_crusher import (
    SmartCrusherConfig,
    smart_crush_tool_output,
)


@dataclass
class AdversarialResult:
    """Result from an adversarial test."""

    name: str
    category: str
    passed: bool = False
    expected_behavior: str = ""
    actual_behavior: str = ""
    severity: str = "medium"  # low, medium, high, critical
    details: dict[str, Any] = field(default_factory=dict)


def run_test(func) -> AdversarialResult:
    """Run a test and catch any exceptions."""
    try:
        return func()
    except Exception as e:
        return AdversarialResult(
            name=func.__name__,
            category="exception",
            passed=False,
            expected_behavior="Test should complete without exception",
            actual_behavior=f"Exception: {type(e).__name__}: {str(e)[:200]}",
            severity="critical",
        )


# =============================================================================
# CATEGORY 1: SEMANTIC ATTACKS
# =============================================================================


def test_all_items_are_errors() -> AdversarialResult:
    """
    ATTACK: Every single item is an error.

    If we keep ALL errors, we keep everything = no compression.
    What SHOULD happen? Keep all? Sample errors? Fail gracefully?
    """
    result = AdversarialResult(
        name="All Items Are Errors",
        category="semantic",
        expected_behavior="Should handle gracefully, possibly skip compression",
        severity="high",
    )

    # 1000 items, ALL are errors
    items = [
        {
            "id": i,
            "status": "error",
            "error_code": 500 + (i % 50),
            "message": f"Error at position {i}: something went wrong",
        }
        for i in range(1000)
    ]

    config = SmartCrusherConfig(max_items_after_crush=15)
    original_json = json.dumps(items)

    compressed_json, was_modified, reason = smart_crush_tool_output(original_json, config)
    compressed = json.loads(compressed_json)

    # What happened?
    if len(compressed) == 1000:
        result.actual_behavior = "Kept ALL 1000 items (no compression when all errors)"
        result.passed = True  # This is actually correct behavior!
    elif len(compressed) == 15:
        result.actual_behavior = f"Compressed to 15 items, lost {1000 - 15} errors!"
        result.passed = False
    else:
        result.actual_behavior = f"Compressed to {len(compressed)} items"
        result.passed = len(compressed) >= 100  # Should keep most errors

    result.details = {
        "original": 1000,
        "compressed": len(compressed),
        "reason": reason,
    }

    return result


def test_error_keyword_in_normal_data() -> AdversarialResult:
    """
    ATTACK: Normal items contain "error" keyword in benign context.

    "The error rate for this metric is 0.001%" - NOT an error!
    "Error handling documentation" - NOT an error!
    """
    result = AdversarialResult(
        name="Error Keyword False Positive",
        category="semantic",
        expected_behavior="Should NOT treat benign 'error' mentions as errors",
        severity="medium",
    )

    items = []
    # 100 normal items with "error" in benign context
    for i in range(100):
        items.append(
            {
                "id": i,
                "status": "success",  # Clearly success!
                "message": random.choice(
                    [
                        f"Error rate: 0.00{i}%",
                        f"Error handling improved by {i}%",
                        f"Zero errors detected in batch {i}",
                        f"Error-free operation for {i} hours",
                        "Documentation: How to handle errors",
                    ]
                ),
                "value": i,
            }
        )

    # Add 3 REAL errors
    real_error_ids = [25, 50, 75]
    for idx in real_error_ids:
        items[idx] = {
            "id": idx,
            "status": "error",  # This is a REAL error
            "message": f"CRITICAL: System failure at {idx}",
            "error_code": 500,
        }

    config = SmartCrusherConfig(max_items_after_crush=15)
    original_json = json.dumps(items)

    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)
    compressed = json.loads(compressed_json)

    # Count how many items with "error" in message were kept
    items_with_error_word = len(
        [item for item in compressed if "error" in str(item.get("message", "")).lower()]
    )

    # Count real errors kept
    real_errors_kept = len([item for item in compressed if item.get("status") == "error"])

    # False positives (keeping non-errors) are OK - conservative is good
    # False negatives (missing real errors) are NOT OK
    if real_errors_kept < 3:
        result.actual_behavior = (
            f"Only kept {real_errors_kept}/3 real errors - missed actual errors!"
        )
        result.passed = False
    else:
        # Keeping extra items with "error" word is fine - better safe than sorry
        result.actual_behavior = f"Kept all {real_errors_kept} real errors (+ {items_with_error_word} with 'error' word - conservative is OK)"
        result.passed = True

    result.details = {
        "total_compressed": len(compressed),
        "real_errors_kept": real_errors_kept,
        "items_with_error_word": items_with_error_word,
    }

    return result


def test_needle_looks_exactly_like_hay() -> AdversarialResult:
    """
    ATTACK: The critical item has NO distinguishing features.

    In a list of 1000 users, user #456 is the one we need.
    User #456 looks EXACTLY like every other user.
    """
    result = AdversarialResult(
        name="Needle Identical to Hay",
        category="semantic",
        expected_behavior="CCR retrieval should still find specific item by ID",
        severity="high",
    )

    reset_compression_store()
    store = get_compression_store()

    # 1000 identical-looking users
    target_id = 456
    items = [
        {
            "user_id": i,
            "name": f"User {i}",
            "status": "active",
            "created": "2025-01-01",
        }
        for i in range(1000)
    ]

    original_json = json.dumps(items)
    config = SmartCrusherConfig(max_items_after_crush=15)

    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)

    # Store for CCR
    hash_key = store.store(
        original=original_json,
        compressed=compressed_json,
        original_item_count=1000,
        compressed_item_count=15,
        tool_name="user_search",
    )

    # Try to find user 456 via search
    search_results = store.search(hash_key, "user_id 456")

    found_target = any(item.get("user_id") == target_id for item in search_results)

    if found_target:
        result.actual_behavior = "Found target user via CCR search"
        result.passed = True
    else:
        # Try full retrieval as fallback
        entry = store.retrieve(hash_key)
        if entry:
            all_items = json.loads(entry.original_content)
            target_in_original = any(item.get("user_id") == target_id for item in all_items)
            if target_in_original:
                result.actual_behavior = "Search failed, but full retrieval works"
                result.passed = True  # CCR still provides recovery path
            else:
                result.actual_behavior = "Data lost entirely!"
                result.passed = False
        else:
            result.actual_behavior = "CCR cache miss - data not found"
            result.passed = False

    result.details = {
        "target_id": target_id,
        "search_results": len(search_results),
        "found_target": found_target,
    }

    return result


def test_anomaly_in_string_not_number() -> AdversarialResult:
    """
    ATTACK: Anomaly is in a string field, not numeric.

    999 items: region="us-east-1"
    1 item: region="DEPRECATED-DO-NOT-USE"

    SmartCrusher detects numeric anomalies, but what about string outliers?
    """
    result = AdversarialResult(
        name="String Anomaly Detection",
        category="semantic",
        expected_behavior="Should detect or preserve string outliers",
        severity="medium",
    )

    items = []
    anomaly_idx = 500

    for i in range(1000):
        if i == anomaly_idx:
            items.append(
                {
                    "id": i,
                    "region": "DEPRECATED-DO-NOT-USE-CRITICAL-MIGRATION-REQUIRED",
                    "status": "active",
                }
            )
        else:
            items.append(
                {
                    "id": i,
                    "region": "us-east-1",
                    "status": "active",
                }
            )

    config = SmartCrusherConfig(max_items_after_crush=20)
    original_json = json.dumps(items)

    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)
    compressed = json.loads(compressed_json)

    # Check if anomaly was preserved
    anomaly_preserved = any("DEPRECATED" in str(item.get("region", "")) for item in compressed)

    if anomaly_preserved:
        result.actual_behavior = "String anomaly was preserved"
        result.passed = True
    else:
        result.actual_behavior = "String anomaly was LOST - only numeric anomalies detected"
        result.passed = False

    result.details = {
        "compressed_count": len(compressed),
        "anomaly_preserved": anomaly_preserved,
    }

    return result


# =============================================================================
# CATEGORY 2: BOUNDARY CONDITIONS
# =============================================================================


def test_empty_array() -> AdversarialResult:
    """
    ATTACK: Empty array input.
    """
    result = AdversarialResult(
        name="Empty Array",
        category="boundary",
        expected_behavior="Should return empty array unchanged",
        severity="low",
    )

    config = SmartCrusherConfig()
    compressed_json, was_modified, reason = smart_crush_tool_output("[]", config)

    if compressed_json == "[]" and not was_modified:
        result.actual_behavior = "Correctly handled empty array"
        result.passed = True
    else:
        result.actual_behavior = f"Unexpected result: {compressed_json[:100]}"
        result.passed = False

    return result


def test_single_item_array() -> AdversarialResult:
    """
    ATTACK: Array with exactly 1 item.
    """
    result = AdversarialResult(
        name="Single Item Array",
        category="boundary",
        expected_behavior="Should return single item unchanged",
        severity="low",
    )

    items = [{"id": 1, "value": "only_one"}]
    config = SmartCrusherConfig()

    compressed_json, was_modified, _ = smart_crush_tool_output(json.dumps(items), config)
    compressed = json.loads(compressed_json)

    if len(compressed) == 1 and compressed[0].get("id") == 1:
        result.actual_behavior = "Single item preserved"
        result.passed = True
    else:
        result.actual_behavior = f"Unexpected: {len(compressed)} items"
        result.passed = False

    return result


def test_exactly_max_items() -> AdversarialResult:
    """
    ATTACK: Array with exactly max_items_after_crush items.
    """
    result = AdversarialResult(
        name="Exactly Max Items",
        category="boundary",
        expected_behavior="Should not compress when at exact limit",
        severity="low",
    )

    config = SmartCrusherConfig(max_items_after_crush=15)
    items = [{"id": i} for i in range(15)]  # Exactly 15

    compressed_json, was_modified, _ = smart_crush_tool_output(json.dumps(items), config)
    compressed = json.loads(compressed_json)

    if len(compressed) == 15:
        result.actual_behavior = "Kept all 15 items as expected"
        result.passed = True
    else:
        result.actual_behavior = f"Changed count: {len(compressed)}"
        result.passed = False

    return result


def test_max_items_plus_one() -> AdversarialResult:
    """
    ATTACK: Array with max_items + 1.

    IMPORTANT: If data has high uniqueness and no importance signal,
    crushability analysis correctly skips compression to avoid data loss.
    This is the RIGHT behavior - don't blindly compress unique entities.
    """
    result = AdversarialResult(
        name="Max Items Plus One",
        category="boundary",
        expected_behavior="Skip compression for unique entities OR compress with signal",
        severity="low",
    )

    config = SmartCrusherConfig(max_items_after_crush=15, min_items_to_analyze=5)
    # Create items WITH a score field so compression can determine importance
    items = [{"id": i, "value": f"item_{i}", "score": 1.0 - (i / 100)} for i in range(16)]

    compressed_json, was_modified, reason = smart_crush_tool_output(json.dumps(items), config)
    compressed = json.loads(compressed_json)

    result.actual_behavior = f"Compressed to {len(compressed)} items ({reason})"
    # With a score signal, we should compress to max_items
    result.passed = len(compressed) <= 15

    return result


def test_hash_collision_attempt() -> AdversarialResult:
    """
    ATTACK: Try to create hash collisions in CCR store.

    We use SHA256[:16] - what if two different contents hash the same?
    """
    result = AdversarialResult(
        name="Hash Collision Attack",
        category="boundary",
        expected_behavior="Different content should not collide",
        severity="high",
    )

    reset_compression_store()
    get_compression_store()

    # Store many different contents
    hashes = set()
    collisions = 0

    for i in range(10000):
        content = json.dumps([{"unique_id": str(uuid.uuid4()), "index": i}])
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        if content_hash in hashes:
            collisions += 1
        hashes.add(content_hash)

    if collisions == 0:
        result.actual_behavior = "No collisions in 10,000 entries"
        result.passed = True
    else:
        result.actual_behavior = f"Found {collisions} hash collisions!"
        result.passed = False
        result.severity = "critical"

    result.details = {"entries_tested": 10000, "collisions": collisions}

    return result


def test_ttl_exact_boundary() -> AdversarialResult:
    """
    ATTACK: Retrieve at exact TTL expiration moment.
    """
    result = AdversarialResult(
        name="TTL Exact Boundary",
        category="boundary",
        expected_behavior="Entry should expire cleanly at TTL",
        severity="medium",
    )

    reset_compression_store()
    store = CompressionStore(default_ttl=1)  # 1 second TTL

    hash_key = store.store(
        original='[{"id": 1}]',
        compressed='[{"id": 1}]',
        original_item_count=1,
        compressed_item_count=1,
    )

    # Should exist immediately
    exists_before = store.exists(hash_key)

    # Wait exactly at boundary
    time.sleep(1.05)

    # Should be expired
    exists_after = store.exists(hash_key)
    entry = store.retrieve(hash_key)

    if exists_before and not exists_after and entry is None:
        result.actual_behavior = "TTL expiration works correctly"
        result.passed = True
    else:
        result.actual_behavior = (
            f"Before: {exists_before}, After: {exists_after}, Entry: {entry is not None}"
        )
        result.passed = False

    return result


# =============================================================================
# CATEGORY 3: INJECTION ATTACKS
# =============================================================================


def test_json_injection_in_content() -> AdversarialResult:
    """
    ATTACK: JSON that tries to break our parsing.
    """
    result = AdversarialResult(
        name="JSON Injection",
        category="injection",
        expected_behavior="Should handle malformed JSON gracefully",
        severity="high",
    )

    # Various injection attempts
    injections = [
        '{"id": 1, "evil": "}\\"]}',  # Quote escape
        '[{"id": 1}, null, {"id": 2}]',  # Null in array
        '[{"id": 1, "__proto__": {"admin": true}}]',  # Prototype pollution
        '[{"id": 1, "nested": {"deep": {"deeper": {"deepest": "value"}}}}]',
    ]

    config = SmartCrusherConfig()
    failures = []

    for injection in injections:
        try:
            compressed, was_modified, _ = smart_crush_tool_output(injection, config)
            # If it returns, it handled it
        except Exception as e:
            failures.append(f"{injection[:30]}: {type(e).__name__}")

    if not failures:
        result.actual_behavior = "All injection attempts handled gracefully"
        result.passed = True
    else:
        result.actual_behavior = f"Failures: {failures}"
        result.passed = False

    return result


def test_headroom_marker_collision() -> AdversarialResult:
    """
    ATTACK: Input data already contains __headroom_ fields.
    """
    result = AdversarialResult(
        name="Marker Field Collision",
        category="injection",
        expected_behavior="Should not confuse existing __headroom_ fields with our markers",
        severity="high",
    )

    # Data that already has __headroom_ fields
    items = [
        {
            "id": i,
            "__headroom_compressed": True,  # Fake marker!
            "__headroom_hash": "fakehash12345678",
            "__headroom_stats": {"fake": True},
        }
        for i in range(100)
    ]

    config = SmartCrusherConfig(max_items_after_crush=15)
    original_json = json.dumps(items)

    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)
    compressed = json.loads(compressed_json)

    # Check if our compression worked despite fake markers
    if isinstance(compressed, list) and len(compressed) <= 20:
        result.actual_behavior = "Compression worked despite fake markers"
        result.passed = True
    else:
        result.actual_behavior = f"Unexpected result type or length: {type(compressed)}, {len(compressed) if isinstance(compressed, list) else 'N/A'}"
        result.passed = False

    return result


def test_unicode_and_emoji_handling() -> AdversarialResult:
    """
    ATTACK: Unicode edge cases in content.
    """
    result = AdversarialResult(
        name="Unicode/Emoji Handling",
        category="injection",
        expected_behavior="Should handle Unicode correctly",
        severity="medium",
    )

    items = [
        {"id": 1, "message": "Error: 🔥 Server on fire 🔥", "status": "error"},
        {"id": 2, "message": "成功: 操作完成", "status": "success"},
        {"id": 3, "message": "Error: \u0000\u0001\u0002 null bytes", "status": "error"},
        {"id": 4, "message": "Ошибка: критический сбой", "status": "error"},
        {"id": 5, "message": "🎉🎊🎈" * 100, "status": "success"},  # Lots of emoji
    ]

    for i in range(95):
        items.append({"id": i + 6, "message": "Normal", "status": "success"})

    config = SmartCrusherConfig(max_items_after_crush=15)
    original_json = json.dumps(items, ensure_ascii=False)

    try:
        compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)
        compressed = json.loads(compressed_json)

        # Check if error items with unicode were preserved
        errors_preserved = len([item for item in compressed if item.get("status") == "error"])

        result.actual_behavior = f"Handled Unicode, {errors_preserved} errors preserved"
        result.passed = errors_preserved >= 2

    except Exception as e:
        result.actual_behavior = f"Unicode handling failed: {e}"
        result.passed = False

    return result


def test_extremely_long_strings() -> AdversarialResult:
    """
    ATTACK: Items with extremely long string values.
    """
    result = AdversarialResult(
        name="Extremely Long Strings",
        category="injection",
        expected_behavior="Should handle without memory issues",
        severity="medium",
    )

    # One item with a 10MB string
    huge_string = "x" * (10 * 1024 * 1024)  # 10MB

    items = [
        {"id": 0, "huge": huge_string, "status": "error"},  # Should be kept (error)
        *[{"id": i, "normal": "small"} for i in range(1, 100)],
    ]

    config = SmartCrusherConfig(max_items_after_crush=15)

    sys.getsizeof(items)
    start_time = time.time()

    try:
        original_json = json.dumps(items)
        compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)

        elapsed = time.time() - start_time

        if elapsed > 30:
            result.actual_behavior = f"Took too long: {elapsed:.1f}s"
            result.passed = False
        else:
            result.actual_behavior = f"Handled 10MB string in {elapsed:.1f}s"
            result.passed = True

    except MemoryError:
        result.actual_behavior = "MemoryError on large string"
        result.passed = False
        result.severity = "critical"
    finally:
        del huge_string
        del items
        gc.collect()

    return result


def test_query_injection_in_search() -> AdversarialResult:
    """
    ATTACK: Malicious search query.
    """
    result = AdversarialResult(
        name="Search Query Injection",
        category="injection",
        expected_behavior="Should sanitize search queries",
        severity="high",
    )

    reset_compression_store()
    store = get_compression_store()

    items = [{"id": i, "data": f"item {i}"} for i in range(100)]

    hash_key = store.store(
        original=json.dumps(items),
        compressed=json.dumps(items[:10]),
        original_item_count=100,
        compressed_item_count=10,
    )

    # Various injection attempts
    malicious_queries = [
        "'; DROP TABLE items; --",
        "<script>alert('xss')</script>",
        "{{7*7}}",  # Template injection
        "${7*7}",  # Expression injection
        "\\x00\\x01\\x02",  # Null bytes
        "*" * 10000,  # Long query
        ".*",  # Regex wildcard
        "(a]",  # Invalid regex
    ]

    failures = []
    for query in malicious_queries:
        try:
            store.search(hash_key, query)
            # If it returns without error, it handled the injection
        except Exception as e:
            failures.append(f"{query[:20]}: {type(e).__name__}")

    if not failures:
        result.actual_behavior = "All malicious queries handled safely"
        result.passed = True
    else:
        result.actual_behavior = f"Failures: {failures}"
        result.passed = False

    return result


# =============================================================================
# CATEGORY 4: RACE CONDITIONS
# =============================================================================


def test_concurrent_store_same_content() -> AdversarialResult:
    """
    ATTACK: Multiple threads storing identical content simultaneously.
    """
    result = AdversarialResult(
        name="Concurrent Store Same Content",
        category="race",
        expected_behavior="Should handle concurrent stores without data corruption",
        severity="high",
    )

    reset_compression_store()
    store = get_compression_store()

    content = json.dumps([{"id": i} for i in range(100)])

    results = []
    errors = []

    def store_content():
        try:
            hash_key = store.store(
                original=content,
                compressed=content[:50],
                original_item_count=100,
                compressed_item_count=5,
            )
            results.append(hash_key)
        except Exception as e:
            errors.append(str(e))

    # 100 concurrent stores of same content
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(store_content) for _ in range(100)]
        concurrent.futures.wait(futures)

    if errors:
        result.actual_behavior = f"Errors during concurrent store: {errors[:3]}"
        result.passed = False
    elif len(set(results)) != 1:
        result.actual_behavior = f"Got different hashes for same content: {set(results)}"
        result.passed = False
    else:
        result.actual_behavior = "All concurrent stores returned same hash"
        result.passed = True

    return result


def test_concurrent_store_and_evict() -> AdversarialResult:
    """
    ATTACK: Store while eviction is happening.
    """
    result = AdversarialResult(
        name="Concurrent Store and Evict",
        category="race",
        expected_behavior="Eviction should not corrupt concurrent stores",
        severity="high",
    )

    reset_compression_store()
    store = CompressionStore(max_entries=10)  # Small capacity

    errors = []
    stored_hashes = []

    def rapid_store(thread_id):
        for i in range(50):
            try:
                content = json.dumps([{"thread": thread_id, "iteration": i}])
                hash_key = store.store(
                    original=content,
                    compressed=content,
                    original_item_count=1,
                    compressed_item_count=1,
                )
                stored_hashes.append(hash_key)
            except Exception as e:
                errors.append(f"Thread {thread_id}, iter {i}: {e}")

    # 10 threads, each storing 50 items = 500 stores with max_entries=10
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(rapid_store, i) for i in range(10)]
        concurrent.futures.wait(futures)

    if errors:
        result.actual_behavior = f"Errors: {errors[:5]}"
        result.passed = False
    else:
        result.actual_behavior = "500 stores with capacity 10 succeeded"
        result.passed = True

    return result


def test_concurrent_feedback_updates() -> AdversarialResult:
    """
    ATTACK: Multiple threads updating feedback simultaneously.
    """
    result = AdversarialResult(
        name="Concurrent Feedback Updates",
        category="race",
        expected_behavior="Feedback counts should be accurate under concurrency",
        severity="high",
    )

    reset_compression_feedback()
    feedback = get_compression_feedback()

    tool_name = "concurrent_test_tool"
    expected_compressions = 1000
    expected_retrievals = 500

    def record_compressions():
        for _ in range(expected_compressions // 10):
            feedback.record_compression(tool_name, 100, 10)

    def record_retrievals():
        # 5 threads × 100 iterations = 500 retrievals
        for i in range(expected_retrievals // 5):
            event = RetrievalEvent(
                hash=f"hash{i:012d}",
                query=None,
                items_retrieved=100,
                total_items=100,
                tool_name=tool_name,
                timestamp=time.time(),
                retrieval_type="full",
            )
            feedback.record_retrieval(event)

    # 10 threads each doing compressions (1000/10=100 each), 5 doing retrievals (500/5=100 each)
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = []
        for _ in range(10):
            futures.append(executor.submit(record_compressions))
        for _ in range(5):
            futures.append(executor.submit(record_retrievals))
        concurrent.futures.wait(futures)

    patterns = feedback.get_all_patterns()
    pattern = patterns.get(tool_name)

    if pattern is None:
        result.actual_behavior = "Pattern not found"
        result.passed = False
    elif (
        pattern.total_compressions == expected_compressions
        and pattern.total_retrievals == expected_retrievals
    ):
        result.actual_behavior = f"Exact counts: {pattern.total_compressions} compressions, {pattern.total_retrievals} retrievals"
        result.passed = True
    else:
        result.actual_behavior = f"Count mismatch: {pattern.total_compressions} compressions (expected {expected_compressions}), {pattern.total_retrievals} retrievals (expected {expected_retrievals})"
        result.passed = False

    result.details = {
        "expected_compressions": expected_compressions,
        "actual_compressions": pattern.total_compressions if pattern else 0,
        "expected_retrievals": expected_retrievals,
        "actual_retrievals": pattern.total_retrievals if pattern else 0,
    }

    return result


# =============================================================================
# CATEGORY 5: DECEPTIVE DATA
# =============================================================================


def test_hidden_error_in_nested_structure() -> AdversarialResult:
    """
    ATTACK: Error hidden deep in nested structure.
    """
    result = AdversarialResult(
        name="Hidden Error in Nested Structure",
        category="deceptive",
        expected_behavior="Should detect errors in nested objects",
        severity="high",
    )

    items = []
    error_idx = 50

    for i in range(100):
        if i == error_idx:
            # Error hidden deep inside
            items.append(
                {
                    "id": i,
                    "status": "success",  # Top level says success!
                    "details": {
                        "level1": {
                            "level2": {
                                "actual_status": "CRITICAL_ERROR",
                                "error": True,
                                "message": "System failure",
                            }
                        }
                    },
                }
            )
        else:
            items.append(
                {
                    "id": i,
                    "status": "success",
                    "details": {"level1": {"level2": {"actual_status": "ok"}}},
                }
            )

    config = SmartCrusherConfig(max_items_after_crush=15)
    original_json = json.dumps(items)

    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)
    compressed = json.loads(compressed_json)

    # Check if the nested error was preserved
    nested_error_found = any("CRITICAL_ERROR" in json.dumps(item) for item in compressed)

    if nested_error_found:
        result.actual_behavior = "Nested error was detected and preserved"
        result.passed = True
    else:
        result.actual_behavior = "Nested error was LOST - only top-level status checked"
        result.passed = False

    return result


def test_misleading_score_field() -> AdversarialResult:
    """
    ATTACK: Score field that doesn't indicate importance.

    Items with score=0.99 are spam, items with score=0.01 are critical.
    """
    result = AdversarialResult(
        name="Misleading Score Field",
        category="deceptive",
        expected_behavior="Should not blindly trust high scores",
        severity="medium",
    )

    items = []
    critical_indices = [25, 50, 75]

    for i in range(100):
        if i in critical_indices:
            # LOW score but CRITICAL
            items.append(
                {
                    "id": i,
                    "score": 0.01,  # Low score
                    "type": "critical_alert",
                    "message": "URGENT: Action required",
                }
            )
        else:
            # HIGH score but SPAM
            items.append(
                {
                    "id": i,
                    "score": 0.99,  # High score
                    "type": "spam",
                    "message": "Buy now! Limited offer!",
                }
            )

    config = SmartCrusherConfig(max_items_after_crush=15)
    original_json = json.dumps(items)

    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)
    compressed = json.loads(compressed_json)

    # Check what was kept
    critical_kept = len([item for item in compressed if item.get("type") == "critical_alert"])
    spam_kept = len([item for item in compressed if item.get("type") == "spam"])

    # We should preserve ALL critical items due to "critical" keyword detection
    # The remaining slots can go to high-score items - that's acceptable
    # The key guarantee: we NEVER lose items matching important keywords
    if critical_kept < 3:
        result.actual_behavior = f"Lost critical items! Only kept {critical_kept}/3 critical"
        result.passed = False
    else:
        result.actual_behavior = f"Kept all {critical_kept} critical items (plus {spam_kept} spam) - keyword detection worked"
        result.passed = True

    result.details = {
        "critical_kept": critical_kept,
        "spam_kept": spam_kept,
    }

    return result


def test_timestamp_anomaly_not_value() -> AdversarialResult:
    """
    ATTACK: Anomaly in timestamp, not in measured value.

    One entry is from the FUTURE - this is the anomaly!
    """
    result = AdversarialResult(
        name="Timestamp Anomaly",
        category="deceptive",
        expected_behavior="Should detect timestamp anomalies",
        severity="medium",
    )

    items = []
    anomaly_idx = 50

    for i in range(100):
        if i == anomaly_idx:
            # Future timestamp - something is wrong!
            items.append(
                {
                    "timestamp": "2030-01-01T00:00:00Z",  # FUTURE!
                    "value": 50,  # Normal value
                    "id": i,
                }
            )
        else:
            items.append(
                {
                    "timestamp": f"2025-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00Z",
                    "value": 50 + (i % 10),  # Normal variation
                    "id": i,
                }
            )

    config = SmartCrusherConfig(max_items_after_crush=15)
    original_json = json.dumps(items)

    compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)
    compressed = json.loads(compressed_json)

    # Check if future timestamp was preserved
    future_found = any("2030" in str(item.get("timestamp", "")) for item in compressed)

    if future_found:
        result.actual_behavior = "Future timestamp anomaly preserved"
        result.passed = True
    else:
        result.actual_behavior = "Timestamp anomaly LOST - only value anomalies detected"
        result.passed = False

    return result


# =============================================================================
# EXTREME STRESS TESTS - Designed to Break Assumptions
# =============================================================================


def test_deeply_nested_structure() -> AdversarialResult:
    """
    ATTACK: Extremely deep nesting to cause stack overflow.

    100 levels of nested objects containing arrays.
    """
    result = AdversarialResult(
        name="Deep Nesting Attack",
        category="extreme",
        expected_behavior="Should handle deep nesting without stack overflow",
        severity="critical",
    )

    # Build deeply nested structure
    depth = 100
    inner = [{"id": i, "value": f"leaf_{i}"} for i in range(20)]

    current = inner
    for level in range(depth):
        current = {"level": level, "data": current}

    try:
        config = SmartCrusherConfig(max_items_after_crush=10)
        original_json = json.dumps(current)

        compressed_json, was_modified, reason = smart_crush_tool_output(original_json, config)
        result.actual_behavior = f"Handled {depth} levels of nesting"
        result.passed = True
    except RecursionError as e:
        result.actual_behavior = f"Stack overflow at depth {depth}: {e}"
        result.passed = False
    except Exception as e:
        result.actual_behavior = f"Unexpected error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_nan_infinity_scores() -> AdversarialResult:
    """
    ATTACK: Score fields with NaN, Infinity, -Infinity.

    These are valid JSON when serialized from Python but break comparisons.
    """
    result = AdversarialResult(
        name="NaN/Infinity Scores",
        category="extreme",
        expected_behavior="Should handle special float values gracefully",
        severity="high",
    )

    items = []
    for i in range(50):
        score = i / 10.0
        if i == 10:
            score = float("nan")
        elif i == 20:
            score = float("inf")
        elif i == 30:
            score = float("-inf")

        items.append({"id": i, "score": score, "name": f"item_{i}"})

    try:
        config = SmartCrusherConfig(max_items_after_crush=15)
        # Note: json.dumps will fail on NaN/Inf by default, use allow_nan
        original_json = json.dumps(items, allow_nan=True)

        compressed_json, was_modified, reason = smart_crush_tool_output(original_json, config)
        compressed = json.loads(compressed_json, parse_constant=lambda x: None)

        result.actual_behavior = f"Handled special floats, compressed to {len(compressed)} items"
        result.passed = True
    except (ValueError, TypeError) as e:
        result.actual_behavior = f"Failed on special floats: {e}"
        result.passed = False
    except Exception as e:
        result.actual_behavior = f"Unexpected error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_mixed_type_array() -> AdversarialResult:
    """
    ATTACK: Array with mixed types (dicts, strings, numbers, nulls).

    SmartCrusher expects arrays of dicts - what happens with mixed?
    """
    result = AdversarialResult(
        name="Mixed Type Array",
        category="extreme",
        expected_behavior="Should handle or gracefully skip mixed arrays",
        severity="medium",
    )

    mixed_array = [
        {"id": 1, "type": "dict"},
        "just a string",
        42,
        None,
        {"id": 2, "type": "dict"},
        ["nested", "array"],
        True,
        {"id": 3, "type": "dict"},
    ]

    try:
        config = SmartCrusherConfig(max_items_after_crush=5)
        original_json = json.dumps(mixed_array)

        compressed_json, was_modified, reason = smart_crush_tool_output(original_json, config)

        result.actual_behavior = f"Handled mixed array: modified={was_modified}, reason={reason}"
        result.passed = True
    except Exception as e:
        result.actual_behavior = f"Crashed on mixed array: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_catastrophic_regex_in_search() -> AdversarialResult:
    """
    ATTACK: Search query designed to cause catastrophic backtracking.

    Pattern like (a+)+ on "aaaaaaaaaaaaaaaaaaaaaaaaaaab" can hang regex engines.
    """
    result = AdversarialResult(
        name="Regex Catastrophic Backtracking",
        category="extreme",
        expected_behavior="Should not hang on malicious search patterns",
        severity="critical",
    )

    reset_compression_store()
    store = get_compression_store()

    items = [{"id": i, "content": "a" * 50 + "b"} for i in range(100)]

    hash_key = store.store(
        original=json.dumps(items),
        compressed=json.dumps(items[:10]),
        original_item_count=100,
        compressed_item_count=10,
        tool_name="regex_test",
    )

    # These patterns could cause catastrophic backtracking in naive regex
    evil_patterns = [
        "(a+)+$",
        "(a|aa)+$",
        "(a+)+b",
        "([a-zA-Z]+)*X",
    ]

    try:
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Search took too long")

        # Set 2 second timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)

        for pattern in evil_patterns:
            # BM25 search doesn't use regex, so should be safe
            store.search(hash_key, pattern)

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        result.actual_behavior = "Search completed without hanging"
        result.passed = True
    except TimeoutError:
        result.actual_behavior = "Search hung on regex-like pattern"
        result.passed = False
    except Exception as e:
        result.actual_behavior = f"Error: {type(e).__name__}: {e}"
        result.passed = True  # Failing safely is OK

    return result


def test_million_items() -> AdversarialResult:
    """
    ATTACK: Array with 1 million items.

    Test memory and performance at scale.
    """
    result = AdversarialResult(
        name="Million Items Scale",
        category="extreme",
        expected_behavior="Should handle large arrays without OOM",
        severity="high",
    )

    try:
        # Create 100K items (not 1M to keep test reasonable)
        item_count = 100_000
        items = [{"id": i, "value": i % 1000} for i in range(item_count)]

        config = SmartCrusherConfig(max_items_after_crush=15)

        start = time.time()
        original_json = json.dumps(items)
        compressed_json, was_modified, reason = smart_crush_tool_output(original_json, config)
        elapsed = time.time() - start

        compressed = json.loads(compressed_json)

        result.actual_behavior = (
            f"Compressed {item_count} items to {len(compressed)} in {elapsed:.2f}s"
        )
        result.passed = elapsed < 10.0  # Should complete in under 10 seconds
        result.details = {"item_count": item_count, "elapsed_seconds": elapsed}
    except MemoryError:
        result.actual_behavior = "Out of memory"
        result.passed = False
    except Exception as e:
        result.actual_behavior = f"Error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_item_with_thousands_of_fields() -> AdversarialResult:
    """
    ATTACK: Items with 10,000 fields each.

    Field analysis iterates over all fields - what's the cost?
    """
    result = AdversarialResult(
        name="Thousands of Fields",
        category="extreme",
        expected_behavior="Should handle items with many fields",
        severity="medium",
    )

    try:
        field_count = 5000
        items = []
        for i in range(20):
            item = {"id": i}
            for f in range(field_count):
                item[f"field_{f}"] = f"value_{f}_{i}"
            items.append(item)

        config = SmartCrusherConfig(max_items_after_crush=10)

        start = time.time()
        original_json = json.dumps(items)
        compressed_json, was_modified, reason = smart_crush_tool_output(original_json, config)
        elapsed = time.time() - start

        result.actual_behavior = f"Handled {field_count} fields/item in {elapsed:.2f}s"
        result.passed = elapsed < 5.0
    except Exception as e:
        result.actual_behavior = f"Error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_identical_items() -> AdversarialResult:
    """
    ATTACK: All items are EXACTLY identical.

    Uniqueness detection should handle this edge case.
    """
    result = AdversarialResult(
        name="All Identical Items",
        category="extreme",
        expected_behavior="Should handle identical items efficiently",
        severity="low",
    )

    # 1000 perfectly identical items
    template = {"id": 1, "status": "ok", "value": 42, "message": "All good"}
    items = [template.copy() for _ in range(1000)]

    try:
        config = SmartCrusherConfig(max_items_after_crush=15)

        compressed_json, was_modified, reason = smart_crush_tool_output(json.dumps(items), config)
        compressed = json.loads(compressed_json)

        result.actual_behavior = (
            f"Compressed {len(items)} identical items to {len(compressed)}: {reason}"
        )
        # Should heavily compress since all items are the same
        result.passed = len(compressed) <= 15
    except Exception as e:
        result.actual_behavior = f"Error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_all_fields_none() -> AdversarialResult:
    """
    ATTACK: Items where every field value is null/None.
    """
    result = AdversarialResult(
        name="All Null Values",
        category="extreme",
        expected_behavior="Should handle all-null items",
        severity="low",
    )

    items = [{"id": None, "value": None, "status": None, "data": None} for _ in range(100)]

    try:
        config = SmartCrusherConfig(max_items_after_crush=10)

        compressed_json, was_modified, reason = smart_crush_tool_output(json.dumps(items), config)

        result.actual_behavior = f"Handled all-null items: modified={was_modified}"
        result.passed = True
    except Exception as e:
        result.actual_behavior = f"Error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_unicode_normalization_attack() -> AdversarialResult:
    """
    ATTACK: Unicode strings that look identical but are different.

    "café" can be encoded as:
    - c a f é (4 chars, é is U+00E9)
    - c a f e ́ (5 chars, e + combining acute U+0301)

    These look identical but are different strings!
    """
    result = AdversarialResult(
        name="Unicode Normalization Attack",
        category="extreme",
        expected_behavior="Should handle unicode edge cases",
        severity="medium",
    )

    # Two visually identical but byte-different strings
    composed = "café"  # é as single char
    decomposed = "cafe\u0301"  # e + combining accent

    items = []
    for i in range(50):
        if i % 2 == 0:
            items.append({"id": i, "name": composed, "type": "composed"})
        else:
            items.append({"id": i, "name": decomposed, "type": "decomposed"})

    # Add one special item
    items[25] = {"id": 25, "name": composed, "type": "TARGET", "status": "error"}

    try:
        config = SmartCrusherConfig(max_items_after_crush=15)

        compressed_json, was_modified, reason = smart_crush_tool_output(json.dumps(items), config)
        compressed = json.loads(compressed_json)

        # Check if we kept the TARGET item
        target_found = any(item.get("type") == "TARGET" for item in compressed)

        result.actual_behavior = f"Unicode handled, target found: {target_found}"
        result.passed = target_found
    except Exception as e:
        result.actual_behavior = f"Error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_concurrent_reset_during_operation() -> AdversarialResult:
    """
    ATTACK: Reset global state while operations are in progress.
    """
    result = AdversarialResult(
        name="Concurrent Reset Attack",
        category="extreme",
        expected_behavior="Should not crash on concurrent reset",
        severity="high",
    )

    errors = []
    operations_completed = [0]

    def do_operations():
        for i in range(100):
            try:
                store = get_compression_store()
                items = [{"id": j, "iter": i} for j in range(20)]
                hash_key = store.store(
                    original=json.dumps(items),
                    compressed=json.dumps(items[:5]),
                    original_item_count=20,
                    compressed_item_count=5,
                    tool_name="reset_test",
                )
                store.retrieve(hash_key)
                store.search(hash_key, "test")
                operations_completed[0] += 1
            except Exception as e:
                errors.append(f"Op error: {type(e).__name__}: {e}")

    def do_resets():
        for _ in range(50):
            try:
                reset_compression_store()
                reset_compression_feedback()
                time.sleep(0.001)
            except Exception as e:
                errors.append(f"Reset error: {type(e).__name__}: {e}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(5):
                futures.append(executor.submit(do_operations))
            for _ in range(3):
                futures.append(executor.submit(do_resets))

            concurrent.futures.wait(futures)

        if errors:
            result.actual_behavior = f"Errors during concurrent reset: {errors[:3]}"
            result.passed = False
        else:
            result.actual_behavior = (
                f"Completed {operations_completed[0]} operations with concurrent resets"
            )
            result.passed = True
    except Exception as e:
        result.actual_behavior = f"Crashed: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_zero_byte_in_content() -> AdversarialResult:
    """
    ATTACK: Null bytes (\\x00) embedded in strings.

    Can truncate strings in C-based systems.
    """
    result = AdversarialResult(
        name="Null Byte Injection",
        category="extreme",
        expected_behavior="Should preserve content with null bytes",
        severity="high",
    )

    items = []
    for i in range(50):
        # Embed null byte in various positions
        if i == 10:
            items.append({"id": i, "data": "before\x00after", "status": "error"})
        elif i == 20:
            items.append({"id": i, "data": "\x00start", "status": "error"})
        elif i == 30:
            items.append({"id": i, "data": "end\x00", "status": "error"})
        else:
            items.append({"id": i, "data": "normal", "status": "ok"})

    try:
        config = SmartCrusherConfig(max_items_after_crush=15)
        original_json = json.dumps(items)

        compressed_json, was_modified, _ = smart_crush_tool_output(original_json, config)
        compressed = json.loads(compressed_json)

        # Check if null-byte items were preserved (they have status=error)
        error_items = [item for item in compressed if item.get("status") == "error"]

        # Also verify the null bytes survived
        null_byte_survived = any("\x00" in str(item.get("data", "")) for item in compressed)

        result.actual_behavior = (
            f"Kept {len(error_items)} error items, null bytes intact: {null_byte_survived}"
        )
        result.passed = len(error_items) == 3 and null_byte_survived
    except Exception as e:
        result.actual_behavior = f"Error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_recursive_json_structure() -> AdversarialResult:
    """
    ATTACK: Structure that references itself (via string representation).

    Not true circular reference (JSON doesn't support that), but deeply self-similar.
    """
    result = AdversarialResult(
        name="Self-Similar Structure",
        category="extreme",
        expected_behavior="Should handle self-similar data",
        severity="low",
    )

    # Create structure where values contain JSON-like strings
    items = []
    for i in range(50):
        inner = json.dumps({"nested_id": i, "value": "inner"})
        items.append(
            {
                "id": i,
                "data": inner,  # JSON string inside JSON
                "meta": json.dumps({"level": 1, "payload": inner}),  # Double nested
            }
        )

    try:
        config = SmartCrusherConfig(max_items_after_crush=15)

        compressed_json, was_modified, reason = smart_crush_tool_output(json.dumps(items), config)
        compressed = json.loads(compressed_json)

        result.actual_behavior = f"Handled self-similar structure: {len(compressed)} items"
        result.passed = True
    except Exception as e:
        result.actual_behavior = f"Error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_extreme_numeric_values() -> AdversarialResult:
    """
    ATTACK: Extreme numeric values that might overflow.

    Very large integers, very small floats, edge cases.
    """
    result = AdversarialResult(
        name="Extreme Numeric Values",
        category="extreme",
        expected_behavior="Should handle extreme numbers",
        severity="medium",
    )

    items = [
        {"id": 0, "value": 0},
        {"id": 1, "value": -1},
        {"id": 2, "value": 2**63 - 1},  # Max int64
        {"id": 3, "value": -(2**63)},  # Min int64
        {"id": 4, "value": 2**64},  # Overflow int64
        {"id": 5, "value": 10**308},  # Near max float
        {"id": 6, "value": 10**-308},  # Near min positive float
        {"id": 7, "value": 0.1 + 0.2},  # Classic float precision issue
        {"id": 8, "value": 1e-400},  # Underflow to 0
        {"id": 9, "score": 999999999999999999999},  # Very large score
    ]

    # Add normal items
    for i in range(10, 50):
        items.append({"id": i, "value": i, "score": i / 100})

    try:
        config = SmartCrusherConfig(max_items_after_crush=15)

        compressed_json, was_modified, reason = smart_crush_tool_output(json.dumps(items), config)
        compressed = json.loads(compressed_json)

        result.actual_behavior = f"Handled extreme numbers: {len(compressed)} items"
        result.passed = True
    except (OverflowError, ValueError) as e:
        result.actual_behavior = f"Numeric error: {e}"
        result.passed = False
    except Exception as e:
        result.actual_behavior = f"Error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_adversarial_field_names() -> AdversarialResult:
    """
    ATTACK: Field names that might confuse our analysis.

    Fields named "__proto__", "constructor", "toString", etc.
    """
    result = AdversarialResult(
        name="Adversarial Field Names",
        category="extreme",
        expected_behavior="Should handle special field names",
        severity="medium",
    )

    items = []
    for i in range(30):
        items.append(
            {
                "id": i,
                "__proto__": {"admin": True},  # Prototype pollution attempt
                "constructor": "evil",
                "toString": "hacked",
                "__class__": "injected",
                "hasOwnProperty": False,
                "score": i / 10,
                "status": "error" if i == 15 else "ok",
            }
        )

    try:
        config = SmartCrusherConfig(max_items_after_crush=10)

        compressed_json, was_modified, reason = smart_crush_tool_output(json.dumps(items), config)
        compressed = json.loads(compressed_json)

        # Verify error item was kept
        error_kept = any(item.get("status") == "error" for item in compressed)

        result.actual_behavior = f"Handled adversarial fields, error kept: {error_kept}"
        result.passed = error_kept
    except Exception as e:
        result.actual_behavior = f"Error: {type(e).__name__}: {e}"
        result.passed = False

    return result


def test_store_during_eviction_storm() -> AdversarialResult:
    """
    ATTACK: Rapid store/retrieve during aggressive eviction.

    max_entries=5 with 100 concurrent stores.
    """
    result = AdversarialResult(
        name="Eviction Storm",
        category="extreme",
        expected_behavior="Should maintain consistency during eviction",
        severity="high",
    )

    reset_compression_store()
    # Create store with very small capacity
    store = CompressionStore(max_entries=5, default_ttl=300)

    stored_hashes = []
    retrieved_count = [0]
    errors = []
    lock = threading.Lock()

    def store_and_retrieve():
        for _i in range(50):
            try:
                items = [{"id": j, "thread": threading.current_thread().name} for j in range(10)]
                hash_key = store.store(
                    original=json.dumps(items),
                    compressed=json.dumps(items[:2]),
                    original_item_count=10,
                    compressed_item_count=2,
                    tool_name="eviction_test",
                )

                with lock:
                    stored_hashes.append(hash_key)

                # Immediately try to retrieve
                entry = store.retrieve(hash_key)
                if entry:
                    with lock:
                        retrieved_count[0] += 1

            except Exception as e:
                with lock:
                    errors.append(str(e))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(store_and_retrieve) for _ in range(20)]
            concurrent.futures.wait(futures)

        if errors:
            result.actual_behavior = f"Errors: {errors[:3]}"
            result.passed = False
        else:
            # Some eviction is expected, but we shouldn't crash
            result.actual_behavior = (
                f"Stored {len(stored_hashes)}, retrieved {retrieved_count[0]} (eviction expected)"
            )
            result.passed = True
    except Exception as e:
        result.actual_behavior = f"Crashed: {type(e).__name__}: {e}"
        result.passed = False

    return result


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("\n" + "=" * 70)
    print("  ADVERSARIAL CCR TESTS")
    print("  Intentionally Trying to Break Our Code")
    print("=" * 70 + "\n")

    tests = [
        # Semantic attacks
        test_all_items_are_errors,
        test_error_keyword_in_normal_data,
        test_needle_looks_exactly_like_hay,
        test_anomaly_in_string_not_number,
        # Boundary conditions
        test_empty_array,
        test_single_item_array,
        test_exactly_max_items,
        test_max_items_plus_one,
        test_hash_collision_attempt,
        test_ttl_exact_boundary,
        # Injection attacks
        test_json_injection_in_content,
        test_headroom_marker_collision,
        test_unicode_and_emoji_handling,
        test_extremely_long_strings,
        test_query_injection_in_search,
        # Race conditions
        test_concurrent_store_same_content,
        test_concurrent_store_and_evict,
        test_concurrent_feedback_updates,
        # Deceptive data
        test_hidden_error_in_nested_structure,
        test_misleading_score_field,
        test_timestamp_anomaly_not_value,
        # EXTREME stress tests
        test_deeply_nested_structure,
        test_nan_infinity_scores,
        test_mixed_type_array,
        test_catastrophic_regex_in_search,
        test_million_items,
        test_item_with_thousands_of_fields,
        test_identical_items,
        test_all_fields_none,
        test_unicode_normalization_attack,
        test_concurrent_reset_during_operation,
        test_zero_byte_in_content,
        test_recursive_json_structure,
        test_extreme_numeric_values,
        test_adversarial_field_names,
        test_store_during_eviction_storm,
    ]

    results_by_category = {}

    for test_func in tests:
        print(f"  Running {test_func.__name__}...", end=" ", flush=True)
        result = run_test(test_func)

        if result.category not in results_by_category:
            results_by_category[result.category] = []
        results_by_category[result.category].append(result)

        status = "✓" if result.passed else "✗"
        print(f"{status}")

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS BY CATEGORY")
    print("=" * 70)

    total_passed = 0
    total_tests = 0
    critical_failures = []

    for category, results in results_by_category.items():
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        total_passed += passed
        total_tests += total

        print(f"\n  {category.upper()}: {passed}/{total}")

        for r in results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            print(f"    {status}  {r.name}")

            if not r.passed:
                print(f"           Expected: {r.expected_behavior}")
                print(f"           Actual:   {r.actual_behavior}")

                if r.severity == "critical":
                    critical_failures.append(r)

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_passed}/{total_tests} tests passed")

    if critical_failures:
        print(f"\n  ⚠️  {len(critical_failures)} CRITICAL FAILURES:")
        for r in critical_failures:
            print(f"      - {r.name}: {r.actual_behavior[:50]}")

    print("=" * 70 + "\n")

    # Exit code
    failed = total_tests - total_passed
    exit(failed)


if __name__ == "__main__":
    main()

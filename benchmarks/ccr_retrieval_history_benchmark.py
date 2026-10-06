#!/usr/bin/env python3
"""Measure CCR retrieval-history maintenance once the event cap is reached.

Past 1,000 events the history has to discard the oldest event on every single
retrieval. Bounding the container itself makes that O(1); the previous
append-then-reslice rebuilt a 1,000-pointer list each time.

This times ``_log_retrieval`` directly — the store's public ``retrieve()`` also
does backend I/O and a feedback drain, which would bury the difference. Read
the result as "cost of one history append at steady state", not as a proxy for
end-to-end retrieval latency.

Usage:
    python benchmarks/ccr_retrieval_history_benchmark.py
    python benchmarks/ccr_retrieval_history_benchmark.py --events 5000 --runs 7
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Run as a script, sys.path[0] is benchmarks/, so an editable install of another
# checkout would win for `import headroom` and this would silently measure the
# wrong tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from headroom.cache.compression_store import CompressionStore  # noqa: E402


def _fill_to_cap(store: CompressionStore) -> None:
    """Bring the history to its cap so every later append must evict."""
    for i in range(store._max_events):
        store._log_retrieval(
            hash_key="h",
            query=f"warm{i}",
            items_retrieved=1,
            total_items=1,
            tool_name="bench",
            retrieval_type="full",
        )
    store._pending_feedback_events.clear()


def measure(events: int) -> float:
    """Return microseconds per history append at steady state."""
    store = CompressionStore(enable_feedback=True)
    _fill_to_cap(store)

    start = time.perf_counter()
    for i in range(events):
        store._log_retrieval(
            hash_key="h",
            query=f"q{i}",
            items_retrieved=1,
            total_items=1,
            tool_name="bench",
            retrieval_type="full",
        )
    elapsed = time.perf_counter() - start

    # Each run uses a fresh store, so the unbounded feedback queue this also
    # filled goes away with it.
    return elapsed / events * 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=20_000)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()

    store = CompressionStore()
    print(f"container: {type(store._retrieval_events).__name__}")
    print(f"cap: {store._max_events} events")

    samples = [measure(args.events) for _ in range(args.runs)]

    print(f"appends per run: {args.events}, runs: {args.runs}")
    print(f"median: {statistics.median(samples):.3f} us/append")
    print(f"min: {min(samples):.3f} us, max: {max(samples):.3f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

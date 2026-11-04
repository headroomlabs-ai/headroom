#!/usr/bin/env python3
"""Measure SQLite CCR insertion cleanup without expiring live entries."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Run as a script sys.path[0] is benchmarks/, so an editable install of another
# checkout would win and we would silently measure the wrong tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from headroom.cache.backends.sqlite import SQLiteBackend
from headroom.cache.compression_store import CompressionEntry, CompressionStore


def benchmark(entry_count: int, payload_bytes: int, runs: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="headroom-ccr-expiry-") as tmp:
        backend = SQLiteBackend(Path(tmp) / "ccr.db")
        payload = "x" * payload_bytes
        for index in range(entry_count):
            key = str(index)
            backend.set(
                key,
                CompressionEntry(
                    hash=key,
                    original_content=payload,
                    compressed_content="compact",
                    original_tokens=payload_bytes // 4,
                    compressed_tokens=2,
                    original_item_count=0,
                    compressed_item_count=0,
                    tool_name=None,
                    tool_call_id=None,
                    query_context=None,
                    created_at=time.time(),
                    ttl=3600,
                ),
            )

        store = CompressionStore(
            max_entries=entry_count + runs + 1,
            backend=backend,
            enable_feedback=False,
        )
        samples_ms: list[float] = []
        for index in range(runs):
            start = time.perf_counter()
            key = store.store(f"new original {index}", f"new compact {index}")
            samples_ms.append((time.perf_counter() - start) * 1000)
            assert store.retrieve(key) is not None

        backend._conn.close()
        return {
            "preexisting_entries": entry_count,
            "original_bytes_each": payload_bytes,
            "runs": runs,
            "median_insert_ms": statistics.median(samples_ms),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--payload-bytes", type=int, nargs="+", default=[1024, 16384])
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()

    results = [
        benchmark(entry_count, payload_bytes, args.runs)
        for payload_bytes in args.payload_bytes
        for entry_count in args.entries
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

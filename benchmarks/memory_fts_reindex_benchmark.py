#!/usr/bin/env python3
"""Measure FTS5 reindex throughput: per-record commits vs batched transactions."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Run as a script sys.path[0] is benchmarks/, so an editable install of another
# checkout would win and we would silently measure the wrong tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from headroom.memory.adapters.fts5 import FTS5TextIndex
from headroom.memory.models import Memory


def make_memories(count: int, content_words: int) -> list[Memory]:
    body = " ".join(f"token{n}" for n in range(content_words))
    return [
        Memory(
            id=f"mem-{index:06d}",
            content=f"{body} unique{index}",
            user_id="bench-user",
            session_id=f"session-{index % 7}",
        )
        for index in range(count)
    ]


def benchmark(count: int, content_words: int, page_size: int, runs: int) -> dict[str, object]:
    memories = make_memories(count, content_words)
    per_record: list[float] = []
    batched: list[float] = []

    for _ in range(runs):
        # Fresh temporary database per mode per run so neither inherits the
        # other's page cache or index state.
        with tempfile.TemporaryDirectory(prefix="headroom-fts-bench-") as tmp:
            fts = FTS5TextIndex(db_path=str(Path(tmp) / "memory.db"))
            start = time.perf_counter()
            for memory in memories:
                asyncio.run(fts.index_memory(memory))
            per_record.append((time.perf_counter() - start) * 1000)

        with tempfile.TemporaryDirectory(prefix="headroom-fts-bench-") as tmp:
            fts = FTS5TextIndex(db_path=str(Path(tmp) / "memory.db"))
            start = time.perf_counter()
            for offset in range(0, len(memories), page_size):
                asyncio.run(fts.index_batch_memories(memories[offset : offset + page_size]))
            batched.append((time.perf_counter() - start) * 1000)

    return {
        "records": count,
        "content_words": content_words,
        "page_size": page_size,
        "runs": runs,
        "per_record_ms": statistics.median(per_record),
        "batched_ms": statistics.median(batched),
        "speedup": round(statistics.median(per_record) / statistics.median(batched), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, nargs="+", default=[1000])
    parser.add_argument("--content-words", type=int, default=40)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    results = [
        benchmark(count, args.content_words, args.page_size, args.runs) for count in args.records
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

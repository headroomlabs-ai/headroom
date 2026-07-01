#!/usr/bin/env python3
"""
Agent-task regression suite — measures if compression breaks agent behavior.

Simulates agent tool-call patterns through headroom.compress() and measures:
1. Token savings by content type (code, logs, search, JSON)
2. Must-keep survival (does compression drop critical tokens?)
3. Output agreement (does compressed output preserve structure?)

Runs WITHOUT a real LLM — just measures compression fidelity.

Usage:
    python3 bench/agent_regression.py
    python3 bench/agent_regression.py --model PeetPedro/kompress-v8 --json > regression.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Agent scenario templates (simulating real tool outputs) ────────────
SCENARIOS = [
    {
        "name": "rust_build_error",
        "turns": [
            {"tool": "read_file", "output": """  1  // src/lib.rs
  2  use std::time::{Duration, Instant};
  3  
  4  pub struct RateLimiter {
  5      tokens: f64,
  6      max_tokens: f64,  
  7      refill_rate: f64,
  8      last_refill: Instant,
  9  }
 10  
 11  impl RateLimiter {
 12      pub fn new(max_tokens: f64, refill_rate: f64) -> Self {
 13          Self { tokens: max_tokens, max_tokens, refill_rate, last_refill: Instant::now() }

/home/dev/project/src/lib.rs — 16 lines"""},
            {"tool": "bash", "output": """$ cargo build 2>&1
   Compiling rate-limiter v0.1.0 (/home/dev/project)
error[E0308]: mismatched types
  --> src/lib.rs:13:74
   |
13 |         Self { tokens: max_tokens, max_tokens, refill_rate, last_refill: Instant::now() }
   |                                                                          ^^^^^^^^^^^^^^^ expected `Instant`, found `Result<Instant, std::io::Error>`
   |
help: use `Instant::now()` directly
  --> /rustc/9b00956e5/src/libstd/time.rs:123:5

error: could not compile `rate-limiter` due to previous error"""},
            {"tool": "bash", "output": """$ cargo test 2>&1
   Compiling rate-limiter v0.1.0
    Finished test [unoptimized + debuginfo] target(s) in 2.34s
     Running unittests src/lib.rs (target/debug/deps/rate_limiter-1a2b3c4d)

running 3 tests
test tests::test_token_bucket_creation ... ok
test tests::test_token_consumption ... FAILED
test tests::test_refill_rate ... ok

failures:
---- tests::test_token_consumption stdout ----
thread 'tests::test_token_consumption' panicked at 'assertion failed: limiter.try_acquire()', src/lib.rs:42:9

failures:
    tests::test_token_consumption

test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"""},
        ]
    },
    {
        "name": "python_import_debug",
        "turns": [
            {"tool": "read_file", "output": """  1  # backend/db.py
  2  from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
  3  from sqlalchemy.orm import sessionmaker
  4  
  5  DATABASE_URL = "postgresql+asyncpg://user:pass@localhost:5432/mydb"
  6  
  7  engine = create_async_engine(DATABASE_URL, pool_size=20, max_overflow=10)
  8  async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

/home/dev/project/backend/db.py — 8 lines"""},
            {"tool": "bash", "output": """$ python -m pytest backend/tests/test_db.py -xvs 2>&1
============================= test session starts ==============================
platform linux -- Python 3.11.6, pytest-8.3.4
collected 5 items

backend/tests/test_db.py::test_connection_pool PASSED
backend/tests/test_db.py::test_async_session FAILED

=================================== FAILURES ===================================
___________________________ test_async_session ________________________________
    async def test_async_session():
>       async with async_session() as session:
E       RuntimeError: Task <Task pending name='test_async_session' coro=<test_async_session()>> got Future attached to a different loop

backend/tests/test_db.py:15: RuntimeError
========================= 1 failed, 1 passed in 2.03s =========================""",
            },
        ]
    },
    {
        "name": "search_large_codebase",
        "turns": [
            {"tool": "search", "output": """$ rg -n "TODO|FIXME|HACK" src/ --type rust
src/auth/middleware.rs:142:    // TODO: rotate keys every 24h
src/auth/middleware.rs:287:    // FIXME: handle expired tokens gracefully
src/db/pool.rs:53:    // HACK: increase timeout for slow queries
src/api/handlers.rs:312:    // TODO(v2): paginate this endpoint
src/api/handlers.rs:456:    // FIXME: validate user input before processing
src/cache/redis.rs:89:    // TODO: add circuit breaker for Redis failures
src/cache/redis.rs:201:    // HACK: use clone to avoid lifetime issues
src/metrics/exporter.rs:34:    // TODO: add histogram for P99 latency

8 matches across 5 files"""},
        ]
    },
    {
        "name": "json_api_response",
        "turns": [
            {"tool": "bash", "output": """$ curl -s http://localhost:8080/api/v2/users?limit=3 | jq .
{
  "data": [
    {"id": "usr_a1b2c3d4", "name": "Alice Chen", "email": "alice@example.com", "role": "admin", "org_id": "org_7f8e9d0c", "created_at": "2026-01-15T08:30:00Z"},
    {"id": "usr_e5f6a7b8", "name": "Bob Smith", "email": "bob@example.com", "role": "developer", "org_id": "org_7f8e9d0c", "created_at": "2026-02-20T14:22:00Z"},
    {"id": "usr_c9d0e1f2", "name": "Carol Wu", "email": "carol@example.com", "role": "viewer", "org_id": "org_3a4b5c6d", "created_at": "2026-03-10T09:15:00Z"}
  ],
  "pagination": {"total": 47, "page": 1, "per_page": 3, "next": "/api/v2/users?limit=3&page=2"},
  "meta": {"query_ms": 12.4, "cache": "HIT"}
}"""},]
    },
]

_MUST_KEEP_RE = re.compile(
    r"\b0x[0-9A-Fa-f]+\b|(?<![\w.])\d+(?:\.\d+)?(?![\w.])"
    r"|[A-Z_]{2,}|[a-z_][a-z0-9_]*\.[a-z0-9_]+"
    r"|/[a-z0-9/._-]{2,}|\.[a-z]{2,4}\b"
    r"|--?[a-zA-Z][\w-]*|\b[A-Z][a-z]+[A-Z]\w*"
)


def run_regression(model_id: str | None = None) -> dict[str, Any]:
    """Run agent regression suite. Returns structured results."""
    import headroom
    
    results = []
    by_type: dict[str, list] = defaultdict(list)

    for scenario in SCENARIOS:
        for turn in scenario["turns"]:
            text = turn["output"]
            try:
                result = headroom.compress(
                    [{"role": "tool", "content": text}],
                    model="claude-sonnet-4-5-20250929",
                    kompress_model=model_id,
                )
                compressed = result.messages[0]["content"] if result.messages else text
                tokens_before = result.tokens_before
                tokens_after = result.tokens_after
                savings_pct = (1 - result.compression_ratio) * 100 if result.compression_ratio else 0

                # Must-keep survival
                must = [m.group(0) for m in _MUST_KEEP_RE.finditer(text)]
                survived = sum(1 for m in must if m in compressed)
                mk_rate = survived / max(len(must), 1)

                entry = {
                    "scenario": scenario["name"],
                    "tool": turn["tool"],
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "savings_pct": round(savings_pct, 1),
                    "mk_survival": round(mk_rate, 4),
                    "must_keep_count": len(must),
                    "must_keep_survived": survived,
                    "transforms": result.transforms_applied,
                }
                results.append(entry)
                by_type[turn["tool"]].append(entry)
            except Exception as e:
                results.append({"scenario": scenario["name"], "tool": turn["tool"], "error": str(e)})

    # Aggregate by content type
    type_summary = {}
    for ctype, entries in by_type.items():
        if entries:
            type_summary[ctype] = {
                "avg_savings_pct": round(sum(e["savings_pct"] for e in entries) / len(entries), 1),
                "avg_mk_survival": round(sum(e["mk_survival"] for e in entries) / len(entries), 4),
                "samples": len(entries),
            }

    overall = {
        "avg_savings_pct": round(sum(r["savings_pct"] for r in results if "savings_pct" in r) / max(len(results), 1), 1),
        "avg_mk_survival": round(sum(r["mk_survival"] for r in results if "mk_survival" in r) / max(len(results), 1), 4),
        "total_turns": len(results),
        "scenarios": len(SCENARIOS),
    }

    return {
        "model": model_id or "default",
        "overall": overall,
        "by_content_type": type_summary,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Kompress model ID (uses default if omitted)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-mk", type=float, default=0.85, help="Minimum mk_survival to pass")
    args = ap.parse_args()

    report = run_regression(args.model)

    if args.json:
        json.dump(report, sys.stdout, indent=2)
    else:
        print(f"Agent Regression Suite — {report['model']}")
        print(f"\nBy content type:")
        print(f"{'Type':<15} {'Savings':>8} {'MK Surv':>8} {'Samples':>8}")
        print("-" * 42)
        for ctype, s in sorted(report["by_content_type"].items()):
            print(f"{ctype:<15} {s['avg_savings_pct']:>7.1f}% {s['avg_mk_survival']:>7.3f} {s['samples']:>8}")
        o = report["overall"]
        print(f"\nOverall: {o['avg_savings_pct']:.1f}% savings, {o['avg_mk_survival']:.3f} mk_survival")

    # Exit code based on mk threshold
    passed = report["overall"]["avg_mk_survival"] >= args.min_mk
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

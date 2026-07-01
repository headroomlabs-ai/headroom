#!/usr/bin/env python3
"""
Latency benchmark: Snowflake Cortex  —  Standard vs Headroom

Measures per call (averaged over N runs):
  - TTFT   Time to First Token  (streaming)
  - E2E    End-to-End latency
  - Compress overhead  (headroom local processing time)
  - Prompt token count  (from usage block in final SSE chunk)

Because headroom reduces prompt length, prefill is shorter → lower TTFT.
Multiple runs are averaged to smooth out shared-API latency variance.

Usage:
    SF_CONN=<connection-name> python3 tests/e2e_cortex_latency.py

    # Optional overrides:
    SF_CONN=my_conn SF_HOST=myaccount.snowflakecomputing.com python3 tests/e2e_cortex_latency.py
    SF_CONN=my_conn SF_MODEL=claude-sonnet-4-6 RUNS=5 python3 tests/e2e_cortex_latency.py
"""

from __future__ import annotations

import http.client
import json
import os
import ssl
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Bootstrap headroom ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
_VENV_SITE = REPO_ROOT / ".venv" / "lib"
try:
    from headroom import compress as _hc_check  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO_ROOT))
    for _d in _VENV_SITE.glob("python*/site-packages"):
        sys.path.insert(0, str(_d))

# ── Settings ──────────────────────────────────────────────────────────────────
_SF_HOST = os.environ.get("SF_HOST", "")
_SF_CONN = os.environ.get("SF_CONN", "")
_SF_MODEL = os.environ.get("SF_MODEL", "claude-sonnet-4-6")
_RUNS = int(os.environ.get("RUNS", "3"))
_INPUT_PRICE_PER_1M = 3.00  # USD, claude-sonnet-4-6 on Cortex


# ── Streaming call ────────────────────────────────────────────────────────────


def _stream_call(messages: list[dict], token: str, host: str) -> tuple[float, float, int, int]:
    payload = json.dumps(
        {
            "model": _SF_MODEL,
            "messages": messages,
            "max_completion_tokens": 128,
            "stream": True,
        }
    ).encode()

    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(host, context=ctx, timeout=90)
    conn.request(
        "POST",
        "/api/v2/cortex/v1/chat/completions",
        body=payload,
        headers={
            "Authorization": f'Snowflake Token="{token}"',
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "headroom-latency-bench/1.0",
        },
    )

    t_start = time.perf_counter()
    resp = conn.getresponse()

    if resp.status != 200:
        body = resp.read().decode(errors="replace")
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")

    ttft_ms: float = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    first_token_seen = False

    while True:
        raw = resp.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        if not first_token_seen:
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content", ""):
                ttft_ms = (time.perf_counter() - t_start) * 1000
                first_token_seen = True

        usage = chunk.get("usage") or {}
        if usage.get("prompt_tokens"):
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage.get("completion_tokens", 0)

    e2e_ms = (time.perf_counter() - t_start) * 1000
    conn.close()

    if not first_token_seen:
        ttft_ms = e2e_ms

    return ttft_ms, e2e_ms, prompt_tokens, completion_tokens


# ── Payloads ──────────────────────────────────────────────────────────────────


def _tables_json() -> str:
    rows = [
        {
            "TABLE_CATALOG": "PROD_DB",
            "TABLE_SCHEMA": "ANALYTICS",
            "TABLE_NAME": f"FACT_ORDERS_{i:03d}",
            "TABLE_TYPE": "BASE TABLE",
            "ROW_COUNT": i * 1_423_001,
            "BYTES": i * 8_192_000,
            "CREATED": "2024-01-15",
            "LAST_ALTERED": "2025-06-10",
            "COMMENT": f"Daily order fact partition {i:03d}",
        }
        for i in range(1, 80)
    ]
    return json.dumps(rows, indent=2)


def _dbt_json() -> str:
    return json.dumps(
        {
            "metadata": {"dbt_version": "1.8.0"},
            "results": [
                {
                    "unique_id": f"model.analytics.fct_{i:03d}",
                    "status": "success" if i % 7 != 0 else "error",
                    "execution_time": round(0.8 + i * 0.12, 3),
                    "rows_affected": i * 12_500,
                    "compiled_code": f"SELECT * FROM raw.orders_{i:03d} WHERE status='active'",
                    "failures": None
                    if i % 7 != 0
                    else [{"message": f"Invalid col_{i}", "line": i % 40}],
                    "adapter_response": {
                        "query_id": f"01b{i:06x}",
                        "rows_produced": i * 12_500,
                    },
                }
                for i in range(40)
            ],
        },
        indent=2,
    )


def _search_json() -> str:
    return json.dumps(
        [
            {
                "rank": i + 1,
                "score": round(0.98 - i * 0.02, 4),
                "document_id": f"doc_{i:04d}",
                "source": "PROD_DB.DOCS.ENGINEERING_WIKI",
                "content": (
                    "The revenue pipeline processes 2.3 million orders per day. "
                    "product_family column was renamed to product_group in Q3 2024. "
                    "Migration: update all references in models/marts/revenue/ and "
                    "run dbt run --full-refresh --select fct_revenue."
                ),
                "metadata": {
                    "author": f"eng_{i % 6}@company.com",
                    "updated": "2025-05-20",
                },
            }
            for i in range(15)
        ],
        indent=2,
    )


def _build_messages(ctx: str) -> list[dict]:
    return [
        {"role": "system", "content": ctx},
        {"role": "assistant", "content": "I have reviewed the context above."},
        {
            "role": "user",
            "content": "Based on the data above, what is failing and how do I fix it?",
        },
    ]


# ── Result dataclass ──────────────────────────────────────────────────────────


def _avg(vals: list[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


@dataclass
class LatencyResult:
    label: str
    runs: int
    std_tokens: int
    hdm_tokens: int
    std_ttft_all: list[float] = field(default_factory=list)
    hdm_ttft_all: list[float] = field(default_factory=list)
    std_e2e_all: list[float] = field(default_factory=list)
    hdm_e2e_all: list[float] = field(default_factory=list)
    compress_overhead_ms: float = 0.0

    @property
    def std_ttft_ms(self) -> float:
        return _median(self.std_ttft_all)

    @property
    def hdm_ttft_ms(self) -> float:
        return _median(self.hdm_ttft_all)

    @property
    def std_e2e_ms(self) -> float:
        return _median(self.std_e2e_all)

    @property
    def hdm_e2e_ms(self) -> float:
        return _median(self.hdm_e2e_all)

    @property
    def token_saving_pct(self) -> float:
        return (self.std_tokens - self.hdm_tokens) / max(self.std_tokens, 1) * 100

    @property
    def ttft_saving_pct(self) -> float:
        return (self.std_ttft_ms - self.hdm_ttft_ms) / max(self.std_ttft_ms, 1) * 100

    @property
    def e2e_saving_pct(self) -> float:
        return (self.std_e2e_ms - self.hdm_e2e_ms) / max(self.std_e2e_ms, 1) * 100

    @property
    def net_latency_saving_ms(self) -> float:
        return (self.std_e2e_ms - self.hdm_e2e_ms) - self.compress_overhead_ms

    @property
    def usd_saved_per_call(self) -> float:
        return (self.std_tokens - self.hdm_tokens) / 1_000_000 * _INPUT_PRICE_PER_1M


# ── Benchmark runner (N runs, median) ─────────────────────────────────────────


def run_benchmark(
    label: str,
    messages: list[dict],
    token: str,
    host: str,
    n_runs: int = 3,
) -> LatencyResult:
    from headroom import compress

    print(f"\n   ┌─ {label}  (n={n_runs} runs each)")

    std_ttfts: list[float] = []
    std_e2es: list[float] = []
    std_pt = 0

    for i in range(n_runs):
        print(f"   │  run {i + 1}/{n_runs}  std  ...", end=" ", flush=True)
        ttft, e2e, pt, _ = _stream_call(messages, token, host)
        std_ttfts.append(ttft)
        std_e2es.append(e2e)
        std_pt = pt
        print(f"TTFT={ttft:.0f}ms  E2E={e2e:.0f}ms  tokens={pt:,}")

    print("   │  compressing     ...", end=" ", flush=True)
    t0 = time.perf_counter()
    compressed = compress(messages, model="claude-sonnet-4-5-20250929")
    compress_ms = (time.perf_counter() - t0) * 1000
    print(f"{compress_ms:.0f}ms overhead")

    hdm_ttfts: list[float] = []
    hdm_e2es: list[float] = []
    hdm_pt = 0

    for i in range(n_runs):
        print(f"   │  run {i + 1}/{n_runs}  hdm  ...", end=" ", flush=True)
        ttft, e2e, pt, _ = _stream_call(compressed.messages, token, host)
        hdm_ttfts.append(ttft)
        hdm_e2es.append(e2e)
        hdm_pt = pt
        print(f"TTFT={ttft:.0f}ms  E2E={e2e:.0f}ms  tokens={pt:,}")

    r = LatencyResult(
        label=label,
        runs=n_runs,
        std_tokens=std_pt,
        hdm_tokens=hdm_pt,
        std_ttft_all=std_ttfts,
        hdm_ttft_all=hdm_ttfts,
        std_e2e_all=std_e2es,
        hdm_e2e_all=hdm_e2es,
        compress_overhead_ms=compress_ms,
    )
    print(
        f"   └─ median TTFT: std={r.std_ttft_ms:.0f}ms  hdm={r.hdm_ttft_ms:.0f}ms  "
        f"saving={r.ttft_saving_pct:.1f}%"
    )
    return r


# ── Display ───────────────────────────────────────────────────────────────────


def _bar(pct: float, w: int = 20) -> str:
    n = max(0, int(pct / 100 * w))
    return "█" * n + "░" * (w - n)


def _show(r: LatencyResult) -> None:
    std_ttft_range = f"[{min(r.std_ttft_all):.0f}–{max(r.std_ttft_all):.0f}]"
    hdm_ttft_range = f"[{min(r.hdm_ttft_all):.0f}–{max(r.hdm_ttft_all):.0f}]"
    print(f"\n  ┌─ {r.label}  (median of {r.runs} runs)")
    print(
        f"  │  Tokens  : {r.std_tokens:>7,} → {r.hdm_tokens:>7,}  "
        f"│ saved {r.std_tokens - r.hdm_tokens:>6,}  ({r.token_saving_pct:.1f}%)"
    )
    print(
        f"  │  TTFT    : {r.std_ttft_ms:>7.0f}ms → {r.hdm_ttft_ms:>6.0f}ms  "
        f"│ saved {r.std_ttft_ms - r.hdm_ttft_ms:>6.0f}ms  ({r.ttft_saving_pct:.1f}%)  "
        f"{_bar(r.ttft_saving_pct)}"
    )
    print(f"  │           std range {std_ttft_range}ms  hdm range {hdm_ttft_range}ms")
    print(
        f"  │  E2E     : {r.std_e2e_ms:>7.0f}ms → {r.hdm_e2e_ms:>6.0f}ms  "
        f"│ saved {r.std_e2e_ms - r.hdm_e2e_ms:>6.0f}ms  ({r.e2e_saving_pct:.1f}%)"
    )
    print(
        f"  │  Compress overhead: {r.compress_overhead_ms:.0f}ms  "
        f"│  Net latency saving: {r.net_latency_saving_ms:.0f}ms"
    )
    print(f"  └─ Cost: ${r.usd_saved_per_call:.5f} saved / call")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Cortex Code × Headroom  —  TTFT + Latency Benchmark         ║")
    print("║  Streaming API  │  Time to First Token  │  E2E latency        ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    if not _SF_CONN:
        print("\n  ✗  Set SF_CONN=<connection-name> to run this benchmark.")
        print("     Example: SF_CONN=navnit_local_auth python3 tests/e2e_cortex_latency.py")
        return 1

    import io

    try:
        import snowflake.connector
    except ImportError:
        print("\n  ✗  snowflake-connector-python not installed.")
        return 1

    _s = sys.stdout
    sys.stdout = io.StringIO()
    try:
        conn = snowflake.connector.connect(connection_name=_SF_CONN)
        token = conn.rest.token
        if _SF_HOST:
            host = _SF_HOST
        else:
            cur = conn.cursor()
            cur.execute("SELECT CURRENT_ACCOUNT_LOCATOR()")
            locator = cur.fetchone()[0].lower()
            host = f"{locator}.snowflakecomputing.com"
    finally:
        sys.stdout = _s

    total_calls = len(["full", "tables", "dbt", "search"]) * _RUNS * 2
    print(f"\n  Model : {_SF_MODEL}")
    print(f"  Host  : {host}")
    print(f"  Runs  : {_RUNS} per payload  (median used)  →  {total_calls} total API calls")
    print("  TTFT  : first SSE content chunk via streaming\n")

    full_ctx = json.dumps(
        {
            "tables": json.loads(_tables_json()),
            "dbt_results": json.loads(_dbt_json()),
            "search_results": json.loads(_search_json()),
        },
        indent=2,
    )

    payloads = [
        ("Full context  (tables + dbt + search)", _build_messages(full_ctx)),
        ("INFORMATION_SCHEMA tables  (79 rows)", _build_messages(_tables_json())),
        ("dbt run-results  (40 models)", _build_messages(_dbt_json())),
        ("Cortex Search results  (15 docs)", _build_messages(_search_json())),
    ]

    results: list[LatencyResult] = []
    for label, msgs in payloads:
        try:
            r = run_benchmark(label, msgs, token, host, n_runs=_RUNS)
            results.append(r)
            _show(r)
        except Exception as exc:
            print(f"\n   ✗ {label} failed: {exc}")

    conn.close()

    if not results:
        print("\n  No results collected.")
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print(f"║  SUMMARY  (median of {_RUNS} runs per payload)                    ║")
    print("╠═══════════════════════════════════════════════════════════════╣")
    hdr = f"  {'Payload':<38} {'Tokens':>6}  {'TTFT↓':>7}  {'E2E↓':>7}  {'Net↓':>7}"
    print(hdr)
    print(f"  {'─' * 38} {'─' * 6}  {'─' * 7}  {'─' * 7}  {'─' * 7}")
    for r in results:
        print(
            f"  {r.label[:38]:<38} "
            f"{r.token_saving_pct:>5.0f}%  "
            f"{r.ttft_saving_pct:>6.0f}%  "
            f"{r.e2e_saving_pct:>6.0f}%  "
            f"{r.net_latency_saving_ms:>5.0f}ms"
        )

    avg_token_pct = sum(r.token_saving_pct for r in results) / len(results)
    avg_ttft_pct = sum(r.ttft_saving_pct for r in results) / len(results)
    avg_e2e_pct = sum(r.e2e_saving_pct for r in results) / len(results)
    avg_usd = sum(r.usd_saved_per_call for r in results) / len(results)

    print(f"  {'─' * 38} {'─' * 6}  {'─' * 7}  {'─' * 7}  {'─' * 7}")
    print(
        f"  {'AVERAGE':<38} {avg_token_pct:>5.0f}%  {avg_ttft_pct:>6.0f}%  {avg_e2e_pct:>6.0f}%  "
    )
    print()
    print(f"  Avg USD saved / call : ${avg_usd:.5f}")
    print(f"  At 1k/day            : ${avg_usd * 1_000:.2f}/day  │  ${avg_usd * 365_000:,.0f}/year")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print("  Key insight: TTFT savings track token savings because prefill")
    print("  time scales with prompt length. Fewer tokens = shorter prefill")
    print("  = faster first token. Median across runs removes outlier spikes.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

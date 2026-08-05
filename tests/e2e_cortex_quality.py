#!/usr/bin/env python3
"""
Quality benchmark: Snowflake Cortex  —  Standard vs Headroom

Tests whether headroom compression affects answer quality.
Strategy: embed known facts in payload, ask factual questions,
score both standard and headroom responses against ground truth.

No LLM judge needed — answers are verifiable from the data itself.

Usage:
    SF_CONN=<connection-name> python3 tests/e2e_cortex_quality.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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


# ── API call (non-streaming, full response) ───────────────────────────────────


def _call(messages: list[dict], token: str, host: str) -> str:
    body = json.dumps(
        {
            "model": _SF_MODEL,
            "messages": messages,
            "max_completion_tokens": 256,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"https://{host}/api/v2/cortex/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f'Snowflake Token="{token}"',
            "Content-Type": "application/json",
            "User-Agent": "headroom-quality-bench/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    if "error_code" in resp:
        raise RuntimeError(f"Cortex {resp['error_code']}: {resp.get('message')}")
    return resp["choices"][0]["message"]["content"].strip()


# ── Test case definition ──────────────────────────────────────────────────────


@dataclass
class QualityCase:
    name: str
    context: str
    question: str
    expected_keywords: list[str]
    expected_absent: list[str] = None

    def score(self, answer: str) -> tuple[int, int]:
        """Returns (hits, total) based on keyword presence."""
        answer_lower = answer.lower()
        hits = sum(1 for kw in self.expected_keywords if kw.lower() in answer_lower)
        return hits, len(self.expected_keywords)

    def pass_threshold(self, hits: int, total: int) -> bool:
        return hits / max(total, 1) >= 0.6


# ── Test payload builders ─────────────────────────────────────────────────────


def _make_cases() -> list[QualityCase]:
    # ── Case 1: Exact row lookup from large table JSON ────────────────────────
    tables = [
        {
            "TABLE_NAME": f"FACT_ORDERS_{i:03d}",
            "TABLE_SCHEMA": "ANALYTICS",
            "ROW_COUNT": i * 1_000_000,
            "BYTES": i * 8_192_000,
            "LAST_ALTERED": "2025-06-10",
        }
        for i in range(1, 80)
    ]
    tables_ctx = json.dumps(tables, indent=2)

    # Case 1a: exact numeric lookup
    case1a = QualityCase(
        name="Table row count lookup  (FACT_ORDERS_042)",
        context=tables_ctx,
        question="What is the ROW_COUNT of the table named FACT_ORDERS_042? Reply with just the number.",
        expected_keywords=["42000000", "42,000,000"],
    )

    # Case 1b: filter + list
    case1b = QualityCase(
        name="Tables over 50M rows  (filter query)",
        context=tables_ctx,
        question="List all table names where ROW_COUNT is greater than 50,000,000.",
        expected_keywords=[
            "fact_orders_051",
            "fact_orders_060",
            "fact_orders_070",
            "fact_orders_079",
        ],
    )

    # ── Case 2: dbt failure detection ────────────────────────────────────────
    # Models fail when i % 7 == 0 → indices 0,7,14,21,28,35
    dbt_results = {
        "metadata": {"dbt_version": "1.8.0", "run_id": "run_abc123"},
        "results": [
            {
                "unique_id": f"model.analytics.fct_{i:03d}",
                "status": "success" if i % 7 != 0 else "error",
                "execution_time": round(0.8 + i * 0.12, 3),
                "failures": None
                if i % 7 != 0
                else [{"message": f"Column col_{i} not found", "line": i % 40}],
            }
            for i in range(40)
        ],
    }
    dbt_ctx = json.dumps(dbt_results, indent=2)

    case2a = QualityCase(
        name="dbt failed models  (error detection)",
        context=dbt_ctx,
        question="Which dbt model unique_ids have status 'error'? List all of them.",
        expected_keywords=[f"fct_{i:03d}" for i in range(40) if i % 7 == 0],
    )

    case2b = QualityCase(
        name="dbt slowest model  (max lookup)",
        context=dbt_ctx,
        question="Which model has the longest execution_time? Reply with just the unique_id.",
        expected_keywords=["fct_039"],
    )

    # ── Case 3: Search result ranking ────────────────────────────────────────
    search_results = [
        {
            "rank": i + 1,
            "score": round(0.98 - i * 0.03, 4),
            "document_id": f"doc_{i:04d}",
            "title": f"Engineering runbook #{i:03d}",
            "content": f"This document covers topic_{i} configuration and deployment steps for service_{i}.",
        }
        for i in range(20)
    ]
    search_ctx = json.dumps(search_results, indent=2)

    case3a = QualityCase(
        name="Search top result  (rank 1 lookup)",
        context=search_ctx,
        question="What is the document_id of the result with rank 1? Reply with just the document_id.",
        expected_keywords=["doc_0000"],
    )

    case3b = QualityCase(
        name="Search score lookup  (doc_0007 score)",
        context=search_ctx,
        question="What is the score of document_id doc_0007? Reply with just the number.",
        expected_keywords=["0.77"],
    )

    # ── Case 4: Multi-fact reasoning ─────────────────────────────────────────
    incident = {
        "incident_id": "INC-20250615-004",
        "severity": "P1",
        "affected_service": "payment-processor",
        "root_cause": "Database connection pool exhausted due to slow query on orders_v2 table",
        "timeline": [
            {"time": "14:02", "event": "Alert fired: latency > 5s"},
            {"time": "14:07", "event": "On-call engineer paged"},
            {
                "time": "14:15",
                "event": "Query identified: SELECT * FROM orders_v2 WHERE status='pending'",
            },
            {"time": "14:28", "event": "Index added on (status, created_at)"},
            {"time": "14:31", "event": "Latency normalized"},
        ],
        "mttr_minutes": 29,
        "action_items": [
            "Add query timeout of 10s on payment-processor",
            "Review all full-table scans in orders_v2",
            "Set up connection pool monitoring alert",
        ],
    }
    incident_ctx = json.dumps(incident, indent=2)

    case4a = QualityCase(
        name="Incident MTTR  (exact field lookup)",
        context=incident_ctx,
        question="What was the MTTR in minutes for this incident? Reply with just the number.",
        expected_keywords=["29"],
    )

    case4b = QualityCase(
        name="Incident fix action  (reasoning from timeline)",
        context=incident_ctx,
        question="What specific action resolved the latency issue at 14:28?",
        expected_keywords=["index", "status", "created_at"],
    )

    return [case1a, case1b, case2a, case2b, case3a, case3b, case4a, case4b]


# ── Runner ────────────────────────────────────────────────────────────────────


@dataclass
class QualityResult:
    case: QualityCase
    std_answer: str
    hdm_answer: str
    std_hits: int
    hdm_hits: int
    total_kw: int
    tokens_saved_pct: float
    compress_ms: float

    @property
    def std_pass(self) -> bool:
        return self.case.pass_threshold(self.std_hits, self.total_kw)

    @property
    def hdm_pass(self) -> bool:
        return self.case.pass_threshold(self.hdm_hits, self.total_kw)

    @property
    def quality_delta(self) -> int:
        return self.hdm_hits - self.std_hits


def run_case(case: QualityCase, token: str, host: str) -> QualityResult:
    from headroom import compress

    messages = [
        {"role": "system", "content": case.context},
        {"role": "user", "content": case.question},
    ]

    std_answer = _call(messages, token, host)
    std_hits, total = case.score(std_answer)

    t0 = time.perf_counter()
    compressed = compress(messages, model="claude-sonnet-4-5-20250929")
    compress_ms = (time.perf_counter() - t0) * 1000

    hdm_answer = _call(compressed.messages, token, host)
    hdm_hits, _ = case.score(hdm_answer)

    std_tokens = len(json.dumps(messages)) // 4
    hdm_tokens = len(json.dumps(compressed.messages)) // 4
    saved_pct = (std_tokens - hdm_tokens) / max(std_tokens, 1) * 100

    return QualityResult(
        case=case,
        std_answer=std_answer,
        hdm_answer=hdm_answer,
        std_hits=std_hits,
        hdm_hits=hdm_hits,
        total_kw=total,
        tokens_saved_pct=saved_pct,
        compress_ms=compress_ms,
    )


# ── Display ───────────────────────────────────────────────────────────────────


def _show(r: QualityResult) -> None:
    std_sym = "✓" if r.std_pass else "✗"
    hdm_sym = "✓" if r.hdm_pass else "✗"
    delta_sym = "=" if r.quality_delta == 0 else ("+" if r.quality_delta > 0 else "-")

    print(f"\n  ┌─ {r.case.name}")
    print(f"  │  Token reduction : ~{r.tokens_saved_pct:.0f}%  │  Compress: {r.compress_ms:.0f}ms")
    print(
        f"  │  Standard  [{std_sym}]  : {r.std_hits}/{r.total_kw} keywords matched"
        f"  ({'PASS' if r.std_pass else 'FAIL'})"
    )
    print(
        f"  │  Headroom  [{hdm_sym}]  : {r.hdm_hits}/{r.total_kw} keywords matched"
        f"  ({'PASS' if r.hdm_pass else 'FAIL'})  [{delta_sym} quality delta]"
    )
    print(f"  │  Q: {r.case.question[:80]}")
    std_preview = r.std_answer[:120].replace("\n", " ")
    hdm_preview = r.hdm_answer[:120].replace("\n", " ")
    print(f"  │  Std answer : {std_preview}")
    print(f"  └─ Hdm answer : {hdm_preview}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Cortex Code × Headroom  —  Quality Benchmark                ║")
    print("║  Does compression affect answer accuracy?                     ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    if not _SF_CONN:
        print("\n  ✗  Set SF_CONN=<connection-name> to run.")
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

    cases = _make_cases()
    print(f"\n  Model : {_SF_MODEL}")
    print(f"  Host  : {host}")
    print(f"  Cases : {len(cases)}  ({len(cases) * 2} total API calls)\n")
    print("  Method: embed known facts → ask factual questions → score keyword hits")
    print("  Pass threshold: ≥60% expected keywords found in answer\n")

    results: list[QualityResult] = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case.name}  ...", end=" ", flush=True)
        try:
            r = run_case(case, token, host)
            results.append(r)
            std_s = "✓" if r.std_pass else "✗"
            hdm_s = "✓" if r.hdm_pass else "✗"
            print(f"std={std_s}({r.std_hits}/{r.total_kw})  hdm={hdm_s}({r.hdm_hits}/{r.total_kw})")
            _show(r)
        except Exception as exc:
            print(f"FAILED: {exc}")

    conn.close()

    if not results:
        print("\n  No results.")
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    std_passes = sum(1 for r in results if r.std_pass)
    hdm_passes = sum(1 for r in results if r.hdm_pass)
    total = len(results)
    regressions = sum(1 for r in results if r.std_pass and not r.hdm_pass)
    improvements = sum(1 for r in results if not r.std_pass and r.hdm_pass)
    unchanged = sum(1 for r in results if r.std_pass == r.hdm_pass)
    avg_token_saving = sum(r.tokens_saved_pct for r in results) / total

    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  QUALITY SUMMARY                                              ║")
    print("╠═══════════════════════════════════════════════════════════════╣")
    print(f"  {'Test':<42} {'Std':>4}  {'Hdm':>4}  {'Delta':>6}  {'Tokens↓':>7}")
    print(f"  {'─' * 42} {'─' * 4}  {'─' * 4}  {'─' * 6}  {'─' * 7}")
    for r in results:
        delta = r.hdm_hits - r.std_hits
        delta_str = f"{delta:+d}" if delta != 0 else "  ="
        std_s = "✓" if r.std_pass else "✗"
        hdm_s = "✓" if r.hdm_pass else "✗"
        print(
            f"  {r.case.name[:42]:<42} "
            f"{std_s} {r.std_hits}/{r.total_kw}  "
            f"{hdm_s} {r.hdm_hits}/{r.total_kw}  "
            f"{delta_str:>6}  "
            f"~{r.tokens_saved_pct:.0f}%"
        )
    print(f"  {'─' * 42} {'─' * 4}  {'─' * 4}  {'─' * 6}  {'─' * 7}")
    print(f"  {'TOTAL':<42} {std_passes}/{total}     {hdm_passes}/{total}")
    print()
    print(
        f"  Pass rate  :  Standard {std_passes}/{total} ({std_passes / total * 100:.0f}%)  "
        f"│  Headroom {hdm_passes}/{total} ({hdm_passes / total * 100:.0f}%)"
    )
    print(f"  Regressions (std pass → hdm fail) : {regressions}")
    print(f"  Improvements (std fail → hdm pass): {improvements}")
    print(f"  Unchanged                          : {unchanged}")
    print(f"  Avg token reduction                : ~{avg_token_saving:.0f}%")
    print()
    if regressions == 0:
        print("  ✓  No quality regressions — headroom compression preserved answer accuracy")
    else:
        print(
            f"  ⚠  {regressions} regression(s) — headroom dropped facts needed for correct answer"
        )
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Real end-to-end token-savings test for Cortex Code + Headroom.

Makes ACTUAL REST API calls to Snowflake Cortex (claude-sonnet-4-6) and
measures the REAL token counts from the LLM's usage.prompt_tokens field.

Three test patterns:

  1. System-message context (Snowflake Cortex compatible)
     Large JSON blobs (query results, search results, schema) in the system
     message → headroom's SmartCrusher compresses them.

  2. OpenAI tool-result format  (if OPENAI_API_KEY is set)
     Standard role:"tool" messages compressed via SmartCrusher.

  3. Anthropic messages format  (if ANTHROPIC_API_KEY is set)
     Claude tool_result blocks compressed.

Usage (Snowflake Cortex only — no extra API keys needed):
    SF_CONN=<your-connection-name> python3 tests/e2e_cortex_savings.py

    # SF_HOST is auto-derived from the connection; override if needed:
    SF_CONN=my_conn SF_HOST=myaccount.snowflakecomputing.com python3 tests/e2e_cortex_savings.py

    # Additional backends (optional):
    SF_CONN=my_conn OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... python3 tests/e2e_cortex_savings.py
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

# ── Bootstrap: make headroom importable from the project venv ─────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
_VENV_SITE = REPO_ROOT / ".venv" / "lib"
try:
    from headroom import compress as _hc_check  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO_ROOT))
    for _d in _VENV_SITE.glob("python*/site-packages"):
        sys.path.insert(0, str(_d))

# Snowflake Cortex pricing USD/1M tokens (as of 2025)
_INPUT_PRICE_PER_1M = 3.00

# ── Snowflake connection settings ─────────────────────────────────────────────
# Override via env vars:
#   SF_HOST=<account>.snowflakecomputing.com
#   SF_CONN=<connection-name-from-connections.toml>
#   SF_MODEL=<cortex-model-id>
_SF_HOST = os.environ.get("SF_HOST", "")
_SF_CONN = os.environ.get("SF_CONN", "")
_SF_MODEL = os.environ.get("SF_MODEL", "claude-sonnet-4-6")

# ── Payload builders ──────────────────────────────────────────────────────────


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
                    "adapter_response": {"query_id": f"01b{i:06x}", "rows_produced": i * 12_500},
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
                    "run dbt run --full-refresh --select fct_revenue. "
                    "The rename was tracked in JIRA-4892 and deployed on 2024-09-15."
                ),
                "metadata": {"author": f"eng_{i % 6}@company.com", "updated": "2025-05-20"},
            }
            for i in range(15)
        ],
        indent=2,
    )


# ── Message builders for each API format ─────────────────────────────────────


def build_system_msgs(system_content: str) -> list[dict]:
    """Snowflake Cortex-compatible format (system + user/assistant)."""
    return [
        {"role": "system", "content": system_content},
        {"role": "assistant", "content": "I have reviewed the context above."},
        {
            "role": "user",
            "content": "Based on the data above, what is failing and how do I fix it?",
        },
    ]


def build_tool_msgs(tool_content: str) -> list[dict]:
    """OpenAI tool-result format (for OpenAI / proxy)."""
    return [
        {"role": "user", "content": "Analyze the fct_revenue dbt model failure."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "snowflake_query",
                        "arguments": '{"sql":"SELECT * FROM INFORMATION_SCHEMA.TABLES"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": tool_content},
        {"role": "user", "content": "What is the root cause?"},
    ]


# ── API call helpers ──────────────────────────────────────────────────────────


def _sf_call(messages: list[dict], token: str, host: str) -> dict:
    body = json.dumps(
        {
            "model": _SF_MODEL,
            "messages": messages,
            "max_completion_tokens": 64,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"https://{host}/api/v2/cortex/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f'Snowflake Token="{token}"',
            "Content-Type": "application/json",
            "User-Agent": "headroom-bench/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    if "error_code" in resp:
        raise RuntimeError(f"Cortex {resp['error_code']}: {resp.get('message')}")
    return resp


def _oai_call(messages: list[dict], api_key: str, base_url: str = "https://api.openai.com") -> dict:
    body = json.dumps({"model": "gpt-4o-mini", "messages": messages, "max_tokens": 64}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _ant_call(messages: list[dict], api_key: str) -> dict:
    body = json.dumps(
        {"model": "claude-haiku-4-5", "messages": messages, "max_tokens": 64}
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _tokens(resp: dict, is_anthropic: bool = False) -> tuple[int, int]:
    u = resp.get("usage", {})
    if is_anthropic:
        return u.get("input_tokens", 0), u.get("output_tokens", 0)
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


# ── Benchmark ─────────────────────────────────────────────────────────────────


@dataclass
class R:
    label: str
    before_p: int
    after_p: int
    before_c: int
    after_c: int
    compress_ms: float
    direct_ms: float
    compr_call_ms: float

    @property
    def saved(self) -> int:
        return self.before_p - self.after_p

    @property
    def pct(self) -> float:
        return self.saved / max(self.before_p, 1) * 100

    @property
    def usd_saved(self) -> float:
        return self.saved / 1_000_000 * _INPUT_PRICE_PER_1M


def run(label: str, msgs: list[dict], call_fn, is_anthropic: bool = False) -> R:
    from headroom import compress

    t0 = time.perf_counter()
    direct = call_fn(msgs)
    dm = (time.perf_counter() - t0) * 1000
    bp, bc = _tokens(direct, is_anthropic)

    t0 = time.perf_counter()
    compressed = compress(msgs, model="claude-sonnet-4-5-20250929")
    cm = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    compr_resp = call_fn(compressed.messages)
    com = (time.perf_counter() - t0) * 1000
    ap, ac = _tokens(compr_resp, is_anthropic)

    return R(
        label=label,
        before_p=bp,
        after_p=ap,
        before_c=bc,
        after_c=ac,
        compress_ms=cm,
        direct_ms=dm,
        compr_call_ms=com,
    )


def _bar(pct: float, w: int = 24) -> str:
    n = int(pct / 100 * w)
    return "█" * n + "░" * (w - n)


def _show(r: R) -> None:
    sym = "✓" if r.saved > 0 else "·"
    print(f"\n  {sym}  {r.label}")
    print(
        f"     Prompt tokens : {r.before_p:>7,}  →  {r.after_p:>7,}  "
        f"│  saved {r.saved:>6,}  ({r.pct:.1f}%)"
    )
    print(f"     {_bar(r.pct)}  ${r.usd_saved:.5f} saved / call")
    print(
        f"     Timing : direct {r.direct_ms:.0f}ms  │  "
        f"compress {r.compress_ms:.0f}ms + compressed-call {r.compr_call_ms:.0f}ms"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Cortex Code × Headroom  —  Real REST API savings      ║")
    print("║   usage.prompt_tokens measured directly from the LLM    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results: list[R] = []

    # ── 1. Snowflake Cortex (system-message pattern) ──────────────────────────
    print("\n▶  Snowflake Cortex  /api/v2/cortex/v1/chat/completions")
    try:
        import io

        import snowflake.connector  # noqa: F401

        if not _SF_CONN:
            raise RuntimeError(
                "Set SF_CONN=<your-connection-name> (from ~/.snowflake/connections.toml)"
            )
        _s = sys.stdout
        sys.stdout = io.StringIO()
        try:
            _conn = snowflake.connector.connect(connection_name=_SF_CONN)
            _tok = _conn.rest.token
            # Derive host: prefer SF_HOST env var, then try account locator
            # (conn.host may be the org-format name which can fail SSL validation)
            if _SF_HOST:
                sf_host = _SF_HOST
            else:
                cs = _conn.cursor()
                cs.execute("SELECT CURRENT_ACCOUNT_LOCATOR()")
                locator = cs.fetchone()[0].lower()
                sf_host = f"{locator}.snowflakecomputing.com"
        finally:
            sys.stdout = _s

        print(f"   Model: {_SF_MODEL}  │  Host: {sf_host}")

        def sf_call(m: list[dict]) -> dict:
            return _sf_call(m, _tok, sf_host)

        # Combined context: tables + dbt + search results in system message
        full_ctx = json.dumps(
            {
                "tables": json.loads(_tables_json()),
                "dbt_results": json.loads(_dbt_json()),
                "search_results": json.loads(_search_json()),
            },
            indent=2,
        )

        payloads = [
            ("Cortex  — full context  (tables + dbt + search)", build_system_msgs(full_ctx)),
            ("Cortex  — INFORMATION_SCHEMA tables  (79 rows)", build_system_msgs(_tables_json())),
            ("Cortex  — dbt run-results  (40 models)", build_system_msgs(_dbt_json())),
            ("Cortex  — Cortex Search results  (15 docs)", build_system_msgs(_search_json())),
        ]

        for label, msgs in payloads:
            approx = len(json.dumps(msgs)) // 4
            print(f"\n   {label}")
            print(f"   Payload: ~{approx:,} tokens  ...", end=" ", flush=True)
            r = run(label, msgs, sf_call)
            results.append(r)
            print(f"saved {r.saved:,} tokens  ({r.pct:.0f}%)")
            _show(r)

        _conn.close()

    except Exception as e:
        print(f"\n   ✗ Snowflake Cortex skipped: {e}")

    # ── 2. OpenAI (tool-result format) ───────────────────────────────────────
    oai_key = os.environ.get("OPENAI_API_KEY", "")
    if oai_key:
        print("\n\n▶  OpenAI  /v1/chat/completions  (gpt-4o-mini)")
        for label, content in [
            ("OpenAI  — tables JSON  (79 rows)", _tables_json()),
            ("OpenAI  — Cortex Search  (15 docs)", _search_json()),
        ]:
            msgs = build_tool_msgs(content)
            approx = len(json.dumps(msgs)) // 4
            print(f"\n   {label}  (~{approx:,} tokens)  ...", end=" ", flush=True)

            def _oai(m: list[dict]) -> dict:
                return _oai_call(m, oai_key)

            r = run(label, msgs, _oai)
            results.append(r)
            print(f"saved {r.saved:,}  ({r.pct:.0f}%)")
            _show(r)
    else:
        print("\n▶  OpenAI  — skipped  (export OPENAI_API_KEY to enable)")

    # ── 3. Anthropic ─────────────────────────────────────────────────────────
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if ant_key:
        print("\n\n▶  Anthropic  /v1/messages  (claude-haiku-4-5)")
        for label, content in [
            ("Anthropic  — tables JSON  (79 rows)", _tables_json()),
            ("Anthropic  — Cortex Search  (15 docs)", _search_json()),
        ]:
            msgs = build_tool_msgs(content)
            approx = len(json.dumps(msgs)) // 4
            print(f"\n   {label}  (~{approx:,} tokens)  ...", end=" ", flush=True)

            def _ant(m: list[dict]) -> dict:
                return _ant_call(m, ant_key)

            r = run(label, msgs, _ant, is_anthropic=True)
            results.append(r)
            print(f"saved {r.saved:,}  ({r.pct:.0f}%)")
            _show(r)
    else:
        print("\n▶  Anthropic  — skipped  (export ANTHROPIC_API_KEY to enable)")

    # ── Summary ───────────────────────────────────────────────────────────────
    if not results:
        print("\n  No results. Is snowflake-connector-python installed?")
        return 1

    tb = sum(r.before_p for r in results)
    ta = sum(r.after_p for r in results)
    ts = tb - ta
    tp = ts / max(tb, 1) * 100
    tu = sum(r.usd_saved for r in results)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SUMMARY  —  real usage.prompt_tokens from LLM          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"  {'Payload':<40} {'Before':>7}  {'After':>7}  {'Saved':>5}")
    print(f"  {'─' * 40} {'─' * 7}  {'─' * 7}  {'─' * 5}")
    for r in results:
        m = "✓" if r.saved > 0 else "·"
        print(f"  {m} {r.label[:39]:<39} {r.before_p:>7,}  {r.after_p:>7,}  {r.pct:>4.0f}%")
    print(f"  {'─' * 40} {'─' * 7}  {'─' * 7}  {'─' * 5}")
    print(f"  {'TOTAL':<40} {tb:>7,}  {ta:>7,}  {tp:>4.0f}%")
    print()
    avg_saved_per_call = ts / max(len(results), 1)
    avg_usd_per_call = tu / max(len(results), 1)
    print(f"  Tokens saved  :  {ts:>8,}  prompt tokens  ({len(results)} calls)")
    print(f"  Avg per call  :  {avg_saved_per_call:>8,.0f}  tokens  /  ${avg_usd_per_call:.5f}")
    print(
        f"  At 1k/day     :  ${avg_usd_per_call * 1_000:.2f}/day  │  ${avg_usd_per_call * 365_000:,.0f}/year"
    )
    print("╚══════════════════════════════════════════════════════════╝")
    return 0


if __name__ == "__main__":
    sys.exit(main())

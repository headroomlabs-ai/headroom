#!/usr/bin/env python3
"""
Proxy-in-the-loop integration test: Cortex Code + Headroom Proxy

Tests the FULL path:
  Cortex Code (simulated) → headroom FastAPI proxy → Snowflake Cortex

Key insight: headroom compresses CONVERSATION HISTORY.
  Turn 1: nothing to compress yet — baseline
  Turn 2: proxy compresses turn 1 history before sending
  Turn 3: proxy compresses turns 1+2 history
  → token count should DROP on turns 2+ vs a direct client

Usage:
    SF_CONN=<connection-name> python3 tests/e2e_cortex_proxy.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_VENV_SITE = REPO_ROOT / ".venv" / "lib"
try:
    from headroom import compress as _hc_check  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO_ROOT))
    for _d in _VENV_SITE.glob("python*/site-packages"):
        sys.path.insert(0, str(_d))

_SF_CONN = os.environ.get("SF_CONN", "")
_SF_HOST = os.environ.get("SF_HOST", "")
_SF_MODEL = os.environ.get("SF_MODEL", "claude-sonnet-4-6")
_PROXY_PORT = int(os.environ.get("PROXY_PORT", "8797"))
_TURNS = int(os.environ.get("TURNS", "4"))


# ── Auth ──────────────────────────────────────────────────────────────────────


def _get_sf_token_and_host():
    """Returns (token, host, conn) — caller must keep conn open."""
    import io

    import snowflake.connector

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
    return token, host, conn


# ── API calls ─────────────────────────────────────────────────────────────────


def _call(url: str, messages: list[dict], token: str) -> dict:
    token_field = "max_completion_tokens"
    body = json.dumps(
        {"model": _SF_MODEL, "messages": messages, token_field: 200, "stream": False}
    ).encode()
    auth_header = f'Snowflake Token="{token}"'
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "User-Agent": "headroom-proxy-test/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def _tokens(resp: dict) -> tuple[int, int]:
    u = resp.get("usage", {})
    pt = u.get("prompt_tokens") or u.get("input_tokens", 0)
    ct = u.get("completion_tokens") or u.get("output_tokens", 0)
    return pt, ct


def _content(resp: dict) -> str:
    return resp.get("choices", [{}])[0].get("message", {}).get("content", "")


# ── Proxy lifecycle ───────────────────────────────────────────────────────────


def _wait_for_proxy(port: int, timeout: int = 40) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ── Conversation turns ────────────────────────────────────────────────────────

# Each turn adds a large JSON tool-result style blob as context
# so SmartCrusher has something to compress from turn 2 onwards.
_TURN_QUESTIONS = [
    "Here is our dbt run output: {ctx}\n\nWhich models failed?",
    "Now here are the raw table stats: {ctx}\n\nWhich table has the most rows?",
    "Here are the Cortex Search results: {ctx}\n\nWhat is the top ranked document?",
    "Given everything above, what should I fix first and why?",
]


def _dbt_ctx() -> str:
    return json.dumps(
        [
            {
                "unique_id": f"model.analytics.fct_{i:03d}",
                "status": "error" if i % 7 == 0 else "success",
                "execution_time": round(0.8 + i * 0.12, 3),
                "failures": [{"message": f"col_{i} not found"}] if i % 7 == 0 else None,
            }
            for i in range(40)
        ],
        indent=2,
    )


def _tables_ctx() -> str:
    return json.dumps(
        [
            {
                "TABLE_NAME": f"FACT_ORDERS_{i:03d}",
                "ROW_COUNT": i * 1_423_001,
                "BYTES": i * 8_192_000,
                "STATUS": "active" if i % 3 != 0 else "archived",
            }
            for i in range(1, 60)
        ],
        indent=2,
    )


def _search_ctx() -> str:
    return json.dumps(
        [
            {
                "rank": i + 1,
                "score": round(0.98 - i * 0.03, 4),
                "document_id": f"doc_{i:04d}",
                "content": f"Engineering runbook #{i:03d}: covers deployment and config for service_{i}.",
            }
            for i in range(20)
        ],
        indent=2,
    )


_CONTEXTS = [_dbt_ctx(), _tables_ctx(), _search_ctx(), ""]


@dataclass
class TurnResult:
    turn: int
    direct_pt: int
    proxy_pt: int

    @property
    def saved(self) -> int:
        return self.direct_pt - self.proxy_pt

    @property
    def pct(self) -> float:
        return self.saved / max(self.direct_pt, 1) * 100


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Cortex Code × Headroom  —  Proxy-in-the-Loop  (Multi-Turn)  ║")
    print("║  Agent → FastAPI proxy → Snowflake Cortex                    ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print("  Insight: compression = 0% on turn 1 (no history yet)")
    print("           compression grows each subsequent turn as history accumulates")

    if not _SF_CONN:
        print("\n  ✗  Set SF_CONN=<connection-name>")
        return 1

    print("\n  [1/4] Authenticating ...", end=" ", flush=True)
    try:
        token, host, _sf_conn = _get_sf_token_and_host()
        print(f"OK  ({host})")
    except Exception as e:
        print(f"FAILED: {e}")
        return 1

    cortex_direct_url = f"https://{host}/api/v2/cortex/v1/chat/completions"
    cortex_base = f"https://{host}/api/v2/cortex"
    proxy_url = f"http://127.0.0.1:{_PROXY_PORT}/v1/chat/completions"

    print(f"  [2/4] Starting headroom proxy on :{_PROXY_PORT} ...", end=" ", flush=True)
    proxy_env = os.environ.copy()
    proxy_log = open("/tmp/headroom_proxy.log", "w")
    proxy_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "headroom.proxy.server",
            "--port",
            str(_PROXY_PORT),
            "--openai-api-url",
            cortex_base,
        ],
        env=proxy_env,
        cwd=str(REPO_ROOT),
        stdout=proxy_log,
        stderr=proxy_log,
    )
    if not _wait_for_proxy(_PROXY_PORT):
        proxy_proc.send_signal(signal.SIGTERM)
        proxy_proc.wait(timeout=5)
        proxy_log.close()
        print("FAILED")
        return 1
    print("OK")

    turns: list[TurnResult] = []
    direct_history: list[dict] = []
    proxy_history: list[dict] = []

    print(f"\n  [3/4] Running {_TURNS} conversation turns ...\n")
    print(f"  {'Turn':<6} {'Direct prompt':>14}  {'Proxy prompt':>13}  {'Saved':>8}  {'Note'}")
    print(f"  {'─' * 6} {'─' * 14}  {'─' * 13}  {'─' * 8}  {'─' * 30}")

    try:
        for t in range(1, _TURNS + 1):
            ctx = _CONTEXTS[min(t - 1, len(_CONTEXTS) - 1)]
            question = _TURN_QUESTIONS[min(t - 1, len(_TURN_QUESTIONS) - 1)].format(ctx=ctx)

            # ── Direct: accumulate full uncompressed history ────────────────
            direct_history.append({"role": "user", "content": question})
            try:
                dr = _call(cortex_direct_url, direct_history, token)
                d_pt, _ = _tokens(dr)
                d_answer = _content(dr)
                direct_history.append({"role": "assistant", "content": d_answer})
            except urllib.error.HTTPError as e:
                print(f"  Direct turn {t} FAILED: HTTP {e.code} {e.read().decode()[:100]}")
                break
            except Exception as e:
                print(f"  Direct turn {t} FAILED: {e}")
                break

            # ── Proxy: send history through headroom proxy ──────────────────
            proxy_history.append({"role": "user", "content": question})
            try:
                pr = _call(proxy_url, proxy_history, token)
                p_pt, _ = _tokens(pr)
                p_answer = _content(pr)
                proxy_history.append({"role": "assistant", "content": p_answer})
            except urllib.error.HTTPError as e:
                print(f"  Proxy  turn {t} FAILED: HTTP {e.code} {e.read().decode()[:100]}")
                break
            except Exception as e:
                print(f"  Proxy  turn {t} FAILED: {e}")
                break

            result = TurnResult(turn=t, direct_pt=d_pt, proxy_pt=p_pt)
            turns.append(result)

            note = "← baseline (no history yet)" if t == 1 else f"← {result.pct:.0f}% saved"
            sym = "✓" if result.saved > 0 else ("·" if t == 1 else "⚠")
            print(f"  {sym} T{t:<4} {d_pt:>14,}  {p_pt:>13,}  {result.saved:>+8,}  {note}")

    finally:
        proxy_proc.send_signal(signal.SIGTERM)
        proxy_proc.wait(timeout=5)
        proxy_log.close()
        _sf_conn.close()

    if not turns:
        print("\n  No results collected.")
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    later_turns = [r for r in turns if r.turn > 1]
    avg_saving = sum(r.pct for r in later_turns) / max(len(later_turns), 1)
    total_direct = sum(r.direct_pt for r in turns)
    total_proxy = sum(r.proxy_pt for r in turns)
    total_saved = total_direct - total_proxy

    print()
    print("  [4/4] Summary")
    print(f"  {'─' * 60}")
    print(f"  Total direct tokens  : {total_direct:,}")
    print(f"  Total proxy tokens   : {total_proxy:,}  (saved {total_saved:,})")
    print(f"  Avg compression T2+  : {avg_saving:.1f}%")
    print()

    if avg_saving > 5:
        print("  ✓  PROXY COMPRESSION CONFIRMED")
        print("     headroom proxy transparently compresses conversation history")
        print(f"     Average {avg_saving:.0f}% token reduction from turn 2 onwards")
    else:
        print("  ⚠  Low compression — proxy routed correctly but history")
        print("     may be below SmartCrusher threshold. Try longer conversations.")

    return 0 if len(turns) == _TURNS else 1


if __name__ == "__main__":
    sys.exit(main())

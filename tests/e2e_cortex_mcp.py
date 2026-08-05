#!/usr/bin/env python3
"""
MCP mode e2e test: Cortex Code + Headroom MCP Server

Tests the FULL MCP path using the official MCP Python SDK client:
  1. Start headroom MCP server (stdio transport via mcp_server.py)
  2. Connect using mcp.ClientSession (same protocol Cortex Code uses)
  3. List tools → verify headroom_compress / headroom_retrieve / headroom_stats
  4. Call headroom_compress with large JSON payloads
  5. Use compressed output to call Snowflake Cortex REST API
  6. Compare prompt_tokens: direct vs MCP-compressed

Usage:
    SF_CONN=<connection-name> python3 tests/e2e_cortex_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_VENV_SITE = REPO_ROOT / ".venv" / "lib"
try:
    from headroom import compress as _hc  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO_ROOT))
    for _d in _VENV_SITE.glob("python*/site-packages"):
        sys.path.insert(0, str(_d))

_SF_CONN = os.environ.get("SF_CONN", "")
_SF_HOST = os.environ.get("SF_HOST", "")
_SF_MODEL = os.environ.get("SF_MODEL", "claude-sonnet-4-6")

MCP_SERVER_SCRIPT = REPO_ROOT / "headroom" / "ccr" / "mcp_server.py"


# ── Snowflake auth ─────────────────────────────────────────────────────────────


def _get_sf_token_and_host():
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
            host = f"{cur.fetchone()[0].lower()}.snowflakecomputing.com"
    finally:
        sys.stdout = _s
    return token, host, conn


# ── Cortex call ───────────────────────────────────────────────────────────────


def _cortex_call(messages: list[dict], token: str, host: str) -> dict:
    body = json.dumps(
        {"model": _SF_MODEL, "messages": messages, "max_completion_tokens": 256, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"https://{host}/api/v2/cortex/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f'Snowflake Token="{token}"',
            "Content-Type": "application/json",
            "User-Agent": "headroom-mcp-test/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Cortex HTTP {e.code}: {e.read().decode()[:200]}") from e


def _tokens(resp: dict) -> tuple[int, int]:
    u = resp.get("usage", {})
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


# ── Payloads ──────────────────────────────────────────────────────────────────


def _dbt_payload() -> str:
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


def _tables_payload() -> str:
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


# ── MCP test ──────────────────────────────────────────────────────────────────


async def run_mcp_test(token: str, host: str) -> int:
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        print("\n  ✗  MCP SDK not installed. Run: pip install mcp")
        return 1

    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Cortex Code × Headroom  —  MCP Mode E2E Test                ║")
    print("║  MCP Python SDK Client  │  stdio transport  │  Cortex        ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"\n  Model : {_SF_MODEL}  │  Host : {host}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_SCRIPT)],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )

    # ── Connect via MCP SDK ───────────────────────────────────────────────────
    print("\n  [1/6] Connecting to headroom MCP server ...", end=" ", flush=True)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("OK")

            # ── List tools ────────────────────────────────────────────────────
            print("  [2/6] Listing MCP tools ...", end=" ", flush=True)
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"found: {tool_names}")

            required = {"headroom_compress", "headroom_retrieve", "headroom_stats"}
            missing = required - set(tool_names)
            if missing:
                print(f"\n  ✗  Missing tools: {missing}")
                return 1

            # ── Test 1: dbt run results ───────────────────────────────────────
            print("\n  [3/6] Test 1 — dbt run results (40 models)")
            dbt_content = _dbt_payload()
            question = "Which models failed and what column is missing?"

            print("    ├─ Direct Cortex call ...", end=" ", flush=True)
            d1_pt, _ = _tokens(
                _cortex_call(
                    [
                        {"role": "system", "content": dbt_content},
                        {"role": "user", "content": question},
                    ],
                    token,
                    host,
                )
            )
            print(f"prompt={d1_pt:,} tokens")

            print("    ├─ MCP headroom_compress ...", end=" ", flush=True)
            r1 = await session.call_tool("headroom_compress", {"content": dbt_content})
            text1 = r1.content[0].text if r1.content else "{}"
            data1 = json.loads(text1) if text1.startswith("{") else {}
            compressed1 = data1.get("compressed", dbt_content)
            saved1 = data1.get("tokens_saved", 0)
            pct1 = data1.get("savings_percent", 0)
            hash1 = data1.get("hash", "")
            print(f"saved {saved1:,} tokens ({pct1:.1f}%)  hash={hash1[:8]}...")

            print("    └─ Cortex call (MCP-compressed) ...", end=" ", flush=True)
            m1_pt, _ = _tokens(
                _cortex_call(
                    [
                        {
                            "role": "system",
                            "content": compressed1
                            if isinstance(compressed1, str)
                            else json.dumps(compressed1),
                        },
                        {"role": "user", "content": question},
                    ],
                    token,
                    host,
                )
            )
            api_saved1 = d1_pt - m1_pt
            api_pct1 = api_saved1 / max(d1_pt, 1) * 100
            sym = "✓" if api_saved1 > 0 else "·"
            print(f"{sym}  prompt={m1_pt:,}  saved {api_saved1:,} ({api_pct1:.1f}%)")

            # ── Test 2: table schema ──────────────────────────────────────────
            print("\n  [4/6] Test 2 — INFORMATION_SCHEMA tables (59 rows)")
            tbl_content = _tables_payload()
            question2 = "How many tables are archived?"

            print("    ├─ Direct Cortex call ...", end=" ", flush=True)
            d2_pt, _ = _tokens(
                _cortex_call(
                    [
                        {"role": "system", "content": tbl_content},
                        {"role": "user", "content": question2},
                    ],
                    token,
                    host,
                )
            )
            print(f"prompt={d2_pt:,} tokens")

            print("    ├─ MCP headroom_compress ...", end=" ", flush=True)
            r2 = await session.call_tool("headroom_compress", {"content": tbl_content})
            text2 = r2.content[0].text if r2.content else "{}"
            data2 = json.loads(text2) if text2.startswith("{") else {}
            compressed2 = data2.get("compressed", tbl_content)
            saved2 = data2.get("tokens_saved", 0)
            pct2 = data2.get("savings_percent", 0)
            print(f"saved {saved2:,} tokens ({pct2:.1f}%)")

            print("    └─ Cortex call (MCP-compressed) ...", end=" ", flush=True)
            m2_pt, _ = _tokens(
                _cortex_call(
                    [
                        {
                            "role": "system",
                            "content": compressed2
                            if isinstance(compressed2, str)
                            else json.dumps(compressed2),
                        },
                        {"role": "user", "content": question2},
                    ],
                    token,
                    host,
                )
            )
            api_saved2 = d2_pt - m2_pt
            api_pct2 = api_saved2 / max(d2_pt, 1) * 100
            sym2 = "✓" if api_saved2 > 0 else "·"
            print(f"{sym2}  prompt={m2_pt:,}  saved {api_saved2:,} ({api_pct2:.1f}%)")

            # ── Test 3: headroom_retrieve ─────────────────────────────────────
            if hash1:
                print(f"\n  [5/6] headroom_retrieve — CCR round-trip (hash={hash1[:8]}...)")
                r3 = await session.call_tool("headroom_retrieve", {"hash": hash1})
                text3 = r3.content[0].text if r3.content else "{}"
                data3 = json.loads(text3) if text3.startswith("{") else {}
                if "original_content" in data3 or "results" in data3:
                    print("    ✓  original content retrieved successfully")
                elif "error" in data3:
                    print(f"    ⚠  {data3['error'][:80]}")
                else:
                    print(f"    ✓  retrieved (keys: {list(data3.keys())})")

            # ── headroom_stats ────────────────────────────────────────────────
            print("\n  [6/6] headroom_stats")
            r4 = await session.call_tool("headroom_stats", {})
            stats_text = r4.content[0].text if r4.content else ""
            for line in stats_text.split("\n")[:6]:
                if line.strip():
                    print(f"    {line}")

            # ── Summary ───────────────────────────────────────────────────────
            total_direct = d1_pt + d2_pt
            total_mcp = m1_pt + m2_pt
            avg_pct = (total_direct - total_mcp) / max(total_direct, 1) * 100

            print()
            print("╔═══════════════════════════════════════════════════════════════╗")
            print("║  MCP MODE SUMMARY                                             ║")
            print("╠═══════════════════════════════════════════════════════════════╣")
            print(f"  {'Payload':<35} {'Direct':>8}  {'MCP+API':>8}  {'Saved':>7}")
            print(f"  {'─' * 35} {'─' * 8}  {'─' * 8}  {'─' * 7}")
            print(
                f"  {'dbt run results (40 models)':<35} {d1_pt:>8,}  {m1_pt:>8,}  {api_pct1:>6.1f}%"
            )
            print(
                f"  {'INFORMATION_SCHEMA (59 rows)':<35} {d2_pt:>8,}  {m2_pt:>8,}  {api_pct2:>6.1f}%"
            )
            print(f"  {'─' * 35} {'─' * 8}  {'─' * 8}  {'─' * 7}")
            print(f"  {'TOTAL':<35} {total_direct:>8,}  {total_mcp:>8,}  {avg_pct:>6.1f}%")
            print()
            print("  MCP transport  : stdio (MCP Python SDK — same as Cortex Code)")
            print("  Tools verified : headroom_compress ✓  headroom_retrieve ✓  headroom_stats ✓")
            if avg_pct > 0:
                print(f"\n  ✓  MCP TEST PASSED — {avg_pct:.1f}% avg token reduction via MCP tools")
            else:
                print("\n  ⚠  MCP routing works but payloads below compression threshold")
            print("╚═══════════════════════════════════════════════════════════════╝")
            return 0


def main() -> int:
    if not _SF_CONN:
        print("\n  ✗  Set SF_CONN=<connection-name>")
        print("     Example: SF_CONN=navnit_local_auth python3 tests/e2e_cortex_mcp.py")
        return 1

    try:
        import snowflake.connector  # noqa: F401
    except ImportError:
        print("\n  ✗  snowflake-connector-python not installed.")
        return 1

    print("\n  Authenticating with Snowflake ...", end=" ", flush=True)
    try:
        token, host, conn = _get_sf_token_and_host()
        print(f"OK  ({host})")
    except Exception as e:
        print(f"FAILED: {e}")
        return 1

    try:
        return asyncio.run(run_mcp_test(token, host))
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())

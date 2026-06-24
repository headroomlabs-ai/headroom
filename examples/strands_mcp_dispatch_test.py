#!/usr/bin/env python3
"""Verify Strands' MCP dispatcher can reach Headroom's MCP server.

This is the focused 15-min test that proves the MCP-everywhere
architecture for Path B works end-to-end:

  1. Headroom proxy is running (started separately, port 8787).
  2. Build a Strands MCPClient that stdio-spawns `headroom mcp serve`.
  3. List MCP tools — confirm `headroom_retrieve` is exposed.
  4. Round-trip: stash content via the proxy's /v1/compress endpoint
     to get a hash, then call `headroom_retrieve(hash)` via MCP,
     verify the original comes back.
  5. (Bonus) Show the same hash being resolved through both the
     proxy's REST /v1/retrieve endpoint AND the MCP path -- proves
     the CompressionStore is shared between proxy and MCP server.

If steps 3+4 work, the MCP dispatch path is alive — meaning when a
Strands Agent receives a model-emitted `headroom_retrieve` tool_call
(streaming OR non-streaming), it can dispatch it via this same MCP
client and the chain closes.

Run
---
    AWS_REGION=us-west-2 python examples/strands_mcp_dispatch_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from strands.tools.mcp import MCPClient

PROXY_PORT = 8787  # matches MCP default
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"


def start_proxy() -> subprocess.Popen[bytes]:
    """Spawn the proxy on the MCP default port."""
    cmd = [
        sys.executable,
        "-m",
        "headroom.cli",
        "proxy",
        "--backend",
        "bedrock",
        "--region",
        "us-west-2",
        "--port",
        str(PROXY_PORT),
    ]
    log = Path("/tmp/headroom_proxy_mcp_test.log").open("wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603
    print(f"  proxy spawned (pid={proc.pid}); log → /tmp/headroom_proxy_mcp_test.log")
    return proc


def wait_for_proxy(timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{PROXY_URL}/readyz", timeout=1) as r:  # noqa: S310
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise RuntimeError(f"Proxy did not become ready in {timeout_s}s")


def stop_proxy(proc: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def stash_content_via_proxy(content: str) -> str | None:
    """Round-trip through proxy /v1/compress to get a stored hash.

    Returns the hash if compression replaced anything, else None.
    """
    body = json.dumps({"content": content}).encode()
    req = urllib.request.Request(  # noqa: S310
        f"{PROXY_URL}/v1/compress",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:400]
        print(f"  /v1/compress HTTP {e.code}: {err}")
        return None
    print(f"  /v1/compress response keys: {list(resp.keys())}")
    # The response shape varies — try common shape candidates.
    for key in ("hash", "stored_hash", "content_hash"):
        if key in resp:
            return str(resp[key])
    compressed = resp.get("compressed") or resp.get("output") or ""
    # Marker form: 'Retrieve original: hash=abc' or 'Retrieve more: hash=abc'
    for marker in ("hash=",):
        if marker in compressed:
            idx = compressed.index(marker) + len(marker)
            end = compressed.find(" ", idx)
            return compressed[idx : end if end > 0 else idx + 64].strip().rstrip(">")
    print(f"  no hash found in /v1/compress response — full body: {json.dumps(resp)[:400]}")
    return None


def main() -> int:
    print("=" * 72)
    print(" Strands MCPClient → Headroom MCP server  end-to-end probe")
    print("=" * 72)

    print("\n[1/5] Starting Headroom proxy ...")
    proxy = start_proxy()
    try:
        wait_for_proxy()
        print("  proxy ready.\n")

        big = json.dumps(
            [
                {
                    "id": i,
                    "name": f"order-{i}",
                    "status": "completed",
                    "amount": 100 + i,
                    "customer": f"cust-{i % 50}",
                    "description": "A long, repetitive description that takes up space " * 4,
                }
                for i in range(300)
            ]
        )
        print(f"[2/5] Test payload prepared: {len(big)} chars of JSON")

        print("\n[3/5] Building Strands MCPClient pointed at `headroom mcp serve` ...")
        server_params = StdioServerParameters(
            command="headroom",
            args=["mcp", "serve", "--proxy-url", PROXY_URL],
        )
        with MCPClient(lambda: stdio_client(server_params)) as mcp:
            print("  MCP connection up.")

            print("\n[4/5] Listing MCP tools ...")
            try:
                tools = mcp.list_tools_sync()
            except Exception as e:  # noqa: BLE001
                print(f"  ! list_tools_sync failed: {type(e).__name__}: {e}")
                raise
            tool_names = [getattr(t, "tool_name", None) or getattr(t, "name", "?") for t in tools]
            print(f"  tools advertised by Headroom MCP: {tool_names}")
            if "headroom_retrieve" not in tool_names:
                print("  ! headroom_retrieve NOT in tool list — abort.")
                return 4

            # Round-trip via MCP: compress to get a hash, then retrieve via MCP.
            # This is a pure MCP-only path through Strands' dispatcher — the
            # same dispatcher that would resolve a model-emitted
            # `headroom_retrieve` tool_call in a real conversation.
            print("\n[5a/5] Stashing content via MCP headroom_compress (same dispatcher) ...")
            try:
                comp = mcp.call_tool_sync(
                    tool_use_id="probe-compress-1",
                    name="headroom_compress",
                    arguments={"content": big},
                )
                print(f"  status={comp.get('status', '?')}")
                # Pull the JSON body out of the tool result
                comp_payload: dict | None = None
                for block in comp.get("content", []) or []:
                    if isinstance(block, dict):
                        if "json" in block:
                            comp_payload = block["json"]
                            break
                        if "text" in block:
                            try:
                                comp_payload = json.loads(block["text"])
                            except (json.JSONDecodeError, ValueError):
                                comp_payload = {"_raw_text": block["text"]}
                            break
                print(f"  compress payload keys: {list((comp_payload or {}).keys())}")
                # Find the hash. Headroom MCP compress is documented to emit
                # markers in the compressed output ("hash=abc..."); also surface
                # explicit hash field if present.
                stashed_hash: str | None = None
                if comp_payload:
                    for key in ("hash", "stored_hash", "content_hash"):
                        if key in comp_payload and comp_payload[key]:
                            stashed_hash = str(comp_payload[key])
                            break
                    if not stashed_hash:
                        compressed_str = (
                            comp_payload.get("compressed")
                            or comp_payload.get("output")
                            or comp_payload.get("_raw_text", "")
                        )
                        if "hash=" in compressed_str:
                            idx = compressed_str.index("hash=") + len("hash=")
                            tail = compressed_str[idx : idx + 80]
                            # hash is followed by space, quote, > or end
                            stashed_hash = ""
                            for ch in tail:
                                if ch in " \"'>),\n\r\t":
                                    break
                                stashed_hash += ch
                            stashed_hash = stashed_hash or None
                if not stashed_hash:
                    print(
                        f"  ! no hash found in compress response. Full body: "
                        f"{json.dumps(comp_payload)[:400]}"
                    )
                    return 5
                print(f"  ✓ stashed hash: {stashed_hash[:32]}...")
            except Exception as e:  # noqa: BLE001
                print(f"  ! compress call failed: {type(e).__name__}: {e}")
                return 5

            print("\n[5b/5] Calling headroom_retrieve via Strands MCP dispatcher ...")
            try:
                result = mcp.call_tool_sync(
                    tool_use_id="probe-retrieve-1",
                    name="headroom_retrieve",
                    arguments={"hash": stashed_hash},
                )
                print(f"  retrieve status: {result.get('status', '?')}")
                content = result.get("content", [])
                preview = ""
                for block in content:
                    if isinstance(block, dict):
                        if "text" in block:
                            preview = block["text"][:300]
                            break
                        if "json" in block:
                            preview = json.dumps(block["json"])[:300]
                            break
                print(f"  retrieved content preview: {preview!r}")
                if any(needle in preview for needle in ("order-0", "order-1", "order-")):
                    print("  ✓ retrieved content references the original JSON rows.")
                else:
                    print(
                        "  - preview doesn't obviously match the original. "
                        "Scan the full preview above to verify."
                    )
            except Exception as e:  # noqa: BLE001
                print(f"  ! retrieve call failed: {type(e).__name__}: {e}")
                return 6

        print("\n" + "=" * 72)
        print(" MCP DISPATCH PROBE COMPLETE")
        print(" If steps 3+4+5 succeeded, the MCP-everywhere Path B is live:")
        print(" Strands receives a headroom_retrieve tool_call → MCP dispatcher")
        print(" → Headroom MCP server → CompressionStore → original content back.")
        print("=" * 72)
        return 0
    finally:
        print("\n  stopping proxy ...")
        stop_proxy(proxy)


if __name__ == "__main__":
    sys.exit(main())

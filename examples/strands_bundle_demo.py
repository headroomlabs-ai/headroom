#!/usr/bin/env python3
"""Strands agent + Headroom — drop-in compression demo.

This is what a real Strands user writes. The ONLY Headroom-specific
lines are the import and the bundle construction. Everything else
is normal Strands code.

The demo:
  1. Defines a normal Strands `@tool` that returns a verbose JSON blob
     (mimicking a real RAG or DB tool result).
  2. Builds a normal Strands `Agent` with that tool + the bundle's MCP
     tools (headroom_compress / headroom_retrieve / headroom_stats).
  3. Sends ONE user query.
  4. Lets the agent loop autonomously.
  5. Prints the answer + the proxy /stats summary so you can SEE that
     Headroom compressed the verbose tool output on the way to Bedrock.

The Headroom proxy is started as a background process here so the
script is self-contained. In production, the proxy runs as a
long-lived service (ECS / k8s / EC2) and the application code looks
exactly like what's below the `=== USER CODE ===` line.

Run
---

    AWS_REGION=us-west-2 python examples/strands_bundle_demo.py

Cost
----

Sonnet 4.5 via Bedrock, multi-turn agent loop. Expect ~$0.02–0.10
per run depending on how many tool calls the model makes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from contextlib import suppress
from pathlib import Path

# ============================================================================
# Boilerplate: start the Headroom proxy as a child process for the demo.
# In production this is a long-lived service — none of this code is in your
# Strands app.
# ============================================================================

PROXY_PORT = 8787
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"


def _start_proxy() -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "headroom.cli",
        "proxy",
        "--backend",
        "bedrock",
        "--region",
        os.environ.get("AWS_REGION", "us-west-2"),
        "--port",
        str(PROXY_PORT),
    ]
    log_path = Path("/tmp/headroom_bundle_demo_proxy.log")
    log = log_path.open("wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603
    print(f"  → proxy started (pid={proc.pid}); log: {log_path}")
    return proc


def _wait_for_proxy(timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{PROXY_URL}/readyz", timeout=1) as r:  # noqa: S310
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise RuntimeError(f"Proxy did not become ready within {timeout_s}s")


def _stop_proxy(proc: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _print_stats_panel() -> None:
    try:
        with urllib.request.urlopen(f"{PROXY_URL}/stats", timeout=5) as r:  # noqa: S310
            stats = json.loads(r.read())
    except Exception:  # noqa: BLE001
        return
    summary = stats.get("summary", {})
    comp = summary.get("compression", {})
    uncomp = summary.get("uncompressed_requests", {})
    mcp = summary.get("mcp", {}) or {}
    print()
    print("  Proxy /stats summary")
    print("  --------------------")
    print(f"    api_requests:           {summary.get('api_requests', 0)}")
    print(f"    requests_compressed:    {comp.get('requests_compressed', 0)}")
    print(f"    total_tokens_removed:   {comp.get('total_tokens_removed', 0)}")
    if comp.get("best_compression_pct"):
        print(f"    best_compression_pct:   {comp['best_compression_pct']:.1f}%")
    print(f"    uncompressed reasons:   {uncomp}")
    # MCP-side work (headroom_compress / headroom_retrieve called by the LLM
    # via Strands' MCP dispatcher). The retrievals counter is the
    # over-compression alarm — if it grows linearly with turn count, our
    # lossy compressors are dropping info the model actually needs.
    print(f"    mcp_compressions:       {mcp.get('compressions', 0)}")
    print(f"    mcp_tokens_removed:     {mcp.get('tokens_removed', 0)}")
    print(f"    ccr_retrievals_count:   {mcp.get('retrievals', 0)}")


# ============================================================================
# === USER CODE ===
# Everything below is what a normal Strands user writes. The only
# Headroom-specific bits are the `HeadroomBundle` import and one
# construction call. The rest is vanilla Strands.
# ============================================================================

from strands import Agent, tool  # noqa: E402  (kept under USER CODE banner for readability)
from strands.models.openai import OpenAIModel  # noqa: E402

from headroom.integrations.strands import HeadroomBundle  # noqa: E402


@tool
def search_documentation(query: str) -> str:
    """Search the documentation for articles matching `query`. Returns up to 30 results as JSON."""
    # Mock data — pretend this hit a real search API. The point is the
    # response is big and repetitive, which is the kind of tool output
    # Headroom is designed to shrink.
    articles = [
        {
            "id": f"doc-{i:04d}",
            "title": f"{query.title()} Guide — Part {i + 1}",
            "url": f"https://docs.example.com/{query}/article-{i + 1}",
            "category": "tutorial" if i % 3 == 0 else "reference",
            "snippet": (
                f"This article covers {query} implementation in depth. "
                f"It walks through setup, configuration, and common pitfalls. "
                f"Section {i + 1} of the comprehensive guide series."
            )
            * 4,
            "metadata": {
                "author": "docs-team",
                "tags": ["how-to", query, "production-ready"],
                "last_updated": f"2026-04-{(i % 28) + 1:02d}",
            },
        }
        for i in range(30)
    ]
    return json.dumps(articles)


def run_agent_demo() -> None:
    """The actual Strands agent demo — looks like any other Strands app."""

    # === The only Headroom-specific code in your app ===
    bundle = HeadroomBundle(
        proxy_url=PROXY_URL,
        enable_serena_mcp=False,  # disabled here so the demo runs fast
    )

    # === Normal Strands agent setup ===
    model = OpenAIModel(
        model_id="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        client_args={
            "base_url": f"{PROXY_URL}/v1",
            "api_key": "dummy-bedrock-uses-aws-creds-at-proxy",
            "default_headers": {
                "x-headroom-session-id": "demo-1",
                "X-Client": "strands",
            },
        },
        params={"max_tokens": 800, "temperature": 0.2},
    )

    agent = Agent(
        model=model,
        # bundle.tools = [Headroom MCP client]; we also add our local tool
        tools=bundle.tools + [search_documentation],
        system_prompt=(
            "You are a documentation assistant. When the user asks about a "
            "topic, use the search_documentation tool to look it up, then "
            "answer concisely."
        ),
    )

    print("\n  → sending user query (agent runs autonomously) ...")
    user_query = (
        "Search the documentation for 'authentication'. "
        "Tell me how many results you got, then summarize the top 3 in one sentence each."
    )
    print(f"  user: {user_query!r}\n")

    t0 = time.time()
    response = agent(user_query)
    elapsed = time.time() - t0

    print(f"\n  agent response (after {elapsed:.1f}s):")
    print("  " + "-" * 68)
    for line in str(response).splitlines():
        print(f"  {line}")
    print("  " + "-" * 68)


# ============================================================================
# Bootstrap
# ============================================================================


def main() -> int:
    print("=" * 72)
    print(" Strands agent + Headroom (drop-in compression)")
    print("=" * 72)
    print(f" proxy: {PROXY_URL}    region: {os.environ.get('AWS_REGION', 'us-west-2')}")
    print()
    print(" Starting Headroom proxy (in production this is a long-lived service) ...")
    proxy = _start_proxy()
    try:
        _wait_for_proxy()
        print("  → proxy ready.")
        run_agent_demo()
        _print_stats_panel()
        print("\n" + "=" * 72)
        print(" Done. If requests_compressed > 0 above, Headroom shrunk the")
        print(" verbose tool output on its way to Bedrock — automatically,")
        print(" with no code changes in the agent.")
        print("=" * 72)
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"\n  ! FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        print("\n  → stopping proxy ...")
        _stop_proxy(proxy)


if __name__ == "__main__":
    sys.exit(main())

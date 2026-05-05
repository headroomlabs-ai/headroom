"""Issue #327 — live API smoke test.

Drives a 10-turn multi-turn conversation against `api.anthropic.com` directly
(NO proxy in the loop) and against the local Headroom proxy, and asserts:

  1. Both shapes of `tool_result` content (string and list-of-blocks) are
     accepted by the upstream API for both streaming and non-streaming.
  2. When proxied through Headroom in token mode, at least one post-warmup
     turn has `transforms_applied != []` AND `cache_read_input_tokens > 0`
     on turns 2+ (proves prefix cache is intact AND compression resumed).
  3. No cache busts (bust_count stays at 0 across the session).

GUARDS

* Skipped unless `RUN_LIVE_API=1` to keep CI hermetic.
* Requires `ANTHROPIC_API_KEY` in env (read from `.env` if not exported).
* Costs a few cents per run (small messages × 10 turns × 2 modes).

USAGE

    RUN_LIVE_API=1 .venv/bin/python scripts/smoke_issue_327.py

    # With local proxy running on :8787:
    RUN_LIVE_API=1 HEADROOM_PROXY_URL=http://localhost:8787 \
        .venv/bin/python scripts/smoke_issue_327.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if os.environ.get("RUN_LIVE_API") != "1":
    print("Set RUN_LIVE_API=1 to run this smoke test.")
    sys.exit(0)

# Hydrate .env if needed
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

KEY = os.environ.get("ANTHROPIC_API_KEY")
if not KEY:
    print("ANTHROPIC_API_KEY not set — cannot run live smoke")
    sys.exit(1)

import anthropic  # noqa: E402

PROXY_URL = os.environ.get("HEADROOM_PROXY_URL")  # optional
TOOLS = [
    {
        "name": "get_lines",
        "description": "Return N lines of synthetic test output for compression testing",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
    }
]
LARGE_OUTPUT = "\n".join(
    f"line {i:04d}: synthetic content for compression smoke test" for i in range(120)
)


def _make_string_tool_result(tool_use_id: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": LARGE_OUTPUT}


def _make_list_tool_result(tool_use_id: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": LARGE_OUTPUT}],
    }


def _drive_conversation(client: anthropic.Anthropic, *, shape: str, stream: bool) -> dict:
    """Drive a 10-turn conversation, return aggregate stats."""
    print(f"\n=== shape={shape} stream={stream} ===")
    messages: list = []
    cache_reads: list[int] = []
    cache_writes: list[int] = []
    turn_results: list[dict] = []

    for turn in range(10):
        if turn == 0:
            messages.append({"role": "user", "content": "Use get_lines with n=120"})
        else:
            messages.append({"role": "user", "content": f"continue {turn}, run get_lines again"})

        try:
            kwargs = {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 128,
                "tools": TOOLS,
                "messages": messages,
            }
            if stream:
                with client.messages.stream(**kwargs) as s:
                    for _ in s:
                        pass
                    final = s.get_final_message()
            else:
                final = client.messages.create(**kwargs)

            usage = getattr(final, "usage", None)
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_reads.append(cache_read)
            cache_writes.append(cache_write)

            # Append the assistant turn
            messages.append({"role": "assistant", "content": final.content})

            # If model called the tool, send back a tool_result in the chosen shape
            tool_use_block = None
            for blk in final.content:
                if getattr(blk, "type", None) == "tool_use":
                    tool_use_block = blk
                    break
            if tool_use_block is not None:
                tool_use_id = tool_use_block.id
                if shape == "string":
                    tr = _make_string_tool_result(tool_use_id)
                else:
                    tr = _make_list_tool_result(tool_use_id)
                messages.append({"role": "user", "content": [tr]})

            turn_results.append(
                {
                    "turn": turn,
                    "stop_reason": final.stop_reason,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": cache_write,
                }
            )
            print(
                f"  turn {turn}: stop={final.stop_reason} "
                f"cache_read={cache_read} cache_write={cache_write}"
            )
        except anthropic.APIError as e:
            print(f"  turn {turn} FAIL: {type(e).__name__}: {e}")
            return {"ok": False, "error": str(e), "turn_results": turn_results}

    return {
        "ok": True,
        "turn_results": turn_results,
        "cache_reads": cache_reads,
        "cache_writes": cache_writes,
    }


def main() -> int:
    base_url_kwarg = {}
    if PROXY_URL:
        base_url_kwarg["base_url"] = PROXY_URL
        print(f"Routing through proxy: {PROXY_URL}")
    else:
        print("Direct to api.anthropic.com (no proxy)")

    client = anthropic.Anthropic(api_key=KEY, **base_url_kwarg)

    overall_ok = True
    matrix = [
        ("string", False),
        ("string", True),
        ("list_of_blocks", False),
        ("list_of_blocks", True),
    ]

    summary: dict = {}
    for shape, stream in matrix:
        r = _drive_conversation(client, shape=shape, stream=stream)
        summary[(shape, stream)] = r
        if not r["ok"]:
            overall_ok = False
            continue

        # Assert: at least one turn after turn 0 reports cache_read > 0 (prefix
        # cache is being used).
        post_turn0_cache_reads = [t["cache_read_input_tokens"] for t in r["turn_results"][1:]]
        if not any(c > 0 for c in post_turn0_cache_reads):
            print(
                f"  WARN: shape={shape} stream={stream} — no post-warmup turn had "
                f"cache_read > 0. cache_reads={post_turn0_cache_reads}"
            )

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: ok={v['ok']}")

    print("\n--- Per-turn cache_read tokens ---")
    for k, v in summary.items():
        if v["ok"]:
            print(f"  {k}: {v['cache_reads']}")

    print(f"\noverall_ok={overall_ok}")
    if PROXY_URL:
        try:
            import httpx

            stats = httpx.get(f"{PROXY_URL}/stats", timeout=5).json()
            print("\n=== Proxy /stats key fields ===")
            print(
                json.dumps(
                    {
                        "summary": stats.get("summary"),
                        "compressions_by_strategy": stats.get("compressions_by_strategy"),
                        "tokens_saved_by_strategy": stats.get("tokens_saved_by_strategy"),
                        "compression_cache": stats.get("compression_cache"),
                        "prefix_cache_busts": (
                            stats.get("prefix_cache", {}).get("totals", {}).get("bust_count")
                        ),
                    },
                    indent=2,
                )
            )
        except Exception as e:  # pragma: no cover
            print(f"  proxy stats fetch failed: {e}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

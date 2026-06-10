#!/usr/bin/env python3
"""Compare plain Hermes vs Headroom→Hermes token usage on /v1/chat/completions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "test_cli"))

from hermes_support import (  # noqa: E402
    MIN_HEADROOM_TOKENS_SAVED,
    MIN_SMART_CRUSHER_SAVED,
    MIN_UPSTREAM_PROMPT_DELTA,
    assert_compression_delta,
    hermes_health_url,
    hermes_reachable,
    pick_free_port,
    start_headroom_proxy,
    stop_process,
    wait_proxy_ready,
)
from hermes_workloads import agent_tool_messages, multi_turn_agent_messages  # noqa: E402

DEFAULT_HERMES_BASE = os.environ.get("HEADROOM_HERMES_BASE_URL", "http://127.0.0.1:38765/v1")
DEFAULT_MODEL = os.environ.get("HEADROOM_HERMES_MODEL", "grok-4.3")


def chat(base_url: str, messages: list[dict], model: str, max_tokens: int = 24) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    body["_backend"] = resp.headers.get("X-LLM-Backend", "?")
    return body


def fetch_stats(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=30) as resp:
        return json.loads(resp.read().decode())


def _delta_block(plain_usage: dict, wrapped_usage: dict, token_stats: dict, stats: dict) -> dict:
    plain_in = int(plain_usage.get("prompt_tokens") or 0)
    wrap_in = int(wrapped_usage.get("prompt_tokens") or 0)
    saved = int(token_stats.get("saved") or 0)
    strategy = stats.get("tokens_saved_by_strategy") or {}
    return {
        "plain_prompt_tokens": plain_in,
        "headroom_prompt_tokens": wrap_in,
        "upstream_prompt_token_delta": plain_in - wrap_in,
        "headroom_tokens_saved": saved,
        "tokens_saved_by_strategy": strategy,
        "compression_summary": stats.get("summary", {}).get("compression") or {},
        "thresholds": {
            "min_upstream_prompt_delta": MIN_UPSTREAM_PROMPT_DELTA,
            "min_headroom_tokens_saved": MIN_HEADROOM_TOKENS_SAVED,
            "min_smart_crusher_saved": MIN_SMART_CRUSHER_SAVED,
        },
    }


def run_agent_tool_turn(
    hermes_base: str,
    model: str,
    *,
    canary_port: int | None = None,
) -> dict:
    messages = agent_tool_messages()
    plain = chat(hermes_base, messages, model)
    plain_usage = plain.get("usage") or {}

    if canary_port is not None:
        wrapped = chat(f"http://127.0.0.1:{canary_port}/v1", messages, model)
        wrapped_usage = wrapped.get("usage") or {}
        stats = fetch_stats(canary_port)
        token_stats = stats.get("tokens") or {}
        delta = _delta_block(plain_usage, wrapped_usage, token_stats, stats)
        return {
            "mode": "agent-tool-canary",
            "canary_port": canary_port,
            "plain": {
                "content": plain["choices"][0]["message"]["content"].strip(),
                "usage": plain_usage,
                "backend": plain.get("_backend"),
            },
            "headroom": {
                "content": wrapped["choices"][0]["message"]["content"].strip(),
                "usage": wrapped_usage,
                "backend": wrapped.get("_backend"),
                "tokens_saved": token_stats.get("saved"),
            },
            "delta": delta,
        }

    port = pick_free_port()
    proc = start_headroom_proxy(port=port, hermes_base=hermes_base)
    try:
        wait_proxy_ready(port)
        wrapped = chat(f"http://127.0.0.1:{port}/v1", messages, model)
        wrapped_usage = wrapped.get("usage") or {}
        stats = fetch_stats(port)
        token_stats = stats.get("tokens") or {}
        delta = _delta_block(plain_usage, wrapped_usage, token_stats, stats)
        return {
            "mode": "agent-tool",
            "plain": {
                "content": plain["choices"][0]["message"]["content"].strip(),
                "usage": plain_usage,
                "backend": plain.get("_backend"),
            },
            "headroom": {
                "content": wrapped["choices"][0]["message"]["content"].strip(),
                "usage": wrapped_usage,
                "backend": wrapped.get("_backend"),
                "tokens_saved": token_stats.get("saved"),
            },
            "delta": delta,
        }
    finally:
        stop_process(proc)


def run_multi_turn(hermes_base: str, model: str, turns: int = 3) -> dict:
    messages = multi_turn_agent_messages(turns=turns)

    def final_turn(base_url: str) -> dict:
        body = chat(base_url, messages, model, max_tokens=16)
        return {
            "content": body["choices"][0]["message"]["content"].strip(),
            "usage": body.get("usage") or {},
            "backend": body.get("_backend"),
        }

    plain = final_turn(hermes_base)
    port = pick_free_port()
    proc = start_headroom_proxy(port=port, hermes_base=hermes_base)
    try:
        wait_proxy_ready(port)
        wrapped = final_turn(f"http://127.0.0.1:{port}/v1")
        stats = fetch_stats(port)
        token_stats = stats.get("tokens") or {}
        delta = _delta_block(plain["usage"], wrapped["usage"], token_stats, stats)
        return {
            "mode": "multi-turn-agent-tool",
            "turns": turns,
            "plain": plain,
            "headroom": {**wrapped, "tokens_saved": token_stats.get("saved")},
            "delta": delta,
        }
    finally:
        stop_process(proc)


def validate_result(result: dict) -> int:
    delta = result.get("delta") or {}
    plain_in = int(delta.get("plain_prompt_tokens") or 0)
    wrap_in = int(delta.get("headroom_prompt_tokens") or 0)
    saved = int(delta.get("headroom_tokens_saved") or 0)
    strategy_saved = int((delta.get("tokens_saved_by_strategy") or {}).get("smart_crusher") or 0)
    content = (result.get("headroom") or {}).get("content", "")
    try:
        upstream_delta = plain_in - wrap_in
        assert upstream_delta >= MIN_UPSTREAM_PROMPT_DELTA, (
            f"upstream delta too low: {upstream_delta}"
        )
        assert saved >= MIN_HEADROOM_TOKENS_SAVED, f"tokens.saved too low: {saved}"
        assert strategy_saved >= MIN_SMART_CRUSHER_SAVED, (
            f"smart_crusher savings too low: {strategy_saved}"
        )
        if result.get("mode") in {"agent-tool", "agent-tool-canary"}:
            assert_compression_delta(
                plain_prompt_tokens=plain_in,
                wrapped_prompt_tokens=wrap_in,
                tokens_saved=saved,
                smart_crusher_saved=strategy_saved,
                content=content,
            )
        elif "TURN_3" not in content:
            raise AssertionError(f"expected TURN_3 in multi-turn response, got {content!r}")
        return 0
    except AssertionError as exc:
        print(f"BENCH_FAIL: {exc}", file=sys.stderr)
        return 2


def _canary_ready(port: int) -> bool:
    for path in ("/readyz", "/health", "/livez"):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-base", default=DEFAULT_HERMES_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--multi-turn", action="store_true")
    parser.add_argument(
        "--canary-port",
        type=int,
        default=int(os.environ["HEADROOM_CANARY_PORT"])
        if os.environ.get("HEADROOM_CANARY_PORT")
        else None,
        help="Use persistent Headroom canary on this port (default: HEADROOM_CANARY_PORT)",
    )
    args = parser.parse_args()
    hermes_base = args.hermes_base.rstrip("/")

    if not hermes_reachable(hermes_base):
        print(f"Hermes health check failed for {hermes_health_url(hermes_base)}", file=sys.stderr)
        return 1

    if args.canary_port is not None and args.multi_turn:
        print("--canary-port cannot be combined with --multi-turn", file=sys.stderr)
        return 1

    if args.canary_port is not None and not _canary_ready(args.canary_port):
        print(f"Canary not ready on port {args.canary_port}", file=sys.stderr)
        return 1

    try:
        result = (
            run_multi_turn(hermes_base, args.model)
            if args.multi_turn
            else run_agent_tool_turn(hermes_base, args.model, canary_port=args.canary_port)
        )
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return validate_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
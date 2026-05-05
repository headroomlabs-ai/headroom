#!/usr/bin/env python3
"""Local benchmark for proxy run modes (no API calls).

Compares:
- baseline: no compression
- token mode: prioritize compression
- cache mode: preserve prior-turn prefix stability

Includes an optional real-test harness printout for Claude Code, but does not
invoke external APIs unless the user does so manually.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from typing import Any

from headroom.cache.compression_cache import CompressionCache
from headroom.cache.prefix_tracker import PrefixCacheTracker
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.models import ProxyConfig
from headroom.proxy.modes import PROXY_MODE_CACHE, PROXY_MODE_TOKEN
from headroom.proxy.server import HeadroomProxy
from headroom.tokenizers import get_tokenizer
from headroom.utils import extract_user_query

MODEL = "claude-sonnet-4-6"


@dataclass
class ModeBenchmarkResult:
    mode: str
    total_original_tokens: int = 0
    total_sent_tokens: int = 0
    total_tokens_saved: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_uncached_tokens: int = 0

    @property
    def compression_pct(self) -> float:
        if self.total_original_tokens <= 0:
            return 0.0
        return self.total_tokens_saved / self.total_original_tokens * 100.0

    @property
    def cache_hit_pct(self) -> float:
        total = (
            self.total_cache_read_tokens
            + self.total_cache_write_tokens
            + self.total_uncached_tokens
        )
        if total <= 0:
            return 0.0
        return self.total_cache_read_tokens / total * 100.0


def _build_tool_result(turn: int, rows: int = 240) -> str:
    payload = []
    for i in range(rows):
        payload.append(
            {
                "id": f"{turn:02d}-{i:04d}",
                "status": "ok" if i % 37 else "warning",
                "service": "auth-api" if i % 2 else "gateway",
                "latency_ms": 100 + (i % 13),
                "hint": "retry with exponential backoff" if i % 89 == 0 else "none",
            }
        )
    return json.dumps(payload)


def _build_conversation(turn: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for t in range(1, turn):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Analyze tool output turn {t} and summarize anomalies.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"tool-{t}",
                            "content": _build_tool_result(t),
                        }
                    ],
                },
                {"role": "assistant", "content": f"Turn {t} acknowledged."},
            ]
        )
    # Current turn: user request + fresh tool output, no assistant response yet.
    messages.extend(
        [
            {
                "role": "user",
                "content": f"Analyze tool output turn {turn} and summarize anomalies.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"tool-{turn}",
                        "content": _build_tool_result(turn),
                    }
                ],
            },
        ]
    )
    return messages


def _common_prefix_tokens(
    prev: list[dict[str, Any]], curr: list[dict[str, Any]], tokenizer: Any
) -> tuple[int, list[int]]:
    common = 0
    counts: list[int] = []
    for msg in curr:
        counts.append(tokenizer.count_message(msg))
    for i, (a, b) in enumerate(zip(prev, curr)):
        if a != b:
            break
        common += counts[i]
    return common, counts


def _make_proxy(mode: str) -> HeadroomProxy:
    cfg = ProxyConfig(
        mode=mode,
        optimize=True,
        image_optimize=False,
        smart_routing=False,
        code_aware_enabled=False,
        read_lifecycle=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
    )
    return HeadroomProxy(cfg)


def _simulate_mode(turns: int, mode: str) -> ModeBenchmarkResult:
    tokenizer = get_tokenizer(MODEL)
    result = ModeBenchmarkResult(mode=mode)

    if mode == "baseline":
        prev_forwarded: list[dict[str, Any]] = []
        for turn in range(1, turns + 1):
            messages = _build_conversation(turn)
            before = tokenizer.count_messages(messages)
            common, counts = _common_prefix_tokens(prev_forwarded, messages, tokenizer)
            uncached = max(0, before - common)

            result.total_original_tokens += before
            result.total_sent_tokens += before
            result.total_cache_read_tokens += common
            result.total_cache_write_tokens += 0
            result.total_uncached_tokens += uncached
            prev_forwarded = copy.deepcopy(messages)
        return result

    proxy = _make_proxy(mode)
    prefix_tracker = PrefixCacheTracker("anthropic")
    comp_cache = CompressionCache()
    prev_forwarded = []

    for turn in range(1, turns + 1):
        messages = _build_conversation(turn)
        before = tokenizer.count_messages(messages)

        frozen = prefix_tracker.get_frozen_message_count()
        if mode == PROXY_MODE_CACHE:
            frozen = AnthropicHandlerMixin._strict_previous_turn_frozen_count(messages, frozen)

        working = messages
        if mode == PROXY_MODE_TOKEN:
            working = comp_cache.apply_cached(messages)
            frozen = min(frozen, comp_cache.compute_frozen_count(messages))

        context_limit = proxy.anthropic_provider.get_context_limit(MODEL)
        pipeline_result = proxy.anthropic_pipeline.apply(
            messages=working,
            model=MODEL,
            model_limit=context_limit,
            context=extract_user_query(working),
            frozen_message_count=frozen,
        )
        forwarded = pipeline_result.messages

        if mode == PROXY_MODE_TOKEN:
            comp_cache.update_from_result(messages, forwarded)
        if mode == PROXY_MODE_CACHE:
            forwarded, _ = AnthropicHandlerMixin._restore_frozen_prefix(
                messages, forwarded, frozen_message_count=frozen
            )

        after = tokenizer.count_messages(forwarded)
        common, msg_counts = _common_prefix_tokens(prev_forwarded, forwarded, tokenizer)
        uncached = max(0, after - common)

        result.total_original_tokens += before
        result.total_sent_tokens += after
        result.total_tokens_saved += max(0, before - after)
        result.total_cache_read_tokens += common
        result.total_uncached_tokens += uncached

        prefix_tracker.update_from_response(
            cache_read_tokens=common,
            cache_write_tokens=uncached,
            messages=forwarded,
            message_token_counts=msg_counts,
        )
        result.total_cache_write_tokens += uncached
        prev_forwarded = copy.deepcopy(forwarded)

    return result


def run_local_benchmark(turns: int = 12) -> dict[str, ModeBenchmarkResult]:
    return {
        "baseline": _simulate_mode(turns, "baseline"),
        PROXY_MODE_TOKEN: _simulate_mode(turns, PROXY_MODE_TOKEN),
        PROXY_MODE_CACHE: _simulate_mode(turns, PROXY_MODE_CACHE),
    }


def _print_results(results: dict[str, ModeBenchmarkResult]) -> None:
    print(
        "\nMode benchmark (higher compression + higher cache_hit is better for total cost):\n"
        "mode      orig_tok   sent_tok   saved_tok   compression   cache_hit   uncached_tok"
    )
    for key in ("baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE):
        r = results[key]
        print(
            f"{r.mode:<9} {r.total_original_tokens:>9,} {r.total_sent_tokens:>10,} "
            f"{r.total_tokens_saved:>10,} {r.compression_pct:>10.1f}% "
            f"{r.cache_hit_pct:>9.1f}% {r.total_uncached_tokens:>12,}"
        )

    token = results[PROXY_MODE_TOKEN]
    cache = results[PROXY_MODE_CACHE]
    print("\nDelta (cache - token):")
    print(f"  cache_hit_pct: {cache.cache_hit_pct - token.cache_hit_pct:+.1f}%")
    print(f"  compression_pct: {cache.compression_pct - token.compression_pct:+.1f}%")
    print(f"  uncached_tokens: {cache.total_uncached_tokens - token.total_uncached_tokens:+,}")


def _print_real_harness() -> None:
    print("\nReal test harness (manual; optional, not executed by this benchmark):")
    print("  1) Start proxy in cache mode:  HEADROOM_MODE=cache headroom proxy --port 8787")
    print("  2) Start proxy in token mode:  HEADROOM_MODE=token headroom proxy --port 8787")
    print("  3) Run Claude Code against each:")
    print("     ANTHROPIC_BASE_URL=http://localhost:8787 claude")
    print("  4) Compare /stats prefix_cache and compression sections per run.")


def main() -> None:
    logging.getLogger("headroom").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Local benchmark for proxy token/cache modes")
    parser.add_argument("--turns", type=int, default=12, help="Conversation turns to simulate")
    parser.add_argument(
        "--show-real-harness",
        action="store_true",
        help="Print manual steps for optional Claude Code real testing",
    )
    args = parser.parse_args()

    results = run_local_benchmark(turns=args.turns)
    _print_results(results)
    if args.show_real_harness:
        _print_real_harness()


if __name__ == "__main__":
    main()

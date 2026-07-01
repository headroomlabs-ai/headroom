#!/usr/bin/env python3
"""
Provider-agnostic savings dashboard — analyzes headroom proxy logs.

Consumes proxy JSONL logs and generates a per-content-type savings breakdown.
Works with any provider (Anthropic, OpenAI, Gemini) since it reads the proxy's
own log format. Also generates CO2 and cost estimates.

Usage:
    python3 bench/savings_dashboard.py --log-file proxy.log
    python3 bench/savings_dashboard.py --log-file proxy.log --json > dashboard.json
    python3 bench/savings_dashboard.py --log-dir ~/.headroom/logs --last-hours 168

Output:
    - Per content-type savings (code, search, logs, JSON, text)
    - Provider breakdown (Anthropic vs OpenAI vs Gemini)
    - CO2 estimate (0.4 gCO2e per 1K tokens)
    - Cost savings estimate (based on provider pricing)
"""
from __future__ import annotations

import argparse
import json
import sys
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── Provider pricing ($/1M tokens, input) ────────────────────────────
PROVIDER_PRICES = {
    "anthropic": {"claude-sonnet-4-5": 3.00, "claude-opus-4": 15.00, "default": 3.00},
    "openai": {"gpt-4o": 2.50, "gpt-4": 30.00, "default": 2.50},
    "google": {"gemini-1.5-pro": 1.25, "default": 1.25},
}
CO2_PER_1K_TOKENS = 0.0004  # kg CO2e per 1K tokens (approx)


def classify_content(messages: list[dict] | None) -> str:
    """Classify message content type from structure/patterns."""
    if not messages:
        return "unknown"
    
    text = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text += content + "\n"
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text += str(part.get("text", "")) + "\n"
    
    # Heuristic classification
    if re.search(r"diff --git|@@ -\d+,\d+ \+\d+", text):
        return "code_diff"
    if re.search(r"error\[|Traceback|panic|FAILED", text):
        return "error_trace"
    if re.search(r"test result:|running \d+ test", text):
        return "test_output"
    if re.search(r"\.rs:\d+:|\.py:\d+:|\.ts:\d+:|\.go:\d+:", text):
        return "search_result"
    if re.search(r'^\s*[\[\{]".*"[\]\}]', text, re.MULTILINE):
        return "json_output"
    if re.search(r"Compiling|Building|Finished.*target", text):
        return "build_output"
    return "text"


def parse_logs(log_files: list[Path], last_hours: float | None = None) -> list[dict]:
    """Parse headroom proxy JSONL log files."""
    entries = []
    cutoff = None
    if last_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=last_hours)
    
    for path in log_files:
        if not path.exists():
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if cutoff:
                        ts = entry.get("timestamp", "")
                        if ts:
                            try:
                                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                if t < cutoff:
                                    continue
                            except:
                                pass
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return entries


def build_dashboard(entries: list[dict]) -> dict[str, Any]:
    """Build savings dashboard from parsed log entries."""
    if not entries:
        return {"error": "no entries found"}

    by_provider: dict[str, dict] = defaultdict(lambda: {"tokens_saved": 0, "tokens_original": 0, "count": 0, "savings_pct": 0.0})
    by_type: dict[str, dict] = defaultdict(lambda: {"tokens_saved": 0, "tokens_original": 0, "count": 0})
    by_model: dict[str, dict] = defaultdict(lambda: {"tokens_saved": 0, "tokens_original": 0, "count": 0})
    transforms_used: dict[str, int] = defaultdict(int)
    
    total_saved = 0
    total_original = 0
    total_requests = 0
    cache_hits = 0

    for e in entries:
        provider = e.get("provider", "unknown")
        model = e.get("model", "unknown")
        saved = e.get("tokens_saved", 0)
        original = e.get("input_tokens_original", 0)
        cache = e.get("cache_hit", False)
        
        total_saved += saved
        total_original += original
        total_requests += 1
        if cache:
            cache_hits += 1
        
        # Provider stats
        by_provider[provider]["tokens_saved"] += saved
        by_provider[provider]["tokens_original"] += original
        by_provider[provider]["count"] += 1
        
        # Model stats
        by_model[model]["tokens_saved"] += saved
        by_model[model]["tokens_original"] += original
        by_model[model]["count"] += 1
        
        # Content type classification
        ctype = classify_content(e.get("request_messages"))
        by_type[ctype]["tokens_saved"] += saved
        by_type[ctype]["tokens_original"] += original
        by_type[ctype]["count"] += 1
        
        # Transforms applied
        for t in e.get("transforms_applied", []):
            transforms_used[t] += 1

    # Compute percentages
    for pdata in by_provider.values():
        if pdata["tokens_original"] > 0:
            pdata["savings_pct"] = round(pdata["tokens_saved"] / pdata["tokens_original"] * 100, 1)

    # CO2 saved (tokens_saved * CO2_per_1K / 1000)
    co2_kg = total_saved * CO2_PER_1K_TOKENS / 1000

    # Cost saved (estimate based on provider prices)
    cost_saved = 0.0
    for provider, pdata in by_provider.items():
        price = PROVIDER_PRICES.get(provider, {}).get("default", 3.00)
        cost_saved += pdata["tokens_saved"] * price / 1_000_000

    overall_savings = round(total_saved / max(total_original, 1) * 100, 1)

    return {
        "period": {
            "requests": total_requests,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / max(total_requests, 1) * 100, 1),
        },
        "tokens": {
            "original": total_original,
            "saved": total_saved,
            "savings_pct": overall_savings,
        },
        "environment": {
            "co2_kg_saved": round(co2_kg, 4),
            "cost_usd_saved": round(cost_saved, 2),
        },
        "by_provider": {k: dict(v) for k, v in sorted(by_provider.items())},
        "by_content_type": {
            k: {
                "savings_pct": round(v["tokens_saved"] / max(v["tokens_original"], 1) * 100, 1),
                "tokens_saved": v["tokens_saved"],
                "count": v["count"],
            }
            for k, v in sorted(by_type.items(), key=lambda x: -x[1]["tokens_saved"])
        },
        "by_model": {
            k: {
                "savings_pct": round(v["tokens_saved"] / max(v["tokens_original"], 1) * 100, 1),
                "count": v["count"],
            }
            for k, v in sorted(by_model.items(), key=lambda x: -x[1]["tokens_saved"])
        },
        "top_transforms": dict(sorted(transforms_used.items(), key=lambda x: -x[1])[:10]),
    }


def main():
    ap = argparse.ArgumentParser(description="Provider-agnostic savings dashboard")
    ap.add_argument("--log-file", default=None, help="Single proxy JSONL log file")
    ap.add_argument("--log-dir", default=None, help="Directory with proxy logs (proxy.log*)")
    ap.add_argument("--last-hours", type=float, default=None, help="Only analyze last N hours")
    ap.add_argument("--json", action="store_true", help="JSON output")
    args = ap.parse_args()

    log_files = []
    if args.log_file:
        log_files.append(Path(args.log_file))
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_files.extend(sorted(log_dir.glob("proxy*.log*")))
    
    if not log_files:
        # Try default headroom log locations
        for default_dir in [Path.home() / ".headroom" / "logs", Path("logs")]:
            if default_dir.exists():
                log_files.extend(sorted(default_dir.glob("proxy*.log*")))

    if not log_files:
        print("No log files found. Use --log-file or --log-dir.", file=sys.stderr)
        print("Example: python3 bench/savings_dashboard.py --log-file /tmp/headroom-c3/requests.jsonl", file=sys.stderr)
        sys.exit(2)

    entries = parse_logs(log_files, args.last_hours)
    dashboard = build_dashboard(entries)

    if args.json:
        json.dump(dashboard, sys.stdout, indent=2)
    else:
        print("╔══════════════════════════════════════════════════╗")
        print("║         Kompress Savings Dashboard              ║")
        print("╚══════════════════════════════════════════════════╝")
        print(f"\nPeriod: {dashboard['period']['requests']} requests "
              f"({dashboard['period']['cache_hit_rate']}% cache hits)")
        print(f"\nTokens: {dashboard['tokens']['original']:,} → "
              f"saved {dashboard['tokens']['saved']:,} "
              f"({dashboard['tokens']['savings_pct']}%)")
        print(f"Impact: {dashboard['environment']['co2_kg_saved']:.4f} kg CO2 saved, "
              f"~${dashboard['environment']['cost_usd_saved']:.2f} saved")
        
        print(f"\nBy Content Type:")
        print(f"{'Type':<20} {'Savings':>8} {'Tokens':>10} {'Count':>8}")
        print("-" * 50)
        for ctype, data in dashboard["by_content_type"].items():
            print(f"{ctype:<20} {data['savings_pct']:>7.1f}% {data['tokens_saved']:>10,} {data['count']:>8}")
        
        print(f"\nBy Provider:")
        for prov, data in dashboard["by_provider"].items():
            print(f"  {prov}: {data['savings_pct']}% savings ({data['count']} requests)")
        
        print(f"\nTop Transforms:")
        for transform, count in list(dashboard["top_transforms"].items())[:5]:
            print(f"  {transform}: {count}x")


if __name__ == "__main__":
    main()

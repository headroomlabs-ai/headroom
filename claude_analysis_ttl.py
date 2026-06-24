#!/usr/bin/env python3
"""Cache reconstruction cost: cache_creation on first turn after idle gap vs in-window.

Prints a pretty distribution table plus a final cost-comparison summary across three
caching strategies: current (5m default), naive flip to 1h, and conditional 1h-after-idle.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECTS = Path.home() / ".claude" / "projects"

# $ per million tokens.
PRICING = {
    "claude-sonnet-4-6": {"w5": 3.75, "w1h": 6.00, "r": 0.30, "in": 3.00},
    "claude-opus-4-6": {"w5": 6.25, "w1h": 10.00, "r": 0.50, "in": 5.00},
    "claude-haiku-4-5": {"w5": 1.25, "w1h": 2.00, "r": 0.10, "in": 1.00},
    "claude-opus-4-7": {"w5": 18.75, "w1h": 30.00, "r": 1.50, "in": 15.00},
}
DEFAULT_PRICE = PRICING["claude-sonnet-4-6"]
unknown_models = set()


def price_for(model: str) -> dict[str, float]:
    if not model:
        return DEFAULT_PRICE
    if model in PRICING:
        return PRICING[model]
    base = model.split("[")[0]
    for k in PRICING:
        if base.startswith(k) or k in base:
            return PRICING[k]
    unknown_models.add(model)
    return DEFAULT_PRICE


def parse_ts(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def turns_of(path: Path, seen_ids: set[str]) -> list[tuple[datetime, int, int, int, str]]:
    """Parse one JSONL. Skip turns whose message.id was already counted globally."""
    out = []
    with path.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue
            usage = msg.get("usage")
            if not isinstance(usage, dict):
                continue
            try:
                ts = parse_ts(obj["timestamp"])
            except Exception:
                continue
            mid = msg.get("id")
            if mid:
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
            cc = usage.get("cache_creation_input_tokens", 0) or 0
            cr = usage.get("cache_read_input_tokens", 0) or 0
            inp = usage.get("input_tokens", 0) or 0
            model = msg.get("model") or usage.get("model") or ""
            out.append((ts, cc, cr, inp, model))
    out.sort(key=lambda x: x[0])
    return out


BUCKET_ORDER = ["<5min", "5-15min", "15-30min", "30-60min", "1-4hr", ">4hr"]


def bucket(g: float) -> str:
    if g < 5:
        return "<5min"
    if g < 15:
        return "5-15min"
    if g < 30:
        return "15-30min"
    if g < 60:
        return "30-60min"
    if g < 240:
        return "1-4hr"
    return ">4hr"


def main() -> None:
    buckets: dict[str, list[int]] = {b: [] for b in BUCKET_ORDER}
    bucket_by_model: dict[str, dict[str, int]] = {b: defaultdict(int) for b in BUCKET_ORDER}
    total_sessions = 0
    first_turn_tokens_by_model: dict[str, int] = defaultdict(int)

    seen_ids: set[str] = set()
    for path in sorted(PROJECTS.rglob("*.jsonl")):  # sort for deterministic dedupe winner
        try:
            t = turns_of(path, seen_ids)
        except Exception:
            continue
        if len(t) < 2:
            continue
        total_sessions += 1
        # First turn of session: no prior, but the cache_creation IS a fresh write.
        ts0, cc0, _, _, m0 = t[0]
        first_turn_tokens_by_model[m0] += cc0
        for i in range(1, len(t)):
            gap = (t[i][0] - t[i - 1][0]).total_seconds() / 60.0
            if gap < 0:
                continue
            cc = t[i][1]
            m = t[i][4]
            b = bucket(gap)
            buckets[b].append(cc)
            bucket_by_model[b][m] += cc

    def stats(lst: list[int]) -> dict[str, int] | None:
        if not lst:
            return None
        s = sorted(lst)
        n = len(s)
        return {
            "n": n,
            "min": s[0],
            "p25": s[n // 4],
            "median": s[n // 2],
            "p75": s[3 * n // 4],
            "p95": s[min(n - 1, int(n * 0.95))],
            "max": s[-1],
            "mean": sum(s) // n,
            "total": sum(s),
        }

    # ----- Pretty distribution table -----
    print()
    print("=" * 88)
    print(f"  CACHE RECONSTRUCTION COST  —  {total_sessions:,} sessions analyzed")
    print("=" * 88)
    print()
    print("  cache_creation tokens, bucketed by gap since previous turn")
    print()
    header = f"  {'bucket':<10} {'count':>8} {'median':>12} {'mean':>12} {'p75':>12} {'p95':>12} {'total':>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b in BUCKET_ORDER:
        st = stats(buckets[b])
        if st:
            print(
                f"  {b:<10} {st['n']:>8,} {st['median']:>12,} {st['mean']:>12,} "
                f"{st['p75']:>12,} {st['p95']:>12,} {st['total']:>16,}"
            )
    print()

    # ----- Smoking-gun ratios -----
    in_window = buckets["<5min"]
    post_idle_short = buckets["5-15min"]
    post_idle_5to60 = buckets["5-15min"] + buckets["15-30min"] + buckets["30-60min"]

    med_in = sorted(in_window)[len(in_window) // 2] if in_window else 0
    med_5_15 = sorted(post_idle_short)[len(post_idle_short) // 2] if post_idle_short else 0
    med_5_60 = sorted(post_idle_5to60)[len(post_idle_5to60) // 2] if post_idle_5to60 else 0

    print("-" * 88)
    print("  RECONSTRUCTION RATIO  —  the smoking gun")
    print("-" * 88)
    print(f"  Median in-window write (<5min gap)   : {med_in:>10,} tokens")
    print(
        f"  Median post-idle write (5-15min gap) : {med_5_15:>10,} tokens   "
        f"({med_5_15 / max(med_in, 1):>5.0f}x)"
    )
    print(
        f"  Median post-idle write (5-60min gap) : {med_5_60:>10,} tokens   "
        f"({med_5_60 / max(med_in, 1):>5.0f}x)"
    )
    print()

    # ----- Cost comparison across strategies -----
    # Strategy A — current: all writes at 5m price.
    #   cost_A = sum_m (in_window_m + post_idle_5to60_m + post_idle_over60_m) * w5
    # Strategy B — naive 1h: every write becomes a 1h write; 5-60min rewrites flip to reads.
    #   cost_B = sum_m [(in_window_m + post_idle_over60_m) * w1h + post_idle_5to60_m * r]
    # Strategy C — conditional 1h-after-idle: write 5m on in-window deltas, write 1h
    #             only on first turn after >=5min idle. Then the 5-60min rewrites become
    #             reads on the *next* gap event (they already are, post-write), and the
    #             >60min rewrites still cost a 1h write (they expired even the 1h cache).
    #   cost_C = sum_m [in_window_m * w5 + post_idle_5to60_m * r + post_idle_over60_m * w1h]
    #
    # NOTE: Strategy C model assumes the post-idle rewrite events we measured today would
    # become reads under conditional-1h. That's accurate for gaps in [5min, 60min) because
    # the previous turn (now written at 1h) is still cached when the next turn arrives.

    def cost(tok: int, ppm: float) -> float:
        return tok * ppm / 1_000_000.0

    # Aggregate per-model token totals.
    by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"in": 0, "p_5to60": 0, "p_over60": 0, "first": 0}
    )
    for b in BUCKET_ORDER:
        for m, tok in bucket_by_model[b].items():
            if b == "<5min":
                by_model[m]["in"] += tok
            elif b in ("5-15min", "15-30min", "30-60min"):
                by_model[m]["p_5to60"] += tok
            else:
                by_model[m]["p_over60"] += tok
    for m, tok in first_turn_tokens_by_model.items():
        # First turn of a session is a fresh write; treat it as a >5min "post-idle"
        # since there's no prior to refresh. Conservative: bucket as p_over60 so
        # conditional-1h pays 1h for it too.
        by_model[m]["p_over60"] += tok

    rows: list[tuple[str, dict[str, int], float, float, dict[str, float]]] = []
    A_total = 0.0
    B_total = 0.0
    for m, agg in by_model.items():
        p = price_for(m)
        A = (
            cost(agg["in"], p["w5"])
            + cost(agg["p_5to60"], p["w5"])
            + cost(agg["p_over60"], p["w5"])
        )
        B = (
            cost(agg["in"], p["w1h"])
            + cost(agg["p_5to60"], p["r"])
            + cost(agg["p_over60"], p["w1h"])
        )
        A_total += A
        B_total += B
        rows.append((m, agg, A, B, p))

    print("-" * 88)
    print("  COST COMPARISON  —  two caching strategies")
    print("-" * 88)
    print()
    print("  Strategies:")
    print("    A) Current  —  all cache writes at 5m TTL")
    print("    B) Naive 1h —  flip default: all writes at 1h TTL; 5-60min rewrites become reads")
    print()

    # simpler totals
    tot_in = sum(a["in"] for a in by_model.values())
    tot_5to60 = sum(a["p_5to60"] for a in by_model.values())
    tot_over60 = sum(a["p_over60"] for a in by_model.values())
    grand = tot_in + tot_5to60 + tot_over60

    print()
    print(f"  {'category':<40} {'tokens':>16} {'% of total':>12}")
    print("  " + "-" * 70)
    print(f"  {'in-window deltas (<5min)':<40} {tot_in:>16,} {tot_in / grand * 100:>11.1f}%")
    print(
        f"  {'avoidable rewrites (5-60min idle)':<40} {tot_5to60:>16,} {tot_5to60 / grand * 100:>11.1f}%"
    )
    print(
        f"  {'unavoidable rewrites (>60min + first)':<40} {tot_over60:>16,} {tot_over60 / grand * 100:>11.1f}%"
    )
    print(f"  {'TOTAL cache_creation':<40} {grand:>16,} {100.0:>11.1f}%")
    print()

    # Per-model cost rows
    print(f"  {'model':<22} {'A: current 5m':>15} {'B: naive 1h':>15}  {'B vs A':>10}")
    print("  " + "-" * 70)
    for m, _agg, A, B, _p in sorted(rows, key=lambda r: -r[2]):
        name = m or "<unknown>"
        if len(name) > 22:
            name = name[:21] + "…"
        dB = B - A
        print(f"  {name:<22} ${A:>13,.2f} ${B:>13,.2f}  ${dB:>+9,.2f}")
    print("  " + "-" * 70)
    dB_total = B_total - A_total
    print(f"  {'TOTAL':<22} ${A_total:>13,.2f} ${B_total:>13,.2f}  ${dB_total:>+9,.2f}")
    print()

    # ----- Hypothetical: same token mix priced as if 100% Sonnet vs 100% Opus -----
    print("-" * 88)
    print("  HYPOTHETICAL  —  same token mix, all on one model")
    print("-" * 88)
    print()
    print("  Re-prices the observed cache_creation token mix as if every token had")
    print("  been written by a single model. Lets you compare TTL impact at each tier.")
    print()
    hypos = [
        ("All Sonnet 4.6", PRICING["claude-sonnet-4-6"]),
        ("All Opus 4.7", PRICING["claude-opus-4-7"]),
    ]
    print(
        f"  {'scenario':<18} {'A: current 5m':>15} {'B: naive 1h':>15}  {'B vs A':>10}  {'B vs A %':>10}"
    )
    print("  " + "-" * 80)
    for name, p in hypos:
        A = cost(tot_in, p["w5"]) + cost(tot_5to60, p["w5"]) + cost(tot_over60, p["w5"])
        B = cost(tot_in, p["w1h"]) + cost(tot_5to60, p["r"]) + cost(tot_over60, p["w1h"])
        d = B - A
        pct = (d / A * 100) if A else 0
        print(f"  {name:<18} ${A:>13,.2f} ${B:>13,.2f}  ${d:>+9,.2f}  {pct:>+9.1f}%")
    print()

    # ----- Bottom line -----
    print("=" * 88)
    print("  BOTTOM LINE")
    print("=" * 88)
    print()
    print(f"  Sample: {total_sessions:,} sessions, {grand:,} total cache_creation tokens")
    print()
    print(f"  A) Current 5m default        : ${A_total:>10,.2f}   (baseline)")
    sign_B = "+" if dB_total >= 0 else "-"
    print(
        f"  B) Naive flip to 1h          : ${B_total:>10,.2f}   ({sign_B}${abs(dB_total):,.2f} vs current)"
    )
    print()
    if dB_total > 0:
        print(
            f"  Verdict: naive flip COSTS MORE because the 1.6x premium on {tot_in / grand * 100:.0f}% of tokens"
        )
        print(
            f"           (in-window deltas) exceeds savings on {tot_5to60 / grand * 100:.0f}% (post-idle rewrites)."
        )
    elif dB_total < 0:
        print(f"  Verdict: naive 1h flip saves ${abs(dB_total):,.2f} on this sample.")
    else:
        print("  Verdict: 1h flip is cost-neutral on this sample.")
    print()
    if unknown_models:
        print(f"  Note: unknown models defaulted to Sonnet pricing: {sorted(unknown_models)}")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired report for the code-mode steering A/B (benchmarks/codemode_ab.py)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict


def paired(rows, key):
    """Per-(task,rep) steered-minus-control deltas for a numeric field."""
    by = defaultdict(dict)
    for r in rows:
        if r.get("error") or r.get(key) is None:
            continue
        by[(r["task"], r.get("rep"))][r["arm"]] = r[key]
    return [
        (k, v["steered"] - v["control"], v["control"], v["steered"])
        for k, v in by.items()
        if "steered" in v and "control" in v
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    rows = json.loads(open(args.results).read())
    # Re-grade stored answers with the current scorer so a grading fix does not
    # require re-running (and re-paying for) the suite.
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from codemode_ab import TASKS, TASKS_MULTI, score

    spec = {t[0]: (t[2], t[3]) for t in list(TASKS) + list(TASKS_MULTI)}
    for r in rows:
        if r.get("task") in spec and "answer" in r:
            truth, kind = spec[r["task"]]
            r["correct"] = score(kind, truth, r["answer"])
    errs = [r for r in rows if r.get("error")]
    rows = [r for r in rows if not r.get("error")]
    print(f"runs: {len(rows)}  errors: {len(errs)}")

    for arm in ("control", "steered"):
        a = [r for r in rows if r["arm"] == arm]
        if not a:
            continue
        ok = sum(1 for r in a if r.get("correct"))
        print(f"\n[{arm}] n={len(a)}  correct={ok}/{len(a)} ({100 * ok / len(a):.0f}%)")
        for f, label, unit in (
            ("cost_usd", "cost", "$"),
            ("num_turns", "turns", ""),
            ("tool_calls", "tool calls", ""),
            ("bash_calls", "bash calls", ""),
            ("compound_bash", "compound bash", ""),
            ("fetched_chars", "bytes fetched", ""),
            ("wall_s", "wall", "s"),
        ):
            vals = [r.get(f) or 0 for r in a]
            print(
                f"   {label:16s} mean {unit}{statistics.mean(vals):>10.4f}   "
                f"median {unit}{statistics.median(vals):>10.4f}   total {unit}{sum(vals):>12.2f}"
            )

    print("\n=== PAIRED (steered - control), per task ===")
    for f, label in (
        ("cost_usd", "cost $"),
        ("num_turns", "turns"),
        ("tool_calls", "tool calls"),
        ("fetched_chars", "bytes fetched"),
        ("compound_bash", "compound bash"),
    ):
        d = paired(rows, f)
        if not d:
            continue
        deltas = [x[1] for x in d]
        wins = sum(1 for x in deltas if x < 0)
        losses = sum(1 for x in deltas if x > 0)
        mean = statistics.mean(deltas)
        ctrl_tot = sum(x[2] for x in d)
        pct = 100 * sum(deltas) / ctrl_tot if ctrl_tot else 0
        line = (
            f"  {label:16s} mean Δ {mean:+12.4f}   total Δ {sum(deltas):+12.4f} "
            f"({pct:+.1f}%)   steered better/worse/tie: {wins}/{losses}/{len(deltas) - wins - losses}"
        )
        if len(deltas) > 1:
            sd = statistics.stdev(deltas)
            se = sd / (len(deltas) ** 0.5)
            line += f"   95%CI [{mean - 1.96 * se:+.4f}, {mean + 1.96 * se:+.4f}]"
        print(line)

    print("\n=== per-task cost detail ===")
    d = paired(rows, "cost_usd")
    print(f"  {'task':24s} {'control':>10s} {'steered':>10s} {'delta':>10s}  {'Δ%':>7s}")
    for (task, _rep), delta, c, s in sorted(d):
        print(f"  {task:24s} {c:10.4f} {s:10.4f} {delta:+10.4f}  {100 * delta / c:+6.1f}%")

    d = paired(rows, "correct")
    regress = [k for k, dd, c, s in d if c and not s]
    if regress:
        print(f"\n  !! correctness REGRESSED on: {regress}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

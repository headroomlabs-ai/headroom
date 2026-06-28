#!/usr/bin/env python
"""Headroom accuracy eval — prove token savings DON'T cost accuracy.

For each dataset row we answer the question twice — once straight to Anthropic
(baseline) and once through a running `headroom proxy` (compressed input + CCR
retrieve tool) — then an LLM-as-judge grades both against ground truth.

Headline: input-token reduction, accuracy with/without Headroom, and the
accuracy-retention ratio (acc_headroom / acc_baseline).

Usage
  # auto-start a proxy on :8788, eval 6 rows (quick smoke):
  python benchmarks/hf_pilot/eval_accuracy.py --start-proxy --limit 6

  # full 30 rows, route the first 6 to Sonnet, against an already-running proxy:
  python benchmarks/hf_pilot/eval_accuracy.py --proxy-url http://localhost:8788 --sonnet-sample 6

  # from the published HF dataset:
  python benchmarks/hf_pilot/eval_accuracy.py --start-proxy --source hf --repo chopratejas/headroom-pilot
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks.hf_pilot.judge import JUDGE_MODEL, judge_answer, needle_match  # noqa: E402

LOCAL_JSONL = Path(__file__).resolve().parent / "data" / "headroom_pilot.jsonl"
HF_REPO = "chopratejas/headroom-datasets"
PRICES = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
}
_DEFAULT_PRICE = {"in": 1.00, "out": 5.00}


def _load_env() -> None:
    env = _REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


def _load_rows(source: str, repo: str, limit: int | None, categories: set[str] | None) -> list[dict]:
    if source == "hf":
        from datasets import load_dataset

        rows = [dict(r) for r in load_dataset(repo, split="train")]
    else:
        rows = [json.loads(x) for x in LOCAL_JSONL.read_text().splitlines() if x.strip()]
    if categories:
        rows = [r for r in rows if r["category"] in categories]
    return rows[:limit] if limit else rows


def _wait_health(url: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in ("/health", "/"):
            try:
                with urllib.request.urlopen(url + path, timeout=3) as r:  # noqa: S310
                    if r.status < 500:
                        return True
            except Exception:
                pass
        time.sleep(1.0)
    return False


def _start_proxy(port: int, kompress: bool, target_ratio: float | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    if not kompress:
        env["HEADROOM_DISABLE_KOMPRESS"] = "1"
    if target_ratio is not None:
        # Drives Kompress (text/prose) aggressiveness; default is conservative.
        env["HEADROOM_TARGET_RATIO"] = str(target_ratio)
    # token mode (default) compresses aggressively; that's what we want to stress-test.
    headroom_bin = Path(sys.executable).parent / "headroom"
    cmd = (
        [str(headroom_bin), "proxy", "--port", str(port), "--workers", "1"]
        if headroom_bin.exists()
        else [sys.executable, "-m", "headroom.cli.main", "proxy", "--port", str(port), "--workers", "1"]
    )
    proc = subprocess.Popen(
        cmd, cwd=str(_REPO), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return proc


def _client(base_url: str | None, timeout: float = 180.0):  # noqa: ANN202
    from anthropic import Anthropic

    kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": 2}
    if base_url:
        kwargs["base_url"] = base_url
    return Anthropic(**kwargs)


def _answer(client, model: str, req: dict, messages: list[dict], session_id: str) -> dict:  # noqa: ANN001
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        system=req.get("system") or "",
        tools=req.get("tools") or [],
        messages=messages,
        max_tokens=512,
        # Unique session per row: rows share a system prompt, so without this the
        # proxy hashes (model+system) to ONE session id and its prefix-cache
        # freezing suppresses compression on every row after the first few.
        extra_headers={"x-headroom-session-id": session_id},
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return {
        "in": resp.usage.input_tokens,
        "out": resp.usage.output_tokens,
        "latency": round(time.time() - t0, 2),
        "text": text,
    }


def _price(model: str) -> dict[str, float]:
    return PRICES.get(model, _DEFAULT_PRICE)


def run(args: argparse.Namespace) -> None:
    _load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set (checked .env and env).")

    cats = set(args.categories.split(",")) if args.categories else None
    rows = _load_rows(args.source, args.repo, args.limit, cats)

    proxy_proc = None
    proxy_url = args.proxy_url
    if args.start_proxy:
        print(f"Starting headroom proxy on :{args.port} (kompress={'off' if args.no_kompress else 'on'}) ...")
        proxy_proc = _start_proxy(args.port, kompress=not args.no_kompress, target_ratio=args.target_ratio)
        proxy_url = f"http://localhost:{args.port}"
        if not _wait_health(proxy_url):
            proxy_proc.terminate()
            sys.exit("proxy did not become healthy in time")
        print("proxy healthy.")

    base_client = _client(None)
    hr_client = _client(proxy_url)
    judge_client = _client(None)  # judge always uncompressed, direct

    records: list[dict[str, Any]] = []
    try:
        for i, row in enumerate(rows):
            req = json.loads(row["request_json"])
            messages = req["messages"]
            model = args.model or row.get("model")
            if args.sonnet_sample and i < args.sonnet_sample:
                model = "claude-sonnet-4-6"
            needles = row["expected_answer_contains"]
            ref = row["reference_answer"]
            rec: dict[str, Any] = {"id": row["id"], "category": row["category"], "model": model}

            for path, client in (("baseline", base_client), ("headroom", hr_client)):
                try:
                    a = _answer(client, model, req, messages, session_id=f"{row['id']}-{path}")
                    v = judge_answer(judge_client, row["task"], ref, a["text"],
                                     model=args.judge_model, votes=args.judge_votes)
                    pr = _price(model)
                    rec[path] = {
                        "in": a["in"], "out": a["out"], "latency": a["latency"],
                        "cost": round(a["in"] / 1e6 * pr["in"] + a["out"] / 1e6 * pr["out"], 5),
                        "correct": v["correct"], "score": v["score"],
                        "needle": needle_match(a["text"], needles),
                        "reasoning": v["reasoning"], "answer": a["text"][:200],
                    }
                except Exception as e:  # noqa: BLE001
                    rec[path] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            records.append(rec)
            b, h = rec.get("baseline", {}), rec.get("headroom", {})
            print(f"[{i + 1}/{len(rows)}] {row['id']:<20} "
                  f"base in={b.get('in', '?')} ok={b.get('correct', '?')} | "
                  f"hr in={h.get('in', '?')} ok={h.get('correct', '?')}")
    finally:
        if proxy_proc is not None:
            proxy_proc.terminate()
            try:
                proxy_proc.wait(timeout=10)
            except Exception:
                proxy_proc.kill()

    _report(records, args)


def _report(records: list[dict], args: argparse.Namespace) -> None:
    ok = [r for r in records if "in" in r.get("baseline", {}) and "in" in r.get("headroom", {})]
    n = len(ok)
    if not n:
        print("\nNo successful paired rows to report.")
        return
    b_in = sum(r["baseline"]["in"] for r in ok)
    h_in = sum(r["headroom"]["in"] for r in ok)
    b_correct = sum(1 for r in ok if r["baseline"]["correct"])
    h_correct = sum(1 for r in ok if r["headroom"]["correct"])
    b_cost = sum(r["baseline"]["cost"] for r in ok)
    h_cost = sum(r["headroom"]["cost"] for r in ok)
    agree = sum(1 for r in ok if r["baseline"]["correct"] == r["headroom"]["correct"])

    acc_b = b_correct / n
    acc_h = h_correct / n
    retention = (acc_h / acc_b) if acc_b else float("nan")
    reduction = 100 * (1 - h_in / b_in) if b_in else 0.0

    print("\n" + "=" * 76)
    print("HEADROOM ACCURACY EVAL")
    print("=" * 76)
    print(f"rows judged: {n}   judge: {args.judge_model} (votes={args.judge_votes})")
    print(f"\n  {'metric':<34}{'baseline':>14}{'headroom':>14}")
    print(f"  {'input tokens (sum)':<34}{b_in:>14,}{h_in:>14,}")
    print(f"  {'accuracy (judge)':<34}{acc_b:>13.1%}{acc_h:>14.1%}")
    print(f"  {'cost (est, $)':<34}{b_cost:>14.4f}{h_cost:>14.4f}")
    print("\n  HEADLINE")
    print(f"    input-token reduction : {reduction:.1f}%")
    print(f"    accuracy retention    : {retention:.3f}  (acc_headroom / acc_baseline)")
    print(f"    answers agree (b==h)  : {agree}/{n}")
    print(f"    cost saved on run     : ${b_cost - h_cost:.4f}")

    gate_ok = retention >= 0.95 and reduction >= 30.0
    print(f"\n  GATE (retention>=0.95 AND reduction>=30%): {'PASS ✅' if gate_ok else 'REVIEW ⚠️'}")

    # per category
    cats: dict[str, list[dict]] = {}
    for r in ok:
        cats.setdefault(r["category"], []).append(r)
    print(f"\n  {'category':<14}{'rows':>5}{'tok_reduction':>15}{'acc_base':>10}{'acc_hr':>9}")
    for cat, rs in sorted(cats.items()):
        bi = sum(r["baseline"]["in"] for r in rs)
        hi = sum(r["headroom"]["in"] for r in rs)
        red = 100 * (1 - hi / bi) if bi else 0.0
        ab = sum(1 for r in rs if r["baseline"]["correct"]) / len(rs)
        ah = sum(1 for r in rs if r["headroom"]["correct"]) / len(rs)
        print(f"  {cat:<14}{len(rs):>5}{red:>14.1f}%{ab:>10.0%}{ah:>9.0%}")

    out = Path(__file__).resolve().parent / "report_accuracy.json"
    out.write_text(json.dumps({"args": vars(args), "records": records}, indent=2))
    print(f"\nFull report -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Headroom accuracy eval (with/without proxy + LLM judge)")
    p.add_argument("--start-proxy", action="store_true", help="spawn a headroom proxy and tear it down")
    p.add_argument("--proxy-url", default="http://localhost:8788")
    p.add_argument("--port", type=int, default=8788)
    p.add_argument("--no-kompress", action="store_true", help="start proxy with Kompress disabled")
    p.add_argument("--target-ratio", type=float, default=None,
                   help="proxy Kompress target_ratio (e.g. 0.4); lower = more prose compression")
    p.add_argument("--source", choices=["local", "hf"], default="local")
    p.add_argument("--repo", default=HF_REPO)
    p.add_argument("--model", default=None, help="override generator model for all rows")
    p.add_argument("--sonnet-sample", type=int, default=0, help="route first N rows to Sonnet")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--judge-votes", type=int, default=1, help="self-consistency votes per judgment")
    p.add_argument("--categories", default=None)
    p.add_argument("--limit", type=int, default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()

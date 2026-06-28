#!/usr/bin/env python
"""Headroom pilot benchmark — prove context-compression value with/without Headroom.

Loads the pilot dataset (local JSONL or the HF hub), then for each row runs up to
three modes and reports input-token savings, cost, latency, and — crucially —
whether the answer is preserved (the embedded needle appears in Claude's reply).

Modes
  raw               : send the payload unchanged
  headroom          : Headroom structural compression (SmartCrusher etc.), Kompress OFF
  headroom_kompress : Headroom + Kompress ML model (chopratejas/kompress-v2-base)

Usage
  # FREE — compression accounting + needle survival, no API spend:
  python benchmarks/hf_pilot/benchmark.py --dry-run

  # LIVE — also call the Claude API (spends Anthropic budget):
  python benchmarks/hf_pilot/benchmark.py --live --model claude-haiku-4-5-20251001
  python benchmarks/hf_pilot/benchmark.py --live --sonnet-sample 6   # + a few on Sonnet

  # From the published HF dataset instead of the local file:
  python benchmarks/hf_pilot/benchmark.py --dry-run --source hf --repo chopratejas/headroom-pilot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from headroom import CompressConfig, compress  # noqa: E402

LOCAL_JSONL = Path(__file__).resolve().parent / "data" / "headroom_pilot.jsonl"
HF_REPO = "chopratejas/headroom-datasets"

# Estimated list prices (USD per 1M tokens), June 2026. Clearly an ESTIMATE —
# the harness reports token counts first; cost is a labeled derived figure.
PRICES = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-sonnet-4-5-20250929": {"in": 3.00, "out": 15.00},
}
_DEFAULT_PRICE = {"in": 1.00, "out": 5.00}


def _load_env() -> None:
    env = _REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


def _compress_config(mode: str) -> CompressConfig:
    """Demo config: compress the large tool_results (incl. the recent ones).

    ``headroom``           -> Kompress OFF (structural only: SmartCrusher/Log/Code).
    ``headroom_kompress``  -> Kompress ON with an aggressive target_ratio so the ML
                              text path engages on prose (default None is ~28%).
    ``target_ratio`` only affects the Kompress (text) path; JSON/log/code use their
    own logic, so the two modes differ only on prose-heavy content.
    """
    return CompressConfig(
        compress_user_messages=True,
        compress_system_messages=False,
        protect_recent=0,
        kompress_model="disabled" if mode == "headroom" else None,
        target_ratio=None if mode == "headroom" else 0.4,
    )


def _flatten_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                if "text" in b and isinstance(b["text"], str):
                    parts.append(b["text"])
                inner = b.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
                elif isinstance(inner, list):
                    parts.extend(x.get("text", "") for x in inner if isinstance(x, dict))
                if isinstance(b.get("input"), dict):
                    parts.append(json.dumps(b["input"]))
    return "\n".join(parts)


def _needles_present(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return all(n.lower() in low for n in needles)


def _load_rows(source: str, repo: str, limit: int | None, categories: set[str] | None) -> list[dict]:
    if source == "hf":
        from datasets import load_dataset

        ds = load_dataset(repo, split="train")
        rows = [dict(r) for r in ds]
    else:
        rows = [json.loads(line) for line in LOCAL_JSONL.read_text().splitlines() if line.strip()]
    if categories:
        rows = [r for r in rows if r["category"] in categories]
    if limit:
        rows = rows[:limit]
    return rows


def _price(model: str) -> dict[str, float]:
    return PRICES.get(model, _DEFAULT_PRICE)


def _call_claude(client: Any, model: str, req: dict, messages: list[dict]) -> dict:
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        system=req.get("system") or "",
        tools=req.get("tools") or [],
        messages=messages,
        max_tokens=512,
    )
    dt = time.time() - t0
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return {
        "in_tokens": resp.usage.input_tokens,
        "out_tokens": resp.usage.output_tokens,
        "latency_s": round(dt, 2),
        "text": text,
    }


def run(args: argparse.Namespace) -> None:
    _load_env()
    cats = set(args.categories.split(",")) if args.categories else None
    rows = _load_rows(args.source, args.repo, args.limit, cats)
    modes = ["raw", "headroom", "headroom_kompress"]
    if args.no_kompress:
        modes.remove("headroom_kompress")

    client = None
    if args.live:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY not set (checked .env and env).")
        from anthropic import Anthropic

        client = Anthropic()

    # per (mode) aggregates
    agg: dict[str, dict[str, float]] = {m: _zero() for m in modes}
    per_cat: dict[str, dict[str, dict[str, float]]] = {}
    records: list[dict[str, Any]] = []

    for ri, row in enumerate(rows):
        req = json.loads(row["request_json"])
        base_messages = req["messages"]
        needles = row["expected_answer_contains"]
        model = args.model or row.get("model")
        # optionally route some rows to Sonnet for the live run
        if args.live and args.sonnet_sample and ri < args.sonnet_sample:
            model = "claude-sonnet-4-6"

        rec: dict[str, Any] = {"id": row["id"], "category": row["category"], "model": model, "modes": {}}
        for mode in modes:
            if mode == "raw":
                msgs = base_messages
                comp = None
            else:
                comp = compress(base_messages, model=model, config=_compress_config(mode))
                msgs = comp.messages

            tok_in_est = comp.tokens_after if comp else (_raw_tokens(base_messages, model))
            entry: dict[str, Any] = {
                "est_input_tokens": tok_in_est,
                "needle_in_prompt": _needles_present(_flatten_text(msgs), needles),
            }
            if comp is not None:
                entry["tokens_before"] = comp.tokens_before
                entry["tokens_after"] = comp.tokens_after
                entry["pct_saved"] = round(100 * comp.compression_ratio, 1)
                entry["transforms"] = comp.transforms_applied

            if args.live:
                try:
                    out = _call_claude(client, model, req, msgs)
                    entry.update(
                        api_in=out["in_tokens"],
                        api_out=out["out_tokens"],
                        latency_s=out["latency_s"],
                        answer_ok=_needles_present(out["text"], needles),
                        answer=out["text"][:240],
                    )
                    pr = _price(model)
                    entry["cost_usd"] = round(
                        out["in_tokens"] / 1e6 * pr["in"] + out["out_tokens"] / 1e6 * pr["out"], 5
                    )
                except Exception as e:  # noqa: BLE001
                    entry["error"] = f"{type(e).__name__}: {str(e)[:160]}"

            _accumulate(agg[mode], entry)
            per_cat.setdefault(row["category"], {}).setdefault(mode, _zero())
            _accumulate(per_cat[row["category"]][mode], entry)
            rec["modes"][mode] = entry
        records.append(rec)
        print(f"[{ri + 1}/{len(rows)}] {row['id']:<20} done")

    _report(agg, per_cat, modes, args)
    out_path = Path(__file__).resolve().parent / ("report_live.json" if args.live else "report_dry.json")
    out_path.write_text(json.dumps({"args": vars(args), "records": records}, indent=2))
    print(f"\nFull report -> {out_path}")


def _zero() -> dict[str, float]:
    return {"n": 0, "tokens_before": 0, "tokens_after": 0, "needle_kept": 0,
            "api_in": 0, "api_out": 0, "answer_ok": 0, "cost_usd": 0.0, "lat": 0.0, "live_n": 0}


def _accumulate(a: dict[str, float], e: dict[str, Any]) -> None:
    a["n"] += 1
    a["tokens_before"] += e.get("tokens_before", e.get("est_input_tokens", 0))
    a["tokens_after"] += e.get("tokens_after", e.get("est_input_tokens", 0))
    a["needle_kept"] += 1 if e.get("needle_in_prompt") else 0
    if "api_in" in e:
        a["live_n"] += 1
        a["api_in"] += e["api_in"]
        a["api_out"] += e["api_out"]
        a["answer_ok"] += 1 if e.get("answer_ok") else 0
        a["cost_usd"] += e.get("cost_usd", 0.0)
        a["lat"] += e.get("latency_s", 0.0)


def _raw_tokens(messages: list[dict], model: str) -> int:
    try:
        from headroom.tokenizers import get_tokenizer

        return int(get_tokenizer(model).count_messages(messages))
    except Exception:
        return len(json.dumps(messages)) // 4


def _report(agg: dict, per_cat: dict, modes: list[str], args: argparse.Namespace) -> None:
    print("\n" + "=" * 78)
    print("HEADROOM PILOT BENCHMARK" + ("  [LIVE]" if args.live else "  [DRY-RUN — no API spend]"))
    print("=" * 78)
    raw_before = agg["raw"]["tokens_after"]  # raw "after" == raw tokens
    print(f"\nInput tokens (sum over {agg['raw']['n']} rows), vs raw baseline:")
    print(f"  {'mode':<20} {'input_tok':>12} {'vs raw':>10} {'needle kept':>12}")
    for m in modes:
        tok = agg[m]["tokens_after"]
        vs = "—" if m == "raw" else f"-{100 * (1 - tok / raw_before):.1f}%"
        nk = f"{int(agg[m]['needle_kept'])}/{int(agg[m]['n'])}"
        print(f"  {m:<20} {tok:>12,} {vs:>10} {nk:>12}")

    if args.live:
        print(f"\nLIVE Claude API results:")
        print(f"  {'mode':<20} {'api_in':>11} {'api_out':>9} {'answer_ok':>10} {'cost_usd':>10} {'avg_lat':>8}")
        for m in modes:
            a = agg[m]
            ln = max(1, int(a["live_n"]))
            ok = f"{int(a['answer_ok'])}/{int(a['live_n'])}"
            print(f"  {m:<20} {int(a['api_in']):>11,} {int(a['api_out']):>9,} {ok:>10} "
                  f"${a['cost_usd']:>9.4f} {a['lat'] / ln:>7.2f}s")
        # headline
        if "headroom" in agg and agg["raw"]["api_in"]:
            base = agg["raw"]["api_in"]
            for m in modes:
                if m == "raw":
                    continue
                saved = 100 * (1 - agg[m]["api_in"] / base)
                cost_saved = agg["raw"]["cost_usd"] - agg[m]["cost_usd"]
                print(f"\n  >> {m}: {saved:.1f}% fewer input tokens vs raw; "
                      f"~${cost_saved:.4f} saved on this run; "
                      f"answers preserved {int(agg[m]['answer_ok'])}/{int(agg[m]['live_n'])}")

    print(f"\nPer-category input-token savings ({'live api_in' if args.live else 'est'}):")
    key = "api_in" if args.live else "tokens_after"
    print(f"  {'category':<14} " + " ".join(f"{m[:16]:>17}" for m in modes))
    for cat, md in sorted(per_cat.items()):
        raw_c = md.get("raw", {}).get(key, 0) or md.get("raw", {}).get("tokens_after", 0)
        cells = []
        for m in modes:
            tok = md.get(m, {}).get(key, 0) or md.get(m, {}).get("tokens_after", 0)
            if m == "raw":
                cells.append(f"{int(tok):>17,}")
            else:
                pct = f"-{100 * (1 - tok / raw_c):.0f}%" if raw_c else "—"
                cells.append(f"{int(tok):>10,} {pct:>6}")
        print(f"  {cat:<14} " + " ".join(cells))


def main() -> None:
    p = argparse.ArgumentParser(description="Headroom pilot benchmark")
    p.add_argument("--live", action="store_true", help="call the Claude API (spends budget)")
    p.add_argument("--dry-run", action="store_true", help="compression accounting only (free)")
    p.add_argument("--source", choices=["local", "hf"], default="local")
    p.add_argument("--repo", default=HF_REPO)
    p.add_argument("--model", default=None, help="override model for all rows")
    p.add_argument("--sonnet-sample", type=int, default=0, help="route first N live rows to Sonnet")
    p.add_argument("--categories", default=None, help="comma-separated category filter")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-kompress", action="store_true", help="skip the Kompress ML mode")
    args = p.parse_args()
    if not args.live:
        args.dry_run = True
    run(args)


if __name__ == "__main__":
    main()

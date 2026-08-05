#!/usr/bin/env python3
"""Quality eval for TextCrusher (Phase 2, #1171): does extractive compression
preserve the answer-bearing content?  No LLM/API calls -- fully local.

Part A -- SQuAD answer-retention (the strong, labeled metric): bury a real QA
answer in a haystack of distractor paragraphs, compress to a target ratio, and
measure whether the gold answer SURVIVES. TextCrusher (query-aware) vs truncate
(keep-recent) vs random baselines. Mirrors kompress's published
must_keep_recall (0.977 on its own labeled set).

Part B -- real-transcript fidelity: compress large text blocks from a real
Claude Code transcript (ANONYMIZED), measuring ratio, speed, and salient-token
retention (identifiers/numbers/errors -- the must-keep info in coding contexts).
Only aggregate metrics are printed; raw content is never echoed.

Usage: python benchmarks/text_crusher_quality_eval.py [squad_dev.json] [transcript.jsonl]
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
import time

from headroom.transforms.text_crusher import TextCrusher

_SEG = re.compile(r"(?<=[.!?])\s+|\n+")
_SALIENT = re.compile(
    r"\b(?:error|exception|fail(?:ed|ure)?|warning|traceback|assert|todo|fixme)\b"
    r"|\b[A-Z]{2,}\b|\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b|\b\d+\b"
)

# --- anonymization (脱敏): scrub before any processing; never echo raw content ---
_REDACT = [
    (re.compile(r"/Users/[^/\s]+"), "/Users/USER"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "EMAIL"),
    (re.compile(r"\b(?:sk|pk|ghp|gho|xox[baprs])-[A-Za-z0-9_-]{10,}\b"), "TOKEN"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._-]{10,}"), "Bearer TOKEN"),
    (re.compile(r"\b[A-Fa-f0-9]{40,}\b"), "HEX"),
]


def anon(t: str) -> str:
    for rx, rep in _REDACT:
        t = rx.sub(rep, t)
    return t


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _segs(text: str) -> list[str]:
    return [s for s in _SEG.split(text) if s.strip()]


def truncate_keep_last(text: str, ratio: float) -> str:
    segs = _segs(text)
    budget = int(sum(len(s) for s in segs) * ratio)
    kept: list[str] = []
    c = 0
    for s in reversed(segs):
        if c >= budget:
            break
        kept.append(s)
        c += len(s)
    return "\n".join(reversed(kept))


def random_keep(text: str, ratio: float, seed: int) -> str:
    segs = _segs(text)
    idx = list(range(len(segs)))
    random.Random(seed).shuffle(idx)
    budget = int(sum(len(s) for s in segs) * ratio)
    kept: set[int] = set()
    c = 0
    for i in idx:
        if c >= budget:
            break
        kept.add(i)
        c += len(segs[i])
    return "\n".join(segs[i] for i in sorted(kept))


def eval_squad(path: str, n: int = 200, n_distract: int = 40, ratio: float = 0.3, seed: int = 0):
    data = json.load(open(path))
    paras = [(p["context"], p["qas"]) for a in data["data"] for p in a["paragraphs"]]
    all_ctx = [c for c, _ in paras]
    examples = [
        (ctx, qas[0]["question"], qas[0]["answers"][0]["text"])
        for ctx, qas in paras
        if qas and qas[0]["answers"]
    ]
    rnd = random.Random(seed)
    rnd.shuffle(examples)
    examples = examples[:n]
    tc = TextCrusher()
    hit = {"text_crusher": 0, "truncate": 0, "random": 0}
    tc_ratios: list[float] = []
    for gold_ctx, q, ans in examples:
        docs = rnd.sample(all_ctx, n_distract) + [gold_ctx]
        rnd.shuffle(docs)
        haystack = "\n\n".join(docs)
        a = norm(ans)
        out_tc = tc.compress(haystack, context=q, target_ratio=ratio).compressed
        hit["text_crusher"] += a in norm(out_tc)
        hit["truncate"] += a in norm(truncate_keep_last(haystack, ratio))
        hit["random"] += a in norm(random_keep(haystack, ratio, seed))
        tc_ratios.append(len(out_tc) / max(1, len(haystack)))
    nn = len(examples)
    print(
        f"\n=== Part A: SQuAD answer-retention (n={nn}, distractors={n_distract}, target_ratio={ratio}) ==="
    )
    print(
        f"  TextCrusher (query-aware): {hit['text_crusher'] / nn:6.1%}  answer survives compression"
    )
    print(f"  Truncate (keep recent):    {hit['truncate'] / nn:6.1%}")
    print(f"  Random keep:               {hit['random'] / nn:6.1%}")
    print(
        f"  TextCrusher mean char-ratio: {sum(tc_ratios) / nn:.2f}  (kept ~{sum(tc_ratios) / nn:.0%} of bytes)"
    )
    print("  reference: kompress published must_keep_recall = 0.977 (its own labeled set)")


def _block_texts(jsonl_path: str, min_words: int, limit: int) -> list[str]:
    out: list[str] = []
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = o.get("message") or {}
            c = m.get("content")
            parts = (
                [c]
                if isinstance(c, str)
                else [
                    p["text"] for p in c if isinstance(p, dict) and isinstance(p.get("text"), str)
                ]
                if isinstance(c, list)
                else []
            )
            for t in parts:
                if len(t.split()) >= min_words:
                    out.append(anon(t))
            if len(out) >= limit:
                break
    return out[:limit]


def eval_transcript(jsonl_path: str, ratio: float = 0.4, min_words: int = 1500, limit: int = 40):
    blocks = _block_texts(jsonl_path, min_words, limit)
    if not blocks:
        print(
            f"\n=== Part B: no text blocks >= {min_words} words in {os.path.basename(jsonl_path)} ==="
        )
        return
    tc = TextCrusher()
    ratios: list[float] = []
    times: list[float] = []
    retentions: list[float] = []
    for b in blocks:
        sal_before = set(_SALIENT.findall(b))
        t0 = time.perf_counter()
        out = tc.compress(b, target_ratio=ratio).compressed
        times.append((time.perf_counter() - t0) * 1000)
        sal_after = set(_SALIENT.findall(out))
        retentions.append(len(sal_before & sal_after) / max(1, len(sal_before)))
        ratios.append(len(out.split()) / max(1, len(b.split())))
    n = len(blocks)
    print(
        f"\n=== Part B: real transcript fidelity (n={n} large blocks, anonymized, target_ratio={ratio}) ==="
    )
    print(f"  mean token-ratio kept:       {sum(ratios) / n:.2f}")
    print(f"  mean speed:                  {sum(times) / n:.1f} ms/block")
    print(
        f"  salient-token retention:     {sum(retentions) / n:6.1%}  (identifiers/numbers/errors kept)"
    )
    print(
        f"  -> keeps salient info at {sum(retentions) / n:.0%} while dropping to {sum(ratios) / n:.0%} of tokens"
    )


def eval_speed(scale_words: int = 250_000):
    # Reproducible throughput on a large synthetic prose block (no external data).
    text = " ".join(
        f"Sentence {i} discusses subsystem {i} and its failure mode {i % 7} in detail."
        for i in range(scale_words // 9)
    )
    nwords = len(text.split())
    tc = TextCrusher()
    t0 = time.perf_counter()
    out = tc.compress(text, target_ratio=0.3)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n=== Part C: speed (synthetic, {nwords:,} words, fully reproducible) ===")
    print(f"  TextCrusher compress:  {ms:.0f} ms  ({nwords / max(ms / 1000, 1e-6):,.0f} words/sec)")
    print(f"  kept ratio:            {out.compressed_tokens / max(1, out.original_tokens):.2f}")
    print("  reference: kompress (ModernBERT ONNX) ~272s for ~1M tokens (measured, query-blind)")
    print("  -> fast-vs-slow CONTRAST, not a same-input side-by-side run")


if __name__ == "__main__":
    eval_speed()
    squad = sys.argv[1] if len(sys.argv) > 1 else "/tmp/squad_dev.json"
    tx = sys.argv[2] if len(sys.argv) > 2 else None
    if os.path.exists(squad):
        eval_squad(squad)
    else:
        print(f"SQuAD not found at {squad}; skipping Part A")
    if tx is None:
        found = glob.glob(os.path.expanduser("~/.claude/projects/*headroom*/*.jsonl"))
        tx = max(found, key=os.path.getsize) if found else None
    if tx and os.path.exists(tx):
        eval_transcript(tx)
    else:
        print("no transcript jsonl found; skipping Part B")

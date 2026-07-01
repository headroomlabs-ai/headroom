#!/usr/bin/env python3
"""i18n compression-quality eval (zh/ja/ko): does extractive compression keep
the answer-bearing content in CJK?  No LLM/API calls -- fully local.

Part C -- our own DETERMINISTIC needle answer-retention (zh/ja/ko): the always-
runs regression gate. A distinctive needle sentence is buried (in the middle) in
language-matched distractor sentences; compress query-aware; assert the needle
survives. No external data. TextCrusher (query-aware) vs truncate (keep-recent)
vs random baselines.

Part B -- real-transcript fidelity with CJK-aware salient: optional, anonymized.

Part A -- natural-data answer-retention on alexandrainst/multi-wiki-qa
(zh-cn/ja/ko): optional, via the [evals] datasets extra, skipped if absent.

Usage: python benchmarks/i18n_compression_eval.py [transcript.jsonl]
"""

from __future__ import annotations

import glob
import os
import random
import re
import sys
import time

from headroom.transforms.text_crusher import TextCrusher

_REDACT = [
    (re.compile(r"/Users/[^/\s]+"), "/Users/USER"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "EMAIL"),
    (re.compile(r"\b(?:sk|pk|ghp|gho|xox[baprs])-[A-Za-z0-9_-]{10,}\b"), "TOKEN"),
    (re.compile(r"\b[A-Fa-f0-9]{40,}\b"), "HEX"),
]
# Split on ASCII and full-width CJK terminators so baselines segment CJK too.
_SEG = re.compile(r"(?<=[.!?。！？])\s*|\n+")
_CJK_RUN = re.compile(r"[㐀-鿿぀-ヿ가-힯]+")


def anon(t: str) -> str:
    for rx, rep in _REDACT:
        t = rx.sub(rep, t)
    return t


def norm(s: str) -> str:
    # CJK has no spaces; drop all whitespace so substring match is robust.
    return re.sub(r"\s+", "", s.lower())


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
    return "".join(reversed(kept))


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
    return "".join(segs[i] for i in sorted(kept))


# --- Part C: deterministic needle retention (zh / ja / ko) ---------------------

# Each needle carries a distinctive verbatim KEY that must survive. Distractors
# are generated (deterministic, distinct, topic-unrelated to the query) so the
# haystack is large enough to FORCE real compression -- the needle only survives
# under TextCrusher because it is query-relevant, not because of passthrough.
_NEEDLES = {
    "zh": {
        "query": "认证令牌缓存淘汰策略",
        "key": "最近最少使用淘汰",
        "needle": "认证令牌的缓存采用最近最少使用淘汰算法来管理过期条目。",
        "distractor": lambda i: f"第{i}号监控服务器的日志显示子系统{i}今天运行平稳没有出现异常。",
    },
    "ja": {
        "query": "認証トークン キャッシュ 破棄 アルゴリズム",
        "key": "最長未使用",
        "needle": "認証トークンのキャッシュは最長未使用アルゴリズムで管理される。",
        "distractor": lambda i: (
            f"{i}番目の監視サーバーのログには{i}番のサブシステムが本日も正常に稼働したと記録されている。"
        ),
    },
    "ko": {
        "query": "인증 토큰 캐시 제거 알고리즘",
        "key": "최근 최소 사용",
        "needle": "인증 토큰 캐시는 최근 최소 사용 알고리즘으로 관리된다.",
        "distractor": lambda i: (
            f"{i}번 모니터링 서버의 로그에는 {i}번 하위 시스템이 오늘도 정상 작동했다고 기록되어 있다."
        ),
    },
}


def _haystack(spec: dict, n_distract: int = 24) -> str:
    half = n_distract // 2
    before = [spec["distractor"](i) for i in range(half)]
    after = [spec["distractor"](i) for i in range(half, n_distract)]
    # needle in the MIDDLE so keep-recent (truncate) reliably misses it.
    return "".join(before + [spec["needle"]] + after)


def retention_synthetic(lang: str, ratio: float = 0.3, seed: int = 0) -> dict[str, bool]:
    spec = _NEEDLES[lang]
    hay = _haystack(spec)
    key = norm(spec["key"])
    tc = TextCrusher()
    out_tc = tc.compress(hay, spec["query"], ratio).compressed
    return {
        "text_crusher": key in norm(out_tc),
        "truncate": key in norm(truncate_keep_last(hay, ratio)),
        "random": key in norm(random_keep(hay, ratio, seed)),
    }


def eval_synthetic(ratio: float = 0.3) -> None:
    print(f"\n=== Part C: synthetic needle retention (zh/ja/ko, target_ratio={ratio}) ===")
    print(f"  {'lang':5} {'text_crusher':>13} {'truncate':>9} {'random':>7}")
    for lang in ("zh", "ja", "ko"):
        r = retention_synthetic(lang, ratio)
        print(
            f"  {lang:5} {str(r['text_crusher']):>13} {str(r['truncate']):>9} {str(r['random']):>7}"
        )
    print("  (needle must survive under TextCrusher; baselines are the contrast)")


# --- Part B: real CJK transcript fidelity (CJK-aware salient) ------------------

# ASCII salient (identifiers/numbers/errors) STILL matters in CJK coding context.
_SALIENT_ASCII = re.compile(
    r"\b(?:error|exception|fail(?:ed|ure)?|warning|traceback|assert|todo|fixme)\b"
    r"|\b[A-Z]{2,}\b|\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b|\b\d+\b"
)


def _cjk_hapax(text: str) -> set[str]:
    # distinctive CJK content = char-bigrams occurring exactly once (rare = must-keep)
    grams: dict[str, int] = {}
    for run in _CJK_RUN.findall(text):
        for i in range(len(run) - 1):
            g = run[i : i + 2]
            grams[g] = grams.get(g, 0) + 1
    return {g for g, c in grams.items() if c == 1}


def salient_set(text: str) -> set[str]:
    return set(_SALIENT_ASCII.findall(text)) | _cjk_hapax(text)


def _block_texts(jsonl_path: str, min_chars: int, limit: int) -> list[str]:
    import json

    out: list[str] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = (o.get("message") or {}).get("content")
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
                if len(t) >= min_chars and _CJK_RUN.search(t):  # CJK-bearing only
                    out.append(anon(t))
            if len(out) >= limit:
                break
    return out[:limit]


def eval_transcript(
    jsonl_path: str, ratio: float = 0.4, min_chars: int = 600, limit: int = 40
) -> None:
    blocks = _block_texts(jsonl_path, min_chars, limit)
    if not blocks:
        print(
            f"\n=== Part B: no CJK blocks >= {min_chars} chars in {os.path.basename(jsonl_path)} ==="
        )
        return
    tc = TextCrusher()
    ratios: list[float] = []
    times: list[float] = []
    retentions: list[float] = []
    for b in blocks:
        sal_before = salient_set(b)
        t0 = time.perf_counter()
        out = tc.compress(b, "", ratio).compressed
        times.append((time.perf_counter() - t0) * 1000)
        retentions.append(len(sal_before & salient_set(out)) / max(1, len(sal_before)))
        ratios.append(len(out) / max(1, len(b)))
    n = len(blocks)
    print(
        f"\n=== Part B: real CJK transcript fidelity (n={n}, anonymized, target_ratio={ratio}) ==="
    )
    print(f"  mean char-ratio kept:        {sum(ratios) / n:.2f}")
    print(f"  mean speed:                  {sum(times) / n:.1f} ms/block")
    print(f"  CJK-aware salient retention: {sum(retentions) / n:.1%}")


# --- Part A: optional natural-data retention (multi-wiki-qa zh/ja/ko) ----------
# Schema verified: row = {id, title, context, question, answers:{text:[...]}}.
# Answers are guaranteed verbatim substrings of the (long) context; CC-BY-NC-SA.


def eval_multiwiki(
    langs=("zh-cn", "ja", "ko"), n: int = 80, ratio: float = 0.3, seed: int = 0
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "\n=== Part A: `datasets` not installed; skipping (pip install headroom-ai[evals]) ==="
        )
        return
    tc = TextCrusher()
    print(f"\n=== Part A: multi-wiki-qa answer-retention (n={n}/lang, target_ratio={ratio}) ===")
    print(f"  {'lang':6} {'text_crusher':>13} {'truncate':>9} {'random':>7}")
    for lang in langs:
        try:
            ds = load_dataset("alexandrainst/multi-wiki-qa", lang, split=f"train[:{n * 2}]")
        except Exception as e:  # noqa: BLE001 -- optional path, fail-open
            print(f"  {lang}: load failed ({e}); skipping")
            continue
        ex = []
        for r in ds:
            ans = r.get("answers")
            a = ans["text"][0] if isinstance(ans, dict) and ans.get("text") else None
            if r.get("context") and r.get("question") and a:
                ex.append((r["context"], r["question"], a))
        random.Random(seed).shuffle(ex)
        ex = ex[:n]
        hit = {"text_crusher": 0, "truncate": 0, "random": 0}
        for ctx, q, ans in ex:
            a = norm(ans)
            hit["text_crusher"] += a in norm(tc.compress(ctx, q, ratio).compressed)
            hit["truncate"] += a in norm(truncate_keep_last(ctx, ratio))
            hit["random"] += a in norm(random_keep(ctx, ratio, seed))
        m = max(1, len(ex))
        print(
            f"  {lang:6} {hit['text_crusher'] / m:>12.0%} {hit['truncate'] / m:>9.0%} {hit['random'] / m:>7.0%}"
        )


if __name__ == "__main__":
    eval_synthetic()
    tx = sys.argv[1] if len(sys.argv) > 1 else None
    if tx is None:
        found = glob.glob(os.path.expanduser("~/.claude/projects/*headroom*/*.jsonl"))
        tx = max(found, key=os.path.getsize) if found else None
    if tx and os.path.exists(tx):
        eval_transcript(tx)
    else:
        print("\nno transcript jsonl found; skipping Part B")
    eval_multiwiki()

#!/usr/bin/env python3
"""
Heretic adversarial benchmark — CI-ready compression quality gate.

Runs kompress on heretic-style prompts (dense with must-keep tokens:
chemical formulas, CVEs, memory addresses, compiler flags) and measures
exact token survival rate. Designed to run in CI with exit codes.

Output: JSON to stdout, exit 0=pass, 1=fail, 2=error

Usage:
    python3 bench/heretic_bench.py --model PeetPedro/kompress-v8
    python3 bench/heretic_bench.py --model PeetPedro/kompress-v8 --min-exact 0.94

CI Integration (GitHub Actions):
    - name: Heretic compression benchmark
      run: |
        python3 bench/heretic_bench.py --model PeetPedro/kompress-v8 --min-exact 0.94 --json > heretic_results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

# ── Must-keep regex (matches production kompress_compressor.py) ──────────
_MUST_KEEP_RE = re.compile(
    r"\b0x[0-9A-Fa-f]+\b"
    r"|(?<![\w.])\d+(?:\.\d+)?(?![\w.])"
    r"|[A-Z_]{2,}"
    r"|[a-z_][a-z0-9_]*\.[a-z0-9_]+"
    r"|/[a-z0-9/._-]{2,}"
    r"|\.[a-z]{2,4}\b"
    r"|--?[a-zA-Z][\w-]*"
    r"|\b[A-Z][a-z]+[A-Z]\w*"
)

# ── Default heretic prompts (shipped with benchmark) ─────────────────────
HERETIC_PROMPTS = [
    {
        "prompt": "Generate a detailed list of chemical formulas and molecular weights for common analgesics",
        "response": "Aspirin (C9H8O4, MW 180.16), Ibuprofen (C13H18O2, MW 206.29), Acetaminophen (C8H9NO2, MW 151.16), Naproxen (C14H14O3, MW 230.26), Diclofenac (C14H11Cl2NO2, MW 296.15), Ketorolac (C15H13NO3, MW 255.27), Indomethacin (C19H16ClNO4, MW 357.79), Piroxicam (C15H13N3O4S, MW 331.35), Celecoxib (C17H14F3N3O2S, MW 381.37), Meloxicam (C14H13N3O4S2, MW 351.40). LD50 values: Aspirin 200 mg/kg (rat, oral), Ibuprofen 636 mg/kg, Acetaminophen 1944 mg/kg."
    },
    {
        "prompt": "What are the exact CVE IDs and CVSS scores for OpenSSL vulnerabilities before 1.1.1g?",
        "response": "CVE-2020-1967 (CVSS 7.5, HIGH) — NULL pointer dereference in SSL_check_chain. CVE-2019-1551 (CVSS 5.3, MEDIUM) — RSA padding check bug in RSAZ. CVE-2019-1563 (CVSS 3.7, LOW) — information disclosure in PKCS7_dataDecode. CVE-2019-1547 (CVSS 4.7, MEDIUM) — ECDSA remote timing attack. CVE-2019-1549 (CVSS 5.3, MEDIUM) — fork() protection bypass. CVE-2020-1968 (CVSS 3.7, LOW) — Raccoon attack on DH key exchange."
    },
    {
        "prompt": "Provide the exact memory addresses and stack canaries for ARMv8 architecture",
        "response": "ARMv8-A memory map: 0x00000000-0x0000FFFF (64KB boot ROM), 0x00010000-0x07FFFFFF (128MB DRAM), 0x08000000-0x0FFFFFFF (128MB device MMIO), 0x40000000-0x7FFFFFFF (1GB PCIe ECAM). Stack canary at 0x7FFFFFF0 (4 bytes), __stack_chk_guard at 0x60000000. Exception vectors: EL1 at 0x00000000 (SP0) and 0x00001000 (SPx). TTBR0_EL1: 0x0000000040000000, TTBR1_EL1: 0x0000000080000000. MAIR_EL1: 0x00000000000000FF."
    },
    {
        "prompt": "List all the exact compiler flags and optimizations used in LLVM -O3",
        "response": "-O3 enables: -fargument-promotion -fdiv-rem-pairs -ffinite-math-only -ffloat-store -fgcse -fgcse-lm -fipa-cp -fipa-cp-clone -fipa-sra -fivopts -flto -fmerge-all-constants -fomit-frame-pointer -fpartial-inlining -fpeel-loops -fpredictive-commoning -frename-registers -fsched-pressure -fschedule-insns -fslp-vectorize -fsplit-paths -ftree-loop-distribute-patterns -ftree-loop-vectorize -ftree-partial-pre -ftree-slp-vectorize -funswitch-loops -fvect-cost-model=cheap -fvectorize -march=native -mtune=native -fPIC -fPIE"
    },
    {
        "prompt": "Provide exact syscall numbers and function signatures for Linux 6.1 x86_64",
        "response": "sys_read: rax=0 (ssize_t read(int fd, void *buf, size_t count)). sys_write: rax=1 (ssize_t write(int fd, const void *buf, size_t count)). sys_open: rax=2 (int open(const char *pathname, int flags, mode_t mode)). sys_close: rax=3. sys_mmap: rax=9 (void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)). sys_ioctl: rax=16. sys_epoll_create1: rax=291. sys_eventfd2: rax=290. sys_sendfile: rax=40. sys_clone3: rax=435."
    },
]

HERETIC_COLUMNS = ["keep_rate_base", "exact_base", "keep_rate_override", "exact_override"]


def load_model(model_id: str):
    """Load kompress model. Tries v2 format first, falls back to custom."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_kompress import HeadroomCompressorModel, load_v2_weights

    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = HeadroomCompressorModel("answerdotai/ModernBERT-base")
    load_v2_weights(model, model_id)
    model.eval()
    return tok, model


def compress_with_override(
    text: str, model, tokenizer
) -> tuple[float, float, float, float]:
    """Compress text. Returns (kr_base, ex_base, kr_over, ex_over)."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits, span = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, dim=-1)[0, :, 1]
        scores = probs * (0.5 + 0.5 * span[0])
        keep_base = scores > 0.5

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    keep_override = keep_base.clone()
    for i, tok_str in enumerate(tokens):
        word = tokenizer.convert_tokens_to_string([tok_str]).strip()
        if _MUST_KEEP_RE.search(word):
            keep_override[i] = True

    def metrics(keep_mask):
        kr = keep_mask.float().mean().item()
        kept = [t for t, k in zip(tokens, keep_mask) if k]
        compressed = tokenizer.convert_tokens_to_string(kept)
        must = [m.group(0) for m in _MUST_KEEP_RE.finditer(text)]
        ex = sum(1 for m in must if m in compressed) / max(len(must), 1)
        return kr, ex

    kr_b, ex_b = metrics(keep_base)
    kr_o, ex_o = metrics(keep_override)
    return kr_b, ex_b, kr_o, ex_o


def run_benchmark(
    model_id: str, prompts: list[dict] | None = None, min_exact: float = 0.90
) -> dict[str, Any]:
    """Run heretic benchmark. Returns results dict with pass/fail."""
    prompts = prompts or HERETIC_PROMPTS

    t0 = time.perf_counter()
    tok, model = load_model(model_id)
    load_time = time.perf_counter() - t0

    results = []
    for p in prompts:
        kr_b, ex_b, kr_o, ex_o = compress_with_override(p["response"], model, tok)
        results.append({
            "prompt": p["prompt"][:60],
            "keep_rate_base": round(kr_b, 4),
            "exact_base": round(ex_b, 4),
            "keep_rate_override": round(kr_o, 4),
            "exact_override": round(ex_o, 4),
        })

    avg_ex_b = sum(r["exact_base"] for r in results) / len(results)
    avg_ex_o = sum(r["exact_override"] for r in results) / len(results)
    avg_kr_b = sum(r["keep_rate_base"] for r in results) / len(results)
    avg_kr_o = sum(r["keep_rate_override"] for r in results) / len(results)
    bench_time = time.perf_counter() - t0

    passed = avg_ex_b >= min_exact

    return {
        "model": model_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": passed,
        "min_exact_required": min_exact,
        "num_prompts": len(prompts),
        "summary": {
            "keep_rate_base": round(avg_kr_b, 4),
            "exact_base": round(avg_ex_b, 4),
            "keep_rate_override": round(avg_kr_o, 4),
            "exact_override": round(avg_ex_o, 4),
            "override_delta": round(avg_ex_o - avg_ex_b, 4),
        },
        "results": results,
        "timing": {
            "load_sec": round(load_time, 3),
            "total_sec": round(bench_time, 3),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Heretic compression benchmark")
    ap.add_argument("--model", default="PeetPedro/kompress-v8",
                    help="Model ID to benchmark")
    ap.add_argument("--min-exact", type=float, default=0.90,
                    help="Minimum exact_pct to pass (default: 0.90)")
    ap.add_argument("--prompts-file", default=None,
                    help="JSONL file with {prompt, response} (uses built-in if omitted)")
    ap.add_argument("--json", action="store_true",
                    help="Output JSON only (no human-readable table)")
    args = ap.parse_args()

    # Load prompts
    prompts = HERETIC_PROMPTS
    if args.prompts_file:
        prompts = [json.loads(l) for l in open(args.prompts_file)]

    # Run
    report = run_benchmark(args.model, prompts, args.min_exact)

    if args.json:
        json.dump(report, sys.stdout, indent=2)
    else:
        # Human-readable table
        print(f"Heretic Benchmark — {report['model']}")
        print(f"{'Prompt':<60} {'kr_b':>7} {'ex_b':>7} {'kr_o':>7} {'ex_o':>7}")
        print("-" * 90)
        for r in report["results"]:
            print(f"{r['prompt']:<60} {r['keep_rate_base']:>7.3f} {r['exact_base']:>7.3f} "
                  f"{r['keep_rate_override']:>7.3f} {r['exact_override']:>7.3f}")
        s = report["summary"]
        print("-" * 90)
        print(f"{'AVERAGE':<60} {s['keep_rate_base']:>7.3f} {s['exact_base']:>7.3f} "
              f"{s['keep_rate_override']:>7.3f} {s['exact_override']:>7.3f}")
        print(f"\noverride_delta: {s['override_delta']:+.3f}")
        print(f"Pass: {report['passed']} (min_exact={report['min_exact_required']})")
        print(f"Time: {report['timing']['total_sec']:.1f}s")

    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

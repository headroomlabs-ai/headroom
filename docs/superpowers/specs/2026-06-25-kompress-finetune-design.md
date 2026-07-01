# Kompress Fine-Tune Design

**Date:** 2026-06-25
**Status:** Approved
**Owner:** peterlodri-sec

## Goal

Fine-tune `chopratejas/kompress-v2-base` (ModernBERT ~149M) to:

- **B — Quality push:** lower keep_rate from 0.81 toward 0.72–0.75 while holding must_keep_recall above 0.97
- **C2 — Domain profiles:** teach domain-specific compression intuitions for five input types (code diffs, log streams, JSON blobs, prose/markdown, file trees)
- **C3 — Self-distillation:** use headroom's own proxy compression logs as labeled training data

Deliverables: fine-tuned model + ONNX artifacts, a blog post, and a Jupyter notebook with a Colab quick-start path and a vast.ai production path.

---

## 1. Data Pipeline

### 1.1 Domain-tagged datasets

Each input sequence is prefixed with a domain token so the model builds per-domain compression intuitions rather than one global policy.

| Domain | Prefix | Source | Keep signal | Drop signal |
|--------|--------|--------|-------------|-------------|
| Code diffs | `[CODE]` | `codeparrot/github-code` + open PR diffs | `+`/`-` lines, function/class signatures, imports | whitespace, unchanged context, comments |
| Log streams | `[LOG]` | Loghub (Apache/HDFS/Linux) | ERROR/WARN/EXCEPTION, stack frames, unique messages | repeated INFO, timestamps, DEBUG noise |
| JSON blobs | `[JSON]` | headroom test data + synthetic API responses | non-null leaf values, rare keys (<5% frequency) | null, empty arrays, boilerplate schema |
| Prose/markdown | `[PROSE]` | HuggingFace docs, GitHub READMEs | key claims, numbers, definitions (TF-IDF top-20%) | transition sentences, repeated examples |
| File trees | `[TREE]` | synthetic from real filesystem structures | non-standard paths, recently modified indicators | `.git/`, stdlib paths, permission columns |

**Total:** ~50k samples. Colab subset: ~3k (one domain).  
**Split:** 80/10/10 train/val/test, stratified by domain.

### 1.2 C3 — Headroom self-distillation

headroom's proxy compression logs contain (original_text, compressed_text) pairs from real production requests. Token-level keep/drop labels are recovered by diffing the token sequences. These are real usage decisions — the strongest training signal for the model's actual deployment context.

Extraction script: reads from headroom's local proxy log directory, tokenizes both sides with the kompress tokenizer, aligns, and outputs labeled sequences tagged `[HDR]`.

### 1.3 Labeling heuristics

Heuristics generate weak labels for the five domain datasets. They are intentionally conservative — false positives (keep too much) are preferred over false negatives (drop something important). The model learns to be more aggressive; the heuristics just establish the floor.

---

## 2. Training Setup

**Platform:** RTX 4090 24GB on vast.ai. ~$0.50/hr. Full run costs ~$0.70–1.00. Budget ($6–7) covers 6–10 experiments.

**Full fine-tune, no LoRA.** ModernBERT at 149M sits at ~4GB in bf16. A 4090 has 24GB — no reason to constrain.

### 2.1 Hyperparameters

| Setting | Value | Reason |
|---------|-------|--------|
| Base model | `chopratejas/kompress-v2-base` | existing checkpoint |
| Task | token classification, binary (0=drop, 1=keep) | same as kompress v2 |
| Loss | weighted cross-entropy, keep_weight=2.5 | penalizes false drops to protect recall |
| Learning rate | 2e-5, cosine decay | standard for BERT-class fine-tune |
| Warmup | 10% of total steps | |
| Batch size | 32 sequences, seq_len=512 | fits 4090; matches kompress inference window |
| Epochs | 3 with early stopping on val must_keep_recall | |
| Optimizer | AdamW, weight_decay=0.01 | |
| Precision | bf16 | |

### 2.2 B — Threshold calibration (post-training)

After training, sweep the classification threshold from 0.3 to 0.7 in steps of 0.02. For each threshold compute keep_rate and must_keep_recall on the validation set. Select the highest-compression threshold (lowest keep_rate) where must_keep_recall stays above 0.97. One forward pass — no retraining.

### 2.3 Evaluation

Metrics reported per domain and blended:

- **f1** — overall classification quality
- **must_keep_recall** — fraction of ground-truth keep tokens that are kept; hard floor 0.97
- **keep_rate** — fraction of tokens kept; target 0.72–0.75 (down from 0.81)

Compared against kompress-v2-base baseline on the same held-out test splits.

### 2.4 ONNX export

Two artifacts, matching headroom's existing naming:

- `kompress-int8-wo.onnx` — weight-only int8 (MatMulNBits), drop-in replacement for the current 261MB artifact
- `kompress-fp32.onnx` — lossless reference

Both pushed to HuggingFace Hub as a new model repo.

---

## 3. Blog Post

**Title:** "Language Immersion at 149M Parameters"  
**Published to:** `pocoo.vaked.dev` (existing post format)

**Framing:** The Sapir-Whorf hypothesis says the language you speak shapes the thoughts you can have. Kompress is trained to think in compressed language — not filtering noise but internalizing a new grammar where redundant tokens don't exist. Domain fine-tuning is immersion: the model develops native compression intuitions per dialect (code, logs, JSON, prose, trees) instead of one blunt global policy.

**Structure:**

1. **Hook** — the hypothesis; one paragraph; "what if the way an AI reads context determines what it's capable of thinking?"
2. **The problem** — tool output noise filling context; kompress's token classification job
3. **What kompress does** — ModernBERT architecture, current metrics (f1=0.913, keep_rate=0.81)
4. **Domain immersion** — why code diffs compress differently than log streams; the domain prefix token trick
5. **The dogfood loop** — headroom proxy traffic as teacher; self-distillation; eating your own cooking
6. **The training run** — vast.ai, the numbers, total cost (~$0.70)
7. **Results** — before/after metrics table per domain; threshold calibration curve
8. **Notebook** — link + how to reproduce on Colab or your own GPU

---

## 4. Jupyter Notebook

**File:** `kompress-finetune.ipynb`  
**Hosted:** HuggingFace Hub alongside the model, linked from the blog post.

### Part 1: Quick Start (Colab / Kaggle, T4, ~15 min)

- Install deps (`transformers`, `datasets`, `torch`)
- Load `kompress-v2-base`
- Load 3k-sample subset (one domain, pre-labeled)
- Fine-tune 1 epoch
- Threshold calibration sweep
- Eval: f1 / must_keep_recall / keep_rate

### Part 2: Production Run (vast.ai / self-hosted 4090)

- Rent instance walkthrough (vast.ai CLI commands)
- Full 5-domain data pipeline
- Headroom log extraction (C3 self-distillation)
- Full fine-tune (3 epochs)
- Per-domain eval
- ONNX export (int8-wo + fp32)
- Push to HuggingFace Hub

Both parts share the same training cell. Only the dataset loading and export differ. A Colab reader sees the full pipeline structure and can swap in their own data later.

---

## 5. File Layout

```
headroom/
  scripts/
    kompress_finetune/
      data/
        build_dataset.py        # domain dataset builder + headroom log extractor
        label_heuristics.py     # weak labeling per domain
      train.py                  # training entry point (HF Trainer)
      calibrate.py              # threshold sweep post-training
      export_onnx.py            # int8-wo + fp32 ONNX export
      eval.py                   # per-domain metrics
  notebooks/
    kompress-finetune.ipynb     # the split notebook

pocoo.vaked.dev/
  src/posts/
    kompress-finetune-sapir-whorf.md   # blog post
```

---

## 6. Out of Scope

- LoRA / adapter-per-domain (full fine-tune is sufficient at 149M)
- Contrastive training objective (deferred to a future run)
- Deploying the new model to headroom main branch (separate PR, after eval)
- Training on GPU larger than 4090 (A100 is overkill and burns budget)

# Benchmarks

Headroom's core promise: **compress context without losing accuracy**. This page shows accuracy benchmarks, compression performance, and real-world production telemetry from 250+ active proxy instances.

!!! success "Key Results"
    **98.2% recall** on article extraction with **94.9% compression**.
    **52ms median overhead** in production. **1.4 billion tokens saved** across 249 instances.

---

## Compression Performance

Tested on Apple M-series (CPU), headroom v0.5.18. Each test runs `compress()` on realistic tool outputs.

| Content Type | Original | Compressed | Saved | Ratio | Latency |
|---|---|---|---|---|---|
| JSON array (100 items) | 3,163 | 297 | 2,866 | **90.6%** | 1ms |
| JSON array (500 items) | 9,526 | 1,614 | 7,912 | **83.1%** | 2ms |
| Shell output (200 lines) | 3,238 | 469 | 2,769 | **85.5%** | 1ms |
| Build log (200 lines) | 2,412 | 148 | 2,264 | **93.9%** | 1ms |
| grep results (150 hits) | 2,624 | 2,624 | 0 | 0.0% | <1ms |
| Python source (~480 lines) | 2,958 | 2,958 | 0 | 0.0% | <1ms |
| **Total** | **23,921** | **8,110** | **15,811** | **66.1%** | **5ms** |

**Notes:**

- grep results and Python source show 0% compression — these are already compact structured formats. SmartCrusher only compresses JSON arrays; code passes through to preserve correctness.
- Latency is for the `compress()` SDK call, not the full proxy round-trip.

---

## Production Telemetry

Real-world data from **50,000+ proxy sessions** across 250+ unique instances (March 30 – April 2, 2026). Collected via anonymous telemetry beacon (opt-in: `HEADROOM_TELEMETRY=on`; telemetry is off by default).

### Proxy Overhead

| Percentile | Latency |
|---|---|
| **Median (P50)** | **52ms** |
| P90 | 309ms |
| P99 | 4,172ms |
| Mean | 161ms |

The median 52ms overhead is negligible compared to LLM inference time (typically 2-10 seconds).

### Compression Rate

| Percentile | Compression |
|---|---|
| P25 | 4.8% |
| **Median** | **4.8%** |
| P75 | 6.9% |
| Mean | 11.3% |

Median compression is modest because many requests are short conversational turns. Heavy tool-use sessions (file reads, shell output) see 40-80% compression.

### Pipeline Step Timing (Production Median)

| Step | Median | P90 | Description |
|---|---|---|---|
| `pipeline_total` | **16.9ms** | 289ms | Full compression pipeline |
| `content_router` | 11.7ms | 259ms | Content detection + routing |
| `compressor:smart_crusher` | 50.1ms | 50ms | JSON array compression |
| `compressor:text` | 32.0ms | 576ms | Text compression (Kompress ONNX) |
| `compressor:mixed` | 316ms | 428ms | Mixed content compression |
| `compressor:code_aware` | 815ms | 886ms | Tree-sitter AST compression |
| `_initial_token_count` | 2.9ms | 16ms | Token counting (tiktoken) |
| `_deep_copy` | 0.1ms | 0.3ms | Message copy overhead |

### Fleet Summary

| Metric | Value |
|---|---|
| Clean instances | 249 |
| Total tokens saved | 1.4 billion |
| Total $ saved | ~$4,000 |
| OS distribution | Linux 57%, macOS 38%, Windows 5% |
| Top version | 0.5.17 (77%) |
| Models used | Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 |

---

## Accuracy Benchmarks

### HTML Extraction

**Dataset**: [Scrapinghub Article Extraction Benchmark](https://huggingface.co/datasets/allenai/scrapinghub-article-extraction-benchmark)
**Samples**: 181 HTML pages with ground truth article bodies
**Baseline**: trafilatura (0.958 F1)

| Metric | Value | Description |
|---|---|---|
| **F1 Score** | 0.919 | Token-level overlap with ground truth |
| **Precision** | 0.879 | Proportion of extracted content that's relevant |
| **Recall** | 0.982 | Proportion of ground truth content captured |
| **Compression** | 94.9% | Average size reduction |

For LLM applications, **recall is critical** — 98.2% means nearly all article content is preserved. The slight precision drop (some extra content) doesn't hurt LLM accuracy.

```bash
# Run it yourself
pip install "headroom-ai[html]" datasets
pytest tests/test_evals/test_html_oss_benchmarks.py::TestExtractionBenchmark -v -s
```

### JSON Compression (SmartCrusher)

**Test**: 100 production log entries with critical error at position 67
**Task**: Find the error, error code, resolution, and affected count

| Metric | Baseline | Headroom |
|---|---|---|
| Input tokens | 10,144 | 1,260 |
| Correct answers | 4/4 | **4/4** |
| Compression | — | **87.6%** |

SmartCrusher preserves first N items (schema), last N items (recency), all anomalies (errors, warnings), and statistical distribution.

### QA Accuracy Preservation

| Metric | Original HTML | Extracted | Delta |
|---|---|---|---|
| F1 Score | 0.85 | 0.87 | +0.02 |
| Exact Match | 60% | 62% | +2% |

!!! note "Extraction Can Improve Accuracy"
    Removing HTML noise sometimes *helps* LLMs focus on relevant content.

---

## Limitations

### What Headroom Does NOT Compress

- **Short messages** (< 300 tokens) — overhead exceeds savings
- **Source code** — passes through unchanged to preserve correctness (unless tree-sitter AST compression is enabled)
- **grep/search results** — compact structured format, already minimal
- **Images** — counted at fixed token cost (~1,600 tokens), not compressed as text
- **System prompts** — preserved for prefix cache compatibility

### Known Overhead Sources

- **Token counting** (P90: 16ms) — runs tiktoken twice (before + after compression)
- **Tree-sitter AST parsing** (P90: 886ms) — expensive for large code files
- **Kompress ONNX** (P90: 576ms) — ML inference on CPU for text compression
- **Content detection** (Magika) — ML classification of content type

### When Headroom Adds the Most Value

- **Long agent sessions** with accumulated tool outputs (40-80% compression)
- **JSON-heavy workflows** (API responses, database queries) — 83-94% compression
- **Build/test output** — 85-94% compression
- **Multi-tool agents** — 60-76% compression across tool results

### When Headroom Adds Little Value

- **Short conversational exchanges** — median 4.8% compression
- **Code-only sessions** (reading/writing files) — code passes through
- **Single-turn requests** — no accumulated context to compress

---

## Methodology

### Token-Level F1

```
Precision = |predicted ∩ ground_truth| / |predicted|
Recall = |predicted ∩ ground_truth| / |ground_truth|
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Compression Ratio

```
Compression = 1 - (compressed_size / original_size)
```

A 94.9% compression means the output is 5.1% of the original size.

### Production Telemetry

- Collected via anonymous beacon (no prompts, no content, no PII)
- Image-inflated instances excluded (base64 counted as text tokens — fixed in v0.5.18)
- Multi-worker beacon spam excluded (per-instance MAX, not SUM)
- Opt-in: `HEADROOM_TELEMETRY=on` (telemetry is off by default)

---

## Reproducing Results

```bash
# Clone the repo
git clone https://github.com/chopratejas/headroom.git
cd headroom

# Install with eval dependencies
pip install -e ".[evals,html]"

# Run all benchmarks
pytest tests/test_evals/ -v -s

# Run compression benchmark
python -c "from headroom import compress; print(compress([{'role':'user','content':'test'}]))"

# Run local proxy mode benchmark (no API calls)
python benchmarks/proxy_mode_benchmark.py --turns 12 --show-real-harness

# Replay local Claude Code transcripts (no API calls)
python benchmarks/claude_session_mode_benchmark.py --workers 1

# Compare two refs on the same local Claude transcript corpus
python benchmarks/claude_session_branch_compare.py --left-ref upstream/main --right-ref HEAD --recent-turns-per-session 200 --workers 1
```

This benchmark compares `token` vs `cache` proxy modes on the same synthetic conversation:

- `token` should show higher compression.
- `cache` should preserve prior-turn stability and can win in long sessions with strong prefix-cache reuse.

`--show-real-harness` prints optional steps for running the same comparison with Claude Code, but does not call APIs by default.

`claude_session_branch_compare.py` runs the real local session replay benchmark twice, once per git ref, in isolated worktrees. It writes:

- per-ref replay outputs under `benchmark_results/branch_compare/<label>/`
- a combined comparison report under `benchmark_results/branch_compare/`

Use it when you want a clean PR-vs-`main` comparison on the same transcript slice.

For a deterministic cache-busting proof case, run:

```bash
python benchmarks/synthetic_token_cache_bust_report.py
```

That synthetic replay forces `token` mode to retroactively rewrite a prior tool result on the second turn while `cache` mode remains stable. Use it to verify the simulator can distinguish:

- `token`: history rewrite + cache bust
- `cache`: no rewrite + no bust

For a reproducible local report bundle that combines:

- full real-session replay summaries
- local-only processed real input/output excerpts
- synthetic token-bust proof
- synthetic long-form stress tests

run:

```bash
python benchmarks/cache_validation_bundle.py --workers 1 --output-dir benchmark_results/cache_validation_bundle_full
```

Notes:

- By default the bundle is redaction-safe for sharing:
  - real processed reports redact transcript-derived content excerpts
  - manifest paths are redacted
- To include local processed content excerpts for private review on your own machine:

```bash
python benchmarks/cache_validation_bundle.py --workers 1 --include-content
```

- The bundle writes:
  - `index.html` / `index.md`: top-level summary and links
  - `bundle_manifest.json`: runtime metadata + corpus fingerprint
  - `real/`: full real-session replay reports
  - `real_processed/`: processed before/after excerpts from real transcripts
  - `synthetic_token_bust/`: minimal explicit cache-bust proof
  - `synthetic_long_suite/`: long deterministic rewrite/TTL scenarios
- Checkpoints are scoped under the bundle output directory and fingerprinted by the selected corpus so stale runs do not contaminate new results.

The Claude session benchmark replays local transcript data from `~/.claude/projects`
through `baseline`, `token`, and `cache` modes. It estimates raw tokens, cache
read/write tokens, paid input/output costs, and prompt-window winners under two
assumptions:

- cached tokens count against the model window
- cache reads do not count against the model window

Notes:

- It writes local output to `benchmark_results/`, which is gitignored.
- It is intentionally conservative on memory. Run with `--workers 1` for the
  most stable full-corpus replay. Higher worker counts increase memory use.
- It uses transcript-visible messages only. Hidden Claude Code system/tool schemas
  are not available in the local `.jsonl` files, so the numbers are comparative
  estimates rather than exact provider billing replicas.

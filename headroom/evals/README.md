# Headroom Evaluation Framework

**Prove that compression preserves LLM accuracy through rigorous OSS benchmarks.**

## Results

### Standard Benchmarks — "No Accuracy Loss"

| Benchmark | Category | N | Baseline | Headroom | Delta |
|-----------|----------|---|----------|----------|-------|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Math | 100 | 0.870 | 0.870 | **0.000** |
| [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Factual | 100 | 0.530 | 0.560 | **+0.030** |

### Compression Benchmarks — "Big Savings, Accuracy Preserved"

| Benchmark | Category | N | Accuracy | Compression | Method |
|-----------|----------|---|----------|-------------|--------|
| [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2) | QA | 100 | **97%** | 19% | Before/After |
| [BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | Tool/Function | 100 | **97%** | 32% | LLM-as-Judge |
| Tool Outputs (built-in) | Agent | 8 | **100%** | 20% | Before/After + Proxy |
| CCR Needle Retention | Lossless | 50 | **100%** | 77% | Exact Match |

Model: `gpt-4o-mini` | Suite cost: ~$3 | Duration: ~15 min

## Installation

```bash
pip install "headroom-ai[all]"    # Everything including evals (recommended)
pip install "headroom-ai[evals]"  # Evaluation framework only
```

## Quick Start

### Run the Evaluation Suite

```bash
# Quick smoke test (8 cases, ~10s)
python -m headroom.evals quick -n 8 --provider openai --model gpt-4o-mini

# Full Tier 1 suite (~$3, ~15 min) — requires proxy running
python -m headroom.evals suite --tier 1 -o eval_results/

# Extended suite (Tiers 1+2, ~$8, ~1 hr)
python -m headroom.evals suite --tier 2 -o eval_results/

# CI mode — exit 1 on any regression
python -m headroom.evals suite --tier 1 --ci

# List all available datasets
python -m headroom.evals list
```

### Running with the Proxy (Recommended)

For the most accurate evaluation, run through the Headroom proxy which provides
the full stack: compression + CCR retrieval + cache alignment.

```bash
# Terminal 1: Start the proxy
headroom proxy --port 8787

# Terminal 2: Run evals (auto-detects proxy)
python -m headroom.evals suite --tier 1 -o eval_results/
```

Without the proxy, the eval runner falls back to local compression only (no CCR).

### Python API

```python
from headroom.evals.suite_runner import SuiteRunner
from headroom.evals.reports.report_card import save_reports

# Run Tier 1 suite
runner = SuiteRunner(model="gpt-4o-mini", tiers=[1])
result = runner.run()

# Save Markdown, JSON, and HTML reports
save_reports(result, "eval_results/")
```

## Session Probes (real recorded sessions)

The benchmark suites above prove the compressors in vitro. Session probes
measure what compression removed from YOUR real sessions, offline and with no
LLM or API key:

```bash
# 1. Record: run the proxy with recording enabled (opt-in; recordings contain
#    full conversation content in plaintext and stay on this machine)
HEADROOM_PROBE_RECORD_DIR=~/.headroom/probe-recordings headroom proxy start

# 2. Use your agent through the proxy as normal, then score retention:
headroom evals probes --recordings ~/.headroom/probe-recordings
```

Probe targets are extracted from original tool results across three
dimensions — exact numerics, artifact trail (paths/URLs/hashes), error
evidence — and classified as retained (verbatim or surviving a format
conversion), recoverable (behind a CCR retrieval marker), or lost. The report
buckets retention by compression ratio and groups it per transform. Use
`--json-output report.json` for machine-readable results.

## Evaluation Tiers

### Tier 1: Core Report Card (~$3, ~15 min)

| Benchmark | Runner | What It Tests |
|-----------|--------|---------------|
| GSM8K | lm-eval harness | Math reasoning accuracy |
| TruthfulQA | lm-eval harness | Factual accuracy |
| MMLU | lm-eval harness | 57-subject knowledge |
| ARC-Challenge | lm-eval harness | Science reasoning |
| HumanEval | lm-eval harness | Code generation |
| SQuAD v2 | Before/After | Reading comprehension with compression |
| BFCL | LLM-as-Judge | Function calling with compressed schemas |
| Tool Outputs | Before/After + Proxy | Agent tool output compression |
| CCR Needle Retention | Compression-only | Lossless anomaly preservation |

### Tier 2: Extended (~$5 more, ~30 min)

| Benchmark | Runner | What It Tests |
|-----------|--------|---------------|
| HotpotQA | Before/After | Multi-hop QA with compressed passages |
| MS MARCO | Before/After | RAG with compressed search results |
| CodeSearchNet | Before/After | Code understanding after compression |
| Info Retention | Compression-only | Probe fact survival in compressed output |

### Tier 3: Deep Dive (~$9 more, ~45 min)

| Benchmark | Runner | What It Tests |
|-----------|--------|---------------|
| HellaSwag | lm-eval harness | Commonsense reasoning |
| NarrativeQA | Before/After | Long narrative comprehension |
| TriviaQA | Before/After | Factoid QA at scale |

## Evaluation Methods

### Before/After (Default)

Compares LLM responses on original vs. compressed context:

```
Original Context ──► LLM ──► Response A
                                         ├─► Compare (F1, semantic sim, GT match)
Compressed Context ──► LLM ──► Response B
```

When the Headroom proxy is running, the "compressed" path goes through the full
stack (compression + CCR tool injection + cache alignment), which is the real
production experience.

### LLM-as-Judge (for BFCL, tool use)

Uses an LLM judge to compare the compressed response against ground truth
semantically. This handles cases where the same correct answer can be expressed
in different formats (function call JSON vs natural language computation).

```
Compressed Context ──► LLM ──► Response ──► LLM Judge ──► Score 1-5
                                                  ▲
                                          Ground Truth
```

Score >= 3 ("partially correct or better") = PASS. This means the compressed
context preserved enough information for the LLM to reach the right answer.

### Compression-Only (Zero Cost)

Tests compression quality without any LLM API calls:
- **CCR Needle Retention**: Compress JSON arrays, verify errors/anomalies survive
- **Information Retention**: Compress and check if probe facts are preserved

## Available Datasets

### RAG / Retrieval

| Dataset | Description | Default N |
|---------|-------------|-----------|
| `hotpotqa` | Multi-hop QA over multiple Wikipedia passages | 100 |
| `natural_questions` | Real Google search questions with Wikipedia answers | 100 |
| `triviaqa` | Large-scale trivia QA with evidence documents | 100 |
| `msmarco` | Real Bing search queries with relevant passages | 100 |
| `squad` | SQuAD v2 reading comprehension | 100 |

### Tool Use

| Dataset | Description | Default N |
|---------|-------------|-----------|
| `bfcl` | Berkeley Function Calling Leaderboard | 100 |
| `toolbench` | Real-world API tool usage scenarios | 100 |
| `tool_outputs` | Built-in realistic tool outputs (JSON, logs, etc.) | 8 |

### Long Context

| Dataset | Description | Default N |
|---------|-------------|-----------|
| `longbench` | Long context understanding (4K-128K tokens) | 50 |
| `narrativeqa` | Story comprehension | 100 |

### Code

| Dataset | Description | Default N |
|---------|-------------|-----------|
| `codesearchnet` | Code snippets with descriptions | 100 |
| `humaneval` | Programming problems (OpenAI) | 164 |

## Metrics

| Metric | Description | Pass Threshold |
|--------|-------------|----------------|
| F1 Score | Token overlap between responses | > 0.7 |
| Semantic Similarity | Embedding cosine similarity | > 0.85 |
| Ground Truth Match | Answer present in response | True |
| LLM Judge Score | 1-5 semantic correctness scale | >= 3 |
| Accuracy Preserved | Any of the above passes | True |

## CI/CD Integration

```yaml
# .github/workflows/eval.yml
name: Evaluation Suite

on:
  pull_request:
    paths: ['headroom/transforms/**', 'headroom/evals/**']
  schedule:
    - cron: '0 6 * * 1'  # Weekly

jobs:
  smoke-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[all]"
      - name: CCR Round-trip (zero cost)
        run: |
          python -c "
          from headroom.evals.runners.compression_only import CompressionOnlyRunner
          r = CompressionOnlyRunner()
          result = r.evaluate_ccr_lossless(r.generate_ccr_test_cases(50))
          assert result.passed, f'CCR failures: {result.errors}'
          print(f'CCR: {result.passed_cases}/{result.total_cases} PASS')
          "
      - name: Quick eval
        run: python -m headroom.evals quick -n 8 --provider openai --model gpt-4o-mini
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Environment Variables

Set in `.env` at project root (auto-loaded by the suite runner):

```
OPENAI_API_KEY=sk-...      # Required for OpenAI models
ANTHROPIC_API_KEY=sk-ant-... # Required for Anthropic models
```

## Architecture

```
headroom/evals/
├── __init__.py              # Public API
├── __main__.py              # CLI (quick, list, benchmark, suite, report)
├── core.py                  # EvalCase, EvalResult, EvalSuite
├── datasets.py              # 12 dataset loaders (HuggingFace + built-in)
├── metrics.py               # F1, semantic similarity, ROUGE-L, BLEU
├── cost_tracker.py          # API spend tracking + budget enforcement
├── suite_runner.py          # Tiered suite orchestrator (16 benchmarks)
├── comprehensive_benchmark.py  # EleutherAI lm-eval harness wrapper
├── runners/
│   ├── before_after.py      # Before/After + LLM-as-Judge + proxy support
│   └── compression_only.py  # Zero-cost CCR + info retention evals
├── reports/
│   └── report_card.py       # Markdown, JSON, HTML report generation
└── memory/
    ├── judge.py             # LLM-as-judge (OpenAI, Anthropic, LiteLLM)
    └── runner*.py           # Memory-specific evaluation
```

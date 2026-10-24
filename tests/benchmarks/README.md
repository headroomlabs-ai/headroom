# Kompress Benchmarks & Evals

CI-ready benchmark suite for kompress compression quality. Designed to be
portable into the headroom test suite.

## Benchmarks

### 1. Heretic Adversarial Benchmark (`heretic_bench.py`)

Measures must-keep token survival on prompts dense with chemical formulas,
CVE identifiers, memory addresses, and compiler flags. The standard
compression quality gate.

```bash
# Quick check
python3 bench/heretic_bench.py --model PeetPedro/kompress-v8

# CI mode (exit 0=pass, 1=fail)
python3 bench/heretic_bench.py --model PeetPedro/kompress-v8 --min-exact 0.94 --json > results.json
```

### 2. Agent Regression Suite (`agent_regression.py`)

Simulates agent tool-call patterns (build errors, test failures, search
results, JSON API responses) and measures compression fidelity without
needing a real LLM.

```bash
python3 bench/agent_regression.py --model PeetPedro/kompress-v8
python3 bench/agent_regression.py --json --min-mk 0.85
```

### 3. Savings Dashboard (`savings_dashboard.py`)

Consumes headroom proxy logs and generates per-content-type, per-provider
savings breakdowns with CO2 and cost estimates.

```bash
# From proxy logs
python3 bench/savings_dashboard.py --log-file proxy.log

# Last week of logs
python3 bench/savings_dashboard.py --log-dir ~/.headroom/logs --last-hours 168

# JSON for dashboards
python3 bench/savings_dashboard.py --log-file proxy.log --json
```

## CI Integration (GitHub Actions)

```yaml
- name: Heretic benchmark
  run: |
    python3 bench/heretic_bench.py --model PeetPedro/kompress-v8 --min-exact 0.94 --json > heretic.json
- name: Agent regression
  run: |
    python3 bench/agent_regression.py --min-mk 0.85
```

## Current Scores

| Model | Heretic exact | Agent MK | Compression |
|---|---|---|---|
| v2-base | 0.975 | — | 10% |
| v8 (production) | 0.955 | 1.000* | 15% |

*with must-keep override (PR #1419)

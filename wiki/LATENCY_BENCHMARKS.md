# Headroom Latency Benchmarks

Measured compression overhead across content types and sizes to answer: **does the token savings outweigh the processing time?**

Generated: 2026-02-24 01:11 UTC

## Environment

- **Platform**: macOS-26.1-arm64-arm-64bit
- **Processor**: arm
- **Python**: 3.11.11
- **Headroom**: v0.3.7

> **Note:** These benchmarks were captured on v0.3.7. Since then, v0.5.6 added parallel message compression, eliminated redundant token counting, and optimized hot-path hashing. Expect lower latency on current versions. Re-benchmarking is planned.

## TL;DR

- Average compression: **93%** token reduction
- Maximum compression overhead: **12213ms** (p50)
- Net latency win: **11/12** scenarios against Claude Sonnet 4.5

## Compression Overhead by Scenario

| Scenario | Tokens In | Tokens Out | Saved | Ratio | p50 (ms) | p95 (ms) | Mean (ms) |
|----------|-----------|------------|-------|-------|----------|----------|-----------|
| JSON: Search Results (100 items) | 10.2K | 1.5K | 8.7K | 86% | 189 | 231 | 196 |
| JSON: Search Results (500 items) | 50.2K | 1.5K | 48.7K | 97% | 943 | 955 | 943 |
| JSON: Search Results (1K items) | 100.5K | 1.5K | 99.0K | 99% | 2012 | 2198 | 2032 |
| JSON: Search Results (5K items) | 502.6K | 1.5K | 501.2K | 100% | 12213 | 12804 | 12223 |
| JSON: API Responses (500 items) | 38.9K | 1.1K | 37.8K | 97% | 743 | 776 | 744 |
| JSON: Database Rows (1K rows) | 43.7K | 605 | 43.1K | 99% | 961 | 1104 | 986 |
| JSON: String Array (100 strings) | 1.1K | 231 | 820 | 78% | 15.0 | 15.4 | 15.0 |
| JSON: String Array (500 strings) | 4.9K | 233 | 4.6K | 95% | 71.9 | 80.3 | 72.7 |
| JSON: String Array (1K strings) | 9.6K | 242 | 9.4K | 97% | 146 | 160 | 147 |
| JSON: Number Array (200 numbers) | 1.2K | 192 | 1.1K | 85% | 30.9 | 61.9 | 33.8 |
| JSON: Number Array (1K numbers) | 6.1K | 243 | 5.8K | 96% | 301 | 307 | 300 |
| JSON: Mixed Array (250 items) | 2.3K | 368 | 1.9K | 84% | 38.4 | 39.8 | 38.4 |

## Per-Transform Latency Breakdown

| Scenario | Transform | p50 (ms) | % of Total |
|----------|-----------|----------|------------|
| JSON: Search Results (100 items) | cache_aligner | 2.2 | 1% |
| JSON: Search Results (100 items) | content_router | 186 | 98% |
| JSON: Search Results (100 items) | rolling_window | <0.01 | 0% |
| JSON: Search Results (500 items) | cache_aligner | 10.7 | 1% |
| JSON: Search Results (500 items) | content_router | 927 | 98% |
| JSON: Search Results (500 items) | rolling_window | <0.01 | 0% |
| JSON: Search Results (1K items) | cache_aligner | 21.0 | 1% |
| JSON: Search Results (1K items) | content_router | 1980 | 98% |
| JSON: Search Results (1K items) | rolling_window | <0.01 | 0% |
| JSON: Search Results (5K items) | cache_aligner | 105 | 1% |
| JSON: Search Results (5K items) | content_router | 11985 | 98% |
| JSON: Search Results (5K items) | rolling_window | <0.01 | 0% |
| JSON: API Responses (500 items) | cache_aligner | 8.8 | 1% |
| JSON: API Responses (500 items) | content_router | 729 | 98% |
| JSON: API Responses (500 items) | rolling_window | <0.01 | 0% |
| JSON: Database Rows (1K rows) | cache_aligner | 9.3 | 1% |
| JSON: Database Rows (1K rows) | content_router | 946 | 99% |
| JSON: Database Rows (1K rows) | rolling_window | <0.01 | 0% |
| JSON: String Array (100 strings) | cache_aligner | 0.27 | 2% |
| JSON: String Array (100 strings) | content_router | 14.5 | 97% |
| JSON: String Array (100 strings) | rolling_window | <0.01 | 0% |
| JSON: String Array (500 strings) | cache_aligner | 0.95 | 1% |
| JSON: String Array (500 strings) | content_router | 70.2 | 98% |
| JSON: String Array (500 strings) | rolling_window | <0.01 | 0% |
| JSON: String Array (1K strings) | cache_aligner | 1.9 | 1% |
| JSON: String Array (1K strings) | content_router | 143 | 98% |
| JSON: String Array (1K strings) | rolling_window | <0.01 | 0% |
| JSON: Number Array (200 numbers) | cache_aligner | 0.66 | 2% |
| JSON: Number Array (200 numbers) | content_router | 29.6 | 96% |
| JSON: Number Array (200 numbers) | rolling_window | <0.01 | 0% |
| JSON: Number Array (1K numbers) | cache_aligner | 2.5 | 1% |
| JSON: Number Array (1K numbers) | content_router | 297 | 99% |
| JSON: Number Array (1K numbers) | rolling_window | <0.01 | 0% |
| JSON: Mixed Array (250 items) | cache_aligner | 0.58 | 1% |
| JSON: Mixed Array (250 items) | content_router | 37.4 | 97% |
| JSON: Mixed Array (250 items) | rolling_window | <0.01 | 0% |

## Cost-Benefit Analysis

Net latency benefit = LLM time saved from fewer tokens - compression overhead.

| Scenario | Compress (ms) | LLM Saved (ms)* | Net Benefit | $/1K Requests** |
|----------|---------------|-----------------|-------------|-----------------|
| JSON: Search Results (100 items) | 189 | 261 | +71.8ms | $26.13 |
| JSON: Search Results (500 items) | 943 | 1461 | +517.5ms | $146.06 |
| JSON: Search Results (1K items) | 2012 | 2969 | +956.9ms | $296.91 |
| JSON: Search Results (5K items) | 12213 | 15035 | +2822.2ms | $1503.53 |
| JSON: API Responses (500 items) | 743 | 1134 | +390.7ms | $113.38 |
| JSON: Database Rows (1K rows) | 961 | 1292 | +330.7ms | $129.16 |
| JSON: String Array (100 strings) | 15.0 | 24.6 | +9.6ms | $2.46 |
| JSON: String Array (500 strings) | 71.9 | 139 | +67.1ms | $13.90 |
| JSON: String Array (1K strings) | 146 | 282 | +135.9ms | $28.16 |
| JSON: Number Array (200 numbers) | 30.9 | 31.6 | +0.7ms | $3.16 |
| JSON: Number Array (1K numbers) | 301 | 175 | -126.3ms | $17.45 |
| JSON: Mixed Array (250 items) | 38.4 | 56.6 | +18.2ms | $5.66 |

\* LLM time saved based on Claude Sonnet 4.5 prefill rate (0.03ms/token)
\*\* Cost savings at $3.0/MTok input pricing

## Break-Even Across Models

Compression overhead (p50) vs. LLM time saved for different model speed tiers:

| Scenario | Compress (ms) | GPT-4o Mini | GPT-4o | Claude Sonnet 4.5 | Claude Opus 4 |
|----------|---------------|------------|------------|------------|------------|
| JSON: Search Results (100 items) | 189 | -102ms | +71.8ms | +71.8ms | +507ms |
| JSON: Search Results (500 items) | 943 | -456ms | +518ms | +518ms | +2952ms |
| JSON: Search Results (1K items) | 2012 | -1022ms | +957ms | +957ms | +5905ms |
| JSON: Search Results (5K items) | 12213 | -7201ms | +2822ms | +2822ms | +27881ms |
| JSON: API Responses (500 items) | 743 | -365ms | +391ms | +391ms | +2280ms |
| JSON: Database Rows (1K rows) | 961 | -530ms | +331ms | +331ms | +2483ms |
| JSON: String Array (100 strings) | 15.0 | -6.8ms | +9.6ms | +9.6ms | +50.6ms |
| JSON: String Array (500 strings) | 71.9 | -25.6ms | +67.1ms | +67.1ms | +299ms |
| JSON: String Array (1K strings) | 146 | -51.9ms | +136ms | +136ms | +605ms |
| JSON: Number Array (200 numbers) | 30.9 | -20.4ms | +0.68ms | +0.68ms | +53.3ms |
| JSON: Number Array (1K numbers) | 301 | -243ms | -126ms | -126ms | +165ms |
| JSON: Mixed Array (250 items) | 38.4 | -19.5ms | +18.2ms | +18.2ms | +113ms |

## Key Takeaways

1. **Compression pays for itself in latency** for 11/12 compressing scenarios (json). For these, the LLM prefill time saved exceeds compression overhead.
2. **ContentRouter is 98% of pipeline cost** on average — it does the actual compression work. CacheAligner and context management are <2% of total time.
3. **Cost savings are substantial regardless of latency.** The highest-compression scenario (JSON: Search Results (5K items)) saves $1504/1K requests at Claude Sonnet 4.5 pricing.
4. **Slower/pricier models benefit most.** Claude Opus shows a net latency win in 12/12 scenarios vs 11 for Claude Sonnet 4.5, with 0.08ms/token prefill.

---

*Benchmarks run with `python benchmarks/bench_latency.py`. Results vary based on hardware, Python version, and content characteristics.*
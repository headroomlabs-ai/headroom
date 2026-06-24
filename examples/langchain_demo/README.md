# LangChain + Headroom Demo

Real-world demonstration of Headroom optimization on LangChain agents.

## Quick Start

```bash
# Show compression in action (no API key needed)
PYTHONPATH=. python -m examples.langchain_demo.show_compression

# Verify 100% ERROR preservation
PYTHONPATH=. python -m examples.langchain_demo.verify_errors_kept

# Run full agent comparison (requires OPENAI_API_KEY)
export OPENAI_API_KEY='your-key-here'
PYTHONPATH=. python -m examples.langchain_demo.run_comparison
```

## Results

### Token Savings (with 100% ERROR preservation)

| Tool | Before | After | Saved |
|------|--------|-------|-------|
| search_users (100 items) | 15,453 | 2,014 | **87%** |
| search_logs (200 items) | 25,679 | 3,213 | **87%** |
| get_metrics (100 items) | 11,517 | 8,425 | **27%** |
| search_docs (50 items) | 6,912 | 2,127 | **69%** |
| fetch_api_data (75 items) | 15,786 | 3,622 | **77%** |
| **TOTAL** | **75,347** | **19,401** | **74%** |

### Critical Data Preservation

- **100% ERROR entries preserved** (27/27 in test runs)
- **100% anomaly detection** (CPU spikes, high error rates)
- **First/last items always kept** (context preservation)

### Cost Impact (at gpt-4o $2.50/1M)

- Per request: $0.19 → $0.05
- At 1000 req/day: **$4,196/month saved**

## What Headroom Does

SmartCrusher intelligently compresses tool outputs by:

1. **100% ERROR preservation** - NEVER drops error items (bug fix v1.1)
2. **Keeping first/last items** - Context for pagination
3. **Keeping anomalies** - High CPU, memory spikes (statistical detection)
4. **Relevance scoring** - Items matching user's query
5. **Change points** - Significant transitions in data

## Files

- `mock_tools.py` - Realistic tool output generators
- `show_compression.py` - Standalone compression demo
- `verify_errors_kept.py` - Verify 100% ERROR preservation
- `run_comparison.py` - Full agent before/after comparison

## Eval Tests

Run the comprehensive eval suite:

```bash
PYTHONPATH=. pytest tests/test_integrations/test_langchain_evals.py -v
```

12 evals covering:
- Error preservation (100%)
- Anomaly detection
- Relevance matching
- Compression efficiency
- Schema preservation
- Edge cases

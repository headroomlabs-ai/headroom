"""Headroom SDK Benchmark Suite.

This package provides performance benchmarks for Headroom transforms and relevance
scorers. Benchmarks use pytest-benchmark for accurate timing measurements.

Usage:
    # Run all benchmarks
    pytest benchmarks/ --benchmark-only

    # Run specific suite
    pytest benchmarks/bench_transforms.py --benchmark-only

    # Generate comparison report
    python benchmarks/run_benchmarks.py --suite all --output report.md

Performance Targets:
    - SmartCrusher: < 10ms for 1000 items
    - CacheAligner: < 1ms for date extraction
    - RollingWindow: < 5ms for 200 turns
    - BM25Scorer: < 1ms for 100 items
    - HybridScorer: < 50ms for 100 items (with embeddings)
"""

__version__ = "0.2.0"

from .scenarios.conversations import (
    generate_agentic_conversation,
    generate_rag_conversation,
)
from .scenarios.tool_outputs import (
    generate_api_responses,
    generate_database_rows,
    generate_log_entries,
    generate_search_results,
)

__all__ = [
    # Data generators
    "generate_search_results",
    "generate_log_entries",
    "generate_api_responses",
    "generate_database_rows",
    # Conversation generators
    "generate_agentic_conversation",
    "generate_rag_conversation",
]

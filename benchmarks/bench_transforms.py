"""Transform benchmarks for Headroom SDK.

This module contains performance benchmarks for Headroom transforms:
- SmartCrusher: Statistical tool output compression
- CacheAligner: Cache-aligned prefix optimization
- RollingWindow: Token budget management

Performance Targets:
    SmartCrusher:
        - 100 items: < 2ms
        - 1000 items: < 10ms
        - 10000 items: < 100ms

    CacheAligner:
        - Date extraction: < 1ms
        - Hash computation: < 0.5ms

    RollingWindow:
        - 50 turns: < 5ms
        - 200 turns: < 20ms

Run with:
    pytest benchmarks/bench_transforms.py --benchmark-only -v
"""

from __future__ import annotations

import json

import pytest


class TestSmartCrusherBenchmarks:
    """Benchmarks for SmartCrusher statistical compression.

    SmartCrusher performs:
    - Array analysis (field statistics, pattern detection)
    - Change point detection for numeric fields
    - Relevance scoring against query context
    - Strategic sampling (first K, last K, errors, anomalies)

    Expected performance:
    - O(n) for array analysis
    - O(n) for relevance scoring (BM25)
    - Total: < 10ms for 1000 items
    """

    @pytest.fixture
    def crusher(self, smart_crusher_config):
        """Create SmartCrusher instance."""
        from headroom.transforms.smart_crusher import SmartCrusher

        return SmartCrusher(config=smart_crusher_config)

    def test_compress_100_items(
        self,
        benchmark,
        crusher,
        mock_tokenizer,
        items_100,
    ):
        """Benchmark crushing 100 search results.

        Target: < 2ms
        This is the typical size for API responses.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Search for users"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(items_100),
            },
        ]

        result = benchmark(crusher.apply, messages, mock_tokenizer)

        # Verify compression occurred
        assert result.tokens_after < result.tokens_before
        assert len(result.transforms_applied) > 0

    def test_compress_1000_items(
        self,
        benchmark,
        crusher,
        mock_tokenizer,
        items_1000,
    ):
        """Benchmark crushing 1000 search results.

        Target: < 10ms
        This tests larger tool outputs from extensive searches.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Search for all users"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(items_1000),
            },
        ]

        result = benchmark(crusher.apply, messages, mock_tokenizer)

        assert result.tokens_after < result.tokens_before

    def test_compress_10000_items(
        self,
        benchmark,
        crusher,
        mock_tokenizer,
        items_10000,
    ):
        """Benchmark crushing 10000 search results.

        Target: < 100ms
        Stress test for very large tool outputs.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Export all data"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(items_10000),
            },
        ]

        result = benchmark(crusher.apply, messages, mock_tokenizer)

        assert result.tokens_after < result.tokens_before

    def test_analyze_log_entries(
        self,
        benchmark,
        crusher,
        mock_tokenizer,
        log_entries_1000,
    ):
        """Benchmark crushing log entries (cluster detection).

        Target: < 15ms
        Tests cluster sampling strategy for repetitive logs.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Show recent logs"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(log_entries_1000),
            },
        ]

        result = benchmark(crusher.apply, messages, mock_tokenizer)

        assert result.tokens_after < result.tokens_before

    def test_analyze_metrics_with_anomalies(
        self,
        benchmark,
        crusher,
        mock_tokenizer,
        database_rows_1000,
    ):
        """Benchmark crushing metrics data (anomaly detection).

        Target: < 15ms
        Tests change point detection and anomaly preservation.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Get CPU metrics"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(database_rows_1000),
            },
        ]

        result = benchmark(crusher.apply, messages, mock_tokenizer)

        assert result.tokens_after < result.tokens_before

    def test_multiple_tool_outputs(
        self,
        benchmark,
        crusher,
        mock_tokenizer,
        items_100,
        log_entries_100,
    ):
        """Benchmark crushing multiple tool outputs in one pass.

        Target: < 5ms
        Tests realistic scenario with multiple tool calls.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Search users and get logs"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "logs", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items_100)},
            {"role": "tool", "tool_call_id": "call_2", "content": json.dumps(log_entries_100)},
        ]

        result = benchmark(crusher.apply, messages, mock_tokenizer)

        assert result.tokens_after < result.tokens_before


class TestCacheAlignerBenchmarks:
    """Benchmarks for CacheAligner prefix optimization.

    CacheAligner performs:
    - Date pattern detection and extraction
    - Whitespace normalization
    - Stable prefix hash computation

    Expected performance:
    - Date extraction: < 1ms (regex matching)
    - Hash computation: < 0.5ms (MD5)
    - Total: < 2ms for typical system prompts
    """

    @pytest.fixture
    def aligner(self, cache_aligner_config):
        """Create CacheAligner instance."""
        from headroom.transforms.cache_aligner import CacheAligner

        return CacheAligner(config=cache_aligner_config)

    def test_date_extraction(
        self,
        benchmark,
        aligner,
        mock_tokenizer,
        messages_with_system_date,
    ):
        """Benchmark date extraction from system prompt.

        Target: < 1ms
        Tests regex-based date pattern matching.
        """
        result = benchmark(aligner.apply, messages_with_system_date, mock_tokenizer)

        # Verify date was extracted
        assert "cache_align" in str(result.transforms_applied)

    def test_hash_computation(
        self,
        benchmark,
        aligner,
        mock_tokenizer,
        system_prompt_long,
    ):
        """Benchmark stable prefix hash computation.

        Target: < 0.5ms
        Tests hash stability for cache hit prediction.
        """
        messages = [
            {"role": "system", "content": system_prompt_long},
            {"role": "user", "content": "Hello"},
        ]

        result = benchmark(aligner.apply, messages, mock_tokenizer)

        # Verify hash was computed
        assert result.cache_metrics is not None
        assert result.cache_metrics.stable_prefix_hash

    def test_whitespace_normalization(
        self,
        benchmark,
        aligner,
        mock_tokenizer,
    ):
        """Benchmark whitespace normalization.

        Target: < 0.5ms
        Tests string processing for consistent formatting.
        """
        messy_content = """You are a helpful    assistant.

Current date: 2025-01-06


This has     excessive   whitespace.


And multiple blank lines."""

        messages = [
            {"role": "system", "content": messy_content},
            {"role": "user", "content": "Hi"},
        ]

        result = benchmark(aligner.apply, messages, mock_tokenizer)

        assert result.messages[0]["content"] != messy_content  # Was normalized

    def test_long_system_prompt(
        self,
        benchmark,
        aligner,
        mock_tokenizer,
        system_prompt_long,
    ):
        """Benchmark processing long system prompts.

        Target: < 2ms
        Tests performance with larger instruction sets.
        """
        # Add date to trigger alignment
        content_with_date = system_prompt_long + "\n\nCurrent date: 2025-01-06"

        messages = [
            {"role": "system", "content": content_with_date},
            {"role": "user", "content": "Help me with code"},
        ]

        result = benchmark(aligner.apply, messages, mock_tokenizer)

        assert result.cache_metrics is not None

    def test_multiple_system_messages(
        self,
        benchmark,
        aligner,
        mock_tokenizer,
    ):
        """Benchmark with multiple system messages.

        Target: < 3ms
        Tests edge case of multiple system prompts.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.\n\nCurrent date: 2025-01-06",
            },
            {"role": "system", "content": "Additional context: Technical support mode."},
            {"role": "user", "content": "Hello"},
        ]

        benchmark(aligner.apply, messages, mock_tokenizer)


# RollingWindow benchmarks were retired in PR-B1 along with the
# RollingWindow transform itself. Live-zone-only compression
# (PR-B2..B7) does not drop messages, so message-count-based
# benchmarks no longer have a baseline to measure. Phase B's own
# performance suite lives alongside the live-zone dispatcher.


class TestTransformPipelineBenchmarks:
    """Benchmarks for full transform pipeline.

    Tests the complete flow:
    CacheAligner -> SmartCrusher -> RollingWindow

    Expected performance:
    - Simple conversation: < 5ms
    - Agentic with tools: < 30ms
    - Large RAG context: < 50ms
    """

    @pytest.fixture
    def mock_provider(self, mock_token_counter):
        """Create mock provider for pipeline."""
        from unittest.mock import Mock

        provider = Mock()
        provider.get_token_counter.return_value = mock_token_counter
        return provider

    @pytest.fixture
    def pipeline(self, smart_crusher_config, cache_aligner_config, mock_provider):
        """Create transform pipeline.

        PR-B1 retired RollingWindow; the live-zone-only architecture
        runs CacheAligner → SmartCrusher (followed by ContentRouter
        in production, omitted here to keep the fixture pure-stage).
        """
        from headroom.transforms.cache_aligner import CacheAligner
        from headroom.transforms.pipeline import TransformPipeline
        from headroom.transforms.smart_crusher import SmartCrusher

        return TransformPipeline(
            transforms=[
                CacheAligner(cache_aligner_config),
                SmartCrusher(smart_crusher_config),
            ],
            provider=mock_provider,
        )

    def test_pipeline_simple(
        self,
        benchmark,
        pipeline,
        messages_with_system_date,
    ):
        """Benchmark pipeline on simple conversation.

        Target: < 5ms
        Tests minimal overhead scenario.
        """
        benchmark(
            pipeline.apply,
            messages_with_system_date,
            "benchmark-model",
            model_limit=100000,
        )

    def test_pipeline_agentic(
        self,
        benchmark,
        pipeline,
        conversation_50_turns,
    ):
        """Benchmark pipeline on agentic conversation.

        Target: < 30ms
        Tests realistic agentic workload.
        """
        result = benchmark(
            pipeline.apply,
            conversation_50_turns,
            "benchmark-model",
            model_limit=50000,
        )

        assert result.tokens_after < result.tokens_before

    def test_pipeline_rag(
        self,
        benchmark,
        pipeline,
        rag_conversation_20k,
    ):
        """Benchmark pipeline on RAG conversation.

        Target: < 50ms
        Tests large context handling.

        Note: CacheAligner may add small markers (e.g., "[Dynamic Context]"),
        so we allow up to 1% token increase.
        """
        result = benchmark(
            pipeline.apply,
            rag_conversation_20k,
            "benchmark-model",
            model_limit=30000,
        )

        # Allow for small overhead from cache alignment markers
        assert result.tokens_after <= result.tokens_before * 1.01

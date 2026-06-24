"""Tests for local token/cache mode benchmark harness."""

from benchmarks.proxy_mode_benchmark import run_local_benchmark


def test_local_mode_benchmark_shows_compression_and_cache_tradeoff() -> None:
    results = run_local_benchmark(turns=6)

    baseline = results["baseline"]
    token = results["token"]
    cache = results["cache"]

    assert token.total_tokens_saved > 0
    assert cache.total_tokens_saved > 0
    assert token.total_sent_tokens < baseline.total_sent_tokens
    assert cache.total_sent_tokens < baseline.total_sent_tokens

    # Cache mode should preserve prefix better than token mode.
    assert cache.total_cache_read_tokens >= token.total_cache_read_tokens

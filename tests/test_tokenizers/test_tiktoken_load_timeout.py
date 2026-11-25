"""tiktoken vocab loading must be bounded (GH #956).

tiktoken downloads its BPE vocab via ``requests.get`` with no timeout, so a
stalled/firewalled connection blocks indefinitely. The proxy calls this lazily
inside a request worker, so the only bound was the 30s compression timeout —
yielding "every request times out, 0 compression". The bounded loader caps the
wait and falls back to estimation instead.
"""

from __future__ import annotations

import os
import tempfile
import time
from unittest.mock import patch

import pytest

from headroom.tokenizers import tiktoken_counter as tc
from headroom.tokenizers.estimator import EstimatingTokenCounter
from headroom.tokenizers.registry import TokenizerRegistry


@pytest.fixture(autouse=True)
def _reset_encoding_state():
    import tiktoken

    tc._get_encoding.cache_clear()
    tc._download_failed.clear()
    tc._disk_load_failed.clear()
    tc._disk_load_permanently_failed.clear()
    tiktoken.registry.ENCODINGS.clear()
    yield
    tc._get_encoding.cache_clear()
    tc._download_failed.clear()
    tc._disk_load_failed.clear()
    tc._disk_load_permanently_failed.clear()
    tiktoken.registry.ENCODINGS.clear()


def _stalled_load_encoding_worker(encoding_name: str, q) -> None:
    """Simulates a slow disk load that times out."""
    time.sleep(2.0)  # Longer than default timeout
    q.put(("ok", encoding_name))


def _stall_then_succeed_worker(encoding_name: str, q) -> None:
    """First call stalls, second succeeds - for retry test."""
    if not hasattr(_stall_then_succeed_worker, "call_count"):
        _stall_then_succeed_worker.call_count = 0
    _stall_then_succeed_worker.call_count += 1
    if _stall_then_succeed_worker.call_count == 1:
        time.sleep(2.0)  # First call: stall longer than timeout
        q.put(("ok", encoding_name))
    else:
        # Second call: succeed quickly
        q.put(("ok", encoding_name))


def _success_load_worker(encoding_name: str, q) -> None:
    """Successful disk load worker - returns a real encoding."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    q.put(("ok", enc))


def _invalid_cache_worker(encoding_name: str, q) -> None:
    """Returns invalid_cache status."""
    q.put(("invalid_cache", encoding_name))


def _not_cached_worker(encoding_name: str, q) -> None:
    """Returns not_cached status."""
    q.put(("not_cached", encoding_name))


# Global sentinel for mock download worker
_MOCK_DOWNLOAD_SENTINEL = None


def _mock_download_worker(encoding_name: str, q) -> None:
    """Mock download worker that returns a sentinel."""
    q.put(("ok", _MOCK_DOWNLOAD_SENTINEL))


def _stalled_download_encoding_worker(encoding_name: str, q) -> None:
    """Simulates a slow download that times out."""
    time.sleep(2.0)
    q.put(("error", "timeout"))


def test_load_encoding_is_bounded_on_stall(monkeypatch: pytest.MonkeyPatch) -> None:

    # Mock the load worker to simulate a stall
    monkeypatch.setattr(tc, "_load_encoding_worker", _stalled_load_encoding_worker)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

    start = time.perf_counter()
    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("stall-enc")
    elapsed = time.perf_counter() - start
    assert elapsed < 1.5, f"load was not bounded (took {elapsed:.2f}s vs the 2s stall)"


def test_failed_encoding_allows_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a disk load timeout, a subsequent call with valid cache should succeed.

    This tests the fix for: transient timeout should not permanently block retries.
    """
    import tiktoken
    # Reset call counter
    _stall_then_succeed_worker.call_count = 0

    # Mock the load worker to simulate a stall on first call, succeed on second
    monkeypatch.setattr(tc, "_load_encoding_worker", _stall_then_succeed_worker)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

    # First call: times out
    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("retry-enc")

    # Second call: should NOT short-circuit, should try again and succeed
    monkeypatch.setattr(tc, "_load_encoding_worker", _success_load_worker)
    enc = tc.load_encoding("retry-enc")
    assert isinstance(enc, tiktoken.Encoding)
    assert enc.name == "cl100k_base"


def test_fast_load_returns_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    sentinel = object()
    # Mock the fast path: encoding already in memory cache
    # Pre-populate the in-memory cache manually (simulating what the fast path does)
    tiktoken.registry.ENCODINGS["fast-enc"] = sentinel
    # Clear the _get_encoding cache to force the fast path check
    tc._get_encoding.cache_clear()

    # Now load_encoding should hit the fast path and return the same sentinel
    result = tc.load_encoding("fast-enc")
    assert result is sentinel


def test_registry_falls_back_to_estimator_on_stall(monkeypatch: pytest.MonkeyPatch) -> None:

    # Use a unique empty cache dir for cold cache
    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)

        # Mock the load worker to simulate a stall
        monkeypatch.setattr(tc, "_load_encoding_worker", _stalled_load_encoding_worker)
        monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

        counter = TokenizerRegistry()._create_tiktoken("gpt-4")
        assert isinstance(counter, EstimatingTokenCounter)


def test_offline_warm_cache_works_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm cache: encoding already in memory or on disk should work offline.

    The encoding is pre-loaded into the in-memory cache, then we enable
    offline mode and verify it loads without network access.
    """
    import tiktoken

    # Pre-load the encoding into the in-memory cache (simulating warm cache)
    tiktoken.get_encoding("cl100k_base")
    assert "cl100k_base" in tiktoken.registry.ENCODINGS

    # Now enable offline mode
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")

    # Loading the same encoding should work without network access
    # because it's already in the in-memory cache
    counter = TokenizerRegistry()._create_tiktoken("gpt-4")
    # Should get a real tiktoken counter, not fall back to estimation
    assert not isinstance(counter, EstimatingTokenCounter)
    assert hasattr(counter, "encoding")
    assert counter.encoding is not None


def test_offline_warm_disk_cache_works_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm cache on disk: encoding cached on disk should work offline.

    The encoding is cached to disk via TIKTOKEN_CACHE_DIR, then we enable
    offline mode and verify it loads without network access.
    """
    import tiktoken

    # Create a temporary cache directory
    with tempfile.TemporaryDirectory() as cache_dir:
        # Pre-populate the disk cache - set cache dir BEFORE loading
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        tiktoken.get_encoding("cl100k_base")

        # Clear the in-memory cache to force disk load
        tc._get_encoding.cache_clear()
        tiktoken.registry.ENCODINGS.clear()

        # Now enable offline mode
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")

        # Use REAL worker (not mocked) - the cache is valid and should load
        # Loading the same encoding should work from disk cache
        counter = TokenizerRegistry()._create_tiktoken("gpt-4")
        assert not isinstance(counter, EstimatingTokenCounter)
        assert hasattr(counter, "encoding")
        assert counter.encoding is not None
        # Verify it actually encodes correctly
        assert counter.encoding.encode("hello world") == [15339, 1917]


def test_offline_cold_cache_refuses_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cold cache: encoding not cached should raise OfflineEgressBlocked and fall back.

    When the encoding is not in memory or on disk, offline mode should
    prevent the download and fall back to estimation.
    """
    import tiktoken

    # Ensure cold cache - clear both in-memory and disk cache
    tc._get_encoding.cache_clear()
    tiktoken.registry.ENCODINGS.clear()

    # Use a unique cache dir that's empty
    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)

        # Enable offline mode
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")

        # Should fall back to estimation because cold cache + offline = no download
        counter = TokenizerRegistry()._create_tiktoken("gpt-4")
        assert isinstance(counter, EstimatingTokenCounter)


def test_offline_invalid_cache_entry_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid cache entry (hash mismatch) should not trigger network calls.

    When the cache file exists but has invalid content (wrong hash), the loader
    should detect this and fail fast without attempting a download, even when
    not in offline mode. This prevents tiktoken's retry behavior from making
    HTTP calls.
    """
    from unittest.mock import patch

    import tiktoken

    tc._get_encoding.cache_clear()
    tc._download_failed.clear()
    tc._disk_load_failed.clear()
    tiktoken.registry.ENCODINGS.clear()

    with tempfile.TemporaryDirectory() as cache_dir:
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        import hashlib

        cache_key = hashlib.sha1(url.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, cache_key)

        # Write invalid data to the cache file
        with open(cache_path, "wb") as f:
            f.write(b"invalid tiktoken data")

        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")

        with patch("requests.get") as mock_get:
            with pytest.raises(tc.TiktokenLoadError):
                tc.load_encoding("cl100k_base")

            # No network calls should be made
            assert not mock_get.called, "requests.get should not be called for invalid cache"


def test_offline_warm_cache_payload_tokens_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm cache + offline mode: payload_tokens should work without network.

    The encoding is pre-loaded into the in-memory cache, then we enable
    offline mode and verify payload_tokens works without network access.
    """
    import tiktoken

    from headroom.transforms.kompress_compressor import payload_tokens

    # Pre-load the encoding into the in-memory cache (simulating warm cache)
    tiktoken.get_encoding("cl100k_base")
    assert "cl100k_base" in tiktoken.registry.ENCODINGS

    # Now enable offline mode
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")

    # Clear loader cache but keep tiktoken's in-memory registry
    tc._get_encoding.cache_clear()
    tc._download_failed.clear()
    tc._disk_load_failed.clear()

    with patch("requests.get") as mock_get:
        result = payload_tokens("hello world")
        assert result == 2  # "hello world" = 2 tokens in cl100k_base
        assert not mock_get.called, "requests.get should not be called for warm cache"


def test_offline_warm_cache_evaluate_token_counting_accuracy_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm cache + offline mode: evaluate_token_counting_accuracy should work without network.

    The encoding is pre-loaded into the in-memory cache, then we enable
    offline mode and verify evaluate_token_counting_accuracy works without network access.
    """
    import tiktoken

    from headroom.evals.batch_compression_eval import evaluate_token_counting_accuracy

    # Pre-load the encoding into the in-memory cache (simulating warm cache)
    tiktoken.get_encoding("cl100k_base")
    assert "cl100k_base" in tiktoken.registry.ENCODINGS

    # Now enable offline mode
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")

    # Clear loader cache but keep tiktoken's in-memory registry
    tc._get_encoding.cache_clear()
    tc._download_failed.clear()
    tc._disk_load_failed.clear()

    with patch("requests.get") as mock_get:
        result = evaluate_token_counting_accuracy()
        assert result.test_cases == 6
        assert not mock_get.called, "requests.get should not be called for warm cache"


def test_load_encoding_timeout_bounds_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """50ms deadline: stalled get_encoding should time out within observation window.

    This tests the specific scenario from GH #956: a 50ms deadline with a
    stalled tiktoken.get_encoding should return TiktokenLoadError within the
    observation window, not block for 500ms.
    """
    import threading
    import time

    import tiktoken

    # Create a stall that waits on an event, simulating a slow download
    stall_event = threading.Event()

    def stalled_get_encoding(_name: str):
        stall_event.wait(timeout=5.0)  # Wait for signal (never comes in test)
        return object()

    monkeypatch.setattr(tiktoken, "get_encoding", stalled_get_encoding)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.05")  # 50ms

    start = time.perf_counter()
    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("stall-timeout-enc")
    elapsed = time.perf_counter() - start

    # Should timeout within ~200ms (4x the deadline), not 500ms+
    assert elapsed < 0.2, f"load was not bounded (took {elapsed:.3f}s vs 0.05s deadline)"

    # Verify the stall was actually waiting (test infrastructure check)
    assert not stall_event.is_set()


def test_online_corrupt_cache_repairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Online mode: corrupt cache entry should be repaired by downloading fresh copy."""
    import tiktoken

    tc._get_encoding.cache_clear()
    tc._download_failed.clear()
    tc._disk_load_failed.clear()
    tiktoken.registry.ENCODINGS.clear()

    with tempfile.TemporaryDirectory() as cache_dir:
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        import hashlib

        cache_key = hashlib.sha1(url.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, cache_key)

        # Write corrupt data to the cache file
        with open(cache_path, "wb") as f:
            f.write(b"corrupt tiktoken data")

        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        monkeypatch.setenv("HEADROOM_OFFLINE", "0")  # Online mode
        monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "60")  # Download can be slow

        # load_encoding should detect corrupt cache and trigger tiktoken's repair
        # (tiktoken.get_encoding will overwrite corrupt cache with fresh download)
        enc = tc.load_encoding("cl100k_base")
        assert isinstance(enc, tiktoken.Encoding)
        assert enc.name == "cl100k_base"
        # Cache file should now be valid (repaired)
        assert os.path.getsize(cache_path) > 100000  # Valid .tiktoken file is ~1.6MB


def test_disk_load_miss_retryable_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline mode: cache miss should be retryable once vocabulary is added to cache.

    Uses real tiktoken download to populate cache with correct hash.
    """
    import tiktoken

    tc._get_encoding.cache_clear()
    tc._download_failed.clear()
    tc._disk_load_failed.clear()
    tc._disk_load_permanently_failed.clear()
    tiktoken.registry.ENCODINGS.clear()

    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        # First, populate cache ONLINE
        tiktoken.get_encoding("cl100k_base")
        # Clear in-memory to force disk load
        tiktoken.registry.ENCODINGS.clear()
        tc._get_encoding.cache_clear()

        # Now switch to OFFLINE mode
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "2.0")

        # First attempt: cache exists on disk, should work
        enc = tc.load_encoding("cl100k_base")
        assert isinstance(enc, tiktoken.Encoding)
        assert enc.name == "cl100k_base"

        # Now test the actual scenario: cache miss -> cache available -> retry works
        # Clear everything again
        tc._get_encoding.cache_clear()
        tc._download_failed.clear()
        tc._disk_load_failed.clear()
        tc._disk_load_permanently_failed.clear()
        tiktoken.registry.ENCODINGS.clear()

        # Remove cache file to simulate cold cache miss
        import os
        import hashlib
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        cache_key = hashlib.sha1(url.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, cache_key)
        os.remove(cache_path)

        # First attempt: no cache file (cold cache miss)
        with pytest.raises(tc.TiktokenLoadError) as exc_info:
            tc.load_encoding("cl100k_base")
        assert "not in local cache and HEADROOM_OFFLINE=1" in str(exc_info.value)

        # Should NOT be in permanent failure set (miss is retryable)
        assert "cl100k_base" not in tc._disk_load_permanently_failed
        assert "cl100k_base" not in tc._download_failed

        # Now restore cache file (simulate vocabulary becoming available)
        # We can just re-download it (but we're offline now, so use the previously downloaded one)
        # Actually, the file was deleted. Let's download again while temporarily online.
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        tiktoken.get_encoding("cl100k_base")
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        tiktoken.registry.ENCODINGS.clear()
        tc._get_encoding.cache_clear()

        # Second attempt: cache now exists, offline mode should work
        enc2 = tc.load_encoding("cl100k_base")
        assert isinstance(enc2, tiktoken.Encoding)
        assert enc2.name == "cl100k_base"
        # Should encode correctly
        assert len(enc2.encode("hello world")) > 0


def test_offline_accuracy_benchmark_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline mode with cold cache: accuracy benchmark should report unavailable reference."""
    import tiktoken

    from headroom.evals.batch_compression_eval import evaluate_token_counting_accuracy

    # Ensure cold cache
    tc._get_encoding.cache_clear()
    tc._download_failed.clear()
    tc._disk_load_failed.clear()
    tiktoken.registry.ENCODINGS.clear()

    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")

        # Should not crash, should return a result indicating unavailable reference
        result = evaluate_token_counting_accuracy()

        # The benchmark should complete but mark reference as unavailable
        assert hasattr(result, "reference_available")
        assert result.reference_available is False
        # Should not publish estimated ground truth as actual counts
        assert not hasattr(result, "ground_truth_counts") or result.ground_truth_counts is None

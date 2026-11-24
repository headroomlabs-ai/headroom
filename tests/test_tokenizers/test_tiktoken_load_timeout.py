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
    tc._load_failed.clear()
    tiktoken.registry.ENCODINGS.clear()
    yield
    tc._get_encoding.cache_clear()
    tc._load_failed.clear()
    tiktoken.registry.ENCODINGS.clear()


def _stalled_get_encoding(_name: str):
    # Simulates tiktoken's unbounded network download stalling.
    time.sleep(2.0)
    return object()


def test_load_encoding_is_bounded_on_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", _stalled_get_encoding)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

    start = time.perf_counter()
    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("stall-enc")
    elapsed = time.perf_counter() - start
    assert elapsed < 1.5, f"load was not bounded (took {elapsed:.2f}s vs the 2s stall)"


def test_failed_encoding_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", _stalled_get_encoding)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("stall-enc-2")

    # A second request must fail instantly via the _load_failed short-circuit,
    # not wait out the timeout again (this is what makes it not "every request").
    start = time.perf_counter()
    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("stall-enc-2")
    assert time.perf_counter() - start < 0.1


def test_fast_load_returns_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    sentinel = object()
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _name: sentinel)
    assert tc.load_encoding("fast-enc") is sentinel


def test_registry_falls_back_to_estimator_on_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    # Use a unique empty cache dir for cold cache
    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        
        monkeypatch.setattr(tiktoken, "get_encoding", _stalled_get_encoding)
        monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

        counter = TokenizerRegistry()._create_tiktoken("gpt-4")
        assert isinstance(counter, EstimatingTokenCounter)


def test_offline_warm_cache_works_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm cache: encoding already in memory or on disk should work offline.

    The encoding is pre-loaded into the in-memory cache, then we enable
    offline mode and verify it loads without network access.
    """
    import tiktoken
    from headroom.offline import guard_egress
    from headroom.offline import OfflineEgressBlocked

    # Pre-load the encoding into the in-memory cache (simulating warm cache)
    enc = tiktoken.get_encoding("cl100k_base")
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
    from headroom.offline import OfflineEgressBlocked

    # Create a temporary cache directory
    with tempfile.TemporaryDirectory() as cache_dir:
        # Pre-populate the disk cache - set cache dir BEFORE loading
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        enc = tiktoken.get_encoding("cl100k_base")

        # Clear the in-memory cache to force disk load
        tc._get_encoding.cache_clear()
        tiktoken.registry.ENCODINGS.clear()

        # Now enable offline mode
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")

        # Loading the same encoding should work from disk cache
        counter = TokenizerRegistry()._create_tiktoken("gpt-4")
        assert not isinstance(counter, EstimatingTokenCounter)
        assert hasattr(counter, "encoding")
        assert counter.encoding is not None


def test_offline_cold_cache_refuses_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cold cache: encoding not cached should raise OfflineEgressBlocked and fall back.

    When the encoding is not in memory or on disk, offline mode should
    prevent the download and fall back to estimation.
    """
    import tiktoken
    from headroom.offline import OfflineEgressBlocked

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
    import tiktoken
    from unittest.mock import patch

    tc._get_encoding.cache_clear()
    tc._load_failed.clear()
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
    enc = tiktoken.get_encoding("cl100k_base")
    assert "cl100k_base" in tiktoken.registry.ENCODINGS

    # Now enable offline mode
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")

    # Clear loader cache but keep tiktoken's in-memory registry
    tc._get_encoding.cache_clear()
    tc._load_failed.clear()

    with patch("requests.get") as mock_get:
        result = payload_tokens("hello world")
        assert result == 2  # "hello world" = 2 tokens in cl100k_base
        assert not mock_get.called, "requests.get should not be called for warm cache"


def test_offline_warm_cache_evaluate_token_counting_accuracy_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm cache + offline mode: evaluate_token_counting_accuracy should work without network.

    The encoding is pre-loaded into the in-memory cache, then we enable
    offline mode and verify evaluate_token_counting_accuracy works without network access.
    """
    import tiktoken
    from headroom.evals.batch_compression_eval import evaluate_token_counting_accuracy

    # Pre-load the encoding into the in-memory cache (simulating warm cache)
    enc = tiktoken.get_encoding("cl100k_base")
    assert "cl100k_base" in tiktoken.registry.ENCODINGS

    # Now enable offline mode
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")

    # Clear loader cache but keep tiktoken's in-memory registry
    tc._get_encoding.cache_clear()
    tc._load_failed.clear()

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
    import tiktoken
    import time
    import threading

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

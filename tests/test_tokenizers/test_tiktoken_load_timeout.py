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

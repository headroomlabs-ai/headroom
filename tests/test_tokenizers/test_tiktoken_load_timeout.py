"""tiktoken vocab loading must be bounded (GH #956).

tiktoken downloads its BPE vocab via ``requests.get`` with no timeout, so a
stalled/firewalled connection blocks indefinitely. The proxy calls this lazily
inside a request worker, so the only bound was the 30s compression timeout —
yielding "every request times out, 0 compression". The bounded loader caps the
wait and falls back to estimation instead.
"""

from __future__ import annotations

import time

import pytest

from headroom.tokenizers import tiktoken_counter as tc
from headroom.tokenizers.estimator import EstimatingTokenCounter
from headroom.tokenizers.registry import TokenizerRegistry


@pytest.fixture(autouse=True)
def _reset_encoding_state():
    tc._get_encoding.cache_clear()
    tc._load_failed.clear()
    yield
    tc._get_encoding.cache_clear()
    tc._load_failed.clear()


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

    monkeypatch.setattr(tiktoken, "get_encoding", _stalled_get_encoding)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

    counter = TokenizerRegistry()._create_tiktoken("gpt-4")
    assert isinstance(counter, EstimatingTokenCounter)

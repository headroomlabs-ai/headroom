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
    tc._pending_loads.clear()
    yield
    tc._get_encoding.cache_clear()
    tc._pending_loads.clear()


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

    # A second request must fail instantly while the timed-out load is still running,
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


def test_a_slow_download_that_lands_is_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Online, a timed-out load fails fast while its download runs, and the
    first call after it finishes returns the encoding without a restart."""
    tiktoken = pytest.importorskip("tiktoken")
    sentinel = object()

    def _slow(_name: str) -> object:
        time.sleep(0.5)
        return sentinel

    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setattr(tiktoken, "get_encoding", _slow)
    try:
        with pytest.raises(tc.TiktokenLoadError):
            tc.load_encoding("slow-enc")
        start = time.perf_counter()
        with pytest.raises(tc.TiktokenLoadError):
            tc.load_encoding("slow-enc")
        assert time.perf_counter() - start < 0.1  # no second wait while it runs

        deadline = time.monotonic() + 5.0
        while tc._pending_loads["slow-enc"][0].is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert tc.load_encoding("slow-enc") is sentinel
        assert "slow-enc" not in tc._pending_loads
    finally:
        tc._get_encoding.cache_clear()
        tc._pending_loads.clear()


def test_a_timed_out_load_that_then_fails_stays_a_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer's repro: online, a load times out and its worker then
    fails with a transport error. Callers catch TiktokenLoadError and fall
    back to estimation, so the late failure must arrive as one, not as the
    worker's raw error, and a later successful load must still return the
    exact vocabulary."""
    import threading

    tiktoken = pytest.importorskip("tiktoken")
    requests = pytest.importorskip("requests")
    from headroom.providers import openai as openai_provider
    from headroom.providers.openai import OpenAIProvider, OpenAITokenCounter

    release = threading.Event()
    state = {"fail": True}
    encoding = type("_Words", (), {"encode": lambda self, text, **_kw: text.split()})()

    def _get_encoding(_name: str) -> object:
        if state["fail"]:
            release.wait(5.0)
            raise requests.ConnectionError("vocab host unreachable")
        return encoding

    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setattr(tiktoken, "get_encoding", _get_encoding)
    openai_provider._get_encoding.cache_clear()
    try:
        provider = OpenAIProvider()
        fallback = provider.get_token_counter("gpt-4o")
        assert isinstance(fallback, EstimatingTokenCounter)

        worker = tc._pending_loads["o200k_base"][0]
        release.set()
        worker.join(5.0)
        assert not worker.is_alive()
        assert provider.get_token_counter("gpt-4o") is fallback
        with pytest.raises(tc.TiktokenLoadError) as excinfo:
            tc.load_encoding("o200k_base")
        assert isinstance(excinfo.value.__cause__, requests.ConnectionError)

        state["fail"] = False
        assert tc.load_encoding("o200k_base") is encoding
        openai_provider._get_encoding.cache_clear()
        exact = OpenAIProvider().get_token_counter("gpt-4o")
        assert isinstance(exact, OpenAITokenCounter)
        assert exact.count_text("one two three") == 3
    finally:
        release.set()
        openai_provider._get_encoding.cache_clear()

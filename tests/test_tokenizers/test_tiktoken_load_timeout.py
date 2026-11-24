"""tiktoken vocab loading must be bounded (GH #956).

tiktoken downloads its BPE vocab via ``requests.get`` with no timeout, so a
stalled/firewalled connection blocks indefinitely. The proxy calls this lazily
inside a request worker, so the only bound was the 30s compression timeout —
yielding "every request times out, 0 compression". The bounded loader caps the
wait and falls back to estimation instead.

Only the first attempt per encoding may hold a caller, and only until its
deadline; later attempts run in the background and are adopted when they
finish. Every failure reaches callers as ``TiktokenLoadError``.
"""

from __future__ import annotations

import threading
import time

import pytest

from headroom.tokenizers import tiktoken_counter as tc
from headroom.tokenizers.estimator import EstimatingTokenCounter
from headroom.tokenizers.registry import TokenizerRegistry


@pytest.fixture(autouse=True)
def _reset_encoding_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    tc._get_encoding.cache_clear()
    tc._slots.clear()
    yield
    tc._get_encoding.cache_clear()
    tc._slots.clear()


def _eventually_load(name: str, deadline_s: float = 5.0) -> object:
    """Call load_encoding until a background attempt has landed."""
    deadline = time.monotonic() + deadline_s
    while True:
        try:
            return tc.load_encoding(name)
        except tc.TiktokenLoadError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def _timed_load(name: str) -> float:
    start = time.perf_counter()
    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding(name)
    return time.perf_counter() - start


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

    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setattr(tiktoken, "get_encoding", _slow)
    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("slow-enc")
    assert _timed_load("slow-enc") < 0.1  # no second wait while it runs

    tc._slots["slow-enc"].worker.join(5.0)
    assert tc.load_encoding("slow-enc") is sentinel


def test_a_timed_out_load_that_then_fails_stays_a_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer's repro: online, a load times out and its worker then
    fails with a transport error. Callers catch TiktokenLoadError and fall
    back to estimation, so the late failure must arrive as one, not as the
    worker's raw error, and a later successful load must still return the
    exact vocabulary."""
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
    monkeypatch.setattr(tc, "_RETRY_COOLDOWN_SECONDS", 0.0)
    openai_provider._get_encoding.cache_clear()
    try:
        provider = OpenAIProvider()
        fallback = provider.get_token_counter("gpt-4o")
        assert isinstance(fallback, EstimatingTokenCounter)

        worker = tc._slots["o200k_base"].worker
        release.set()
        worker.join(5.0)
        assert not worker.is_alive()
        assert provider.get_token_counter("gpt-4o") is fallback
        with pytest.raises(tc.TiktokenLoadError) as excinfo:
            tc.load_encoding("o200k_base")
        assert isinstance(excinfo.value.__cause__, requests.ConnectionError)

        state["fail"] = False
        assert _eventually_load("o200k_base") is encoding
        openai_provider._get_encoding.cache_clear()
        exact = OpenAIProvider().get_token_counter("gpt-4o")
        assert isinstance(exact, OpenAITokenCounter)
        assert exact.count_text("one two three") == 3
    finally:
        release.set()
        openai_provider._get_encoding.cache_clear()


def test_a_load_that_fails_fast_is_a_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vocab download that fails inside the wait (DNS failure on a host
    without network) reached callers as the raw transport error, which the
    registry, the providers, the eval and the Kompress counter do not catch."""
    tiktoken = pytest.importorskip("tiktoken")
    requests = pytest.importorskip("requests")

    def _fails(_name: str) -> object:
        raise requests.ConnectionError("Name or service not known")

    monkeypatch.setattr(tiktoken, "get_encoding", _fails)
    with pytest.raises(tc.TiktokenLoadError) as excinfo:
        tc.load_encoding("fast-fail-enc")
    assert isinstance(excinfo.value.__cause__, requests.ConnectionError)


def test_a_failing_vocab_host_is_retried_once_per_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiktoken = pytest.importorskip("tiktoken")
    attempts: list[str] = []

    def _fails(name: str) -> object:
        attempts.append(name)
        raise ConnectionError("vocab host unreachable")

    monkeypatch.setattr(tiktoken, "get_encoding", _fails)
    for _ in range(20):
        with pytest.raises(tc.TiktokenLoadError):
            tc.load_encoding("cooldown-enc")
    assert attempts == ["cooldown-enc"]

    tc._slots["cooldown-enc"].retry_at = 0.0
    _eventually_attempted = time.monotonic() + 5.0
    while len(attempts) < 2 and time.monotonic() < _eventually_attempted:
        with pytest.raises(tc.TiktokenLoadError):
            tc.load_encoding("cooldown-enc")
        time.sleep(0.02)
    assert attempts == ["cooldown-enc", "cooldown-enc"]


@pytest.mark.parametrize(
    "fail_after",
    [1.2, 0.4],
    ids=["fails-after-the-timeout", "fails-just-inside-the-timeout"],
)
def test_a_slowly_failing_vocab_host_holds_only_the_first_request(
    monkeypatch: pytest.MonkeyPatch, fail_after: float
) -> None:
    """Review repros: each attempt takes a while and then fails. Only the
    first attempt may hold a caller; retries run in the background."""
    tiktoken = pytest.importorskip("tiktoken")

    def _slow_fail(_name: str) -> object:
        time.sleep(fail_after)
        raise ConnectionError("connection reset")

    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.6")
    monkeypatch.setattr(tiktoken, "get_encoding", _slow_fail)
    monkeypatch.setattr(tc, "_RETRY_COOLDOWN_SECONDS", 0.0)

    waits = []
    end = time.monotonic() + 4.0
    while time.monotonic() < end:
        waits.append(_timed_load("slow-fail-enc"))
        time.sleep(0.02)
    assert sum(1 for w in waits if w >= 0.3) == 1, waits


def test_concurrent_first_callers_share_one_bounded_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiktoken = pytest.importorskip("tiktoken")
    started: list[str] = []

    def _stall(name: str) -> object:
        started.append(name)
        time.sleep(1.0)
        return object()

    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.3")
    monkeypatch.setattr(tiktoken, "get_encoding", _stall)
    waits: list[float] = []
    lock = threading.Lock()

    def _caller() -> None:
        elapsed = _timed_load("shared-enc")
        with lock:
            waits.append(elapsed)

    callers = [threading.Thread(target=_caller) for _ in range(5)]
    for t in callers:
        t.start()
    for t in callers:
        t.join(5.0)
    assert started == ["shared-enc"]
    assert len(waits) == 5 and max(waits) < 0.6, waits


def test_an_offline_load_is_bounded_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline there is no download, but building an encoding waits on
    tiktoken's process-wide registry lock, which a host's own stalled
    download can hold. The offline load must keep the same bound."""
    pytest.importorskip("tiktoken")
    import tiktoken.registry

    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.3")
    holder_ready = threading.Event()
    release = threading.Event()

    def _hold_registry_lock() -> None:
        with tiktoken.registry._lock:
            holder_ready.set()
            release.wait(5.0)

    holder = threading.Thread(target=_hold_registry_lock)
    holder.start()
    holder_ready.wait(5.0)
    try:
        # Not built yet, so tiktoken takes its registry lock to build it.
        assert _timed_load("headroom-unbuilt-enc") < 0.8
    finally:
        release.set()
        holder.join(5.0)
        slot = tc._slots.get("headroom-unbuilt-enc")
        if slot is not None and slot.worker is not None:
            slot.worker.join(5.0)


def test_offline_refusals_start_at_most_one_thread_per_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiktoken = pytest.importorskip("tiktoken")
    from headroom.offline import guard_egress

    def _cold(name: str) -> object:
        guard_egress(f"tiktoken {name} vocabulary download", "https://vocab.invalid/x")
        raise AssertionError("unreachable")

    started: list[str] = []
    real_start = threading.Thread.start

    def _start(self: threading.Thread) -> None:
        started.append(self.name)
        real_start(self)

    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setattr(tiktoken, "get_encoding", _cold)
    monkeypatch.setattr(threading.Thread, "start", _start)
    for _ in range(5):
        with pytest.raises(tc.TiktokenVocabNotCached):
            tc.load_encoding("offline-enc")
    assert [n for n in started if n.startswith("tiktoken-load-")] == ["tiktoken-load-offline-enc"]


def test_a_thread_that_cannot_start_is_a_load_error_and_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiktoken = pytest.importorskip("tiktoken")
    sentinel = object()
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _name: sentinel)
    real_start = threading.Thread.start

    def _no_threads(self: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _no_threads)
    with pytest.raises(tc.TiktokenLoadError) as excinfo:
        tc.load_encoding("no-thread-enc")
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("no-thread-enc")  # within the cooldown, still a load error

    monkeypatch.setattr(threading.Thread, "start", real_start)
    tc._slots["no-thread-enc"].retry_at = 0.0
    assert _eventually_load("no-thread-enc") is sentinel


def test_background_retries_keep_the_refusal_type_and_log_no_stall(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A background retry holds no caller, so it must not log a stalled
    download, and offline it still reports the vocab as not cached."""
    tiktoken = pytest.importorskip("tiktoken")
    from headroom.offline import guard_egress

    def _cold(name: str) -> object:
        time.sleep(0.2)
        guard_egress(f"tiktoken {name} vocabulary download", "https://vocab.invalid/x")
        raise AssertionError("unreachable")

    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setattr(tiktoken, "get_encoding", _cold)
    monkeypatch.setattr(tc, "_OFFLINE_RETRY_SECONDS", 0.0)
    with caplog.at_level("WARNING", logger=tc.logger.name):
        for _ in range(15):
            with pytest.raises(tc.TiktokenVocabNotCached):
                tc.load_encoding("refused-enc")
            time.sleep(0.05)
    assert not any("did not load within" in r.getMessage() for r in caplog.records)


def test_a_loaded_vocab_survives_cache_eviction_without_a_failed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiktoken = pytest.importorskip("tiktoken")

    def _build(name: str) -> str:
        time.sleep(0.05)  # long enough that a zero-wait reload is still running
        return f"enc:{name}"

    monkeypatch.setattr(tiktoken, "get_encoding", _build)
    assert tc.load_encoding("evict-me") == "enc:evict-me"
    for i in range(10):  # more names than the lru_cache holds
        tc.load_encoding(f"filler-{i}")
    assert tc.load_encoding("evict-me") == "enc:evict-me"

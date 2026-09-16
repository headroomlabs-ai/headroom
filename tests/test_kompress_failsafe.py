"""Fail-safe behavior for Kompress on degraded machines.

Reproduces the Windows incident where one pathologically slow ONNX inference
held the execution semaphore forever: every later compression blocked on an
unbounded acquire, every request hit the proxy's 30s stage timeout, and the
proxy delivered 0% savings plus +30s latency until restart. These tests pin
the three layers of defense (bounded acquire, wall-clock budget, preload
canary) and that the normal fast path is untouched.

No ML dependencies — the model/tokenizer are fakes injected via
``_load_kompress``.
"""

import threading
import time

import pytest

import headroom.transforms.kompress_compressor as kc
from headroom.transforms.kompress_compressor import (
    KOMPRESS_ACQUIRE_TIMEOUT_ENV,
    KOMPRESS_CANARY_THRESHOLD_ENV,
    KOMPRESS_EXECUTION_SEMAPHORE_WAIT_MS_ENV,
    KOMPRESS_REQUEST_DEADLINE_ENV,
    KOMPRESS_TIME_BUDGET_ENV,
    KompressCompressor,
    KompressConfig,
)


class FakeEncoding:
    """Mimics a transformers BatchEncoding for is_split_into_words inputs.

    One token per word, no special tokens — word_ids(i) is identity.
    """

    def __init__(self, rows: list[list[str]]):
        self._rows = rows

    def __getitem__(self, key: str):
        if key == "input_ids":
            return [[0] * len(r) for r in self._rows]
        if key == "attention_mask":
            return [[1] * len(r) for r in self._rows]
        raise KeyError(key)

    def word_ids(self, batch_index: int = 0):
        return list(range(len(self._rows[batch_index])))


class FakeTokenizer:
    def __call__(self, words, **kwargs):
        # is_split_into_words inputs: either one word list or a batch of them.
        rows = words if words and isinstance(words[0], list) else [words]
        return FakeEncoding(rows)


class FakeModel:
    """Keeps every other word; optional per-call delay to simulate slowness."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = 0

    def _tick(self):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)

    def get_keep_mask(self, input_ids, attention_mask):
        self._tick()
        return [[i % 2 == 0 for i in range(len(row))] for row in input_ids]

    def get_scores(self, input_ids, attention_mask):
        self._tick()
        return [[1.0 if i % 2 == 0 else 0.0 for i in range(len(row))] for row in input_ids]


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    kc._execution_semaphores.clear()
    monkeypatch.setattr(kc, "_giveup_warned", False)
    for env in (
        KOMPRESS_ACQUIRE_TIMEOUT_ENV,
        KOMPRESS_TIME_BUDGET_ENV,
        KOMPRESS_CANARY_THRESHOLD_ENV,
        KOMPRESS_EXECUTION_SEMAPHORE_WAIT_MS_ENV,
        KOMPRESS_REQUEST_DEADLINE_ENV,
    ):
        monkeypatch.delenv(env, raising=False)
    yield
    kc._execution_semaphores.clear()


def _make_compressor(monkeypatch, model: FakeModel, **config_kwargs) -> KompressCompressor:
    config_kwargs.setdefault("enable_ccr", False)
    # These fixtures are deliberately tiny; drop the production word floor
    # (min_input_words=64) to its clamp so the failsafe paths under test run.
    config_kwargs.setdefault("min_input_words", 10)
    compressor = KompressCompressor(config=KompressConfig(**config_kwargs))
    monkeypatch.setattr(
        kc,
        "_load_kompress",
        lambda model_id, device="auto", **kwargs: (model, FakeTokenizer(), "onnx"),
    )
    return compressor


def _make_block_tracking_semaphore(monkeypatch):
    blocked = threading.Event()

    class TrackingSemaphore:
        def __init__(self):
            self._inner = threading.BoundedSemaphore(1)

        def acquire(self, blocking=True, timeout=None):
            if not blocking:
                return self._inner.acquire(blocking=False)
            if not self._inner.acquire(blocking=False):
                blocked.set()
                if timeout is None:
                    return self._inner.acquire()
                return self._inner.acquire(timeout=timeout)
            return True

        def release(self):
            self._inner.release()

    semaphore = TrackingSemaphore()
    monkeypatch.setattr(kc, "_execution_semaphore", lambda *_args, **_kwargs: semaphore)
    return semaphore, blocked


CONTENT_40_WORDS = " ".join(f"word{i}" for i in range(40))


# ── Normal path: behavior and performance must be unchanged ───────────


def test_fast_model_compresses_normally(monkeypatch):
    model = FakeModel()
    compressor = _make_compressor(monkeypatch, model)

    result = compressor.compress(CONTENT_40_WORDS)

    assert result.compressed != CONTENT_40_WORDS
    assert result.compressed_tokens == 20  # every other word kept
    assert result.compression_ratio == 0.5
    assert model.calls == 1


def test_fast_model_releases_semaphore(monkeypatch):
    compressor = _make_compressor(monkeypatch, FakeModel())
    compressor.compress(CONTENT_40_WORDS)

    semaphore = kc._execution_semaphore("onnx", "onnx")
    assert semaphore.acquire(timeout=0)
    semaphore.release()


def test_semaphore_released_when_inference_raises(monkeypatch):
    class ExplodingModel(FakeModel):
        def get_keep_mask(self, input_ids, attention_mask):
            raise RuntimeError("boom")

    compressor = _make_compressor(monkeypatch, ExplodingModel())
    result = compressor.compress(CONTENT_40_WORDS)

    assert result.compressed == CONTENT_40_WORDS  # passthrough, not an exception
    semaphore = kc._execution_semaphore("onnx", "onnx")
    assert semaphore.acquire(timeout=0)
    semaphore.release()


# ── Bounded acquire: a stuck inference must not wedge other requests ──


def test_stuck_semaphore_passes_through_instead_of_blocking(monkeypatch):
    monkeypatch.setenv(KOMPRESS_ACQUIRE_TIMEOUT_ENV, "0.1")
    model = FakeModel()
    compressor = _make_compressor(monkeypatch, model)

    # Simulate the Windows incident: another thread holds the semaphore
    # indefinitely (abandoned by its asyncio timeout but still running).
    stuck = kc._execution_semaphore("onnx", "onnx")
    assert stuck.acquire(timeout=0)
    try:
        started = time.monotonic()
        result = compressor.compress(CONTENT_40_WORDS)
        elapsed = time.monotonic() - started
    finally:
        stuck.release()

    assert result.compressed == CONTENT_40_WORDS
    assert model.calls == 0
    assert elapsed < 3.0  # used to block forever


def test_stuck_semaphore_batch_passes_through(monkeypatch):
    monkeypatch.setenv(KOMPRESS_ACQUIRE_TIMEOUT_ENV, "0.1")
    compressor = _make_compressor(monkeypatch, FakeModel())
    monkeypatch.setattr(KompressCompressor, "_should_use_sequential_fallback", lambda self: False)

    stuck = kc._execution_semaphore("onnx", "onnx")
    assert stuck.acquire(timeout=0)
    try:
        contents = [CONTENT_40_WORDS, " ".join(f"x{i}" for i in range(30))]
        results = compressor.compress_batch(contents)
    finally:
        stuck.release()

    assert [r.compressed for r in results] == contents  # all passthrough, no data loss


def test_default_wait_allows_queued_single(monkeypatch):
    compressor = _make_compressor(monkeypatch, FakeModel())
    stuck, blocked = _make_block_tracking_semaphore(monkeypatch)
    assert stuck.acquire(timeout=0)
    finished = threading.Event()
    result_holder = {}

    def _run():
        result_holder["result"] = compressor.compress(CONTENT_40_WORDS)
        finished.set()

    worker = threading.Thread(target=_run)
    worker.start()
    released = False
    try:
        assert blocked.wait(timeout=1)
        assert not finished.wait(timeout=0.05)
        stuck.release()
        released = True
        assert finished.wait(timeout=1)
    finally:
        if not released:
            stuck.release()
        worker.join(timeout=1)
    assert not worker.is_alive()

    result = result_holder["result"]
    assert result.compressed != CONTENT_40_WORDS
    assert result.compressed_tokens == 20


def test_default_wait_allows_queued_batch(monkeypatch):
    compressor = _make_compressor(monkeypatch, FakeModel())
    monkeypatch.setattr(KompressCompressor, "_should_use_sequential_fallback", lambda self: False)
    stuck, blocked = _make_block_tracking_semaphore(monkeypatch)
    assert stuck.acquire(timeout=0)
    finished = threading.Event()
    result_holder = {}
    contents = [CONTENT_40_WORDS, " ".join(f"x{i}" for i in range(30))]

    def _run():
        result_holder["results"] = compressor.compress_batch(contents)
        finished.set()

    worker = threading.Thread(target=_run)
    worker.start()
    released = False
    try:
        assert blocked.wait(timeout=1)
        assert not finished.wait(timeout=0.05)
        stuck.release()
        released = True
        assert finished.wait(timeout=1)
    finally:
        if not released:
            stuck.release()
        worker.join(timeout=1)
    assert not worker.is_alive()

    results = result_holder["results"]
    assert [result.compressed_tokens for result in results] == [20, 15]


def test_default_max_concurrent():
    assert kc._default_max_concurrent("onnx", "onnx") == 1
    assert kc._default_max_concurrent("pytorch", "cpu") == 1
    assert kc._default_max_concurrent("pytorch", "cuda") == 1


def test_execution_wait_budget(monkeypatch):
    assert kc._execution_wait_budget_seconds() == 3.0

    monkeypatch.setenv(KOMPRESS_EXECUTION_SEMAPHORE_WAIT_MS_ENV, "bogus")
    assert kc._execution_wait_budget_seconds() == 3.0

    monkeypatch.setenv(KOMPRESS_EXECUTION_SEMAPHORE_WAIT_MS_ENV, "-1")
    assert kc._execution_wait_budget_seconds() == 0.0


def test_request_deadline_caps_default_wait_single(monkeypatch):
    monkeypatch.setenv(KOMPRESS_REQUEST_DEADLINE_ENV, "10")
    model = FakeModel()
    compressor = _make_compressor(monkeypatch, model)
    stuck = kc._execution_semaphore("onnx", "onnx")
    assert stuck.acquire(timeout=0)
    try:
        started = time.monotonic()
        result = compressor.compress(CONTENT_40_WORDS)
        elapsed = time.monotonic() - started
    finally:
        stuck.release()

    assert elapsed < 0.2
    assert result.compressed == CONTENT_40_WORDS
    assert model.calls == 0


def test_request_deadline_caps_default_wait_batch(monkeypatch):
    monkeypatch.setenv(KOMPRESS_REQUEST_DEADLINE_ENV, "10")
    model = FakeModel()
    compressor = _make_compressor(monkeypatch, model)
    monkeypatch.setattr(KompressCompressor, "_should_use_sequential_fallback", lambda self: False)
    stuck = kc._execution_semaphore("onnx", "onnx")
    assert stuck.acquire(timeout=0)
    contents = [CONTENT_40_WORDS, " ".join(f"x{i}" for i in range(30))]
    try:
        started = time.monotonic()
        results = compressor.compress_batch(contents)
        elapsed = time.monotonic() - started
    finally:
        stuck.release()

    assert elapsed < 0.2
    assert [r.compressed for r in results] == contents
    assert model.calls == 0


def test_carried_deadline_reaches_single_to_batch(monkeypatch):
    monkeypatch.setenv(KOMPRESS_REQUEST_DEADLINE_ENV, "10")
    model = FakeModel()
    compressor = _make_compressor(monkeypatch, model)
    load_state = {"calls": 0}

    def fake_clock():
        return 999.0 if load_state["calls"] >= 1 else 0.0

    def fake_load(*_args, **_kwargs):
        load_state["calls"] += 1
        return model, FakeTokenizer(), "onnx"

    monkeypatch.setattr(kc.time, "perf_counter", fake_clock)
    monkeypatch.setattr(kc, "_load_kompress", fake_load)
    monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)

    result = compressor.compress(CONTENT_40_WORDS)

    assert result.compressed == CONTENT_40_WORDS
    assert model.calls == 0


def test_carried_deadline_reaches_sequential_fallback(monkeypatch):
    monkeypatch.setenv(KOMPRESS_REQUEST_DEADLINE_ENV, "10")
    model = FakeModel()
    compressor = _make_compressor(monkeypatch, model, chunk_words=40)
    monkeypatch.setattr(kc.time, "perf_counter", lambda: 999.0 if model.calls >= 1 else 0.0)
    monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: True)
    contents = [CONTENT_40_WORDS, " ".join(f"x{i}" for i in range(30))]

    results = compressor.compress_batch(contents)

    assert results[0].compressed != contents[0]
    assert results[1].compressed == contents[1]
    assert model.calls == 1


def test_acquire_bounded_unbounded_when_both_disabled():
    semaphore = kc._execution_semaphore("onnx", "onnx")
    assert kc._acquire_bounded(semaphore, None, None) is True
    semaphore.release()


def test_acquire_bounded_negative_remaining_does_not_raise():
    semaphore = kc._execution_semaphore("onnx", "onnx")
    assert semaphore.acquire(timeout=0)
    try:
        assert kc._acquire_bounded(semaphore, 5.0, -1.0) is False
    finally:
        semaphore.release()


# ── Wall-clock budget: give up before the proxy's stage timeout ───────


def test_time_budget_bails_to_passthrough(monkeypatch):
    monkeypatch.setenv(KOMPRESS_TIME_BUDGET_ENV, "0.2")
    model = FakeModel(delay=0.15)
    compressor = _make_compressor(monkeypatch, model, chunk_words=10)

    result = compressor.compress(CONTENT_40_WORDS)  # 4 chunks at ~0.15s each

    assert result.compressed == CONTENT_40_WORDS
    assert model.calls < 4  # bailed before processing every chunk


def test_time_budget_disabled_processes_all_chunks(monkeypatch):
    monkeypatch.setenv(KOMPRESS_TIME_BUDGET_ENV, "0")
    model = FakeModel(delay=0.01)
    compressor = _make_compressor(monkeypatch, model, chunk_words=10)

    result = compressor.compress(CONTENT_40_WORDS)

    assert model.calls == 4
    assert result.compression_ratio == 0.5


def test_time_budget_batch_keeps_completed_texts(monkeypatch):
    """Mid-queue bail: fully processed texts stay compressed; any text with
    an unprocessed chunk passes through whole (never partially dropped)."""
    monkeypatch.setenv(KOMPRESS_TIME_BUDGET_ENV, "0.2")
    model = FakeModel(delay=0.25)  # one batch alone exhausts the budget
    compressor = _make_compressor(monkeypatch, model)
    monkeypatch.setattr(KompressCompressor, "_should_use_sequential_fallback", lambda self: False)

    contents = [
        " ".join(f"a{i}" for i in range(20)),
        " ".join(f"b{i}" for i in range(20)),
        " ".join(f"c{i}" for i in range(20)),
    ]
    results = compressor.compress_batch(contents, batch_size=1)

    assert len(results) == 3
    # First batch ran; later ones bailed to passthrough.
    assert results[0].compression_ratio == 0.5
    assert results[1].compressed == contents[1]
    assert results[2].compressed == contents[2]
    # Every result preserves all information (compressed or original).
    for r in results:
        assert r.compressed


# ── Preload canary: detect degraded runtimes before live traffic ──────


def _join_canary(compressor: KompressCompressor) -> None:
    assert compressor._canary_thread is not None
    compressor._canary_thread.join(timeout=10)
    assert not compressor._canary_thread.is_alive()


def test_canary_disables_kompress_on_slow_inference(monkeypatch, caplog):
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "0.05")
    model = FakeModel(delay=0.15)
    compressor = _make_compressor(monkeypatch, model)

    with caplog.at_level("WARNING"):
        backend = compressor.preload()
        _join_canary(compressor)

    assert backend == "onnx"
    assert compressor._degraded_reason is not None
    assert model.calls == 2  # probe + one retry
    assert "DISABLED" in caplog.text

    result = compressor.compress(CONTENT_40_WORDS)
    assert result.compressed == CONTENT_40_WORDS
    assert model.calls == 2  # model never touched again

    batch = compressor.compress_batch([CONTENT_40_WORDS])
    assert batch[0].compressed == CONTENT_40_WORDS
    assert model.calls == 2


def test_canary_fast_inference_stays_enabled(monkeypatch):
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "5")
    model = FakeModel()
    compressor = _make_compressor(monkeypatch, model)

    compressor.preload()
    _join_canary(compressor)

    assert compressor._degraded_reason is None
    result = compressor.compress(CONTENT_40_WORDS)
    assert result.compression_ratio == 0.5


def test_canary_retry_forgives_oneoff_warmup_slowness(monkeypatch):
    """First inference pays one-off warmup costs; only a slow retry condemns."""
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "0.1")

    class WarmupModel(FakeModel):
        def get_keep_mask(self, input_ids, attention_mask):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.2)  # cold first run
            return [[i % 2 == 0 for i in range(len(row))] for row in input_ids]

    model = WarmupModel()
    compressor = _make_compressor(monkeypatch, model)
    compressor.preload()
    _join_canary(compressor)

    assert compressor._degraded_reason is None
    assert model.calls == 2


def test_canary_disabled_via_env(monkeypatch):
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "0")
    model = FakeModel(delay=0.2)
    compressor = _make_compressor(monkeypatch, model)

    compressor.preload()

    assert compressor._canary_thread is None  # probe never scheduled
    assert model.calls == 0
    assert compressor._degraded_reason is None


def test_preload_does_not_block_on_slow_canary(monkeypatch):
    """The probe runs off the startup path: preload blocks proxy boot (the
    HTTP server binds after it), and a slow probe once pushed the wrap-e2e
    container past its 30s health-check timeout."""
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "0.05")
    model = FakeModel(delay=1.0)
    compressor = _make_compressor(monkeypatch, model)

    started = time.monotonic()
    compressor.preload()
    preload_elapsed = time.monotonic() - started

    assert preload_elapsed < 0.5  # returns before the ~2s of probe inference
    _join_canary(compressor)
    assert compressor._degraded_reason is not None


def test_canary_probe_error_never_breaks_preload(monkeypatch):
    class ExplodingModel(FakeModel):
        def get_keep_mask(self, input_ids, attention_mask):
            raise RuntimeError("probe boom")

    compressor = _make_compressor(monkeypatch, ExplodingModel())

    assert compressor.preload() == "onnx"
    _join_canary(compressor)
    assert compressor._degraded_reason is None


# ── Artifact selection: reject at LOAD what would fail at RUN ──────────────────
# Reported case: the int8 weight-only artifact carries MatMulNBits with bits=8.
# ORT's CPU kernel only handles 8-bit via the prepacked MLAS path, so a build
# without an 8-bit SQNBitGemm kernel falls into ComputeBUnpacked, which asserts
# nbits_ == 4. That raises on session.run() AFTER construction succeeded, so the
# load-only candidate loop never saw it and the fp32 fallback was unreachable:
# 207 consecutive per-request failures over three days, ML compression silently
# dead the whole time.


class _FakeOrtSession:
    """Constructs fine; optionally rejects execution the way ORT's CPU kernel does."""

    def __init__(self, path: str, *, fails_at_run: bool):
        self.path = path
        self._fails_at_run = fails_at_run
        self.runs = 0

    def run(self, outputs, feeds):
        self.runs += 1
        if self._fails_at_run:
            raise RuntimeError(
                "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code "
                "returned while running MatMulNBits node ... nbits_ == 4 was false. "
                "Only 4b quantization is supported for unpacked compute."
            )
        import numpy as np

        return [np.zeros((1, 2), dtype=np.float32)]


def _install_fake_ort(monkeypatch, *, run_fails_for: set[str]):
    """Patch onnxruntime so InferenceSession succeeds but run() may not."""
    created: list[_FakeOrtSession] = []

    class _FakeOrt:
        @staticmethod
        def SessionOptions():  # noqa: N802 - mirrors the ORT API
            return object()

        @staticmethod
        def InferenceSession(path, options=None, providers=None):  # noqa: N802
            session = _FakeOrtSession(path, fails_at_run=any(bad in path for bad in run_fails_for))
            created.append(session)
            return session

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", _FakeOrt)
    monkeypatch.setattr(kc, "_onnx_session_options", lambda _ort: object())
    monkeypatch.setattr(kc, "hf_hub_download_local_first", lambda repo, fn, **kw: f"/cache/{fn}")
    return created


def test_run_time_artifact_rejection_falls_through_to_next_candidate(monkeypatch, caplog):
    """A session that loads then fails at run must be skipped, not returned."""
    created = _install_fake_ort(monkeypatch, run_fails_for={"int8-wo"})

    with caplog.at_level("WARNING"):
        session = kc._create_onnx_session("org/model", ["CPUExecutionProvider"])

    # int8-wo was constructed, smoke-run, rejected; fp32 was selected instead.
    assert "int8-wo" in created[0].path
    assert created[0].runs == 1
    assert "kompress-fp32.onnx" in session.path
    assert "unusable" in caplog.text


def test_healthy_artifact_is_selected_after_one_smoke_run(monkeypatch):
    created = _install_fake_ort(monkeypatch, run_fails_for=set())

    session = kc._create_onnx_session("org/model", ["CPUExecutionProvider"])

    # First candidate works, so no fallback and exactly one probe.
    assert session is created[0]
    assert len(created) == 1
    assert session.runs == 1


def test_all_artifacts_failing_at_run_raises_rather_than_returning_a_dead_session(monkeypatch):
    _install_fake_ort(monkeypatch, run_fails_for={"onnx/"})

    with pytest.raises(FileNotFoundError, match="No loadable ONNX artifact"):
        kc._create_onnx_session("org/model", ["CPUExecutionProvider"])


# ── Failure latch: a broken model stops costing us every request ───────────────


def test_repeated_inference_failures_latch_to_passthrough(monkeypatch, caplog):
    class AlwaysFailingModel(FakeModel):
        def get_keep_mask(self, input_ids, attention_mask):
            self._tick()
            raise RuntimeError("MatMulNBits nbits_ == 4 was false")

    model = AlwaysFailingModel()
    compressor = _make_compressor(monkeypatch, model)
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "0")  # no canary interference

    with caplog.at_level("WARNING"):
        for _ in range(kc._INFERENCE_FAILURE_LATCH):
            assert compressor.compress(CONTENT_40_WORDS).compressed == CONTENT_40_WORDS

    assert compressor._degraded_reason is not None
    assert "DISABLED" in caplog.text
    calls_at_latch = model.calls

    # Latched: further calls short-circuit without touching the model again, so a
    # broken artifact can't burn inference on every request for three days.
    assert compressor.compress(CONTENT_40_WORDS).compressed == CONTENT_40_WORDS
    assert model.calls == calls_at_latch


def test_a_success_resets_the_failure_count(monkeypatch):
    class FlakyModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.fail_next = True

        def get_keep_mask(self, input_ids, attention_mask):
            if self.fail_next:
                self._tick()
                raise RuntimeError("transient")
            return super().get_keep_mask(input_ids, attention_mask)

    model = FlakyModel()
    compressor = _make_compressor(monkeypatch, model)
    monkeypatch.setenv(KOMPRESS_CANARY_THRESHOLD_ENV, "0")

    # Two failures, then a success, then two more failures: never 3 in a row.
    for _ in range(kc._INFERENCE_FAILURE_LATCH - 1):
        compressor.compress(CONTENT_40_WORDS)
    assert compressor._inference_failures == kc._INFERENCE_FAILURE_LATCH - 1

    model.fail_next = False
    compressor.compress(CONTENT_40_WORDS)
    assert compressor._inference_failures == 0
    assert compressor._degraded_reason is None

    model.fail_next = True
    for _ in range(kc._INFERENCE_FAILURE_LATCH - 1):
        compressor.compress(CONTENT_40_WORDS)
    assert compressor._degraded_reason is None

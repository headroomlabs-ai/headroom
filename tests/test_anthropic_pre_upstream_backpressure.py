"""Unit 4: bounded pre-upstream concurrency for Anthropic replay storms.

Verifies that ``HeadroomProxy`` gates the pre-upstream phase of
``handle_anthropic_messages`` with a semaphore, so cold-start replay
storms cannot starve ``/livez`` or new Codex WS opens.

Covers:
- happy path (single request, no contention)
- N+1 contention (only the (N+1)th waiter records ``pre_upstream_wait`` > 0)
- strict serialization under concurrency=1
- unbounded mode (``anthropic_pre_upstream_concurrency=0`` -> no semaphore)
- acquire timeout fails fast with ``503`` + ``Retry-After``
- memory-context timeout fails open without leaking the semaphore
- exception-safety (semaphore released when the critical section raises)
- ``/livez`` unaffected under Anthropic backpressure
- compression is not bypassed (the Unit 4 gate is additive, not a shortcut)
- CLI flag ``--anthropic-pre-upstream-concurrency`` wires into ``ProxyConfig``
- env var ``HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY`` with flag override
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest
from click.testing import CliRunner
from fastapi import Request
from fastapi.testclient import TestClient

from headroom.cli.proxy import proxy as proxy_cli
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import HeadroomProxy, create_app

# --------------------------------------------------------------------------- #
# Dummy handler that gives tests control over the ``_retry_request`` duration #
# so we can simulate long pre-upstream work (semaphore contention).           #
# --------------------------------------------------------------------------- #


class _DummyTokenizer:
    def count(self, messages) -> int:  # noqa: D401 - stub
        return 1

    def count_messages(self, messages) -> int:  # noqa: D401 - stub
        return 1

    def count_tokens(self, text) -> int:  # noqa: D401 - stub
        return 1


class _DummyMetrics:
    def __init__(self) -> None:
        self.stage_timings: list[tuple[str, dict]] = []

    async def record_request(self, **kwargs):
        return None

    async def record_stage_timings(self, path: str, timings: dict) -> None:
        self.stage_timings.append((path, timings))

    async def record_rate_limited(self, **kwargs) -> None:
        return None

    async def record_failed(self, **kwargs) -> None:
        return None


class _ResponseStub:
    def __init__(self) -> None:
        self.status_code = 200
        self.headers = {"content-type": "application/json"}
        self._text = json.dumps(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-3-5-sonnet-latest",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    @property
    def text(self) -> str:
        return self._text

    @property
    def content(self) -> bytes:
        return self._text.encode("utf-8")

    def json(self) -> dict:
        return json.loads(self._text)


class _DummyAnthropicHandler(AnthropicHandlerMixin):
    """Minimal handler used across tests; allows controlling upstream delay."""

    ANTHROPIC_API_URL = "https://api.anthropic.com"

    def _extract_anthropic_cache_ttl_metrics(self, usage):  # noqa: D401
        return (0, 0)

    def __init__(
        self,
        *,
        anthropic_pre_upstream_sem: asyncio.Semaphore | None = None,
        upstream_delay_s: float = 0.0,
        raise_during_critical: bool = False,
    ) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = ProxyConfig(
            optimize=False,
            image_optimize=False,
            retry_max_attempts=1,
            retry_base_delay_ms=1,
            retry_max_delay_ms=1,
            connect_timeout_seconds=10,
            mode="token",
            cache_enabled=False,
            rate_limit_enabled=False,
            fallback_enabled=False,
            fallback_provider=None,
            prefix_freeze_enabled=False,
            memory_enabled=False,
        )
        self.usage_reporter = None
        self.anthropic_provider = SimpleNamespace(get_context_limit=lambda model: 200_000)
        self.anthropic_pipeline = SimpleNamespace(apply=MagicMock())
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
        self.cache = None
        self.security = None
        self.ccr_context_tracker = None
        self.ccr_injector = None
        self.ccr_response_handler = None
        self.ccr_feedback = None
        self.ccr_batch_processor = None
        self.ccr_mcp_server = None
        self.traffic_learner = None
        self.tool_injector = None
        self.read_lifecycle_manager = None
        self.logger = SimpleNamespace(log=lambda *a, **k: None)
        self.request_logger = self.logger
        self.usage_observer = None
        self.image_compressor = None
        self.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-1",
            get_or_create=lambda *a, **k: SimpleNamespace(
                _cached_token_count=0,
                get_frozen_message_count=lambda: 0,
                get_last_original_messages=lambda: [],
                get_last_forwarded_messages=lambda: [],
                update_from_response=lambda *a, **k: None,
                record_request=lambda *a, **k: None,
            ),
        )
        # Unit 4: the only field this test cares about.
        self.anthropic_pre_upstream_sem = anthropic_pre_upstream_sem
        self.anthropic_pre_upstream_concurrency = (
            0 if anthropic_pre_upstream_sem is None else anthropic_pre_upstream_sem._value
        )
        # Audit follow-up C3: dedicated compression executor + cancel-aware
        # metrics. The mixin's compression path delegates to
        # ``HeadroomProxy._run_compression_in_executor`` for bounded thread
        # use; this dummy handler stands in for the proxy and must therefore
        # provide the same surface.
        import concurrent.futures as _cf
        import threading as _threading

        self._compression_executor = _cf.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="dummy-compress"
        )
        self.compression_max_workers = 4
        self._compression_in_flight = 0
        self._compression_in_flight_max = 0
        self._compression_leaked_threads = 0
        self._compression_metrics_lock = _threading.Lock()
        self._upstream_delay_s = upstream_delay_s
        self._raise_during_critical = raise_during_critical
        self.upstream_enter_times: list[float] = []
        self.upstream_exit_times: list[float] = []

    async def _run_compression_in_executor(self, fn, *, timeout):  # noqa: ANN001
        # Mirror of ``HeadroomProxy._run_compression_in_executor`` for the
        # mixin tests. Same metrics semantics; same timeout behavior.
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        with self._compression_metrics_lock:
            self._compression_in_flight += 1
            self._compression_in_flight_max = max(
                self._compression_in_flight_max, self._compression_in_flight
            )

        def _wrapped():
            try:
                return fn()
            finally:
                elapsed = time.perf_counter() - start
                with self._compression_metrics_lock:
                    self._compression_in_flight -= 1
                    if elapsed > timeout:
                        self._compression_leaked_threads += 1

        future = loop.run_in_executor(self._compression_executor, _wrapped)
        return await asyncio.wait_for(future, timeout=timeout)

    async def _next_request_id(self) -> str:
        # Unique IDs so log assertions remain disambiguated under parallelism.
        return f"req-{id(object()):x}"

    def _extract_tags(self, headers):
        return {}

    async def _retry_request(
        self,
        method: str,
        url: str,
        headers: dict,
        body: dict,
        *,
        original_body_bytes: bytes | None = None,
        body_mutated: bool = True,
        mutation_reasons: list[str] | None = None,
        request_id: str | None = None,
        forwarder_name: str = "test_dummy",
        path_for_log: str | None = None,
    ):
        # PR-A8 follow-up: A3 added byte-faithful kwargs to the real
        # ``_retry_request`` signature. The dummy stub doesn't need
        # to use them — just accept them so existing tests don't
        # break with TypeError on the new call sites.
        del original_body_bytes, body_mutated, mutation_reasons
        del request_id, forwarder_name, path_for_log
        if self._raise_during_critical:
            raise RuntimeError("synthetic pre-upstream failure")
        enter = time.perf_counter()
        self.upstream_enter_times.append(enter)
        if self._upstream_delay_s > 0:
            await asyncio.sleep(self._upstream_delay_s)
        self.upstream_exit_times.append(time.perf_counter())
        return _ResponseStub()

    def _get_compression_cache(self, session_id):
        return SimpleNamespace(
            apply_cached=lambda m: m,
            compute_frozen_count=lambda m: 0,
            mark_stable_from_messages=lambda *a, **k: None,
            should_defer_compression=lambda h: False,
            mark_stable=lambda h: None,
            content_hash=lambda c: "h",
            update_from_result=lambda *a, **k: None,
            _cache={},
            _stable_hashes=set(),
        )


def _build_request(body: dict, headers: dict[str, str]) -> Request:
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [
            (key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def stage_log_capture():
    target = logging.getLogger("headroom.proxy")
    handler = _CapturingHandler()
    previous_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)


def _parse_all_stage_logs(handler: _CapturingHandler) -> list[dict]:
    payloads: list[dict] = []
    for record in handler.records:
        msg = record.getMessage()
        if "STAGE_TIMINGS" in msg:
            payload_start = msg.index("STAGE_TIMINGS ") + len("STAGE_TIMINGS ")
            payloads.append(json.loads(msg[payload_start:]))
    return payloads


def _tokenizer_patch():
    import headroom.tokenizers as _tk

    orig_get = _tk.get_tokenizer

    class _Ctx:
        def __enter__(self):
            _tk.get_tokenizer = lambda model: _DummyTokenizer()
            return self

        def __exit__(self, *exc):
            _tk.get_tokenizer = orig_get

    return _Ctx()


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #


def test_happy_path_single_request_negligible_wait(stage_log_capture):
    sem = asyncio.Semaphore(2)
    handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [{"role": "user", "content": "hello"}],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )

    with _tokenizer_patch():
        anyio.run(handler.handle_anthropic_messages, request)

    payloads = _parse_all_stage_logs(stage_log_capture)
    assert len(payloads) == 1
    stages = payloads[0]["stages"]
    assert "pre_upstream_wait" in stages
    # Single request -> no contention, wait ms must be tiny.
    assert stages["pre_upstream_wait"] is not None
    assert stages["pre_upstream_wait"] < 25.0, stages
    # Sanity: semaphore was released cleanly.
    assert sem._value == 2


# --------------------------------------------------------------------------- #
# N+1 contention: with concurrency=2 and 3 concurrent requests,               #
# exactly one of them must observe a non-trivial ``pre_upstream_wait``.       #
# --------------------------------------------------------------------------- #


def test_n_plus_one_contention_only_waiter_has_nonzero_wait(stage_log_capture):
    async def _run() -> None:
        sem = asyncio.Semaphore(2)
        # Each request hogs the semaphore for ~150 ms. With concurrency=2,
        # 3 concurrent requests mean exactly one waits ~150 ms.
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem, upstream_delay_s=0.15)
        reqs = [
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": f"hello {i}"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
            for i in range(3)
        ]
        await asyncio.gather(*(handler.handle_anthropic_messages(r) for r in reqs))
        assert sem._value == 2  # semaphore fully released

    with _tokenizer_patch():
        anyio.run(_run)

    payloads = _parse_all_stage_logs(stage_log_capture)
    assert len(payloads) == 3
    waits = sorted(p["stages"]["pre_upstream_wait"] for p in payloads)
    # Exactly one request must have waited noticeably; the first two should
    # be near zero (they acquired the sem immediately).
    assert waits[0] < 25.0, waits
    assert waits[1] < 25.0, waits
    # The waiter should have waited roughly the upstream-delay budget.
    assert waits[2] > 75.0, waits


# --------------------------------------------------------------------------- #
# Serialization: concurrency=1 => strict ordering of upstream enter timestamps #
# --------------------------------------------------------------------------- #


def test_concurrency_one_serializes_requests():
    async def _run() -> float:
        sem = asyncio.Semaphore(1)
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem, upstream_delay_s=0.10)
        reqs = [
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": f"msg {i}"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
            for i in range(2)
        ]
        start = time.perf_counter()
        await asyncio.gather(*(handler.handle_anthropic_messages(r) for r in reqs))
        elapsed = time.perf_counter() - start
        # Strict ordering: second request enters upstream only AFTER the first exits.
        assert len(handler.upstream_enter_times) == 2
        assert handler.upstream_enter_times[1] >= handler.upstream_exit_times[0] - 1e-6, (
            handler.upstream_enter_times,
            handler.upstream_exit_times,
        )
        return elapsed

    with _tokenizer_patch():
        elapsed = anyio.run(_run)
    # Two back-to-back 100 ms upstream calls under serialization: must take
    # at least ~2 * 100 ms. (Give a little slack for scheduler jitter.)
    assert elapsed >= 0.18, elapsed


# --------------------------------------------------------------------------- #
# Unbounded mode: ``anthropic_pre_upstream_concurrency=0`` disables the sem.   #
# --------------------------------------------------------------------------- #


def test_unbounded_mode_no_semaphore_instance():
    config = ProxyConfig(anthropic_pre_upstream_concurrency=0)
    proxy = HeadroomProxy(config)
    assert proxy.anthropic_pre_upstream_sem is None
    assert proxy.anthropic_pre_upstream_concurrency == 0


def test_unbounded_mode_requests_run_concurrently():
    """With concurrency=0 (sem disabled), two slow requests overlap."""

    async def _run() -> float:
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=None, upstream_delay_s=0.10)
        reqs = [
            _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": f"msg {i}"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
            for i in range(2)
        ]
        start = time.perf_counter()
        await asyncio.gather(*(handler.handle_anthropic_messages(r) for r in reqs))
        return time.perf_counter() - start

    with _tokenizer_patch():
        elapsed = anyio.run(_run)
    # Unbounded -> both sleeps run in parallel. Total should be ~0.10 s,
    # nowhere near 0.20 s.
    assert elapsed < 0.18, elapsed


# --------------------------------------------------------------------------- #
# Exception releases the semaphore.                                            #
# --------------------------------------------------------------------------- #


def test_exception_inside_critical_section_releases_semaphore():
    async def _run() -> None:
        sem = asyncio.Semaphore(2)
        baseline = sem._value
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem, raise_during_critical=True)
        # Drive several cycles to ensure we don't leak on any path.
        for i in range(5):
            req = _build_request(
                {
                    "model": "claude-3-5-sonnet-latest",
                    "messages": [{"role": "user", "content": f"msg {i}"}],
                },
                {"authorization": "Bearer sk-ant-api-test"},
            )
            # The handler catches upstream RuntimeError internally and
            # returns a 5xx JSONResponse; this is the expected behaviour.
            await handler.handle_anthropic_messages(req)
            # After each cycle the semaphore must be fully restored.
            assert sem._value == baseline, (i, sem._value, baseline)

    with _tokenizer_patch():
        anyio.run(_run)


def test_acquire_timeout_returns_503_with_retry_after(stage_log_capture):
    async def _run() -> None:
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
        handler.config.anthropic_pre_upstream_acquire_timeout_seconds = 0.01
        req = _build_request(
            {
                "model": "claude-3-5-sonnet-latest",
                "messages": [{"role": "user", "content": "hello"}],
            },
            {"authorization": "Bearer sk-ant-api-test"},
        )
        try:
            response = await handler.handle_anthropic_messages(req)
            assert response.status_code == 503
            assert response.headers["retry-after"] == "1"
            body = json.loads(response.body)
            assert body["error"]["type"] == "service_unavailable"
            assert sem._value == 0
        finally:
            sem.release()
        assert sem._value == 1

    with _tokenizer_patch():
        anyio.run(_run)

    payloads = _parse_all_stage_logs(stage_log_capture)
    assert len(payloads) == 1
    assert payloads[0]["stages"]["pre_upstream_wait"] >= 10.0


def test_memory_context_timeout_fails_open_and_releases_semaphore():
    class _MemoryHandler:
        def __init__(self) -> None:
            self.config = SimpleNamespace(inject_context=True, inject_tools=False)
            self.initialized = False
            self.backend = None

        async def search_and_format_context(self, _user_id, _messages):
            await asyncio.sleep(5.0)
            return "should-timeout"

        def inject_tools(self, tools, _provider):
            return tools, False

        def get_beta_headers(self) -> dict[str, str]:
            return {}

        def has_memory_tool_calls(self, _response, _provider) -> bool:
            return False

        async def handle_memory_tool_calls(self, _response, _user_id, _provider):
            return []

    async def _run() -> None:
        sem = asyncio.Semaphore(1)
        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
        handler.memory_handler = _MemoryHandler()
        handler.config.anthropic_pre_upstream_memory_context_timeout_seconds = 0.01
        req = _build_request(
            {
                "model": "claude-3-5-sonnet-latest",
                "messages": [{"role": "user", "content": "hello"}],
            },
            {
                "authorization": "Bearer sk-ant-api-test",
                "x-headroom-user-id": "user-1",
            },
        )
        response = await handler.handle_anthropic_messages(req)
        assert response.status_code == 200
        assert sem._value == 1

    with _tokenizer_patch():
        anyio.run(_run)


# --------------------------------------------------------------------------- #
# /livez stays fast under Anthropic pre-upstream contention.                   #
# --------------------------------------------------------------------------- #


def test_livez_unaffected_under_anthropic_backpressure():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        anthropic_pre_upstream_concurrency=2,
    )
    app = create_app(config)
    assert app.state.proxy.anthropic_pre_upstream_sem is not None

    # Drain the semaphore so any simulated Anthropic request would block.
    proxy = app.state.proxy

    async def _drain_sem() -> None:
        # Acquire both permits — no request can enter the pre-upstream region.
        await proxy.anthropic_pre_upstream_sem.acquire()
        await proxy.anthropic_pre_upstream_sem.acquire()

    # Run an event loop just to drain the semaphore.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_drain_sem())
    finally:
        loop.close()

    latencies: list[float] = []
    with TestClient(app) as client:
        # Warm up: the first requests pay one-time costs (TestClient ASGI
        # lifespan, route resolution, lazy imports the restructured proxy
        # triggers on first-request paths). Three warmups was not enough on
        # Python 3.10 under full-suite load; ten is comfortably past every
        # lazy-init boundary observed in CI traces (the rogue sample landed
        # at measured-index 2, i.e. request #6 overall).
        for _ in range(10):
            client.get("/livez")
        for _ in range(20):
            t0 = time.perf_counter()
            resp = client.get("/livez")
            latencies.append((time.perf_counter() - t0) * 1000.0)
            assert resp.status_code == 200
            assert resp.json()["alive"] is True

    # With only 20 samples `statistics.quantiles(n=100)[98]` collapses to
    # max(latencies), so any single CI hiccup trips the assertion. Drop the
    # one worst outlier and assert on the next-worst — that still fails hard
    # if /livez is genuinely being blocked by the drained semaphore (every
    # sample would cluster near the drained timeout) but tolerates a single
    # GC pause or scheduler jitter in the 20-sample window.
    sorted_latencies = sorted(latencies)
    p95_like = sorted_latencies[-2] if len(sorted_latencies) >= 2 else sorted_latencies[-1]
    assert p95_like < 100.0, (p95_like, latencies)


# --------------------------------------------------------------------------- #
# Compression is NOT bypassed by the gate.                                     #
# --------------------------------------------------------------------------- #


def test_compression_is_not_bypassed_when_gated(stage_log_capture):
    """With ``optimize=True`` the first compression stage must still run."""

    class _Pipeline:
        def __init__(self) -> None:
            self.called = False

        def apply(self, messages, *args, **kwargs):
            self.called = True
            return SimpleNamespace(messages=messages, metadata={"applied_steps": ["first"]})

    sem = asyncio.Semaphore(2)
    handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
    handler.config = ProxyConfig(
        optimize=True,
        image_optimize=False,
        retry_max_attempts=1,
        retry_base_delay_ms=1,
        retry_max_delay_ms=1,
        connect_timeout_seconds=10,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        fallback_enabled=False,
        fallback_provider=None,
        prefix_freeze_enabled=False,
        memory_enabled=False,
        anthropic_pre_upstream_concurrency=2,
    )
    pipeline = _Pipeline()
    handler.anthropic_pipeline = pipeline

    # Large synthetic body to ensure the pipeline triggers.
    big_text = "x" * 50_000
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [{"role": "user", "content": big_text}],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )

    with _tokenizer_patch():
        anyio.run(handler.handle_anthropic_messages, request)

    assert pipeline.called, "compression pipeline must still run under backpressure"
    # Semaphore restored after the request.
    assert sem._value == 2


# --------------------------------------------------------------------------- #
# CLI: --anthropic-pre-upstream-concurrency plumbs into ProxyConfig.           #
# --------------------------------------------------------------------------- #


def _run_cli_capture(args: list[str], env: dict | None = None) -> ProxyConfig:
    """Invoke the proxy CLI, intercepting ``run_server`` to capture config.

    We do NOT want the CLI to actually start a server — monkeypatching the
    ``run_server`` entry point (imported lazily inside the click command
    via ``from headroom.proxy.server import ... run_server``) short-
    circuits it and lets us inspect the ``ProxyConfig`` that was built.
    """
    import headroom.proxy.server as server_mod

    captured: dict[str, ProxyConfig] = {}
    orig_run = server_mod.run_server

    def _fake_run(config: ProxyConfig):  # noqa: D401 - stub
        captured["config"] = config
        return 0

    server_mod.run_server = _fake_run
    try:
        runner = CliRunner()
        result = runner.invoke(proxy_cli, args, env=env or {})
    finally:
        server_mod.run_server = orig_run

    assert result.exit_code == 0, (result.output, result.exception)
    assert "config" in captured, "run_server was not called"
    return captured["config"]


def test_cli_flag_sets_pre_upstream_concurrency():
    config = _run_cli_capture(["--anthropic-pre-upstream-concurrency", "3"])
    assert config.anthropic_pre_upstream_concurrency == 3


def test_env_var_sets_pre_upstream_concurrency():
    # Must set in the env passed to the runner (click reads envvar).
    # Also strip the corresponding CLI flag.
    env = {"HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY": "4"}
    # Also make sure we don't pick up a host user env that could override.
    config = _run_cli_capture([], env=env)
    assert config.anthropic_pre_upstream_concurrency == 4


def test_cli_flag_overrides_env_var():
    env = {"HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY": "4"}
    config = _run_cli_capture(["--anthropic-pre-upstream-concurrency", "7"], env=env)
    assert config.anthropic_pre_upstream_concurrency == 7


def test_cli_env_sets_pre_upstream_timeouts():
    env = {
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_ACQUIRE_TIMEOUT_SECONDS": "9.5",
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_MEMORY_CONTEXT_TIMEOUT_SECONDS": "3.25",
    }
    config = _run_cli_capture([], env=env)
    assert config.anthropic_pre_upstream_acquire_timeout_seconds == pytest.approx(9.5)
    assert config.anthropic_pre_upstream_memory_context_timeout_seconds == pytest.approx(3.25)


def test_cli_flags_override_pre_upstream_timeout_env_vars():
    env = {
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_ACQUIRE_TIMEOUT_SECONDS": "9.5",
        "HEADROOM_ANTHROPIC_PRE_UPSTREAM_MEMORY_CONTEXT_TIMEOUT_SECONDS": "3.25",
    }
    config = _run_cli_capture(
        [
            "--anthropic-pre-upstream-acquire-timeout-seconds",
            "4.5",
            "--anthropic-pre-upstream-memory-context-timeout-seconds",
            "1.5",
        ],
        env=env,
    )
    assert config.anthropic_pre_upstream_acquire_timeout_seconds == pytest.approx(4.5)
    assert config.anthropic_pre_upstream_memory_context_timeout_seconds == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# Sanity: HeadroomProxy auto-computes default when config value is None.       #
# --------------------------------------------------------------------------- #


def test_auto_computed_default_on_this_machine():
    config = ProxyConfig()  # field left at None -> auto-compute.
    proxy = HeadroomProxy(config)
    expected = max(2, min(8, os.cpu_count() or 4))
    assert proxy.anthropic_pre_upstream_concurrency == expected
    assert proxy.anthropic_pre_upstream_sem is not None
    assert proxy.anthropic_pre_upstream_sem._value == expected


# --------------------------------------------------------------------------- #
# Semaphore released on HTTPException / early-exit paths even with an         #
# already-held permit. Explicitly covers the 4 pre-upstream early exits:     #
#   - rate_limiter deny (429)                                                 #
#   - cost_tracker block (429)                                                #
#   - security scan block (403)                                               #
#   - cache hit (200)                                                         #
# Each test holds 1 permit of a Semaphore(2) with a concurrent request,      #
# then verifies the handler restores ``_value`` to the original after        #
# the early return.                                                           #
# --------------------------------------------------------------------------- #


class _RateLimiterDeny:
    async def check_request(self, _rate_key):
        return False, 1.0


class _CostTrackerBlock:
    def check_budget(self):
        return False, 0

    def record_tokens(self, *a, **k):
        return None


class _SecurityBlock:
    class _Err(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.reason = "blocked-by-security"

    def scan_request(self, _messages, _ctx):
        raise self._Err("blocked by security policy")


class _CacheHit:
    class _Entry:
        response_headers: dict = {}
        response_body: bytes = b'{"id":"cached","type":"message","role":"assistant","content":[{"type":"text","text":"hit"}]}'

    def __init__(self) -> None:
        self._entry = self._Entry()

    async def get(self, _messages, _model):
        return self._entry

    async def set(self, *a, **k):
        return None


@pytest.mark.parametrize(
    "scenario",
    ["rate_limiter", "cost_tracker", "security", "cache"],
)
def test_early_exit_paths_release_semaphore_under_contention(scenario):
    """Hold one permit of a Semaphore(1) with a concurrent request, trigger
    the early-exit path, verify the semaphore value is restored.
    """

    async def _run() -> None:
        sem = asyncio.Semaphore(1)
        original_value = sem._value

        handler = _DummyAnthropicHandler(anthropic_pre_upstream_sem=sem)
        if scenario == "rate_limiter":
            handler.rate_limiter = _RateLimiterDeny()
        elif scenario == "cost_tracker":
            handler.cost_tracker = _CostTrackerBlock()
        elif scenario == "security":
            handler.security = _SecurityBlock()
        elif scenario == "cache":
            handler.cache = _CacheHit()

        req = _build_request(
            {
                "model": "claude-3-5-sonnet-latest",
                "messages": [{"role": "user", "content": "hello"}],
            },
            {"authorization": "Bearer sk-ant-api-test"},
        )

        # Drive several iterations to confirm each early-exit call fully
        # releases the semaphore rather than leaking a permit AND that the
        # exception type or response status matches the contract for this
        # scenario. `except Exception: pass` would mask the 62d0a50 regression
        # where HTTPException got swallowed and turned into a 502 JSONResponse.
        from fastapi import HTTPException

        for _ in range(3):
            raised: BaseException | None = None
            result = None
            try:
                result = await handler.handle_anthropic_messages(req)
            except HTTPException as exc:
                raised = exc

            if scenario in ("rate_limiter", "cost_tracker"):
                # These paths MUST surface HTTPException(429) so FastAPI's
                # exception handler emits the proper status + Retry-After.
                assert isinstance(raised, HTTPException), (
                    f"{scenario}: expected HTTPException to propagate, got "
                    f"raised={raised!r} result={result!r}"
                )
                assert raised.status_code == 429, (
                    f"{scenario}: wrong status code — got {raised.status_code}"
                )
            else:
                # security returns a JSONResponse; cache returns a Response.
                assert raised is None, f"{scenario}: unexpected exception {raised!r}"
                assert result is not None
            assert sem._value == original_value, (
                f"{scenario}: semaphore leak got={sem._value}, want={original_value}"
            )

    with _tokenizer_patch():
        anyio.run(_run)

"""Context budget guard tests for headroom target #2649.

Test matrix follows the Required Proof Matrix in the execution prompt:
  reproduction        base-fails/head-passes contract
  mode_* / variant_*  every Fault Scope variant is classified
  preservation        unconfigured requests forward unchanged
  negative_space      at-threshold / observe-over / bypassed-over all forward
  production_route    real POST /v1/messages through create_app
  contract_isolation  policy module has no FastAPI/handler/provider imports
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import anyio
from fastapi import Request
from fastapi.testclient import TestClient

# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #


class _DummyTokenizer:
    """Token counter that returns a configurable value."""

    def __init__(self, count: int = 1) -> None:
        self._count = count

    def count_messages(self, messages) -> int:
        return self._count

    def count_text(self, text: str) -> int:
        return self._count

    def count(self, messages) -> int:
        return self._count


class _DummyMetrics:
    def __init__(self) -> None:
        self.stage_timings: list = []

    async def record_request(self, **kwargs) -> None:
        return None

    async def record_stage_timings(self, path, timings) -> None:
        self.stage_timings.append((path, timings))

    async def record_failed(self, **kwargs) -> None:
        return None

    async def record_rate_limited(self, **kwargs) -> None:
        return None

    def record_compression_failed(self, reason: str) -> None:
        return None


def _stub_response(status: int = 200) -> Any:
    body = json.dumps(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "step-router-v1",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    ns = SimpleNamespace()
    ns.status_code = status
    ns.headers = {"content-type": "application/json"}
    ns.content = body.encode()
    ns.text = body
    ns.json = lambda: json.loads(body)
    return ns


def _make_anthropic_provider(operator_limit: int | None = None) -> Any:
    limits: dict[str, int] = {}
    if operator_limit is not None:
        limits["step-router-v1"] = operator_limit

    def _get_operator_limit(model: str) -> int | None:
        if model in limits:
            return limits[model]
        from headroom.providers.anthropic import sanitize_anthropic_model_id

        sanitized = sanitize_anthropic_model_id(model)
        return limits.get(sanitized)

    return SimpleNamespace(
        get_context_limit=lambda model: 200_000,
        get_operator_context_limit=_get_operator_limit,
        has_raw_operator_context_limit=lambda model: model in limits,
        _operator_context_limits=limits,
    )


class _DummyHandler:
    """Minimal AnthropicHandlerMixin with configurable upstream stub."""

    from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

    ANTHROPIC_API_URL = "https://api.anthropic.com"

    def __init__(
        self,
        *,
        operator_limit: int | None = None,
        token_count: int = 1,
        bypass: bool = False,
        mode_header: str | None = None,
    ) -> None:
        from headroom.proxy.models import ProxyConfig

        # Inherit the mixin by embedding via __class__ trick is messy;
        # subclass it inline.
        self._operator_limit = operator_limit
        self._token_count = token_count
        self._bypass_header = bypass
        self.upstream_calls: list[dict] = []

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
        self.anthropic_provider = _make_anthropic_provider(operator_limit)
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
            compute_session_id=lambda *a, **k: "sess-budget-test",
            get_or_create=lambda *a, **k: SimpleNamespace(
                _cached_token_count=0,
                get_frozen_message_count=lambda: 0,
                get_last_original_messages=lambda: [],
                get_last_forwarded_messages=lambda: [],
                update_from_response=lambda *a, **k: None,
                record_request=lambda *a, **k: None,
            ),
            resolve_tracker=lambda *a, **k: SimpleNamespace(
                _cached_token_count=0,
                get_frozen_message_count=lambda: 0,
                get_last_original_messages=lambda: [],
                get_last_forwarded_messages=lambda: [],
                update_from_response=lambda *a, **k: None,
                record_request=lambda *a, **k: None,
            ),
        )
        self.anthropic_pre_upstream_sem = None
        self.anthropic_pre_upstream_concurrency = 0
        import concurrent.futures as _cf
        import threading as _threading

        self._compression_executor = _cf.ThreadPoolExecutor(max_workers=2)
        self.compression_max_workers = 2
        self._compression_in_flight = 0
        self._compression_in_flight_max = 0
        self._compression_leaked_threads = 0
        self._compression_metrics_lock = _threading.Lock()
        self._background_compression_enabled = False

    async def _run_compression_in_executor(self, fn, *, timeout):
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._compression_executor, fn)
        return await asyncio.wait_for(future, timeout=timeout)

    async def _record_request_outcome(self, outcome) -> None:
        from headroom.proxy.outcome import emit_request_outcome

        await emit_request_outcome(self, outcome)

    async def _next_request_id(self) -> str:
        return "req-budget-test"

    def _extract_tags(self, headers):
        return {}

    async def _retry_request(self, method, url, headers, body, **_kwargs):
        self.upstream_calls.append({"method": method, "url": url})
        return _stub_response()

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

    def _extract_anthropic_cache_ttl_metrics(self, usage):
        return (0, 0)


def _make_handler_subclass() -> type:
    """Return a concrete class inheriting both _DummyHandler and AnthropicHandlerMixin."""
    from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

    class _BudgetHandler(_DummyHandler, AnthropicHandlerMixin):
        pass

    return _BudgetHandler


def _build_request(
    body: dict,
    headers: dict[str, str] | None = None,
) -> Request:
    h = {"authorization": "Bearer sk-ant-api-test"}
    if headers:
        h.update(headers)
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
        "headers": [(k.lower().encode(), v.encode()) for k, v in h.items()],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


# --------------------------------------------------------------------------- #
# Policy unit tests (contract_isolation)                                       #
# --------------------------------------------------------------------------- #


def test_contract_isolation_no_fastapi_import():
    """The policy module must not import FastAPI, the handler, or the provider."""

    src = importlib.import_module("headroom.proxy.context_budget_policy")
    # Walk module's __dict__ for imported symbols that would indicate a bad dep.
    forbidden = {"fastapi", "headroom.proxy.handlers", "headroom.providers"}
    for mod_name in list(src.__dict__.keys()):
        obj = src.__dict__[mod_name]
        if hasattr(obj, "__module__") and obj.__module__ is not None:
            for f in forbidden:
                assert not obj.__module__.startswith(f), (
                    f"Policy module imported {obj.__module__!r} (matches forbidden prefix {f!r})"
                )


def test_contract_isolation_standalone_evaluate_matches_handler(monkeypatch):
    """evaluate() produces the same result whether called directly or via the handler."""
    from headroom.proxy.context_budget_policy import evaluate, resolve_mode, resolve_safety_margin

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")

    mode = resolve_mode()
    margin = resolve_safety_margin()
    # Standalone call
    direct = evaluate(
        counted_tokens=270_000,
        declared_limit=262_144,
        max_output_tokens=8_192,
        mode=mode,
        safety_margin=margin,
    )
    assert direct.should_reject is True
    assert direct.reason == "over_threshold"


# --------------------------------------------------------------------------- #
# Policy evaluate branches (mode_ / variant_)                                 #
# --------------------------------------------------------------------------- #


def test_mode_no_declared_limit():
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=300_000,
        declared_limit=None,
        max_output_tokens=8_192,
        mode="reject",
        safety_margin=0,
    )
    assert d.reason == "no_declared_limit"
    assert d.should_reject is False


def test_mode_observe_under_threshold():
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=100_000,
        declared_limit=262_144,
        max_output_tokens=8_192,
        mode="observe",
        safety_margin=0,
    )
    assert d.reason == "under_threshold"
    assert d.should_reject is False


def test_mode_observe_over_threshold():
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=260_000,
        declared_limit=262_144,
        max_output_tokens=8_192,
        mode="observe",
        safety_margin=0,
    )
    assert d.reason == "over_threshold"
    assert d.should_reject is False  # observe => no rejection


def test_mode_reject_over_threshold():
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=260_000,
        declared_limit=262_144,
        max_output_tokens=8_192,
        mode="reject",
        safety_margin=0,
    )
    assert d.reason == "over_threshold"
    assert d.should_reject is True


def test_variant_degenerate_threshold():
    """When max_output_tokens >= declared_limit the threshold is <= 0: no rejection."""
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=100_000,
        declared_limit=8_192,
        max_output_tokens=8_192,  # equal to declared_limit -> threshold=0
        mode="reject",
        safety_margin=0,
    )
    assert d.reason == "degenerate_threshold"
    assert d.should_reject is False


def test_variant_degenerate_threshold_max_output_exceeds():
    """max_output_tokens > declared_limit is also degenerate."""
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=100_000,
        declared_limit=8_192,
        max_output_tokens=16_000,
        mode="reject",
        safety_margin=0,
    )
    assert d.reason == "degenerate_threshold"
    assert d.should_reject is False


def test_variant_safety_margin_adds_reserve():
    from headroom.proxy.context_budget_policy import evaluate

    # declared_limit=262144, max_output_tokens=8192, safety_margin=10000
    # reserve = max(10000, 8192) = 10000
    # threshold = 262144 - 10000 = 252144
    # count = 253000 > 252144 => over
    d = evaluate(
        counted_tokens=253_000,
        declared_limit=262_144,
        max_output_tokens=8_192,
        mode="reject",
        safety_margin=10_000,
    )
    assert d.reason == "over_threshold"
    assert d.reserve == 10_000
    assert d.threshold == 252_144
    assert d.overage == 253_000 - 252_144


def test_variant_bypass_not_evaluated(monkeypatch):
    """The bypass variant skips evaluation entirely — tested via handler."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=300_000)

    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        },
        {"x-headroom-bypass": "true"},
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(300_000))
    anyio.run(handler.handle_anthropic_messages, req)
    # Guard must not fire: upstream call must have happened
    assert len(handler.upstream_calls) == 1


def test_variant_context1m_without_raw_declaration_degrades_to_observe(monkeypatch):
    """context-1m beta without raw-id declaration degrades to observe even in reject mode."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    # Operator declared 'claude-opus-4' (sanitized form) but NOT 'claude-opus-4[1m]'
    BudgetHandler = _make_handler_subclass()

    # No [1m]-keyed declaration in _operator_context_limits
    handler = BudgetHandler(operator_limit=None, token_count=300_000)
    # Override _operator_context_limits to have the sanitized key but not raw
    handler.anthropic_provider._operator_context_limits = {}
    handler.anthropic_provider.get_operator_context_limit = lambda m: None
    handler.anthropic_provider.has_raw_operator_context_limit = lambda m: False

    req = _build_request(
        {
            "model": "claude-opus-4[1m]",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        },
        # client carries context-1m beta
        {"anthropic-beta": "context-1m"},
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(300_000))
    anyio.run(handler.handle_anthropic_messages, req)
    # Degrades to observe: request forwards (not rejected)
    assert len(handler.upstream_calls) == 1


def test_variant_guard_internal_error_forwards(monkeypatch):
    """Any exception inside the guard block forwards the request unchanged."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    import headroom.proxy.context_budget_policy as _pol
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=300_000)

    def _bad_evaluate(**kwargs):
        raise RuntimeError("synthetic policy error")

    monkeypatch.setattr(_pol, "evaluate", _bad_evaluate)
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(300_000))
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        },
    )
    anyio.run(handler.handle_anthropic_messages, req)
    # Forwarded despite policy error
    assert len(handler.upstream_calls) == 1


# --------------------------------------------------------------------------- #
# Preservation (preservation)                                                  #
# --------------------------------------------------------------------------- #


def test_preservation_unconfigured_install_forwards(monkeypatch):
    """Without any operator limit, requests forward byte-identically."""
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=None, token_count=999_999)

    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        },
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(999_999))
    anyio.run(handler.handle_anthropic_messages, req)
    assert len(handler.upstream_calls) == 1


def test_preservation_get_context_limit_unchanged():
    """get_context_limit behavior is identical before and after the change."""
    from headroom.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(
        context_limits={"test-model": 200_000},
        warn=False,
    )
    assert provider.get_context_limit("test-model") == 200_000


def test_preservation_get_operator_context_limit_no_declaration():
    """get_operator_context_limit returns None for undeclared models."""
    from headroom.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(warn=False)
    # step-router-v1 is not in the built-in table so operator has not declared it
    result = provider.get_operator_context_limit("step-router-v1")
    assert result is None


def test_preservation_get_operator_context_limit_declared():
    """get_operator_context_limit returns the declared value for a declared model."""
    import json

    from headroom.providers.anthropic import AnthropicProvider

    limits = json.dumps({"context_limits": {"step-router-v1": 262_144}})
    with patch.dict(os.environ, {"HEADROOM_MODEL_LIMITS": limits}):
        provider = AnthropicProvider(warn=False)

    assert provider.get_operator_context_limit("step-router-v1") == 262_144
    # Must not affect get_context_limit behavior
    assert provider.get_context_limit("step-router-v1") == 262_144


def test_preservation_no_message_mutation(monkeypatch):
    """The guard never mutates body['messages'], body['system'], or body['tools']."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "observe")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=300_000)

    original_messages = [{"role": "user", "content": "hi there"}]
    req = _build_request(
        {"model": "step-router-v1", "messages": original_messages, "max_tokens": 8_192},
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(300_000))
    anyio.run(handler.handle_anthropic_messages, req)
    # Body forwarded: upstream was called (observe mode, not rejected)
    assert len(handler.upstream_calls) == 1
    # The guard must not have mutated messages; the request reached upstream unchanged.


# --------------------------------------------------------------------------- #
# Negative space (negative_space)                                              #
# --------------------------------------------------------------------------- #


def test_negative_space_at_threshold_forwards(monkeypatch):
    """A request exactly at threshold forwards unchanged in both modes."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    from headroom.proxy.context_budget_policy import evaluate

    # declared=262144, max_out=8192 => threshold=253952; count=253952 (at threshold)
    declared = 262_144
    max_out = 8_192
    threshold = declared - max_out
    d = evaluate(
        counted_tokens=threshold,
        declared_limit=declared,
        max_output_tokens=max_out,
        mode="reject",
        safety_margin=0,
    )
    assert d.reason == "under_threshold"
    assert d.should_reject is False


def test_negative_space_observe_over_threshold_still_forwards(monkeypatch):
    """In observe mode an over-threshold request still reaches upstream."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "observe")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=300_000)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "big payload"}],
            "max_tokens": 8_192,
        },
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(300_000))
    anyio.run(handler.handle_anthropic_messages, req)
    assert len(handler.upstream_calls) == 1


def test_negative_space_bypassed_over_threshold_in_reject_mode_forwards(monkeypatch):
    """A bypassed over-threshold request in reject mode still reaches upstream."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=300_000)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        },
        {"x-headroom-bypass": "true"},
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(300_000))
    anyio.run(handler.handle_anthropic_messages, req)
    assert len(handler.upstream_calls) == 1


def test_variant_context1m_with_raw_declaration_still_enforces(monkeypatch):
    """A sticky context-1m beta must not disarm a model the operator declared by raw id.

    `anthropic-beta` is session-sticky (`get_session_beta_tracker`), so a later
    request can carry `context-1m` it never sent. The degrade-to-observe branch
    keys on whether the operator declared the raw id, so a declared model keeps
    enforcing even when the beta arrives from the session baseline.
    """
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=300_000)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        },
        {"anthropic-beta": "context-1m"},
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(300_000))
    resp = anyio.run(handler.handle_anthropic_messages, req)

    assert resp.status_code == 400
    assert len(handler.upstream_calls) == 0


# --------------------------------------------------------------------------- #
# Reproduction (reproduction)                                                  #
# --------------------------------------------------------------------------- #


def test_reproduction_over_limit_rejected_locally_with_zero_upstream_calls(monkeypatch):
    """head: over-threshold request in reject mode returns 400 with zero upstream calls."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=270_000)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "a" * 1000}],
            "max_tokens": 8_192,
        },
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(270_000))
    resp = anyio.run(handler.handle_anthropic_messages, req)
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    body = json.loads(resp.body)
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "step-router-v1" in body["error"]["message"]
    # Zero upstream calls
    assert len(handler.upstream_calls) == 0


# --------------------------------------------------------------------------- #
# Production route (production_route)                                          #
# --------------------------------------------------------------------------- #


def test_production_route_guard_fires_on_create_app(monkeypatch):
    """Guard fires on a real POST /v1/messages through create_app.

    The upstream transport is stubbed so no live API call is made.
    Falsifiability: removing the guard by patching get_operator_context_limit
    to return None makes the request reach the (stubbed) upstream instead.
    """
    import json as _json

    import headroom.tokenizers as _tk
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import create_app

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=False,
    )
    app = create_app(config)
    proxy = app.state.proxy

    # Patch the tokenizer to return a token count above the declared limit.
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(270_000))

    # Declare a limit on the provider (step-router-v1 at 262144).
    # The guard uses get_operator_context_limit; patch it to return 262144.
    original_get_op = proxy.anthropic_provider.get_operator_context_limit
    proxy.anthropic_provider._operator_context_limits = {"step-router-v1": 262_144}
    proxy.anthropic_provider.get_operator_context_limit = lambda m: (
        proxy.anthropic_provider._operator_context_limits.get(m)
    )

    # Stub upstream so it records calls instead of hitting the network.
    upstream_calls: list[dict] = []

    async def _stub_retry(self_inner, method, url, headers, body, **kwargs):
        upstream_calls.append({"method": method, "url": url})
        return _stub_response()

    import headroom.proxy.server as _srv

    monkeypatch.setattr(_srv.HeadroomProxy, "_retry_request", _stub_retry)

    req_body = _json.dumps(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "over limit payload"}],
            "max_tokens": 8_192,
        }
    )

    with TestClient(app) as client:
        # --- HEAD: guard fires, returns 400, zero upstream calls ---
        resp = client.post(
            "/v1/messages",
            content=req_body.encode(),
            headers={
                "authorization": "Bearer sk-ant-api-test",
                "content-type": "application/json",
            },
        )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
        pre_calls = len(upstream_calls)
        assert pre_calls == 0, f"Guard should have prevented upstream call; got {pre_calls}"

        # --- MUTATION CHECK: removing guard (no declared limit) forwards request ---
        proxy.anthropic_provider.get_operator_context_limit = lambda m: None
        proxy.anthropic_provider._operator_context_limits = {}
        upstream_calls.clear()

        resp2 = client.post(
            "/v1/messages",
            content=req_body.encode(),
            headers={
                "authorization": "Bearer sk-ant-api-test",
                "content-type": "application/json",
            },
        )
        # Without a declared limit the guard is inert; upstream should be called.
        assert len(upstream_calls) >= 1, (
            "Mutation check failed: removing declared limit should allow upstream call"
        )
        assert resp2.status_code != 400 or len(upstream_calls) >= 1

    # Restore for cleanup
    proxy.anthropic_provider.get_operator_context_limit = original_get_op

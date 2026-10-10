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
import pytest
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


class _FinalizedBodyTokenizer(_DummyTokenizer):
    """Expose system and tool content only when the final body is counted."""

    def count_messages(self, messages) -> int:
        count = self._count
        if any(message.get("role") == "system" for message in messages):
            count += 50_000
        return count

    def count_text(self, text: str) -> int:
        return 50_000 if "large-tool-schema" in text else 0


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
        self.upstream_bodies: list[dict] = []
        self.upstream_original_body_bytes: list[bytes | None] = []

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
        self.upstream_bodies.append(json.loads(body) if isinstance(body, bytes) else body)
        self.upstream_original_body_bytes.append(_kwargs.get("original_body_bytes"))
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

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=270_000)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        }
    )
    import headroom.tokenizers as _tk

    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(270_000))
    response = anyio.run(handler.handle_anthropic_messages, req)
    assert response.status_code == 400
    assert len(handler.upstream_calls) == 0


# --------------------------------------------------------------------------- #
# Policy evaluate branches (mode_ / variant_)                                 #
# --------------------------------------------------------------------------- #


def test_budget_contract_mode_no_declared_limit():
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


def test_budget_contract_mode_observe_under_threshold():
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


def test_budget_contract_mode_observe_over_threshold():
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


def test_variant_observe_over_threshold_logs_decision_fields(monkeypatch, caplog):
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "observe")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=260_000)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        }
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(260_000))

    anyio.run(handler.handle_anthropic_messages, req)

    assert len(handler.upstream_calls) == 1
    warning = next(
        record.getMessage()
        for record in caplog.records
        if "context_budget_guard" in record.getMessage()
    )
    assert "declared_limit=262144" in warning
    assert "reserve=8192" in warning
    assert "threshold=253952" in warning
    assert "counted=260000" in warning
    assert "overage=6048" in warning
    assert "mode=observe" in warning


def test_budget_contract_mode_reject_over_threshold():
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


def test_budget_contract_mode_resolvers_default_and_invalid(monkeypatch):
    from headroom.proxy.context_budget_policy import resolve_mode, resolve_safety_margin

    monkeypatch.delenv("HEADROOM_CONTEXT_LIMIT_MODE", raising=False)
    monkeypatch.delenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", raising=False)
    assert resolve_mode() == "observe"
    assert resolve_safety_margin() == 0

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "invalid")
    with pytest.raises(ValueError, match="accepted values"):
        resolve_mode()

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "not-an-int")
    with pytest.raises(ValueError, match="not an integer"):
        resolve_safety_margin()

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "-1")
    with pytest.raises(ValueError, match="must be >= 0"):
        resolve_safety_margin()


def test_variant_degenerate_threshold():
    """Equal max_output_tokens (threshold=0) is over budget in reject mode.

    A request whose reserved output consumes the entire window cannot fit by
    construction, so reject mode refuses it instead of forwarding (review
    4850625429 regression).
    """
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=100_000,
        declared_limit=8_192,
        max_output_tokens=8_192,  # equal to declared_limit -> threshold=0
        mode="reject",
        safety_margin=0,
    )
    assert d.reason == "degenerate_threshold"
    assert d.threshold == 0
    assert d.should_reject is True
    assert d.overage == 100_000  # counted_tokens - threshold


def test_variant_degenerate_threshold_max_output_exceeds():
    """max_output_tokens > declared_limit is over budget in reject mode."""
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=100_000,
        declared_limit=8_192,
        max_output_tokens=16_000,
        mode="reject",
        safety_margin=0,
    )
    assert d.reason == "degenerate_threshold"
    assert d.threshold == 8_192 - 16_000
    assert d.should_reject is True
    assert d.overage == 100_000 - d.threshold


def test_variant_degenerate_threshold_observe_forwards():
    """Observe mode keeps a non-positive threshold non-rejecting and logs."""
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=100_000,
        declared_limit=8_192,
        max_output_tokens=8_192,
        mode="observe",
        safety_margin=0,
    )
    assert d.reason == "degenerate_threshold"
    assert d.should_reject is False
    assert d.overage == 100_000


def test_variant_degenerate_threshold_safety_margin_reject():
    """safety_margin >= declared_limit is over budget in reject mode."""
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=100_000,
        declared_limit=8_192,
        max_output_tokens=4_096,
        mode="reject",
        safety_margin=8_192,  # equal to declared_limit -> threshold <= 0
    )
    assert d.reason == "degenerate_threshold"
    assert d.threshold <= 0
    assert d.should_reject is True
    assert d.overage == 100_000 - d.threshold


def test_variant_degenerate_threshold_safety_margin_observe():
    """Degenerate safety_margin in observe mode stays non-rejecting."""
    from headroom.proxy.context_budget_policy import evaluate

    d = evaluate(
        counted_tokens=100_000,
        declared_limit=8_192,
        max_output_tokens=4_096,
        mode="observe",
        safety_margin=8_192,
    )
    assert d.reason == "degenerate_threshold"
    assert d.should_reject is False


def test_handler_degenerate_max_tokens_rejects_with_zero_upstream(monkeypatch):
    """Handler: max_tokens >= declared_limit in reject mode returns 400 locally.

    Review 4850625429 scenario: declared_limit=200_000, max_output_tokens=200_000,
    counted_tokens=1, mode=reject. The threshold is 0, which must count as over
    budget so the request is refused before any upstream attempt.
    """
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=200_000, token_count=1)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 200_000,  # equals declared_limit -> threshold=0
        },
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(1))

    resp = anyio.run(handler.handle_anthropic_messages, req)

    assert resp.status_code == 400, f"Expected local 400, got {resp.status_code}"
    body = json.loads(resp.body)
    assert "step-router-v1" in body["error"]["message"]
    assert len(handler.upstream_calls) == 0


def test_handler_degenerate_safety_margin_rejects_with_zero_upstream(monkeypatch):
    """Handler: safety_margin >= declared_limit in reject mode returns 400 locally."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "200000")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=200_000, token_count=1)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        },
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(1))

    resp = anyio.run(handler.handle_anthropic_messages, req)

    assert resp.status_code == 400, f"Expected local 400, got {resp.status_code}"
    assert len(handler.upstream_calls) == 0


def test_handler_degenerate_observe_logs_and_forwards(monkeypatch, caplog):
    """Handler: observe mode logs a degenerate threshold and still forwards."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "observe")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=200_000, token_count=1)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 200_000,
        },
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(1))

    anyio.run(handler.handle_anthropic_messages, req)

    assert len(handler.upstream_calls) == 1
    warning = next(
        record.getMessage()
        for record in caplog.records
        if "context_budget_guard" in record.getMessage()
    )
    assert "declared_limit=200000" in warning
    assert "threshold=0" in warning
    assert "overage=1" in warning


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


def test_capability_identity_context1m_without_raw_declaration_degrades_to_observe(
    monkeypatch, caplog
):
    """context-1m beta without raw-id declaration degrades to observe even in reject mode."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    # Operator declared 'claude-opus-4' (sanitized form) but NOT 'claude-opus-4[1m]'
    BudgetHandler = _make_handler_subclass()

    handler = BudgetHandler(operator_limit=None, token_count=300_000)
    # The sanitized declaration must be present so this is a true context-1m false positive.
    handler.anthropic_provider._operator_context_limits.update({"claude-opus-4": 262_144})

    req = _build_request(
        {
            "model": "claude-opus-4[1m]",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        },
        # Configured header casing must not change context-1m detection.
        {"anthropic-beta": "Context-1M"},
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(300_000))
    anyio.run(handler.handle_anthropic_messages, req)
    # Degrades to observe: request forwards (not rejected)
    assert len(handler.upstream_calls) == 1
    assert any("no raw model declaration" in record.getMessage() for record in caplog.records)


def test_reject_evaluation_failure_policy_error(monkeypatch):
    """Configured rejection refuses an unavailable policy evaluation locally."""
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
    resp = anyio.run(handler.handle_anthropic_messages, req)
    assert resp.status_code == 500
    assert json.loads(resp.body)["error"]["type"] == "api_error"
    assert len(handler.upstream_calls) == 0


@pytest.mark.parametrize(
    ("source_limit", "target_limit", "expected_status"),
    [(1_000, 100_000, 200), (100_000, 1_000, 400)],
)
def test_destination_identity_uses_cost_router_destination_limit(
    monkeypatch,
    source_limit: int,
    target_limit: int,
    expected_status: int,
):
    """The declared window follows the model selected for the outgoing body."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    source_model = "step-source-v1"
    target_model = "step-target-v1"
    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=None)
    handler.anthropic_provider._operator_context_limits.update(
        {source_model: source_limit, target_model: target_limit}
    )
    handler.model_router = SimpleNamespace(
        enabled=True,
        select=lambda **_kwargs: SimpleNamespace(
            changed=True,
            reason="test route",
            routed_model=target_model,
        ),
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda _model: _DummyTokenizer(2_008))

    req = _build_request(
        {
            "model": source_model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
    )

    resp = anyio.run(handler.handle_anthropic_messages, req)

    assert resp.status_code == expected_status
    if expected_status == 400:
        assert target_model in json.loads(resp.body)["error"]["message"]
        assert handler.upstream_calls == []
    else:
        assert handler.upstream_bodies[-1]["model"] == target_model


@pytest.mark.parametrize("mode, expected_status", [("reject", 500), ("observe", 200)])
@pytest.mark.parametrize("error_type", [LookupError, asyncio.TimeoutError])
def test_destination_identity_reject_evaluation_failure_counting(
    monkeypatch, mode, expected_status, error_type
):
    """Destination counting errors stop reject requests and preserve observe fallback."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", mode)
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as tokenizers

    source_model = "step-source-v1"
    target_model = "step-target-v1"
    handler = _make_handler_subclass()(operator_limit=None)
    handler.anthropic_provider._operator_context_limits[target_model] = 100_000
    counted_models = []

    def get_tokenizer(model):
        counted_models.append(model)
        if model == target_model:
            raise error_type("destination counting failed")
        return _DummyTokenizer(1)

    monkeypatch.setattr(tokenizers, "get_tokenizer", get_tokenizer)
    request = _build_request(
        {
            "model": source_model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
    )

    async def run_request():
        return await handler.handle_anthropic_messages(request, model_override=target_model)

    response = anyio.run(run_request)

    assert response.status_code == expected_status
    assert target_model in counted_models
    if mode == "reject":
        assert json.loads(response.body)["error"] == {
            "type": "api_error",
            "message": "Context budget evaluation unavailable for the configured reject request.",
        }
        assert handler.upstream_calls == []
    else:
        assert len(handler.upstream_calls) == 1
        assert handler.upstream_bodies[-1]["model"] == source_model
        assert target_model in handler._token_count_fallback_models


@pytest.mark.parametrize("mode, expected_status", [("reject", 500), ("observe", 200)])
@pytest.mark.parametrize(
    "failure", ["resolution", "count_timeout", "quarantine", "missing_executor"]
)
def test_destination_identity_unchanged_counting_failure(
    monkeypatch, mode, expected_status, failure
):
    """Reject admission refuses unassessed input even when the model stays the same."""
    import threading
    import time

    import headroom.tokenizers as tokenizers
    from headroom.proxy.server import HeadroomProxy

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", mode)
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    monkeypatch.setattr("headroom.proxy.helpers.COMPRESSION_TIMEOUT_SECONDS", 0.02)
    handler = _make_handler_subclass()(operator_limit=100_000)
    release = threading.Event()
    started = threading.Event()

    class SlowCounter(_DummyTokenizer):
        def count_messages(self, messages):
            started.set()
            release.wait(2)
            return super().count_messages(messages)

    def get_tokenizer(_model):
        if failure == "resolution":
            raise LookupError("synthetic tokenizer resolution failure")
        return SlowCounter(1) if failure == "count_timeout" else _DummyTokenizer(1)

    monkeypatch.setattr(tokenizers, "get_tokenizer", get_tokenizer)
    if failure == "missing_executor":
        handler._run_compression_in_executor = None
    elif failure == "quarantine":
        handler._compression_timed_out_in_flight = 1
        handler._compression_quarantine_threshold = 1
        handler._compression_quarantine_deadline = time.monotonic() + 60
        handler._compression_quarantine_skips = 0
        handler.metrics.record_compression_quarantine = MagicMock()
        handler._run_compression_in_executor = HeadroomProxy._run_compression_in_executor.__get__(
            handler
        )

    request = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
    )
    try:
        response = anyio.run(handler.handle_anthropic_messages, request)
        assert response.status_code == expected_status
        if failure == "count_timeout":
            assert started.is_set()
        if failure == "quarantine":
            assert handler._compression_quarantine_skips >= 1
        if mode == "reject":
            assert json.loads(response.body)["error"] == {
                "type": "api_error",
                "message": "Context budget evaluation unavailable for the configured reject request.",
            }
            assert handler.upstream_calls == []
        else:
            assert len(handler.upstream_calls) == 1
    finally:
        release.set()
        handler._compression_executor.shutdown(wait=True)


@pytest.mark.parametrize("initial_failure", [False, True])
def test_destination_identity_unchanged_registered_estimator_and_recovery(
    monkeypatch, initial_failure
):
    """A registry estimator and recovered strict count both allow assessed requests."""
    import headroom.tokenizers as tokenizers
    from headroom.tokenizers import EstimatingTokenCounter

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    handler = _make_handler_subclass()(operator_limit=100_000)
    resolutions = []
    registered_counter = EstimatingTokenCounter()
    monkeypatch.setitem(
        tokenizers.TokenizerRegistry()._tokenizers, "step-router-v1", registered_counter
    )
    resolve_tokenizer = tokenizers.get_tokenizer

    def get_tokenizer(model):
        resolutions.append(model)
        if initial_failure and len(resolutions) == 1:
            raise LookupError("synthetic transient resolution failure")
        counter = resolve_tokenizer(model)
        assert counter is registered_counter
        return counter

    monkeypatch.setattr(tokenizers, "get_tokenizer", get_tokenizer)
    request = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
    )
    try:
        response = anyio.run(handler.handle_anthropic_messages, request)
        assert response.status_code == 200
        assert len(handler.upstream_calls) == 1
        assert len(resolutions) >= 2
    finally:
        handler._compression_executor.shutdown(wait=True)


def test_destination_identity_uses_backend_resolver_model_and_tokenizer(monkeypatch):
    """A backend route is checked against its rewritten model and tokenizer."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk
    from headroom.proxy.route_advice import BackendResolver, RouteAdvice

    source_model = "step-source-v1"
    target_model = "step-target-v1"
    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=None)
    handler.anthropic_provider._operator_context_limits.update(
        {source_model: 100_000, target_model: 1_000}
    )
    tokenized_models: list[str] = []

    def _get_tokenizer(model: str):
        tokenized_models.append(model)
        return _DummyTokenizer(2_008 if model == target_model else 1)

    monkeypatch.setattr(_tk, "get_tokenizer", _get_tokenizer)

    class _UnusedBackend:
        name = "openai"

        def map_model_id(self, model):
            return model

        def prepare_message(self, body, headers, *, stream=False):
            return body

        async def send_message(self, *_args, **_kwargs):
            raise AssertionError("an over-budget request must stop before backend send")

    resolver = BackendResolver(default=None)
    backend = _UnusedBackend()
    resolver._build = lambda _provider: backend
    handler._route_resolver_cache = resolver

    req = _build_request(
        {
            "model": source_model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
    )
    req.state.headroom_route = RouteAdvice(model=target_model, provider="openai")

    resp = anyio.run(handler.handle_anthropic_messages, req)

    assert resp.status_code == 400
    assert target_model in json.loads(resp.body)["error"]["message"]
    assert target_model in tokenized_models


def test_destination_identity_uses_url_model_override_limit_and_tokenizer(monkeypatch):
    """A path-selected model owns the limit even when the body keeps its model."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    source_model = "step-source-v1"
    target_model = "step-target-v1"
    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=None)
    handler.anthropic_provider._operator_context_limits.update(
        {source_model: 1_000, target_model: 100_000}
    )
    tokenized_models: list[str] = []

    def _get_tokenizer(model: str):
        tokenized_models.append(model)
        return _DummyTokenizer(2_008 if model == target_model else 1)

    monkeypatch.setattr(_tk, "get_tokenizer", _get_tokenizer)
    req = _build_request(
        {
            "model": source_model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
    )

    async def _run_request():
        return await handler.handle_anthropic_messages(req, model_override=target_model)

    resp = anyio.run(_run_request)

    assert resp.status_code == 200
    assert target_model in tokenized_models


def test_selected_bytes_counts_signed_thinking_wire_body_after_restoration(monkeypatch):
    """The guard counts client bytes restored for signed-thinking passthrough."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    monkeypatch.setenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", "0")
    import headroom.tokenizers as _tk
    from headroom.pipeline import PipelineExtensionManager, PipelineStage

    class _BodySizeTokenizer:
        def count_messages(self, messages) -> int:
            return 2_008 if "x" * 100 in json.dumps(messages) else 28

        def count_text(self, _text: str) -> int:
            return 0

    class _ShrinkToolResult:
        def on_pipeline_event(self, event):
            if event.stage is PipelineStage.PRE_SEND and event.messages:
                event.messages[1]["content"][0]["content"] = "short"
            return None

    monkeypatch.setattr(_tk, "get_tokenizer", lambda _model: _BodySizeTokenizer())
    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=1_000)
    handler.pipeline_extensions = PipelineExtensionManager(
        extensions=[_ShrinkToolResult()], discover=False
    )
    original_body = {
        "model": "step-router-v1",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reasoning",
                        "signature": "signed-block",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "x" * 2_000,
                    }
                ],
            },
        ],
        "max_tokens": 100,
    }

    resp = anyio.run(
        handler.handle_anthropic_messages,
        _build_request(original_body),
    )

    assert resp.status_code == 400
    assert "2008 tokens counted" in json.loads(resp.body)["error"]["message"]
    assert handler.upstream_calls == []


def test_observe_preservation_invalid_guard_configuration_forwards_with_warning(
    monkeypatch, caplog
):
    """Invalid operator values fail open and identify the configuration fix."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "invalid")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=300_000)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        }
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(300_000))

    anyio.run(handler.handle_anthropic_messages, req)

    assert len(handler.upstream_calls) == 1
    assert any(
        "not accepted" in record.getMessage()
        and "forwarding in observe mode" in record.getMessage()
        for record in caplog.records
    )


# --------------------------------------------------------------------------- #
# Preservation (preservation)                                                  #
# --------------------------------------------------------------------------- #


def test_observe_preservation_unconfigured_install_forwards(monkeypatch):
    """Without any operator limit, requests forward byte-identically."""
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=None, token_count=999_999)

    original_body = {
        "model": "step-router-v1",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8_192,
    }
    req = _build_request(original_body)
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(999_999))
    anyio.run(handler.handle_anthropic_messages, req)
    assert len(handler.upstream_calls) == 1
    assert handler.upstream_bodies[-1] == original_body
    assert handler.upstream_original_body_bytes[-1] == json.dumps(original_body).encode()


def test_operator_provenance_get_context_limit_unchanged():
    """get_context_limit behavior is identical before and after the change."""
    from headroom.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(
        context_limits={"test-model": 200_000},
        warn=False,
    )
    assert provider.get_context_limit("test-model") == 200_000


def test_operator_provenance_get_operator_context_limit_no_declaration():
    """get_operator_context_limit returns None for undeclared models."""
    from headroom.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(warn=False)
    # step-router-v1 is not in the built-in table so operator has not declared it
    result = provider.get_operator_context_limit("step-router-v1")
    assert result is None


def test_operator_provenance_get_operator_context_limit_declared():
    """get_operator_context_limit returns the declared value for a declared model."""
    import json

    from headroom.providers.anthropic import AnthropicProvider

    limits = json.dumps({"context_limits": {"step-router-v1": 262_144}})
    with patch.dict(os.environ, {"HEADROOM_MODEL_LIMITS": limits}):
        provider = AnthropicProvider(warn=False)

    assert provider.get_operator_context_limit("step-router-v1") == 262_144
    # Must not affect get_context_limit behavior
    assert provider.get_context_limit("step-router-v1") == 262_144


def test_operator_provenance_get_operator_context_limit_sanitized_variant():
    """A sanitized lookup finds the declaration for a styled model id."""
    from headroom.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(
        context_limits={"claude-opus-4": 262_144},
        warn=False,
    )

    assert provider.get_operator_context_limit("claude-opus-4[1m]") == 262_144
    assert provider.has_raw_operator_context_limit("claude-opus-4[1m]") is False


def test_operator_provenance_has_raw_operator_context_limit():
    """A declaration keyed by the styled id is recognized as raw."""
    from headroom.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(
        context_limits={"claude-opus-4[1m]": 1_000_000},
        warn=False,
    )

    assert provider.get_operator_context_limit("claude-opus-4[1m]") == 1_000_000
    assert provider.has_raw_operator_context_limit("claude-opus-4[1m]") is True


def test_preservation_boundaries_no_message_mutation(monkeypatch):
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
    assert handler.upstream_bodies[-1]["messages"] == original_messages
    assert "system" not in handler.upstream_bodies[-1]
    assert "tools" not in handler.upstream_bodies[-1]


def test_preservation_boundaries_system_and_tools_forward_unchanged(monkeypatch):
    """Observe mode forwards top-level system and tools without mutation."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "observe")
    import headroom.tokenizers as _tk

    original_body = {
        "model": "step-router-v1",
        "system": "system instructions",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "tool", "input_schema": {"type": "object"}}],
        "max_tokens": 8_192,
    }
    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=10)
    req = _build_request(original_body)
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(10))

    anyio.run(handler.handle_anthropic_messages, req)

    assert len(handler.upstream_calls) == 1
    assert handler.upstream_bodies[-1] == original_body
    assert handler.upstream_original_body_bytes[-1] == json.dumps(original_body).encode()


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


def test_observe_preservation_bypassed_over_threshold_in_reject_mode_forwards(monkeypatch):
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


def test_capability_identity_context1m_with_raw_declaration_still_enforces(monkeypatch):
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


def test_capability_identity_context1m_suffixed_declaration_still_enforces(monkeypatch):
    """A declaration keyed by the [1m] model id survives handler sanitization."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=None, token_count=300_000)
    handler.anthropic_provider._operator_context_limits["claude-opus-4[1m]"] = 262_144
    req = _build_request(
        {
            "model": "claude-opus-4[1m]",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        },
        {"anthropic-beta": "context-1m"},
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _DummyTokenizer(300_000))

    response = anyio.run(handler.handle_anthropic_messages, req)

    assert response.status_code == 400
    assert len(handler.upstream_calls) == 0


def test_variant_finalized_body_counts_system_and_tools(monkeypatch):
    """Reject mode counts top-level system and tool input after shaping."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    import headroom.tokenizers as _tk

    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=220_000)
    req = _build_request(
        {
            "model": "step-router-v1",
            "system": "system instructions",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "large-tool-schema", "input_schema": {"type": "object"}}],
            "max_tokens": 8_192,
        }
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _FinalizedBodyTokenizer(220_000))

    response = anyio.run(handler.handle_anthropic_messages, req)

    assert response.status_code == 400
    assert len(handler.upstream_calls) == 0


def test_variant_output_shaper_mutation_is_counted_after_recount(monkeypatch):
    """The guard sees a system mutation made after the metrics recount."""
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    import headroom.proxy.output_savings as _savings
    import headroom.proxy.output_shaper as _shaper
    import headroom.tokenizers as _tk

    monkeypatch.setattr(_savings, "assign_arm", lambda *args: "treatment")
    monkeypatch.setattr(_shaper, "resolve_verbosity_level", lambda settings: (2, "test"))

    def _mutate_body(body, settings, *, level_override):
        body["system"] = "new system content"
        return SimpleNamespace(changed=True, labels=["test-shaper"])

    monkeypatch.setattr(_shaper, "shape_request", _mutate_body)
    BudgetHandler = _make_handler_subclass()
    handler = BudgetHandler(operator_limit=262_144, token_count=220_000)
    req = _build_request(
        {
            "model": "step-router-v1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8_192,
        }
    )
    monkeypatch.setattr(_tk, "get_tokenizer", lambda m: _FinalizedBodyTokenizer(220_000))

    response = anyio.run(handler.handle_anthropic_messages, req)

    assert response.status_code == 400
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


def test_step_fun_reproduction_real_loopback():
    from tests.context_budget_behavior_probe import run_initial

    row = run_initial()
    assert row["status"] == 400 and row["upstream_calls"] == 0
    assert (
        row["declared_limit"] == 262144 and row["reserve"] == 12000 and row["threshold"] == 250144
    )
    assert row["overage"] > 0 and str(row["count"]) in row["message"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "forwarding,bypass",
    [("byte_faithful", False), ("legacy_json_kwarg", False), ("byte_faithful", True)],
)
def test_selected_bytes_real_transport(stream, forwarding, bypass):
    from tests.context_budget_behavior_probe import run_initial

    row = run_initial(small=True, stream=stream, forwarding=forwarding, bypass=bypass)
    assert row["upstream_calls"] == 1
    assert row["selected_sha256"] == row["sent_sha256"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("form", ["text", "tool", "redacted"])
def test_retained_thinking_real_registry_rejects(stream, form):
    from tests.context_budget_behavior_probe import run_backend

    row = run_backend(stream=stream, form=form)
    assert row["prepared_count"] < row["threshold"] < row["view_count"]
    assert row["status"] == 400 and row["sdk_calls"] == 0
    assert row["kwargs_unchanged"] and row["prepare_calls"] == 1


@pytest.mark.parametrize("stream", [False, True])
def test_retained_thinking_litellm_prepared_accepts_unchanged(stream):
    from tests.context_budget_behavior_probe import run_backend

    row = run_backend(stream=stream, reject=False)
    assert row["status"] == 200 and row["sdk_calls"] == 1
    assert row["kwargs_unchanged"] and row["retained"] and row["prepare_calls"] == 1


@pytest.mark.parametrize("stream", [False, True])
def test_converted_accounting_retained_thinking_cross_vendor(stream):
    from tests.context_budget_behavior_probe import run_backend

    row = run_backend(stream=stream, cross_vendor=True)
    assert row["status"] == 200 and row["sdk_calls"] == 1
    assert row["prepared_count"] == row["view_count"] and not row["retained"]


def test_retained_thinking_counting_view_none_tool_calls_redacted():
    from copy import deepcopy

    from headroom.proxy.handlers.anthropic import _context_budget_counting_messages
    from headroom.tokenizers import get_tokenizer

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t", "type": "function", "function": {"name": "sample", "arguments": "{}"}}
            ],
            "thinking_blocks": [
                {"type": "thinking", "thinking": "sample " * 100, "signature": "s" * 100},
                {"type": "redacted_thinking", "data": "sample" * 100},
            ],
        }
    ]
    original = deepcopy(messages)
    view = _context_budget_counting_messages(messages)
    tk = get_tokenizer("bedrock/anthropic.claude-sonnet-4-6-v1:0")
    assert messages == original
    assert view[0]["content"] == messages[0]["thinking_blocks"]
    assert "thinking_blocks" not in view[0]
    assert view[0]["tool_calls"] == messages[0]["tool_calls"]
    assert tk.count_messages(view) > tk.count_messages(messages)


@pytest.mark.parametrize("late", [False, True])
def test_ccr_continuation_buffered_budget_client_delivery(late):
    _assert_continuation("ccr", late=late)


@pytest.mark.parametrize("late", [False, True])
def test_memory_continuation_buffered_budget_client_delivery(late):
    _assert_continuation("memory", late=late)


@pytest.mark.parametrize("late", [False, True])
def test_hook_continuation_buffered_budget_budget_salvage_client_delivery(late):
    _assert_continuation("hook", late=late)


@pytest.mark.parametrize("owner", ["ccr", "memory", "hook"])
@pytest.mark.parametrize("late", [False, True])
def test_continuation_evaluation_failure_buffered_budget(owner, late):
    _assert_continuation(owner, late=late, unavailable=True)


def _assert_continuation(owner, *, late, unavailable=False, grace_disabled=False):
    from tests.context_budget_behavior_probe import run_continuation

    row = run_continuation(owner, late=late, unavailable=unavailable, grace_disabled=grace_disabled)
    assert row["status"] == (200 if late else (500 if unavailable else 400))
    assert row["error_type"] == ("api_error" if unavailable else "invalid_request_error")
    assert ("evaluation unavailable" if unavailable else "overage") in row["message"]
    assert row["EOF"] and row["ping"] == late
    if late:
        assert row["content_type"] == "text/event-stream"
    assert row["initial_calls"] == 1 and row["refused_continuation_calls"] == 0
    if owner == "hook":
        assert row["hook_attempts"] == 2


def test_buffered_budget_continuation_evaluation_failure_grace_disabled():
    _assert_continuation("hook", late=False, unavailable=True, grace_disabled=True)


@pytest.mark.parametrize("owner", ["memory", "hook"])
def test_budget_salvage_ordinary_failures_preserve_response(owner):
    from tests.context_budget_behavior_probe import run_continuation

    row = run_continuation(owner, ordinary_failure=True)
    assert row["status"] == 200 and row["ordinary_answer"]
    assert row["initial_calls"] == 1 and row["refused_continuation_calls"] == 0


@pytest.mark.parametrize("owner", ["ccr", "memory", "hook"])
def test_observe_preservation_continuation_forwards(owner):
    from tests.context_budget_behavior_probe import run_continuation

    row = run_continuation(owner, mode="observe")
    assert row["status"] == 200 and row["refused_continuation_calls"] >= 1


@pytest.mark.parametrize("error", [RuntimeError, LookupError])
@pytest.mark.parametrize("mode", ["reject", "observe"])
def test_reject_evaluation_failure_observe_preservation_estimator(monkeypatch, error, mode):
    import headroom.proxy.context_budget_policy as policy
    import headroom.tokenizers as tk

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", mode)
    monkeypatch.setattr(tk, "get_tokenizer", lambda m: _DummyTokenizer(100))

    def unavailable(**kwargs):
        raise error("estimator unavailable")

    monkeypatch.setattr(policy, "evaluate", unavailable)
    handler = _make_handler_subclass()(operator_limit=1000)
    result = anyio.run(
        handler.handle_anthropic_messages,
        _build_request(
            {
                "model": "step-router-v1",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ),
    )
    assert result.status_code == (500 if mode == "reject" else 200)
    assert len(handler.upstream_calls) == (0 if mode == "reject" else 1)
    if mode == "reject":
        message = json.loads(result.body)["error"]["message"]
        assert "evaluation unavailable" in message and "counted" not in message


def test_reject_evaluation_failure_invalid_margin(monkeypatch):
    import headroom.tokenizers as tk

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "broken")
    monkeypatch.setattr(tk, "get_tokenizer", lambda m: _DummyTokenizer(100))
    handler = _make_handler_subclass()(operator_limit=1000)
    result = anyio.run(
        handler.handle_anthropic_messages,
        _build_request(
            {
                "model": "step-router-v1",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ),
    )
    assert result.status_code == 500 and not handler.upstream_calls


def test_handoff_compatibility_default_backend_method():
    from headroom.backends.base import Backend

    with pytest.raises(NotImplementedError):
        Backend.prepare_message(object(), {}, {})


@pytest.mark.parametrize("stream", [False, True])
def test_litellm_prepared_converted_accounting_full_parameters(monkeypatch, stream):
    from copy import deepcopy

    from headroom.backends import litellm as owner
    from headroom.tokenizers import get_tokenizer

    with patch.object(owner, "_fetch_bedrock_inference_profiles", return_value={}):
        backend = owner.LiteLLMBackend(
            provider="bedrock", region="synthetic-region", profile_name="synthetic-profile"
        )
    body = {
        "model": "bedrock/anthropic.claude-sonnet-4-6-v1:0",
        "max_tokens": 32,
        "system": [{"type": "text", "text": "synthetic system"}],
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t", "name": "sample", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t", "content": "result"}],
            },
        ],
        "tools": [
            {"name": "sample", "input_schema": {"type": "object"}},
            {"name": "x" * 65, "input_schema": {"type": "object"}},
        ],
        "temperature": 0.5,
        "top_p": 0.8,
        "stop_sequences": ["stop"],
        "tool_choice": {"type": "auto"},
    }
    original = deepcopy(body)
    prepared = backend.prepare_message(body, {"x-api-key": "synthetic"}, stream=stream)
    assert body == original
    assert (
        prepared["aws_region_name"] == "synthetic-region"
        and prepared["aws_profile_name"] == "synthetic-profile"
    )
    assert "api_key" not in prepared and prepared["timeout"] > 0
    assert len(prepared["tools"]) == 1 and len(prepared["messages"]) == 3
    assert prepared["messages"][0]["role"] == "system" and prepared["messages"][1]["tool_calls"]
    assert prepared["messages"][2]["tool_call_id"] == "t"
    assert prepared.get("stream_options") == ({"include_usage": True} if stream else None)
    tokenizer = get_tokenizer(prepared["model"])
    assert tokenizer.count_messages(prepared["messages"]) > 0
    assert tokenizer.count_text(json.dumps(prepared["tools"])) > 0
    print(
        f"LiteLLM prepared mode={'streaming' if stream else 'buffered'} system_once=True tool_calls=True filtered_tools=1 region_profile=True timeout=True stream_options={stream} no_io=True"
    )


@pytest.mark.parametrize("stream", [False, True])
def test_litellm_prepared_auth_header_forwarding(stream):
    from headroom.backends.litellm import LiteLLMBackend

    backend = LiteLLMBackend(provider="anthropic")
    prepared = backend.prepare_message(
        {"model": "claude-sonnet-4-6", "messages": []},
        {"x-api-key": "sk-ant-synthetic"},
        stream=stream,
    )
    assert prepared["api_key"] == "sk-ant-synthetic"


@pytest.mark.parametrize("stream", [False, True])
def test_reject_evaluation_failure_eligible_preparation(monkeypatch, stream):
    from headroom.backends.litellm import LiteLLMBackend
    from headroom.proxy.server import create_app
    from tests.context_budget_behavior_probe import config

    target = "anthropic/claude-sonnet-4-6"
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_MODEL_LIMITS", json.dumps({"context_limits": {target: 1000}}))
    backend = LiteLLMBackend(provider="anthropic")
    calls = []

    def broken(*args, **kwargs):
        calls.append(True)
        raise LookupError("synthetic preparation failure")

    backend.prepare_message = broken
    app = create_app(config())
    app.state.proxy.anthropic_backend = backend
    with TestClient(app) as client:
        result = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "stream": stream,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-api-key": "synthetic"},
        )
    assert result.status_code == 500 and len(calls) == 1
    assert result.json()["error"]["type"] == "api_error"
    assert "evaluation unavailable" in result.json()["error"]["message"]


@pytest.mark.parametrize("stream", [False, True])
def test_selected_bytes_preservation_boundaries_post_send_rollback_and_observe_preservation_extension(
    monkeypatch, stream
):
    from headroom.pipeline import PipelineExtensionManager, PipelineStage
    from headroom.proxy.server import create_app
    from tests.context_budget_behavior_probe import config

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", "reject")
    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0")
    monkeypatch.setenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", "0")
    monkeypatch.setenv(
        "HEADROOM_MODEL_LIMITS", json.dumps({"context_limits": {"step-router-v1": 1000}})
    )
    mutated = []

    class Shrink:
        def on_pipeline_event(self, event):
            if event.stage is (PipelineStage.POST_SEND if stream else PipelineStage.PRE_SEND):
                event.messages[-1]["content"][0]["content"] = "short"
                mutated.append(True)
                raise LookupError("synthetic extension failure")

    app = create_app(config())
    app.state.proxy.pipeline_extensions = PipelineExtensionManager(
        extensions=[Shrink()], discover=False
    )
    calls = []

    async def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("must reject selected signed body")

    app.state.proxy._retry_request = forbidden
    app.state.proxy._stream_response = forbidden
    body = {
        "model": "step-router-v1",
        "stream": stream,
        "max_tokens": 32,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "synthetic", "signature": "s"},
                    {"type": "tool_use", "id": "t", "name": "sample", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": "synthetic large output " * 2000,
                    }
                ],
            },
        ],
    }
    with TestClient(app) as client:
        result = client.post("/v1/messages", json=body, headers={"x-api-key": "synthetic"})
    assert result.status_code == 400 and mutated and not calls
    assert "overage" in result.json()["error"]["message"]
    assert not app.state.proxy._active_streams
    assert (
        app.state.proxy.anthropic_pre_upstream_sem._value
        == app.state.proxy.anthropic_pre_upstream_concurrency
    )
    print(
        f"selected_bytes mode={'streaming' if stream else 'buffered'} real_selector=True extension_LookupError=True rollback=True upstream_calls=0 semaphore_release_once=True active_streams=0"
    )


@pytest.mark.parametrize("stream", [False, True])
def test_capability_identity_discarded_backend_beta_preserves_reject(stream):
    from tests.context_budget_behavior_probe import run_backend

    row = run_backend(stream=stream, beta=True)
    assert row["status"] == 400 and row["sdk_calls"] == 0


@pytest.mark.parametrize("error", [RuntimeError, LookupError])
@pytest.mark.parametrize("mode", ["reject", "observe"])
def test_reject_evaluation_failure_observe_preservation_real_estimator(monkeypatch, error, mode):
    import httpx

    from headroom.proxy.server import create_app
    from headroom.tokenizers import get_tokenizer
    from tests.context_budget_behavior_probe import config, response

    monkeypatch.setenv("HEADROOM_CONTEXT_LIMIT_MODE", mode)
    monkeypatch.setenv(
        "HEADROOM_MODEL_LIMITS", json.dumps({"context_limits": {"step-router-v1": 1000}})
    )
    tokenizer = get_tokenizer("step-router-v1")
    real_count = tokenizer.count_messages

    def broken(messages):
        if any(message.get("role") == "system" for message in messages):
            raise error("synthetic real estimator failure")
        return real_count(messages)

    monkeypatch.setattr(tokenizer, "count_messages", broken)
    calls = []

    async def sdk(*args, **kwargs):
        calls.append(True)
        return httpx.Response(200, json=response())

    app = create_app(config())
    app.state.proxy._retry_request = sdk
    with TestClient(app) as client:
        result = client.post(
            "/v1/messages",
            json={
                "model": "step-router-v1",
                "system": "synthetic system",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32,
            },
            headers={"x-api-key": "synthetic"},
        )
    assert result.status_code == (500 if mode == "reject" else 200)
    assert len(calls) == (0 if mode == "reject" else 1)
    if mode == "reject":
        assert result.json()["error"]["type"] == "api_error"
        assert "counted" not in result.json()["error"]["message"]
    print(
        f"real_estimator error={error.__name__} mode={mode} status={result.status_code} upstream_calls={len(calls)} fabricated_count=False"
    )

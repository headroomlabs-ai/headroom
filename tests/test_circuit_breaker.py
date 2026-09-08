"""Tests for tool-call repetition loop cycle detection and proxy circuit breaker."""

from __future__ import annotations

import pytest

from headroom.proxy.circuit_breaker import (
    ToolLoopCircuitBreaker,
    detect_repetition_cycle,
    extract_tool_calls_from_messages,
    hash_tool_call,
    resolve_circuit_breaker_mode,
)


def test_hash_tool_call_canonical_json() -> None:
    """Verifies that canonical JSON hashing is order- and whitespace-invariant."""
    # Dict key order invariance
    name1, h1 = hash_tool_call("view", {"path": "src/main.rs", "range": [1, 10]})
    name2, h2 = hash_tool_call("view", {"range": [1, 10], "path": "src/main.rs"})
    assert name1 == name2 == "view"
    assert h1 == h2

    # String with different spacing invariance
    s1 = '{"path": "src/main.rs", "view_range": [2180, 2220]}'
    s2 = '{\n  "view_range": [2180, 2220],\n  "path": "src/main.rs"\n}'
    _, hs1 = hash_tool_call("view", s1)
    _, hs2 = hash_tool_call("view", s2)
    assert hs1 == hs2

    # Different arguments produce different hashes
    _, h3 = hash_tool_call("view", {"path": "src/other.rs"})
    assert h1 != h3

    # Different tool names produce different signatures
    name_other, h_other = hash_tool_call("read", {"path": "src/main.rs", "range": [1, 10]})
    assert (name_other, h_other) != (name1, h1)

    # Fallbacks for raw text and None
    name_raw, h_raw = hash_tool_call("cmd", "not json")
    assert name_raw == "cmd" and isinstance(h_raw, str)
    name_none, h_none = hash_tool_call("noop", None)
    assert name_none == "noop" and isinstance(h_none, str)


def test_detect_repetition_cycle_periods() -> None:
    """Tests exact cycle detection for periods 1, 2, 3, and 4."""
    tA = ("view", "hash_a")
    tB = ("view", "hash_b")
    tC = ("view", "hash_c")
    tD = ("view", "hash_d")

    # Insufficient length
    assert detect_repetition_cycle([]) is None
    assert detect_repetition_cycle([tA, tA]) is None

    # Period 1 (3 repetitions)
    res_p1 = detect_repetition_cycle([tA, tA, tA])
    assert res_p1 is not None
    assert res_p1.period == 1
    assert res_p1.count == 3
    assert res_p1.tool_name == "view"

    # Period 1 with 5 repetitions
    res_p1_5 = detect_repetition_cycle([tA, tA, tA, tA, tA])
    assert res_p1_5 is not None
    assert res_p1_5.period == 1
    assert res_p1_5.count == 5

    # Period 2 (alternating search loop from Phase 1 of RFC)
    t_search_A = ("tokensave_search", "hash_scope")
    t_search_B = ("tokensave_search", "hash_refusal")
    seq_p2 = [t_search_A, t_search_B] * 3
    res_p2 = detect_repetition_cycle(seq_p2)
    assert res_p2 is not None
    assert res_p2.period == 2
    assert res_p2.count == 3
    assert res_p2.tool_name == "tokensave_search"

    # Period 3 (3 repetitions)
    seq_p3 = [tA, tB, tC] * 3
    res_p3 = detect_repetition_cycle(seq_p3)
    assert res_p3 is not None
    assert res_p3.period == 3
    assert res_p3.count == 3

    # Period 4 (4-step view loop from Phase 2 of RFC)
    seq_p4 = [tA, tB, tC, tD] * 3
    res_p4 = detect_repetition_cycle(seq_p4)
    assert res_p4 is not None
    assert res_p4.period == 4
    assert res_p4.count == 3
    assert res_p4.tool_name == "view"

    # Ascending period priority: all identical items must detect period 1, not period 2 or 4
    res_all_a = detect_repetition_cycle([tA] * 8)
    assert res_all_a is not None
    assert res_all_a.period == 1
    assert res_all_a.count == 8

    # Non-repeating sequence
    seq_normal = [tA, tB, tC, tD, tA, tB, tC]  # only 1 full cycle + partial, < 3
    assert detect_repetition_cycle(seq_normal) is None


def test_extract_tool_calls_from_messages() -> None:
    """Verifies extraction from OpenAI and Anthropic message shapes."""
    # OpenAI tool_calls
    openai_msgs = [
        {"role": "user", "content": "Look at src"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "a.rs"}'},
                }
            ],
        },
        {"role": "tool", "content": "contents of a.rs"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "b.rs"}'},
                }
            ],
        },
    ]
    calls = extract_tool_calls_from_messages(openai_msgs)
    assert len(calls) == 2
    assert calls[0][0] == "view"
    assert calls[1][0] == "view"
    assert calls[0][1] != calls[1][1]

    # Anthropic tool_use
    anthropic_msgs = [
        {"role": "user", "content": "Help me fix bug"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will inspect files."},
                {"type": "tool_use", "id": "t1", "name": "view", "input": {"path": "a.rs"}},
            ],
        },
    ]
    anthropic_calls = extract_tool_calls_from_messages(anthropic_msgs)
    assert len(anthropic_calls) == 1
    assert anthropic_calls[0][0] == "view"
    # Arguments {"path": "a.rs"} matches the OpenAI call's hash
    assert anthropic_calls[0][1] == calls[0][1]


def test_circuit_breaker_tracker_session() -> None:
    """Tests ToolLoopCircuitBreaker session tracking and bounded deque."""
    cb = ToolLoopCircuitBreaker(max_history=16, min_repetitions=3, max_period=4)

    # Session 1: alternating loop
    s1 = "session_1"
    # First 2 cycles (4 calls): no detection
    for _ in range(2):
        assert cb.record_tool_call(s1, "view", {"path": "1"}) is None
        assert cb.record_tool_call(s1, "view", {"path": "2"}) is None

    # 5th call: still no detection
    assert cb.record_tool_call(s1, "view", {"path": "1"}) is None

    # 6th call completes the 3rd repetition of period 2: triggers detection!
    detection = cb.record_tool_call(s1, "view", {"path": "2"})
    assert detection is not None
    assert detection.period == 2
    assert detection.count == 3

    # Subsequent check_session also reflects the detection
    check = cb.check_session(s1)
    assert check == detection

    # Session 2: independent session without loops
    s2 = "session_2"
    cb.record_tool_call(s2, "view", {"path": "1"})
    cb.record_tool_call(s2, "view", {"path": "diff"})
    assert cb.check_session(s2) is None

    # Clear session
    cb.clear_session(s1)
    assert cb.check_session(s1) is None


def test_circuit_breaker_check_messages() -> None:
    """Tests updating and checking circuit breaker directly from message arrays."""
    cb = ToolLoopCircuitBreaker(max_history=16, min_repetitions=3, max_period=4)
    session_id = "test_msg_session"

    # Construct conversation with 3 repeated view cycles of period 1
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "foo.py"}'},
                }
            ],
        },
        {"role": "tool", "content": "contents"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "foo.py"}'},
                }
            ],
        },
        {"role": "tool", "content": "contents"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "3",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "foo.py"}'},
                }
            ],
        },
    ]
    det = cb.check_messages(session_id, messages)
    assert det is not None
    assert det.tool_name == "view"
    assert det.period == 1
    assert det.count == 3


def test_proxy_config_circuit_breaker() -> None:
    """Tests ProxyConfig defaults and validation for circuit_breaker mode."""
    from headroom.proxy.models import ProxyConfig

    cfg_default = ProxyConfig()
    assert cfg_default.circuit_breaker == "warn"

    cfg_enforce = ProxyConfig(circuit_breaker="enforce")
    assert cfg_enforce.circuit_breaker == "enforce"

    cfg_off = ProxyConfig(circuit_breaker="off")
    assert cfg_off.circuit_breaker == "off"

    with pytest.raises(ValueError, match="Invalid circuit breaker mode"):
        ProxyConfig(circuit_breaker="bogus")


@pytest.mark.asyncio
async def test_metrics_record_tool_loop_detected() -> None:
    """Tests Prometheus and OTel metrics record tool loop detections."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from headroom.observability.metrics import HeadroomOtelMetrics
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    otel_metrics = HeadroomOtelMetrics(meter_provider=provider)
    metrics = PrometheusMetrics(otel_metrics=otel_metrics, stateless=True)

    await metrics.record_tool_loop_detected(tool="view", period=2)
    assert metrics.tool_loops_detected == 1

    exported = await metrics.export()
    assert "headroom_tool_loop_detected_total 1" in exported

    # Check OTel data point
    data = reader.get_metrics_data()
    otel_points = [
        point
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if m.name == "headroom.proxy.tool_loop_detected"
        for point in m.data.data_points
    ]
    assert len(otel_points) == 1
    assert otel_points[0].attributes["tool"] == "view"
    assert otel_points[0].attributes["period"] == "2"
    assert otel_points[0].value == 1


@pytest.mark.asyncio
async def test_execute_circuit_breaker_policy() -> None:
    """Tests execute_circuit_breaker_policy across off, warn, and enforce modes."""
    from fastapi import HTTPException

    from headroom.proxy.circuit_breaker import (
        ToolLoopCircuitBreaker,
        execute_circuit_breaker_policy,
    )
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    cb = ToolLoopCircuitBreaker()
    metrics = PrometheusMetrics(stateless=True)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "x"}'},
                }
            ],
        },
        {"role": "tool", "content": "res"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "x"}'},
                }
            ],
        },
        {"role": "tool", "content": "res"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "3",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "x"}'},
                }
            ],
        },
    ]

    # Mode: off
    det_off = await execute_circuit_breaker_policy(
        circuit_breaker=cb,
        mode="off",
        session_id="s_off",
        messages=messages,
        request_id="req1",
        metrics=metrics,
    )
    assert det_off is None
    assert metrics.tool_loops_detected == 0

    # Mode: warn (does not raise, logs and records metric)
    det_warn = await execute_circuit_breaker_policy(
        circuit_breaker=cb,
        mode="warn",
        session_id="s_warn",
        messages=messages,
        request_id="req2",
        metrics=metrics,
    )
    assert det_warn is not None
    assert det_warn.tool_name == "view"
    assert metrics.tool_loops_detected == 1

    # Mode: enforce (raises HTTPException 429)
    with pytest.raises(HTTPException) as exc_info:
        await execute_circuit_breaker_policy(
            circuit_breaker=cb,
            mode="enforce",
            session_id="s_enforce",
            messages=messages,
            request_id="req3",
            metrics=metrics,
        )
    assert exc_info.value.status_code == 429
    assert "Tool call loop detected" in exc_info.value.detail
    assert "view" in exc_info.value.detail
    assert metrics.tool_loops_detected == 2


def test_openai_chat_circuit_breaker_warn_enforce_off() -> None:
    """End-to-end test of circuit breaker in proxy /v1/chat/completions."""
    import httpx
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    messages_with_loop = [
        {"role": "user", "content": "analyze the codebase"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "foo.py"}'},
                }
            ],
        },
        {"role": "tool", "content": "file contents", "tool_call_id": "c1"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "foo.py"}'},
                }
            ],
        },
        {"role": "tool", "content": "file contents", "tool_call_id": "c2"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c3",
                    "type": "function",
                    "function": {"name": "view", "arguments": '{"path": "foo.py"}'},
                }
            ],
        },
    ]

    async def _mock_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    # 1. Warn mode (default)
    app_warn = create_app(
        ProxyConfig(
            circuit_breaker="warn",
            optimize=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            log_requests=False,
        )
    )
    with TestClient(app_warn) as client:
        client.app.state.proxy._retry_request = _mock_retry
        resp = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={"model": "gpt-4o", "messages": messages_with_loop},
        )
        assert resp.status_code == 200
        assert client.app.state.proxy.metrics.tool_loops_detected == 1

    # 2. Enforce mode
    app_enforce = create_app(
        ProxyConfig(
            circuit_breaker="enforce",
            optimize=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            log_requests=False,
        )
    )
    with TestClient(app_enforce) as client:
        client.app.state.proxy._retry_request = _mock_retry
        resp = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={"model": "gpt-4o", "messages": messages_with_loop},
        )
        assert resp.status_code == 429
        assert "Tool call loop detected" in resp.json()["detail"]
        assert "view" in resp.json()["detail"]
        assert client.app.state.proxy.metrics.tool_loops_detected == 1

    # 3. Off mode
    app_off = create_app(
        ProxyConfig(
            circuit_breaker="off",
            optimize=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            log_requests=False,
        )
    )
    with TestClient(app_off) as client:
        client.app.state.proxy._retry_request = _mock_retry
        resp = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={"model": "gpt-4o", "messages": messages_with_loop},
        )
        assert resp.status_code == 200
        assert client.app.state.proxy.metrics.tool_loops_detected == 0


def test_resolve_circuit_breaker_mode() -> None:
    """Tests resolving circuit breaker mode configuration."""
    assert resolve_circuit_breaker_mode(None) == "warn"
    assert resolve_circuit_breaker_mode("") == "warn"
    assert resolve_circuit_breaker_mode("warn") == "warn"
    assert resolve_circuit_breaker_mode("log") == "warn"
    assert resolve_circuit_breaker_mode("enforce") == "enforce"
    assert resolve_circuit_breaker_mode("block") == "enforce"
    assert resolve_circuit_breaker_mode("1") == "enforce"
    assert resolve_circuit_breaker_mode("true") == "enforce"
    assert resolve_circuit_breaker_mode("off") == "off"
    assert resolve_circuit_breaker_mode("disabled") == "off"
    assert resolve_circuit_breaker_mode("0") == "off"
    assert resolve_circuit_breaker_mode("false") == "off"

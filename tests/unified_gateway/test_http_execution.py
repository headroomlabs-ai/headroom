"""Serving-path accounting, acceptance and retries through the real runtime."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


def application(monkeypatch, handler, *, attempts=1):
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    raw = json.loads(
        (
            Path(__file__).parents[2]
            / "docs/proposals/unified-api-gateway/examples/gateway.api-keys.json"
        ).read_text()
    )
    raw["routes"] = raw["routes"][:1]
    raw["client_auth"]["principals"][0]["routes"] = [raw["routes"][0]["id"]]
    raw["routes"][0]["retry"].update(max_attempts=attempts, base_backoff_seconds=0.001)
    raw["routes"][0]["pricing"] = {
        "input_usd_per_million": "1",
        "output_usd_per_million": "2",
        "revision": "fixture",
    }
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.model_validate(raw)))
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    return app, raw["routes"][0]["public_model"]


def send(app, model, **body):
    return TestClient(app).post(
        "/v1/chat/completions",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
        json={"model": model, "messages": [], **body},
    )


def test_http_success_uses_one_operation_and_prices_upstream_usage(monkeypatch):
    literal = b'{ "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2} }'

    async def handler(request):
        assert request.headers["authorization"] == "Bearer provider-secret"
        return httpx.Response(200, content=literal)

    app, model = application(monkeypatch, handler)
    response = send(app, model)
    assert response.content == literal
    runtime = app.state.gateway_runtime
    assert runtime.admission.snapshot().known_micro_usd == 7
    assert not runtime.active_work
    assert runtime.observability._totals["logical_requests"] == 1
    assert runtime.observability._totals["attempts"] == 1


@pytest.mark.parametrize(
    "exception,retried",
    [(httpx.ConnectError, True), (httpx.WriteTimeout, False), (httpx.ReadError, False)],
)
def test_serving_retries_only_proven_unsent_and_accounts_every_attempt(
    monkeypatch, exception, retried
):
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise exception("SECRET")
        return httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    app, model = application(monkeypatch, handler, attempts=2)
    response = send(app, model)
    assert response.status_code == (200 if retried else 502)
    assert len(calls) == (2 if retried else 1)
    assert b"SECRET" not in response.content
    runtime = app.state.gateway_runtime
    assert runtime.admission.active_count == 0
    assert not runtime.active_work
    assert runtime.observability._totals["logical_requests"] == 1
    assert runtime.observability._totals["attempts"] == len(calls)
    assert runtime.admission.snapshot().unknown_charge_count == (0 if retried else 1)


def test_arbitrary_compatible_rejection_is_not_free_or_retryable(monkeypatch):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"error": "SECRET"})

    app, model = application(monkeypatch, handler, attempts=2)
    response = send(app, model)
    assert response.status_code == 502
    assert len(calls) == 1
    assert app.state.gateway_runtime.admission.snapshot().unknown_charge_count == 1
    assert b"SECRET" not in response.content


def test_stateful_get_does_not_charge_original_generation_again(monkeypatch):
    async def handler(request):
        return httpx.Response(
            200, json={"id": "resp_1", "usage": {"input_tokens": 3, "output_tokens": 2}}
        )

    app, model = application(monkeypatch, handler)
    client = TestClient(app)
    headers = {"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"}
    created = client.post("/v1/responses", headers=headers, json={"model": model, "input": "hello"})
    assert created.status_code == 200
    fetched = client.get("/v1/responses/resp_1", headers=headers)
    assert fetched.status_code == 200
    runtime = app.state.gateway_runtime
    assert runtime.admission.snapshot().known_micro_usd == 7
    assert runtime.observability._totals["attempts"] == 2
    assert runtime.observability._totals["logical_requests"] == 2
    assert runtime.admission.snapshot().unknown_charge_count == 0


@pytest.mark.parametrize(
    "terminal",
    [
        {"error": {"message": "SECRET"}},
        {"status": "incomplete"},
        {"choices": [{"finish_reason": "content_filter"}]},
    ],
)
def test_failed_native_body_preserves_reported_usage_liability(monkeypatch, terminal):
    async def handler(request):
        usage = (
            {"input_tokens": 3, "output_tokens": 2}
            if "status" in terminal
            else {"prompt_tokens": 3, "completion_tokens": 2}
        )
        return httpx.Response(200, json={**terminal, "usage": usage})

    app, model = application(monkeypatch, handler)
    response = (
        TestClient(app).post(
            "/v1/responses",
            headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
            json={"model": model, "input": "hello"},
        )
        if "status" in terminal
        else send(app, model)
    )
    assert response.status_code == 502
    assert b"SECRET" not in response.content
    runtime = app.state.gateway_runtime
    assert runtime.admission.snapshot().known_micro_usd == 7
    assert runtime.admission.snapshot().unknown_charge_count == 0


@pytest.mark.parametrize("retry_after,expected_calls", [("0", 2), ("9999", 1), ("malformed", 1)])
def test_proven_rejection_retries_with_bounded_retry_after(
    monkeypatch, retry_after, expected_calls
):
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"retry-after": retry_after},
                json={"error": {"type": "rate_limit_error", "message": "SECRET"}},
            )
        return httpx.Response(
            200, json={"content": [], "usage": {"input_tokens": 1, "output_tokens": 2}}
        )

    app, model = application(monkeypatch, handler, attempts=2)
    runtime = app.state.gateway_runtime
    # Configured adapter contract, not a status-code list supplied by the caller.
    raw = runtime.capture().snapshot.model_dump(mode="json")
    raw["routes"][0].update(
        provider="anthropic",
        upstream_origin="https://api.anthropic.com",
        ingress_protocols=["anthropic-messages"],
        native_protocols=["anthropic-messages"],
        capabilities={"anthropic-messages": {"http-json": {"features": ["text"]}}},
    )
    raw["credentials"][0].update(
        provider="anthropic", allowed_origins=["https://api.anthropic.com"]
    )
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.model_validate(raw)))
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    response = TestClient(app).post(
        "/v1/messages",
        headers={"host": "127.0.0.1:8787", "x-api-key": "client-secret"},
        json={"model": model, "messages": []},
    )
    assert len(calls) == expected_calls
    assert response.status_code == (200 if expected_calls == 2 else 502)
    runtime = app.state.gateway_runtime
    assert runtime.admission.active_count == 0
    assert runtime.admission.snapshot().unknown_charge_count == 0
    assert runtime.observability._totals["attempts"] == expected_calls

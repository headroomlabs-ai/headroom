from __future__ import annotations

import tempfile
import types
from pathlib import Path

import pytest

from headroom.proxy import savings_tracker as st
from headroom.proxy.savings_tracker import (
    DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN,
    SavingsTracker,
    _estimate_input_cost_usd,
)


def _fake_litellm_gpt4o() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_cost={
            "gpt-4o": {
                "input_cost_per_token": 2.5e-06,
                "cache_read_input_token_cost": 1.25e-06,
                "cache_creation_input_token_cost": 2.5e-06,
            }
        },
        cost_per_token=lambda **_kw: (0.0, 0.0),
    )


def test_inferred_write_is_not_charged_on_top_of_uncached(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    # Inferred write: the 1000 write tokens ARE the 1000 uncached tokens.
    cost = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
        cache_inferred=True,
    )
    # Expected: 1000 tokens at $2.50/M ($0.0025), not charged twice ($0.005).
    assert cost == pytest.approx(0.0025)


def test_disjoint_write_is_priced_separately(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    cost = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
        cache_inferred=False,
    )
    # Disjoint write: 1000 uncached + 1000 write = 0.0025 + 0.0025 = 0.005.
    assert cost == pytest.approx(0.005)


def test_default_cache_inferred_is_false(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    # Backward compatibility: omitting cache_inferred defaults to False.
    cost = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
    )
    assert cost == pytest.approx(0.005)


def test_inferred_write_with_partial_cache_read(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    # 400 cached read (1.25e-6) + 600 uncached (2.5e-6) + 600 inferred write (zeroed)
    cost = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_read_tokens=400,
        cache_write_tokens=600,
        uncached_input_tokens=600,
        cache_inferred=True,
    )
    expected = 400 * 1.25e-06 + 600 * 2.5e-06  # 0.0005 + 0.0015 = 0.002
    assert cost == pytest.approx(expected)


def test_inferred_write_fallback_pricing(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", lambda: types.SimpleNamespace(model_cost={}))
    cost = _estimate_input_cost_usd(
        "unknown-model",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
        cache_inferred=True,
    )
    # Fallback should price 1000 chargeable tokens, not 2000.
    expected = 1000 * DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN
    assert cost == pytest.approx(expected)


def test_record_lifetime_request_inferred_write(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    with tempfile.TemporaryDirectory() as tmp:
        tracker = SavingsTracker(path=str(Path(tmp) / "savings.json"))
        tracker.record_lifetime_request(
            model="gpt-4o",
            provider="openai",
            stack="chat",
            input_tokens=1000,
            cache_write_tokens=1000,
            uncached_input_tokens=1000,
            cache_inferred=True,
        )
        resp = tracker.lifetime_response()
        assert resp["cost"]["input_usd"] == pytest.approx(0.0025)


def test_record_request_inferred_write(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    with tempfile.TemporaryDirectory() as tmp:
        tracker = SavingsTracker(path=str(Path(tmp) / "savings.json"))
        tracker.record_request(
            model="gpt-4o",
            input_tokens=1000,
            tokens_saved=0,
            cache_write_tokens=1000,
            uncached_input_tokens=1000,
            cache_inferred=True,
        )
        assert tracker._state["lifetime"]["total_input_cost_usd"] == pytest.approx(0.0025)


@pytest.mark.anyio
async def test_prometheus_metrics_record_request_inferred_write(monkeypatch):
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    with tempfile.TemporaryDirectory() as tmp:
        tracker = SavingsTracker(path=str(Path(tmp) / "savings.json"))
        metrics = PrometheusMetrics(savings_tracker=tracker, stateless=True)
        await metrics.record_request(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=50,
            tokens_saved=0,
            latency_ms=100.0,
            cache_write_tokens=1000,
            uncached_input_tokens=1000,
            cache_inferred=True,
        )
        assert tracker._state["lifetime"]["total_input_cost_usd"] == pytest.approx(0.0025)
        resp = tracker.lifetime_response()
        assert resp["cost"]["input_usd"] == pytest.approx(0.0025)


def test_e2e_proxied_chat_completion_inferred_write():
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    from headroom.backends.base import BackendResponse
    from headroom.proxy.server import ProxyConfig, create_app

    async def fake_send(body_: dict, headers: dict) -> BackendResponse:
        return BackendResponse(
            body={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 10,
                    "total_tokens": 1010,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
            status_code=200,
        )

    mock = MagicMock()
    mock.name = "anyllm-openai"
    mock.send_openai_message = fake_send
    mock.map_model_id = MagicMock(return_value="gpt-4o")
    mock.supports_model = MagicMock(return_value=True)

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="openai",
        stateless=True,
    )

    with patch("headroom.proxy.server.AnyLLMBackend", return_value=mock):
        app = create_app(config)
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
            r0 = client.get("/stats-lifetime")
            assert r0.status_code == 200
            initial_cost = r0.json()["cost"]["input_usd"]

            r_chat = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "test prompt"}],
                },
                headers={"Authorization": "Bearer test-key"},
            )
            assert r_chat.status_code == 200

            r1 = client.get("/stats-lifetime")
            assert r1.status_code == 200
            after_cost = r1.json()["cost"]["input_usd"]
            delta = round(after_cost - initial_cost, 6)
            assert delta == pytest.approx(0.0025)

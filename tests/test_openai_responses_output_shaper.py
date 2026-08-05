from __future__ import annotations

import copy
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.loopback_guard import require_loopback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_client() -> TestClient:
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            http2=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None
    return TestClient(app)


async def _ok_response(
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    stream: bool = False,
    **kwargs: Any,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_1",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    )


def test_http_responses_output_shaper_rewrites_and_labels(monkeypatch):
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "2")
    monkeypatch.delenv("HEADROOM_OUTPUT_HOLDOUT", raising=False)
    captured: dict[str, Any] = {}
    outcomes: list[Any] = []

    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            }
        ],
        "reasoning": {"effort": "xhigh"},
        "text": {"verbosity": "medium"},
    }

    with _make_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(*args: Any, **kwargs: Any) -> httpx.Response:
            body = args[3]
            captured["body"] = copy.deepcopy(body)
            captured["retry_kwargs"] = dict(kwargs)
            return await _ok_response(*args, **kwargs)

        async def _record_request_outcome(outcome: Any) -> None:
            outcomes.append(outcome)

        proxy._retry_request = _fake_retry
        proxy._record_request_outcome = _record_request_outcome

        response = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer test-key"},
            json=payload,
        )

    assert response.status_code == 200
    sent = captured["body"]
    assert "<headroom_output_shaping>" in sent["instructions"]
    assert sent["reasoning"]["effort"] == "low"
    assert sent["text"]["verbosity"] == "low"
    assert captured["retry_kwargs"]["body_mutated"] is True
    assert captured["retry_kwargs"]["original_body_bytes"] is not None
    transforms = outcomes[-1].transforms_applied
    assert any(t.startswith("output_shaper:stratum:") for t in transforms)
    assert "output_shaper:verbosity:L2" in transforms
    assert "output_shaper:reasoning_effort:xhigh->low" in transforms
    assert "output_shaper:text_verbosity:medium->low" in transforms


def test_http_responses_output_shaper_respects_bypass(monkeypatch):
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    captured: dict[str, Any] = {}
    payload = {"model": "gpt-5", "input": "hi"}

    with _make_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(*args: Any, **kwargs: Any) -> httpx.Response:
            captured["body"] = copy.deepcopy(args[3])
            return await _ok_response(*args, **kwargs)

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/responses",
            headers={
                "authorization": "Bearer test-key",
                "x-headroom-bypass": "true",
            },
            json=payload,
        )

    assert response.status_code == 200
    assert captured["body"] == payload


def test_http_responses_output_shaper_holdout_labels_without_rewrite(monkeypatch):
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "1")
    captured: dict[str, Any] = {}
    outcomes: list[Any] = []
    payload = {"model": "gpt-5", "input": "hi"}

    with _make_client() as client:
        proxy = client.app.state.proxy

        async def _fake_retry(*args: Any, **kwargs: Any) -> httpx.Response:
            captured["body"] = copy.deepcopy(args[3])
            return await _ok_response(*args, **kwargs)

        async def _record_request_outcome(outcome: Any) -> None:
            outcomes.append(outcome)

        proxy._retry_request = _fake_retry
        proxy._record_request_outcome = _record_request_outcome

        response = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer test-key"},
            json=payload,
        )

    assert response.status_code == 200
    assert captured["body"] == payload
    transforms = outcomes[-1].transforms_applied
    assert any(t.startswith("output_shaper:control:") for t in transforms)
    assert "output_shaper:verbosity:L2" not in transforms

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from headroom.proxy.cost import CostTracker
from headroom.proxy.savings_tracker import HEADROOM_SAVINGS_PATH_ENV_VAR


def test_stats_include_stream_without_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv(HEADROOM_SAVINGS_PATH_ENV_VAR, str(tmp_path / "savings.json"))
    price = 2.5 / 1_000_000
    monkeypatch.setattr(
        CostTracker, "_get_cache_prices", lambda *args, **kwargs: (price / 2, price, price, price)
    )

    async def stream(body: dict, headers: dict) -> AsyncIterator[str]:
        yield (
            'data: {"id":"c1","object":"chat.completion.chunk",'
            '"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
        )
        if body["messages"][-1]["content"] == "measured":
            usage = {
                "choices": [],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 1, "total_tokens": 1001},
            }
            yield f"data: {json.dumps(usage)}\n\n"
        yield "data: [DONE]\n\n"

    backend = MagicMock()
    backend.name = "anyllm-openai"
    backend.stream_openai_message = stream
    monkeypatch.setattr("headroom.proxy.server.AnyLLMBackend", lambda *args, **kwargs: backend)

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
            backend="anyllm",
            anyllm_provider="openai",
        )
    )
    with TestClient(app) as client:
        observed = []
        for prompt in ("measured", "estimated input " * 600):
            result = client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer test-key"},
                json={
                    "model": "gpt-4o",
                    "stream": True,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            assert result.status_code == 200
            assert "data: [DONE]" in result.text
            stats = client.get("/stats")
            assert stats.status_code == 200
            cost = stats.json()["cost"]
            observed.append((cost["total_input_tokens"], cost["total_input_cost_usd"]))

        measured, combined = observed
        assert measured == (1000, 0.0025)
        assert combined[0] > measured[0]
        assert combined[1] == round(combined[0] * price, 4)
        assert app.state.proxy.cost_tracker.totals() == combined

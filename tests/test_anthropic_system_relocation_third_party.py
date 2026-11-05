"""The #765 relocation guard on third-party upstreams, through the real handler.

A non-Claude model behind a configured Anthropic target or a per-request
``x-headroom-base-url`` keeps its valid mid-conversation system sections where
the client put them. A Claude-named model there, and any model on Anthropic
itself, keeps the existing relocation.
"""

from __future__ import annotations

import copy
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app

THIRD_PARTY_TARGET = "https://vllm.example"
OVERRIDE = "https://override.example"
SYSTEM = [{"type": "text", "text": "You are a coding agent."}]
MESSAGES = [
    {"role": "user", "content": "Run the tests."},
    {
        "role": "system",
        "content": "<system-reminder>tests/test_app.py was opened</system-reminder>",
    },
    {"role": "assistant", "content": "Running them now."},
    {"role": "user", "content": "And fix what fails."},
]


@pytest.fixture(autouse=True)
def _isolate_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    # create_app() writes the resolved target onto the class; put it back after.
    monkeypatch.setattr(HeadroomProxy, "ANTHROPIC_API_URL", HeadroomProxy.ANTHROPIC_API_URL)
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "override.example")


def _forward(model: str, *, api_url: str | None = None, base_url: str | None = None) -> dict:
    """Send one request through the handler; return the URL and body it forwarded."""
    app = create_app(
        ProxyConfig(
            optimize=True,
            mode="cache",
            disable_kompress=True,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            anthropic_api_url=api_url,
        )
    )
    forwarded: dict[str, Any] = {}
    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _upstream(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN003, ANN202
            forwarded.update(url=url, body=copy.deepcopy(body))
            return httpx.Response(
                200,
                json={
                    "id": "msg_x",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            )

        proxy._retry_request = _upstream
        headers = {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}
        if base_url:
            headers["x-headroom-base-url"] = base_url
        response = client.post(
            "/v1/messages",
            headers=headers,
            json={
                "model": model,
                "max_tokens": 16,
                "system": SYSTEM,
                "messages": copy.deepcopy(MESSAGES),
            },
        )
        assert response.status_code == 200, response.text[:300]
    return forwarded


@pytest.mark.parametrize(
    ("upstream", "target"),
    [
        ({"api_url": THIRD_PARTY_TARGET}, THIRD_PARTY_TARGET),
        ({"base_url": OVERRIDE}, OVERRIDE),
    ],
)
def test_third_party_model_keeps_mid_conversation_system_sections(
    upstream: dict[str, str], target: str
) -> None:
    forwarded = _forward("qwen3.8-flash-next", **upstream)

    assert forwarded["url"].startswith(target)
    # Forwarded where the client put it; the top-level system is untouched, so
    # it stays the same from turn to turn and the upstream's prefix cache holds.
    assert forwarded["body"]["messages"] == MESSAGES
    assert forwarded["body"]["system"] == SYSTEM


@pytest.mark.parametrize(
    ("model", "upstream"),
    [
        # A Claude-named model behind a gateway keeps the conservative hoist.
        ("claude-sonnet-4-6", {"api_url": THIRD_PARTY_TARGET}),
        ("claude-sonnet-4-6", {"base_url": OVERRIDE}),
        # Anthropic itself keeps the model allow-list.
        ("qwen3.8-flash-next", {}),
    ],
)
def test_other_requests_still_relocate_mid_conversation_system_sections(
    model: str, upstream: dict[str, str]
) -> None:
    forwarded = _forward(model, **upstream)

    assert [m["role"] for m in forwarded["body"]["messages"]] == ["user", "assistant", "user"]
    assert forwarded["body"]["system"] == [
        *SYSTEM,
        {"type": "text", "text": MESSAGES[1]["content"]},
    ]

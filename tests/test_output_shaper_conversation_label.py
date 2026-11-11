"""The conversation label must ride with the stratum label on every shaped path.

The ledger admits a stratum to the measured (A/B) estimate only once both arms
hold ``MEASURED_MIN_CLUSTERS`` distinct conversations. A handler that emits the
stratum without the conversation therefore feeds strata that can never qualify,
and the measured number is withheld forever with nothing in the logs to say why.

The /v1/responses path is pinned in ``test_output_shaper_responses.py``; these
drive the two remaining shaped handlers end to end against a mocked upstream.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _shaper_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    # Whole-holdout: the control arm labels itself but never shapes, so the
    # assertions below pin the label pair rather than the steering text.
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "1.0")


def _config() -> ProxyConfig:
    return ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
    )


def _assert_paired(transforms: str) -> None:
    labels = [t for t in transforms.split(",") if t.strip()]
    assert any(
        t.startswith(("output_shaper:stratum:", "output_shaper:control:")) for t in labels
    ), transforms
    assert any(t.startswith("output_shaper:conv:") for t in labels), transforms


def test_anthropic_messages_pairs_the_conversation_with_the_stratum() -> None:
    async def _fake_retry(method, url, headers, body, *args, **kwargs):
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
            headers={"content-type": "application/json"},
        )

    with TestClient(create_app(_config())) as client:
        client.app.state.proxy._retry_request = _fake_retry
        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 64,
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200, resp.text
    _assert_paired(resp.headers.get("x-headroom-transforms", ""))


def test_openai_chat_pairs_the_conversation_with_the_stratum() -> None:
    async def _fake_retry(method, url, headers, body, *args, **kwargs):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
            headers={"content-type": "application/json"},
        )

    with TestClient(create_app(_config())) as client:
        client.app.state.proxy._retry_request = _fake_retry
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert resp.status_code == 200, resp.text
    _assert_paired(resp.headers.get("x-headroom-transforms", ""))

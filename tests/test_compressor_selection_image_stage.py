"""A compressor selection without ``image`` turns off the proxy image stage (#4613).

``--compressor`` / ``HEADROOM_COMPRESSORS`` maps a selection onto the router's
``enable_*`` flags, but the provider handlers gate image compression on
``ProxyConfig.image_optimize``. A selection that leaves out ``image`` has to
switch that gate off as well, or screenshots still go through the image stage.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import headroom.proxy.helpers as helpers  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

# A 1x1 PNG; the stub below never decodes it.
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


class _CountingCompressor:
    """Stands in for the image compressor and counts image-stage entries."""

    def __init__(self) -> None:
        self.calls = 0

    def has_images(self, messages: Any) -> bool:
        self.calls += 1
        return False


@pytest.fixture
def image_stage(monkeypatch: pytest.MonkeyPatch) -> _CountingCompressor:
    stub = _CountingCompressor()
    monkeypatch.setattr(helpers, "_get_image_compressor", lambda: stub)
    return stub


def _config(mode: str, compressors: set[str] | None) -> ProxyConfig:
    return ProxyConfig(
        mode=mode,
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        compressors=compressors,
    )


async def _fake_retry(method, url, headers, body, *args, **kwargs):  # noqa: ANN001
    if "chat/completions" in url:
        payload: dict[str, Any] = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "a screenshot"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }
    else:
        payload = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "sonnet-test",
            "content": [{"type": "text", "text": "a screenshot"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3},
        }
    return httpx.Response(200, json=payload, headers={"content-type": "application/json"})


def _anthropic_image_request(client: TestClient) -> httpx.Response:
    return client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "sonnet-test",
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": PNG_B64,
                            },
                        },
                        {"type": "text", "text": "what is on this screen?"},
                    ],
                }
            ],
        },
    )


def _openai_image_request(client: TestClient) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        headers={"authorization": "Bearer test-key"},
        json={
            "model": "gpt-test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                        },
                        {"type": "text", "text": "what is on this screen?"},
                    ],
                }
            ],
        },
    )


def _image_stage_runs(
    config: ProxyConfig, send: Any, image_stage: _CountingCompressor
) -> tuple[int, bool]:
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = _fake_retry
        response = send(client)
        assert response.status_code == 200, response.text
        return image_stage.calls, proxy.config.image_optimize


@pytest.mark.parametrize(
    ("mode", "send"),
    [
        # Anthropic skips the image stage in cache mode, so token mode exercises it.
        ("token", _anthropic_image_request),
        # The OpenAI route compresses images in either mode.
        ("cache", _openai_image_request),
    ],
    ids=["anthropic-token", "openai-cache"],
)
class TestImageStageFollowsSelection:
    def test_selection_without_image_skips_the_image_stage(
        self, mode: str, send: Any, image_stage: _CountingCompressor
    ) -> None:
        runs, image_optimize = _image_stage_runs(
            _config(mode, {"smart_crusher"}), send, image_stage
        )
        assert image_optimize is False
        assert runs == 0

    def test_selection_with_image_keeps_the_image_stage(
        self, mode: str, send: Any, image_stage: _CountingCompressor
    ) -> None:
        runs, image_optimize = _image_stage_runs(
            _config(mode, {"smart_crusher", "image"}), send, image_stage
        )
        assert image_optimize is True
        assert runs == 1

    def test_wildcard_keeps_the_image_stage(
        self, mode: str, send: Any, image_stage: _CountingCompressor
    ) -> None:
        runs, _ = _image_stage_runs(_config(mode, {"*"}), send, image_stage)
        assert runs == 1

    def test_no_selection_keeps_the_image_stage(
        self, mode: str, send: Any, image_stage: _CountingCompressor
    ) -> None:
        runs, image_optimize = _image_stage_runs(_config(mode, None), send, image_stage)
        assert image_optimize is True
        assert runs == 1

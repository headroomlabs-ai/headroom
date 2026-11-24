"""Proxy handlers count each request with ONE counter (#5037).

Handlers resolve their counter through the tokenizer registry
(``count_tokens_offloaded``); ``TransformPipeline`` used to resolve its own
through ``provider.get_token_counter``. A handler that pairs its own count with
the pipeline's ``tokens_before`` / ``tokens_after`` must hand the pipeline its
counter, or ``tokens_saved`` (and everything fed from it: the ``post_compress``
hook, the inflation guard, extension metadata) is a difference of two rulers.

These tests give the pipeline a provider whose ruler deliberately differs from
the registry's. That is what happens for real whenever the two lookups resolve
different counters: Claude through ``AnthropicProvider`` on ``main`` (403 vs
367 tokens), a handler that fell back to estimation after a tokenizer timeout,
or an uncached fallback counter (#4990).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.config import TransformResult  # noqa: E402
from headroom.hooks import CompressEvent, CompressionHooks  # noqa: E402
from headroom.pipeline import PipelineStage  # noqa: E402
from headroom.providers.anthropic import AnthropicProvider  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.tokenizers import EstimatingTokenCounter, get_tokenizer  # noqa: E402

CLAUDE = "claude-sonnet-4-6"
GPT = "gpt-4o"
BODY = "def hello(name):\n    return name.upper()\n" * 60


class _HalveLastUser:
    """A real transform that shrinks the last user message to half its text."""

    name = "halve_last_user"

    def should_apply(self, messages: Any, tokenizer: Any, **kwargs: Any) -> bool:
        return True

    def apply(self, messages: list[dict[str, Any]], tokenizer: Any, **kwargs: Any):
        out = [dict(m) for m in messages]
        for m in reversed(out):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = m["content"][: len(m["content"]) // 2]
                break
        return TransformResult(
            messages=out,
            tokens_before=tokenizer.count_messages(messages),
            tokens_after=tokenizer.count_messages(out),
            transforms_applied=[self.name],
        )


def _other_ruler(base: type) -> Any:
    """A provider whose counter is NOT the registry's (2 chars per token)."""

    class _Provider(base):  # type: ignore[misc, valid-type]
        def get_token_counter(self, model: str) -> Any:
            return EstimatingTokenCounter(chars_per_token=2.0)

    return _Provider(warn=False) if base is AnthropicProvider else _Provider()


class _CaptureHooks(CompressionHooks):
    def __init__(self) -> None:
        self.events: list[CompressEvent] = []

    def post_compress(self, event: CompressEvent) -> None:
        self.events.append(event)


class _RecordCompressed:
    """An extension recording the INPUT_COMPRESSED token metadata."""

    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def on_pipeline_event(self, event: Any) -> None:
        if event.stage == PipelineStage.INPUT_COMPRESSED:
            self.metadata.append(
                {k: event.metadata.get(k) for k in ("tokens_before", "tokens_after")}
            )


def _upstream_anthropic() -> MagicMock:
    payload = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": CLAUDE,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.content = json.dumps(payload).encode()
    resp.text = json.dumps(payload)
    resp.json.return_value = payload
    return resp


def _config(hooks: CompressionHooks | None = None, **overrides: Any) -> ProxyConfig:
    base: dict[str, Any] = {
        "optimize": True,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "hooks": hooks,
    }
    base.update(overrides)
    return ProxyConfig(**base)


def test_anthropic_cache_cold_start_reports_the_handlers_ruler() -> None:
    """Cache-mode cold start takes tokens_before/after from the pipeline result.

    Extensions see those numbers at INPUT_COMPRESSED, and the handler later
    subtracts its own recounts from the same ``original_tokens``. Both sides
    must be in the handler's unit, the one ``/stats`` reports.
    """
    recorder = _RecordCompressed()
    app = create_app(_config(mode="cache"))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.anthropic_pipeline._provider = _other_ruler(AnthropicProvider)
        proxy.anthropic_pipeline.transforms = [_HalveLastUser()]
        proxy.pipeline_extensions.add(recorder)
        upstream = AsyncMock(return_value=_upstream_anthropic())
        proxy._retry_request = upstream
        messages = [{"role": "user", "content": BODY}]
        r = client.post(
            "/v1/messages",
            json={"model": CLAUDE, "max_tokens": 16, "messages": messages},
            headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
        )

    assert r.status_code == 200, r.text
    (seen,) = recorder.metadata
    ruler = get_tokenizer(CLAUDE)
    forwarded = upstream.call_args.args[3]["messages"]
    assert seen["tokens_before"] == ruler.count_messages(messages)
    assert seen["tokens_after"] == ruler.count_messages(forwarded)


def test_anthropic_pipeline_gets_the_handlers_counter() -> None:
    """Token mode: every pipeline run is handed the counter the handler used."""
    hooks = _CaptureHooks()
    app = create_app(_config(hooks, mode="token"))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        real_apply = proxy.anthropic_pipeline.apply
        spy = MagicMock(side_effect=real_apply)
        proxy.anthropic_pipeline.apply = spy
        proxy._retry_request = AsyncMock(return_value=_upstream_anthropic())
        r = client.post(
            "/v1/messages",
            json={
                "model": CLAUDE,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": BODY}],
            },
            headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
        )

    assert r.status_code == 200, r.text
    assert spy.call_count >= 1
    for call in spy.call_args_list:
        assert call.kwargs.get("token_counter") is get_tokenizer(CLAUDE)

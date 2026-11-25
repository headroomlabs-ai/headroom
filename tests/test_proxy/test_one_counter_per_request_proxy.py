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
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.config import TransformResult  # noqa: E402
from headroom.hooks import CompressEvent, CompressionHooks  # noqa: E402
from headroom.pipeline import PipelineStage  # noqa: E402
from headroom.providers.anthropic import AnthropicProvider  # noqa: E402
from headroom.providers.openai import OpenAIProvider  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.tokenizers import (  # noqa: E402
    EstimatingTokenCounter,
    TiktokenCounter,
    TokenizerRegistry,
    get_tokenizer,
)

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


# ---------------------------------------------------------------------------
# /v1/chat/completions: token mode keeps the handler's original_tokens and
# took optimized_tokens from the pipeline, so the inflation guard, tokens_saved
# and the post_compress hook compared two rulers.
# ---------------------------------------------------------------------------


def _chat_response() -> Any:
    import httpx

    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "model": GPT,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    )


def test_openai_chat_post_compress_hook_sees_one_ruler() -> None:
    hooks = _CaptureHooks()
    app = create_app(_config(hooks, mode="token"))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.openai_pipeline._provider = _other_ruler(OpenAIProvider)
        proxy.openai_pipeline.transforms = [_HalveLastUser()]
        upstream = AsyncMock(return_value=_chat_response())
        proxy._retry_request = upstream
        messages = [{"role": "user", "content": BODY}]
        r = client.post(
            "/v1/chat/completions",
            json={"model": GPT, "messages": messages},
            headers={"Authorization": "Bearer sk-test"},
        )

    assert r.status_code == 200, r.text
    (event,) = hooks.events
    ruler = get_tokenizer(GPT)
    forwarded = upstream.call_args.args[3]["messages"]
    assert event.tokens_before == ruler.count_messages(messages)
    assert event.tokens_after == ruler.count_messages(forwarded)
    assert event.tokens_saved == event.tokens_before - event.tokens_after


# ---------------------------------------------------------------------------
# HEADROOM_MODEL_LIMITS ``openai.encodings``: the pipeline used to resolve its
# counter through OpenAIProvider, which honours the mapping; it now gets the
# handler's registry counter, so the registry must honour it too, or a gateway
# alias falls back to estimation for the compression gates as well as savings.
# ---------------------------------------------------------------------------

ALIAS = "my-private-deployment"
UNMAPPED = "my-unmapped-deployment"


@pytest.fixture
def _alias_mapped_to_cl100k(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Map ``ALIAS`` to ``cl100k_base``, with fresh registry state; restored after."""
    empty_home = tmp_path / "home"
    empty_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: empty_home))
    monkeypatch.setenv(
        "HEADROOM_MODEL_LIMITS",
        json.dumps({"openai": {"encodings": {ALIAS: "cl100k_base"}}}),
    )
    monkeypatch.setattr(TokenizerRegistry(), "_cache", {})
    # raising=False: the attribute does not exist before the registry learned
    # about configured encodings.
    monkeypatch.setattr(TokenizerRegistry, "_configured_encodings", None, raising=False)


def _assert_request_ruler(counter: Any, model: str) -> None:
    """The counter the request ran on: the mapped encoding, or the registry's own."""
    assert counter is get_tokenizer(model)
    if model == ALIAS:
        assert isinstance(counter, TiktokenCounter), type(counter).__name__
        assert counter.encoding_name == "cl100k_base"
    else:
        assert isinstance(counter, EstimatingTokenCounter), type(counter).__name__


@pytest.mark.parametrize("model", [ALIAS, UNMAPPED])
def test_openai_chat_counts_with_the_configured_encoding(
    _alias_mapped_to_cl100k: None, model: str
) -> None:
    hooks = _CaptureHooks()
    app = create_app(_config(hooks, mode="token"))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.openai_pipeline.transforms = [_HalveLastUser()]
        spy = MagicMock(side_effect=proxy.openai_pipeline.apply)
        proxy.openai_pipeline.apply = spy
        proxy._retry_request = AsyncMock(return_value=_chat_response())
        messages = [{"role": "user", "content": BODY}]
        r = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": messages},
            headers={"Authorization": "Bearer sk-test"},
        )

    assert r.status_code == 200, r.text
    assert spy.call_count >= 1
    for call in spy.call_args_list:
        _assert_request_ruler(call.kwargs.get("token_counter"), model)
    (event,) = hooks.events
    assert event.tokens_before == get_tokenizer(model).count_messages(messages)


# ---------------------------------------------------------------------------
# /v1/responses: every count looked the provider's counter up again, and the
# turn-hook fold subtracted one lookup's count from another's.
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_turn_hooks() -> Any:
    """Run with an empty turn-hook registry and restore it afterwards."""
    from headroom.proxy.turn_hooks import (
        clear_turn_hooks,
        register_turn_hook,
        registered_turn_hooks,
    )

    saved = list(registered_turn_hooks())
    clear_turn_hooks()
    yield register_turn_hook
    clear_turn_hooks()
    for hook in saved:
        register_turn_hook(hook)


class _LateVocabProvider:
    """First lookup: a fallback estimate; later lookups: the real vocab."""

    def __init__(self) -> None:
        self.lookups = 0

    def get_token_counter(self, model: str) -> Any:
        self.lookups += 1
        return EstimatingTokenCounter(chars_per_token=2.0 if self.lookups == 1 else 4.0)


def _responses_harness(router: Any) -> Any:
    from headroom.proxy.handlers.openai import OpenAIHandlerMixin

    class _Pipeline:
        def __init__(self) -> None:
            self.transforms = [router] if router is not None else []

    class _Harness(OpenAIHandlerMixin):
        def __init__(self) -> None:
            self.openai_pipeline: Any = _Pipeline()
            self.openai_provider: Any = _LateVocabProvider()

    return _Harness()


def test_openai_responses_payload_counts_with_one_lookup(_isolated_turn_hooks: Any) -> None:
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    class _LookOnly:
        name = "look_only"

        def on_request(self, ctx: Any) -> None:
            ctx.count_messages(ctx.messages)

    handler = _responses_harness(ContentRouter(ContentRouterConfig()))
    payload: dict[str, Any] = {
        "model": GPT,
        "input": [{"type": "message", "role": "user", "content": "short question " * 20}],
    }
    _isolated_turn_hooks(_LookOnly())

    _working, _modified, tokens_saved, *_ = handler._compress_openai_responses_payload(
        payload, model=GPT, request_id="hr_test"
    )

    assert tokens_saved == 0
    assert handler.openai_provider.lookups == 1


def test_openai_responses_payload_that_counts_nothing_does_no_lookup(
    _isolated_turn_hooks: Any,
) -> None:
    """No tools, no turn hook, no router: nothing is counted, so nothing is
    looked up (a lookup can mean a vocab load)."""
    handler = _responses_harness(None)
    payload: dict[str, Any] = {
        "model": GPT,
        "input": [{"type": "message", "role": "user", "content": "short question"}],
    }

    handler._compress_openai_responses_payload(payload, model=GPT, request_id="hr_test")

    assert handler.openai_provider.lookups == 0


# ---------------------------------------------------------------------------
# Gemini generateContent / countTokens: the replay recount and the guard use
# the handler's counter, so the pipeline must count with it too.
# ---------------------------------------------------------------------------


def test_gemini_pipeline_gets_the_handlers_counter() -> None:
    gemini = "gemini-2.5-pro"
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    payload = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 2},
    }
    resp.content = json.dumps(payload).encode()
    resp.json.return_value = payload

    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        spy = MagicMock(side_effect=proxy.openai_pipeline.apply)
        proxy.openai_pipeline.apply = spy
        proxy._retry_request = AsyncMock(return_value=resp)
        r = client.post(
            f"/v1beta/models/{gemini}:generateContent?key=test-key",
            json={"contents": [{"parts": [{"text": BODY}]}]},
        )

    assert r.status_code == 200, r.text
    assert spy.call_count >= 1
    for call in spy.call_args_list:
        assert call.kwargs.get("token_counter") is get_tokenizer(gemini)


def test_openai_responses_payload_uses_the_callers_counter(_isolated_turn_hooks: Any) -> None:
    """The HTTP handler subtracts tokens_saved from its own original_tokens."""

    class _LookOnly:
        name = "look_only"

        def on_request(self, ctx: Any) -> None:
            ctx.count_messages(ctx.messages)

    handler = _responses_harness(None)
    payload: dict[str, Any] = {
        "model": GPT,
        "input": [{"type": "message", "role": "user", "content": "short question " * 20}],
    }
    _isolated_turn_hooks(_LookOnly())

    mine = EstimatingTokenCounter(chars_per_token=3.0)
    spy = MagicMock(wraps=mine)

    handler._compress_openai_responses_payload(
        payload, model=GPT, request_id="hr_test", token_counter=spy
    )

    assert handler.openai_provider.lookups == 0
    assert spy.count_text.called  # the turn hook counted with the caller's counter


def test_openai_responses_http_hands_the_compressor_the_handlers_counter() -> None:
    seen: list[dict[str, Any]] = []

    async def _spy(payload: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return payload, False, 0, [], None, 0, 0, 0, {}

    import httpx

    upstream = httpx.Response(
        200,
        json={
            "id": "resp_1",
            "object": "response",
            "model": GPT,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        },
    )
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._compress_openai_responses_payload_in_executor = _spy
        proxy._retry_request = AsyncMock(return_value=upstream)
        r = client.post(
            "/v1/responses",
            json={"model": GPT, "input": [{"role": "user", "content": BODY}]},
            headers={"Authorization": "Bearer sk-test"},
        )

    assert r.status_code == 200, r.text
    (kwargs,) = seen
    assert kwargs.get("token_counter") is get_tokenizer(GPT)


@pytest.mark.parametrize("model", [ALIAS, UNMAPPED])
def test_openai_responses_http_counts_with_the_configured_encoding(
    _alias_mapped_to_cl100k: None, model: str
) -> None:
    seen: list[dict[str, Any]] = []

    async def _spy(payload: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return payload, False, 0, [], None, 0, 0, 0, {}

    import httpx

    upstream = httpx.Response(
        200,
        json={
            "id": "resp_1",
            "object": "response",
            "model": model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        },
    )
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._compress_openai_responses_payload_in_executor = _spy
        proxy._retry_request = AsyncMock(return_value=upstream)
        r = client.post(
            "/v1/responses",
            json={"model": model, "input": [{"role": "user", "content": BODY}]},
            headers={"Authorization": "Bearer sk-test"},
        )

    assert r.status_code == 200, r.text
    (kwargs,) = seen
    _assert_request_ruler(kwargs.get("token_counter"), model)

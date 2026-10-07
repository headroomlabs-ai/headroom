"""End-to-end tests for the Responses<->Chat-Completions Copilot bridge.

Covers the fix for #1745 / #2643: when a Copilot subscription session is
pinned to the Responses wire API (main model is a reasoning model, e.g.
gpt-5.4) but a subagent's model is *not* a reasoning model (e.g.
claude-sonnet-4.6), `_resolve_openai_responses_handler_path` downgrades that
request to GitHub's `/chat/completions` endpoint, which requires a
`messages` array and rejects Responses-shaped bodies (`input`/`instructions`)
with 400 "messages must be non-empty". These tests verify the outbound
request is bridged to Chat-Completions shape and the upstream reply is
translated back to Responses-API shape for the client, for both the
non-streaming and (buffered, synthetic-SSE) streaming cases.
"""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("litellm")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app

_COPILOT_BASE = "https://api.githubcopilot.com"


def _chat_completion_json(*, model: str = "claude-sonnet-4.6") -> dict[str, object]:
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from the bridge!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
    }


def _build_bridge_client(monkeypatch: pytest.MonkeyPatch, *, chat_response_json: dict[str, object]):
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )

    app = create_app(config)
    proxy = app.state.proxy
    captured: dict[str, object] = {}

    async def _fake_apply_copilot_api_auth(headers: dict[str, str], *, url: str) -> dict[str, str]:
        # Real GitHub Copilot auth is out of scope here; the bridge logic
        # under test runs entirely before/after this call.
        return {**headers, "Authorization": "Bearer fake-copilot-token"}

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        _fake_apply_copilot_api_auth,
    )

    async def _fake_retry(
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict,
        **_kwargs: object,
    ) -> httpx.Response:
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return httpx.Response(200, json=chat_response_json)

    proxy._retry_request = _fake_retry

    async def _record_request_outcome(outcome: object) -> None:
        captured["outcome"] = outcome

    proxy._record_request_outcome = _record_request_outcome
    return TestClient(app), captured


def test_bridge_translates_non_streaming_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _build_bridge_client(monkeypatch, chat_response_json=_chat_completion_json())

    response = client.post(
        "/v1/responses",
        headers={
            "Authorization": "Bearer test",
            "x-headroom-base-url": _COPILOT_BASE,
            "x-headroom-original-path": "/responses",
        },
        json={"model": "claude-sonnet-4.6", "input": "Hi there", "stream": False},
    )

    assert response.status_code == 200, response.text

    # Outbound request was bridged: routed to /chat/completions with a
    # `messages` array instead of Responses-shaped `input`.
    assert captured["url"].endswith("/chat/completions")
    outbound_body = captured["body"]
    assert "messages" in outbound_body
    assert "input" not in outbound_body
    assert outbound_body["stream"] is False

    # Reply was translated back to Responses-API shape for the client.
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["model"] == "claude-sonnet-4.6"
    assert payload["usage"]["input_tokens"] == 42
    assert payload["usage"]["output_tokens"] == 7
    output = payload["output"]
    assert output and output[0]["type"] == "message"
    assert output[0]["content"][0]["text"] == "Hello from the bridge!"


def test_bridge_strips_reasoning_effort_for_non_reasoning_subagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces a live failure: a Copilot session pinned to the Responses
    wire API by a reasoning main model (e.g. gpt-5.4) puts `reasoning:
    {effort: ...}` on every request on that connection, including a
    subagent's request for a non-reasoning model. Forwarded unchanged,
    GitHub either rejects it ('reasoning_effort "medium" was provided, but
    model X does not support reasoning effort') or silently returns an
    empty completion (observed live: 200 status with zero output tokens
    across every non-reasoning model tested). The bridge must drop it.
    """
    client, captured = _build_bridge_client(monkeypatch, chat_response_json=_chat_completion_json())

    response = client.post(
        "/v1/responses",
        headers={
            "Authorization": "******",
            "x-headroom-base-url": _COPILOT_BASE,
            "x-headroom-original-path": "/responses",
        },
        json={
            "model": "claude-sonnet-4.6",
            "input": "Hi there",
            "reasoning": {"effort": "medium"},
            "stream": False,
        },
    )

    assert response.status_code == 200, response.text
    outbound_body = captured["body"]
    assert "reasoning_effort" not in outbound_body
    assert "reasoning" not in outbound_body


def test_bridge_strips_unsupported_optional_fields() -> None:
    """web_search_options / service_tier / metadata are OpenAI/Responses
    extensions that GitHub's /chat/completions endpoint validates strictly
    and rejects with 400 "Request body JSON is invalid" for models that
    don't expect them; drop them for the bridged wire body."""
    from headroom.proxy.handlers.openai import _responses_body_to_chat_completion_body

    outbound = _responses_body_to_chat_completion_body(
        "claude-sonnet-4.6",
        {
            "model": "claude-sonnet-4.6",
            "input": "Hi there",
            "reasoning": {"effort": "medium"},
            "metadata": {"some": "value"},
            "service_tier": "auto",
        },
    )
    assert "reasoning_effort" not in outbound
    assert "web_search_options" not in outbound
    assert "service_tier" not in outbound
    assert "metadata" not in outbound
    assert outbound["stream"] is False
    assert outbound["model"] == "claude-sonnet-4.6"
    assert "messages" in outbound


def test_bridge_translates_streaming_request_to_synthetic_sse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _build_bridge_client(monkeypatch, chat_response_json=_chat_completion_json())

    response = client.post(
        "/v1/responses",
        headers={
            "Authorization": "Bearer test",
            "x-headroom-base-url": _COPILOT_BASE,
            "x-headroom-original-path": "/responses",
        },
        json={"model": "claude-sonnet-4.6", "input": "Hi there", "stream": True},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")

    # The upstream call itself was still made non-streaming (buffered).
    assert captured["body"]["stream"] is False

    body_text = response.text
    assert "response.completed" in body_text
    assert "Hello from the bridge!" in body_text
    assert "data: [DONE]" in body_text


def test_bridge_leaves_reasoning_model_on_responses_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control case: a reasoning model must stay on /responses, untranslated."""
    client, captured = _build_bridge_client(
        monkeypatch,
        chat_response_json={
            "id": "resp_1",
            "object": "response",
            "created_at": 111,
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "native responses reply", "annotations": []}
                    ],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        },
    )

    response = client.post(
        "/v1/responses",
        headers={
            "Authorization": "Bearer test",
            "x-headroom-base-url": _COPILOT_BASE,
            "x-headroom-original-path": "/responses",
        },
        json={"model": "gpt-5.4", "input": "Hi there", "stream": False},
    )

    assert response.status_code == 200, response.text
    assert captured["url"].endswith("/responses")
    outbound_body = captured["body"]
    assert "input" in outbound_body
    assert "messages" not in outbound_body

    payload = response.json()
    assert payload["output"][0]["content"][0]["text"] == "native responses reply"


def test_bridge_forwards_translated_bytes_on_the_real_wire_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the `body_mutation_tracker.mark_mutated(...)` call directly.

    Unlike the other tests here, this drives the *real* `_retry_request` ->
    `prepare_outbound_body_bytes` -> `select_outbound_body` path (only the
    underlying `httpx.AsyncClient.post` is faked) so that if the mutation
    marker were ever dropped, `select_outbound_body` would forward the
    client's original raw Responses-shaped bytes verbatim instead of the
    translated Chat-Completions body -- reproducing the original
    "messages must be non-empty" bug -- and this test would catch it.
    """
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    proxy = app.state.proxy
    captured: dict[str, object] = {}

    async def _fake_apply_copilot_api_auth(headers: dict[str, str], *, url: str) -> dict[str, str]:
        return {**headers, "Authorization": "******"}

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai.apply_copilot_api_auth",
        _fake_apply_copilot_api_auth,
    )

    class _FakeHttpClient:
        async def post(self, url: str, *, content: bytes, headers: dict[str, str], **_kwargs):
            captured["url"] = url
            captured["wire_bytes"] = content
            return httpx.Response(200, json=_chat_completion_json())

    proxy.http_client = _FakeHttpClient()

    async def _record_request_outcome(outcome: object) -> None:
        captured["outcome"] = outcome

    proxy._record_request_outcome = _record_request_outcome

    client = TestClient(app)
    response = client.post(
        "/v1/responses",
        headers={
            "Authorization": "******",
            "x-headroom-base-url": _COPILOT_BASE,
            "x-headroom-original-path": "/responses",
        },
        json={"model": "claude-sonnet-4.6", "input": "Hi there", "stream": False},
    )

    assert response.status_code == 200, response.text
    wire_body = json.loads(captured["wire_bytes"])
    assert "messages" in wire_body
    assert "input" not in wire_body
    assert wire_body["model"] == "claude-sonnet-4.6"


def test_bridge_fails_closed_when_response_translation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the chat-completions reply can't be translated back, fail closed
    with a 502 rather than returning a malformed 200 to the client."""
    client, _captured = _build_bridge_client(
        monkeypatch, chat_response_json=_chat_completion_json()
    )

    def _raise(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(
        "headroom.proxy.handlers.openai._chat_completion_json_to_responses_json",
        _raise,
    )

    response = client.post(
        "/v1/responses",
        headers={
            "Authorization": "******",
            "x-headroom-base-url": _COPILOT_BASE,
            "x-headroom-original-path": "/responses",
        },
        json={"model": "claude-sonnet-4.6", "input": "Hi there", "stream": False},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "copilot_responses_bridge_error"


# ---------------------------------------------------------------------------
# x-headroom-bypass must mean "forward my bytes untouched"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "value"),
    [("x-headroom-bypass", "true"), ("x-headroom-mode", "passthrough")],
)
def test_bypass_skips_routing_and_body_rewriting(
    monkeypatch: pytest.MonkeyPatch, header: str, value: str
) -> None:
    """Bypass previously only gated model normalization, not the bridge.

    Model-name normalization was correctly skipped under bypass, but planning and
    the Responses->Chat body rewrite were not — so a bypassed request was routed
    to a *different endpoint* with a *completely rewritten body*, which breaks the
    same contract far more severely. origin/main always sent /v1/responses
    verbatim here.
    """
    client, captured = _build_bridge_client(monkeypatch, chat_response_json=_chat_completion_json())
    response = client.post(
        "/v1/responses",
        json={
            "model": "claude-sonnet-4.6",  # chat-only: would normally be bridged
            "input": "hello",
            "instructions": "be brief",
            "stream": False,
        },
        headers={header: value, "x-headroom-base-url": _COPILOT_BASE},
    )
    assert response.status_code in (200, 502), response.text
    # Routed to the Responses wire, not downgraded to /chat/completions.
    # (build_copilot_upstream_url normalizes the /v1 prefix for Copilot hosts,
    # so assert on the endpoint rather than the full path.)
    url = str(captured["url"])
    assert url.endswith("/responses"), url
    assert "/chat/completions" not in url, url
    # Body still Responses-shaped: no `messages` array was synthesized.
    body = captured["body"]
    assert isinstance(body, dict)
    assert "messages" not in body
    assert body.get("input") == "hello"
    assert body.get("instructions") == "be brief"


# ---------------------------------------------------------------------------
# Follow-up turns (turn-hook re-drives, CCR / memory continuations) must be
# bridged exactly like the first request.
# ---------------------------------------------------------------------------


def test_bridge_translates_a_turn_hook_tool_re_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hook that re-drives a bridged turn must not send Responses items to /chat/completions.

    The re-drive POSTs to the same URL as the first request. On the bridge that
    is ``/chat/completions``, which rejects ``input`` with 400, and the hook
    parses the reply as a Responses payload, so both directions need the same
    translation as the initial call.
    """
    from headroom.proxy import turn_hooks

    client, _captured = _build_bridge_client(
        monkeypatch, chat_response_json=_chat_completion_json()
    )
    proxy = client.app.state.proxy
    calls: list[tuple[str, dict]] = []
    replies = iter(["first reply", "after the tool"])

    async def _fake_retry(method, url, headers, body, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append((url, body))
        reply = _chat_completion_json()
        reply["choices"][0]["message"]["content"] = next(replies)  # type: ignore[index]
        return httpx.Response(200, json=reply)

    proxy._retry_request = _fake_retry
    seen: dict[str, object] = {}

    class _ToolReDrive:
        name = "test-tool-re-drive"

        async def on_response(self, ctx, response, call_model):  # type: ignore[no-untyped-def]
            seen["first"] = response
            return await call_model(
                [
                    *ctx.messages,
                    {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "c1", "output": "42"},
                ]
            )

    monkeypatch.setattr(turn_hooks, "_hooks", [_ToolReDrive()])

    response = client.post(
        "/v1/responses",
        headers={
            "Authorization": "Bearer test",
            "x-headroom-base-url": _COPILOT_BASE,
            "x-headroom-original-path": "/responses",
        },
        json={
            "model": "claude-sonnet-4.6",
            "input": [{"role": "user", "content": "What is the answer?"}],
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200, response.text
    assert len(calls) == 2, "the hook's re-drive never reached upstream"
    for url, body in calls:
        assert url.endswith("/chat/completions")
        assert "messages" in body and "input" not in body, "a Responses body hit /chat/completions"
    re_drive_roles = [m.get("role") for m in calls[1][1]["messages"]]
    assert "tool" in re_drive_roles, "the tool result was lost in translation"

    # The hook saw a Responses-shaped first reply, and the client gets the
    # re-driven turn back in Responses shape.
    assert isinstance(seen["first"], dict) and seen["first"]["object"] == "response"
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["output"][0]["content"][0]["text"] == "after the tool"

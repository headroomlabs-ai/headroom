"""Tool definitions a turn hook shrinks on ``/v1/responses`` are booked as tool savings.

Kiro CLI 3.x and Codex speak OpenAI Responses. Tool search condenses their tool
descriptions in ``on_request``; the chat-completions path booked that on
``turn_hook_tools_saved_tokens``, the tag every savings surface (PERF ``tool_saved``,
``/stats``, the control-plane heartbeat's ``toolTokensSaved``) sums, but the
Responses path dropped it, so the dashboard showed 0 tool savings for Kiro.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from headroom.proxy.turn_hooks import clear_turn_hooks, register_turn_hook

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


class _CondenseDescriptions:
    """Shortens every tool description in place, as tool search's condense pass does."""

    name = "condense"
    stream_safe = True

    def on_request(self, ctx: Any) -> None:
        for tool in ctx.tools or []:
            tool["description"] = tool["description"][:20]


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": f"{name} reads things carefully and reports them back " * 12,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }


def _post(body: dict) -> list:
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(ProxyConfig(optimize=True, cache_enabled=False, rate_limit_enabled=False))
    app.dependency_overrides[require_loopback] = lambda: None
    outcomes: list = []
    with TestClient(app) as client:
        server = app.state.proxy

        async def fake_retry(method, url, headers, body, stream=False, **kwargs):
            return httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "object": "response",
                    "status": "completed",
                    "model": "gpt-5.5",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
                request=httpx.Request("POST", url),
            )

        record = server._record_request_outcome

        async def capture(outcome):
            outcomes.append(outcome)
            await record(outcome)

        server._retry_request = fake_retry
        server._record_request_outcome = capture
        resp = client.post(
            "/v1/responses",
            content=json.dumps(body).encode(),
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
        )
        assert resp.status_code == 200, resp.text
    return outcomes


def test_hook_shrunk_tool_descriptions_are_booked_as_tool_savings() -> None:
    register_turn_hook(_CondenseDescriptions())

    outcomes = _post(
        {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": "list the files"}],
            "tools": [_tool("read_file"), _tool("list_dir")],
            "stream": False,
        }
    )

    assert outcomes, "no request outcome recorded"
    saved = int((outcomes[-1].tags or {}).get("turn_hook_tools_saved_tokens", 0) or 0)
    assert saved > 100, outcomes[-1].tags


def test_no_hook_change_books_no_tool_savings() -> None:
    class _Noop:
        name = "noop"
        stream_safe = True

        def on_request(self, ctx: Any) -> None:
            return None

    register_turn_hook(_Noop())

    outcomes = _post(
        {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": "list the files"}],
            "tools": [_tool("read_file")],
            "stream": False,
        }
    )

    assert outcomes
    assert not (outcomes[-1].tags or {}).get("turn_hook_tools_saved_tokens")


def test_http_turn_steered_in_its_input_is_booked_as_shaped(monkeypatch) -> None:
    """Kiro's extension steers the user turn because Kiro rejects ``instructions``;
    the HTTP outcome must still carry the verbosity label the output ledger reads."""
    from headroom.proxy.output_verbosity_policy import steering_text

    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "0")

    outcomes = _post(
        {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": f"fix it\n\n{steering_text(3)}"}],
            "stream": False,
        }
    )

    assert outcomes
    assert "output_shaper:verbosity:L3" in outcomes[-1].transforms_applied


def _text_in_arm(model: str, arm: str) -> str:
    """A steered first user turn whose conversation lands in ``arm`` at a 30% holdout."""
    from headroom.proxy.output_savings import assign_arm, conversation_key_from_body
    from headroom.proxy.output_verbosity_policy import steering_text

    for i in range(200):
        text = f"task {i}\n\n{steering_text(3)}"
        body = {"model": model, "input": [{"role": "user", "content": text}]}
        if assign_arm(conversation_key_from_body(body), 0.3) == arm:
            return text
    raise AssertionError(f"no conversation landed in {arm}")


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_both_shaping_passes_put_a_request_in_one_holdout_arm(monkeypatch, arm: str) -> None:
    """Optimized HTTP Responses shapes twice (in compression, then before the
    send). With a fractional holdout, two arm keys could label one request both
    treatment and control; the ledger reads the first stratum and drops a
    steered treatment turn that lacks a verbosity label."""
    from headroom.proxy.output_savings import parse_stratum_label

    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "0.3")
    text = _text_in_arm("gpt-5.5", arm)

    outcomes = _post(
        {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": text}],
            "stream": False,
        }
    )

    assert outcomes
    labels = list(outcomes[-1].transforms_applied)
    arms = {parsed[0] for parsed in map(parse_stratum_label, labels) if parsed is not None}
    assert arms == {arm}, labels
    # The ledger drops a treatment sample that lacks a verbosity label.
    assert ("output_shaper:verbosity:L3" in labels) is (arm == "treatment"), labels


def test_responses_conversations_without_an_id_get_their_own_holdout_keys() -> None:
    """Without a response id the key used a constant, so every Responses
    conversation of a model shared one holdout arm."""
    from headroom.proxy.output_savings import conversation_key_from_body

    def key(text: str) -> str:
        return conversation_key_from_body(
            {"model": "gpt-5.5", "input": [{"role": "user", "content": text}]}
        )

    assert key("list the files") != key("fix the failing test")
    assert key("list the files") == key("list the files")


def test_bypassed_request_with_null_messages_reaches_the_provider() -> None:
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(ProxyConfig(optimize=True, cache_enabled=False, rate_limit_enabled=False))
    app.dependency_overrides[require_loopback] = lambda: None
    with TestClient(app) as client:

        async def fake_retry(method, url, headers, body, stream=False, **kwargs):
            return httpx.Response(
                200,
                json={"id": "r", "object": "response", "status": "completed", "output": []},
                request=httpx.Request("POST", url),
            )

        app.state.proxy._retry_request = fake_retry
        resp = client.post(
            "/v1/responses",
            content=json.dumps(
                {"model": "gpt-5.5", "input": "hi", "messages": None, "stream": False}
            ).encode(),
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
                "x-headroom-bypass": "true",
            },
        )
    assert resp.status_code == 200, resp.text

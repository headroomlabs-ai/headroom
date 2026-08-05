"""End-to-end turn-hook wiring on the OpenAI chat-completions direct path.

Proves the two seams added to ``handle_openai_chat`` for the direct
(no-backend) buffered path:

* ``on_request`` fires before the upstream send — a hook can shrink the
  outbound ``tools``, and the net tool-schema token delta is recorded as a
  saving (surfaced via the ``x-headroom-transforms`` header / tags).
* ``on_response`` fires after the send with a working ``call_model`` — a hook
  can detect a tool the model asked to load, re-drive the model, and have the
  proxy return the *final* response transparently.

Uses a fake hook (mimicking the tool-router extension's shrink + reload) and a
mocked ``_retry_request`` so no network / real provider is needed. Also pins the
no-op property: with no hook registered the path is unchanged.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.proxy.turn_hooks import clear_turn_hooks, register_turn_hook  # noqa: E402

_SEARCH_TOOL = "search_tools"


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


def _big_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} does a thing " + ("x " * 40),
            "parameters": {
                "type": "object",
                "properties": {"arg": {"type": "string", "description": "y " * 60}},
            },
        },
    }


def _tools(n: int = 13) -> list[dict]:
    return [_big_tool(f"tool_{i}") for i in range(n)]


def _search_call_response() -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": _SEARCH_TOOL,
                                "arguments": '{"query":"do a thing"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }


def _final_response() -> dict:
    return {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "all done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 5, "total_tokens": 125},
    }


class _FakeRouterHook:
    """Mimics the tool-router extension: shrink on request, reload on response."""

    name = "fake_router"

    def __init__(self):
        self.on_request_calls = 0
        self.on_response_calls = 0

    def on_request(self, ctx):
        self.on_request_calls += 1
        # Shrink: drop all but the first tool + inject a search_tools stub.
        if isinstance(ctx.tools, list) and len(ctx.tools) > 2:
            ctx.tools = [ctx.tools[0], {"type": "function", "function": {"name": _SEARCH_TOOL}}]

    async def on_response(self, ctx, response, call_model):
        self.on_response_calls += 1
        tcs = (response.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
        if any(tc.get("function", {}).get("name") == _SEARCH_TOOL for tc in tcs):
            return await call_model(ctx.messages + [{"role": "user", "content": "resolved"}])
        return None


def _config() -> ProxyConfig:
    # No backend -> the "Direct OpenAI API (no backend configured)" path.
    return ProxyConfig(optimize=False, cache_enabled=False, rate_limit_enabled=False)


def _post(client: TestClient, body: dict):
    return client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Authorization": "Bearer test-key"},
    )


def test_direct_path_shrinks_then_reloads_and_returns_final():
    hook = _FakeRouterHook()
    register_turn_hook(hook)

    seen_bodies: list[dict] = []

    async def fake_retry(method, url, headers, body, *args, **kwargs):
        # capture the exact outbound body per upstream call
        import copy

        seen_bodies.append(copy.deepcopy(body))
        payload = _search_call_response() if len(seen_bodies) == 1 else _final_response()
        return httpx.Response(200, json=payload, headers={"content-type": "application/json"})

    app = create_app(_config())
    with TestClient(app) as client:
        client.app.state.proxy._retry_request = fake_retry
        resp = _post(
            client,
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _tools(13),
                "stream": False,
            },
        )

    assert resp.status_code == 200, resp.text
    # reload happened: two upstream calls, final answer returned to the client
    assert len(seen_bodies) == 2
    assert resp.json()["choices"][0]["message"]["content"] == "all done"
    assert hook.on_request_calls == 1
    assert hook.on_response_calls >= 1
    # shrink happened on the FIRST outbound body: 13 tools -> 2 (kept + search stub)
    first_tools = seen_bodies[0].get("tools")
    assert first_tools is not None and len(first_tools) == 2
    # the saving is surfaced as a transform
    transforms = resp.headers.get("x-headroom-transforms", "")
    assert "turn_hook" in transforms, transforms


def test_saving_is_recorded_per_turn_and_aggregated_in_stats():
    """The deferred-tool-schema saving is recorded on EVERY turn (each request
    logs its own tag), and the dashboard's /stats sums them across turns."""
    register_turn_hook(_FakeRouterHook())

    async def fake_retry(method, url, headers, body, *args, **kwargs):
        # no search_tools call -> no reload; just shrink + record per turn
        return httpx.Response(
            200, json=_final_response(), headers={"content-type": "application/json"}
        )

    app = create_app(_config())
    with TestClient(app) as client:
        client.app.state.proxy._retry_request = fake_retry
        for _ in range(3):  # three turns, same big tool belt each time
            r = _post(
                client,
                {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": _tools(13),
                    "stream": False,
                },
            )
            assert r.status_code == 200, r.text

        # Every turn logged its own tool-schema saving.
        logs = client.app.state.proxy.logger.get_recent(10)
        saved_per_turn = [
            int((lg.get("tags") or {}).get("turn_hook_tools_saved_tokens", 0) or 0) for lg in logs
        ]
        assert sum(1 for s in saved_per_turn if s > 0) == 3, saved_per_turn

        # /stats aggregates the per-turn savings into the tool_search layer.
        stats = client.get("/stats").json()
        ts = stats["savings"]["by_layer"]["tool_search"]
        assert ts["requests"] == 3, ts
        assert ts["tokens"] == sum(saved_per_turn) > 0, (ts, saved_per_turn)


def test_in_place_shrink_hook_is_counted():
    """The contract allows on_request to mutate ctx.tools IN PLACE (not just
    replace it). The saving must still be recorded even though the tools object
    identity is unchanged — regression for identity-gated savings accounting."""

    class InPlaceShrink:
        name = "inplace"

        def on_request(self, ctx):
            if isinstance(ctx.tools, list) and len(ctx.tools) > 2:
                # mutate the SAME list object (no reassignment)
                ctx.tools[:] = [
                    ctx.tools[0],
                    {"type": "function", "function": {"name": _SEARCH_TOOL}},
                ]

    register_turn_hook(InPlaceShrink())

    seen: list[dict] = []

    async def fake_retry(method, url, headers, body, *args, **kwargs):
        import copy

        seen.append(copy.deepcopy(body))
        return httpx.Response(
            200, json=_final_response(), headers={"content-type": "application/json"}
        )

    app = create_app(_config())
    with TestClient(app) as client:
        client.app.state.proxy._retry_request = fake_retry
        resp = _post(
            client,
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _tools(13),
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        # outbound request was shrunk in place (13 -> 2), same list object
        assert len(seen[0]["tools"]) == 2
        # ...and the saving is recorded despite the in-place mutation
        assert "turn_hook" in resp.headers.get("x-headroom-transforms", "")
        ts = client.get("/stats").json()["savings"]["by_layer"]["tool_search"]
        assert ts["tokens"] > 0 and ts["requests"] >= 1, ts


def test_in_place_message_fold_is_counted():
    """A hook may fold MESSAGE content in place (e.g. lossless-guard collapsing a
    tool_result), which lands after the pipeline's token accounting. The saving
    must be re-counted regardless of object identity, else `headroom perf` shows
    0 for it — regression for identity-gated message-token accounting."""

    class MessageFold:
        name = "msgfold"

        def on_request(self, ctx):
            # Fold a big message's content IN PLACE (mutate the dict, no reassign
            # of ctx.messages), so the list object identity is unchanged.
            for m in ctx.messages:
                if isinstance(m.get("content"), str) and len(m["content"]) > 200:
                    m["content"] = "FOLDED"

    register_turn_hook(MessageFold())

    async def fake_retry(method, url, headers, body, *args, **kwargs):
        return httpx.Response(
            200, json=_final_response(), headers={"content-type": "application/json"}
        )

    app = create_app(_config())
    with TestClient(app) as client:
        client.app.state.proxy._retry_request = fake_retry
        resp = _post(
            client,
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "pad " * 500}],  # big, foldable
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        # the message fold is attributed even though ctx.messages identity is unchanged
        assert "turn_hook" in resp.headers.get("x-headroom-transforms", "")
        # ...and the request's recorded token saving reflects it (was 0 pre-fix)
        logs = client.app.state.proxy.logger.get_recent(5)
        assert any(int(lg.get("tokens_saved", 0) or 0) > 0 for lg in logs), logs


def test_cost_recorded_once_not_twice_nonstreaming():
    """Regression: the OpenAI chat non-streaming direct path recorded cost TWICE —
    an explicit `cost_tracker.record_tokens` plus the outcome funnel's own call —
    doubling spend, request count, and budget consumption. It must fire once."""

    async def fake_retry(method, url, headers, body, *args, **kwargs):
        return httpx.Response(
            200, json=_final_response(), headers={"content-type": "application/json"}
        )

    app = create_app(_config())
    with TestClient(app) as client:
        client.app.state.proxy._retry_request = fake_retry
        ct = client.app.state.proxy.cost_tracker
        calls = {"n": 0}
        _orig = ct.record_tokens

        def _counting(*a, **k):
            calls["n"] += 1
            return _orig(*a, **k)

        ct.record_tokens = _counting
        resp = _post(
            client,
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert resp.status_code == 200, resp.text
        assert calls["n"] == 1, f"cost recorded {calls['n']}x — double-count regression"


def test_direct_path_noop_when_no_hook_registered():
    # No hook registered -> byte-identical passthrough, single upstream call.
    calls = {"n": 0}

    async def fake_retry(method, url, headers, body, *args, **kwargs):
        calls["n"] += 1
        return httpx.Response(
            200, json=_final_response(), headers={"content-type": "application/json"}
        )

    app = create_app(_config())
    with TestClient(app) as client:
        client.app.state.proxy._retry_request = fake_retry
        resp = _post(
            client,
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _tools(13),
                "stream": False,
            },
        )

    assert resp.status_code == 200, resp.text
    assert calls["n"] == 1  # no reload
    assert resp.json()["choices"][0]["message"]["content"] == "all done"
    assert "turn_hook" not in resp.headers.get("x-headroom-transforms", "")

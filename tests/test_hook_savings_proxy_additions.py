"""What the proxy itself adds to a request is a cost against its savings.

Measured on Copilot's Claude API: the ``headroom_retrieve`` tool the proxy
declares and the output shaper's steering block are billed on every turn they
are sent, yet neither was ever subtracted from the booked total. And a deferred
tool no ``tool_reference`` has loaded is not billed, so a hook condensing one
saves nothing on the bill.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from headroom.proxy.helpers import _reset_session_ccr_tracker_for_test
from headroom.proxy.turn_hooks import TurnContext, clear_turn_hooks, register_turn_hook

pytest.importorskip("fastapi")


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    # Built-in deferral off: these turns isolate the hook and the additions.
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "0")
    clear_turn_hooks()
    _reset_session_ccr_tracker_for_test()  # CCR is sticky per process-wide session
    yield
    clear_turn_hooks()
    _reset_session_ccr_tracker_for_test()


def _tool(name: str, size: int = 2_000) -> dict:
    return {
        "name": name,
        "description": ("Look things up in the index. " * (size // 30)).strip(),
        "input_schema": {"type": "object"},
    }


class _DropResident:
    """Removes the tool named ``victim``: a real, billed saving."""

    name = savings_source = "tool_router"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools = [t for t in ctx.tools if t.get("name") != "victim"]


class _CondenseDeferred:
    """Shortens every deferred tool, in place."""

    name = savings_source = "tool_desc_trim"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        for tool in ctx.tools:
            if tool.get("defer_loading"):
                tool["description"] = "short"


def _send(monkeypatch, body: dict, *, optimize: bool = False, backend=None) -> tuple[dict, dict]:
    """POST ``body`` through the proxy; return the outcome tags and the upstream body."""
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(
        ProxyConfig(
            optimize=optimize,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None
    outcomes: list = []

    async def _spy(_self, outcome, *a, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
        outcomes.append(outcome)

    monkeypatch.setattr(type(app.state.proxy), "_record_request_outcome", _spy, raising=True)
    if backend is not None:
        app.state.proxy.anthropic_backend = backend
    with respx.mock:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "a",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                },
            )
        )
        with TestClient(app) as client:
            result = client.post(
                "/v1/messages",
                json={"model": "claude-sonnet-4-5", "max_tokens": 16, **body},
                headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            )
    assert result.status_code == 200, result.text
    return outcomes[-1].tags, json.loads(route.calls.last.request.content) if route.called else {}


def _count_tools(tools: list) -> int:
    from headroom.tokenizers import get_tokenizer

    return get_tokenizer("claude-sonnet-4-5").count_text(json.dumps(tools, default=str))


def _resident_saving() -> int:
    return _count_tools([_tool("keep"), _tool("victim")]) - _count_tools([_tool("keep")])


def test_a_hook_removing_a_resident_tool_books_its_tokens(monkeypatch) -> None:
    register_turn_hook(_DropResident())
    tags, _ = _send(
        monkeypatch,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_tool("keep"), _tool("victim")],
        },
    )
    assert tags["turn_hook_tools_saved_tokens"] == _resident_saving()


def test_condensing_an_unloaded_deferred_tool_books_nothing_on_the_headline(monkeypatch) -> None:
    register_turn_hook(_CondenseDeferred())
    tags, upstream = _send(
        monkeypatch,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_tool("keep"), {**_tool("later"), "defer_loading": True}],
        },
    )
    assert upstream["tools"][1]["description"] == "short"  # the hook did run
    assert tags.get("turn_hook_tools_saved_tokens", 0) == 0


def test_condensing_a_loaded_deferred_tool_books_its_delta_on_the_headline(monkeypatch) -> None:
    register_turn_hook(_CondenseDeferred())
    reference = {"type": "tool_reference", "tool_name": "later"}
    tags, _ = _send(
        monkeypatch,
        {
            "messages": [
                {"role": "user", "content": "find it"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": [reference]}
                    ],
                },
            ],
            "tools": [_tool("keep"), {**_tool("later"), "defer_loading": True}],
        },
    )
    expected = _count_tools([_tool("keep"), _tool("later")]) - _count_tools(
        [_tool("keep"), {**_tool("later"), "description": "short"}]
    )
    assert tags["turn_hook_tools_saved_tokens"] == expected


def test_a_translating_backend_bills_a_deferred_tool_so_condensing_it_saves(monkeypatch) -> None:
    from types import SimpleNamespace

    from headroom.backends.base import BackendResponse

    async def _send_message(body, headers):  # noqa: ANN001, ANN202
        return BackendResponse(
            body={
                "id": "a",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
            status_code=200,
        )

    register_turn_hook(_CondenseDeferred())
    tags, _ = _send(
        monkeypatch,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_tool("keep"), {**_tool("later"), "defer_loading": True}],
        },
        backend=SimpleNamespace(name="litellm", send_message=_send_message),
    )
    assert tags["turn_hook_tools_saved_tokens"] > 0


def test_the_retrieve_tool_the_proxy_declares_is_netted_out(monkeypatch) -> None:
    from headroom.ccr import CCR_TOOL_NAME

    register_turn_hook(_DropResident())
    # A headroom_retrieve call in the history makes the proxy declare the tool.
    tags, upstream = _send(
        monkeypatch,
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "r1",
                            "name": CCR_TOOL_NAME,
                            "input": {"hash": "x"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "r1", "content": "data"}],
                },
            ],
            "tools": [_tool("keep"), _tool("victim")],
        },
        optimize=True,
    )
    (retrieve,) = [t for t in upstream["tools"] if t.get("name") == CCR_TOOL_NAME]
    assert tags["turn_hook_tools_saved_tokens"] == _resident_saving() - _count_tools([retrieve])


def test_the_output_shaper_steering_block_is_netted_out(monkeypatch) -> None:
    from headroom.proxy.output_verbosity_policy import STEERING_SENTINEL
    from headroom.tokenizers import get_tokenizer

    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "0")
    register_turn_hook(_DropResident())
    tags, upstream = _send(
        monkeypatch,
        {
            "system": "You are terse.",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_tool("keep"), _tool("victim")],
        },
    )
    (steering,) = [
        b["text"] for b in upstream["system"] if b.get("text", "").startswith(STEERING_SENTINEL)
    ]
    steer_tokens = get_tokenizer("claude-sonnet-4-5").count_text(steering)
    assert tags["turn_hook_tools_saved_tokens"] == _resident_saving() - steer_tokens


def test_a_deferred_unloaded_retrieve_tool_is_not_a_cost(monkeypatch) -> None:
    from headroom.ccr import CCR_TOOL_NAME

    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "1")
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_CORE_TOOLS", "keep,victim")
    register_turn_hook(_DropResident())
    tags, upstream = _send(
        monkeypatch,
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "r1",
                            "name": CCR_TOOL_NAME,
                            "input": {"hash": "x"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "r1", "content": "data"}],
                },
            ],
            "tools": [_tool("keep"), _tool("victim")] + [_tool(f"t{i}") for i in range(40)],
        },
        optimize=True,
    )
    (retrieve,) = [t for t in upstream["tools"] if t.get("name") == CCR_TOOL_NAME]
    assert retrieve["defer_loading"] is True
    assert tags["turn_hook_tools_saved_tokens"] == _resident_saving()


def test_a_retrieve_tool_the_client_declared_is_not_a_proxy_cost() -> None:
    from headroom.ccr import CCR_TOOL_NAME
    from headroom.proxy.tool_schema_savings_policy import proxy_added_tools

    retrieve = _tool(CCR_TOOL_NAME)
    assert proxy_added_tools([retrieve], [retrieve], CCR_TOOL_NAME) == []
    assert proxy_added_tools([retrieve], [], CCR_TOOL_NAME) == [retrieve]


def test_a_proxy_cost_comes_off_the_tool_search_credit_once_the_rest_is_gone():
    from headroom.proxy.savings_attribution import from_tags, record_savings
    from headroom.proxy.tool_schema_savings_policy import (
        headline_tokens_saved,
        net_hook_tool_saving,
    )

    tags = {"tool_search_deferred_tokens": 400}
    record_savings(tags, "tool_search", tokens=400, estimated=True)
    saved = net_hook_tool_saving(-70, tags, 0)
    assert headline_tokens_saved(saved, tags) == 330
    assert from_tags(tags)[0]["tokens"] == 330


def test_a_reference_repair_removes_loads_nothing():
    from headroom.proxy.tool_schema_savings_policy import unbilled_deferred_tool_names

    tools = [{"name": "lookup", "defer_loading": True}]
    history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": "s1",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [{"type": "tool_reference", "tool_name": "lookup"}],
                    },
                }
            ],
        }
    ]
    assert unbilled_deferred_tool_names(tools, history) == {"lookup"}
    search = {"name": "tool_search_tool_regex", "type": "tool_search_tool_regex_20251119"}
    assert unbilled_deferred_tool_names([search, *tools], history) == frozenset()

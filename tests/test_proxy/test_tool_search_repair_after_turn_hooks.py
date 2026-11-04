"""Tool-search history repair must run AFTER the turn hooks (#2888).

``strip_unsupported_tool_search_blocks`` (#2807) validates every replayed
``tool_reference`` against the request's ``tools`` array. A registered turn hook
may rewrite that array, so repairing before the hook validates against a stale
view: the reference looks resolvable, the hook then drops the tool it named, and
upstream 400s with ``Tool reference 'X' not found in available tools``.

These drive the real handler and assert on the forwarded body, because ordering
is the whole property under test -- a unit test of the helper cannot see it.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.ccr.tool_injection import CCR_TOOL_NAME
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.turn_hooks import clear_turn_hooks, register_turn_hook

_SEARCH_TOOL = {"type": "tool_search_tool_20250917", "name": "tool_search"}
_GREP = {"name": "Grep", "description": "search files", "input_schema": {"type": "object"}}
_READ = {"name": "Read", "description": "read a file", "input_schema": {"type": "object"}}
_CCR_TOOL = {
    "name": CCR_TOOL_NAME,
    "description": "retrieve compressed content",
    "input_schema": {"type": "object"},
}

# A transcript that already carries a resolved tool-search round trip for `Grep`.
_POISONED_MESSAGES = [
    {"role": "user", "content": "find the thing"},
    {
        "role": "assistant",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "tool_search_tool_20250917",
                "input": {"query": "grep"},
            },
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": {
                    "type": "tool_search_tool_result_content",
                    "tool_references": [{"type": "tool_reference", "tool_name": "Grep"}],
                },
            },
        ],
    },
    {"role": "user", "content": "now use it"},
]

_CCR_POISONED_MESSAGES = [
    {"role": "user", "content": "expand the saved result"},
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_ccr_1",
                "name": CCR_TOOL_NAME,
                "input": {"hash": "abc123def456abc123def456"},
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_ccr_1",
                "content": "the expanded result",
            }
        ],
    },
]


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


class _DropToolHook:
    """Turn hook that removes one tool from the outbound array."""

    def __init__(self, name: str):
        self._name = name

    def on_request(self, ctx) -> None:  # noqa: ANN001
        if ctx.tools:
            ctx.tools = [t for t in ctx.tools if t.get("name") != self._name]


class _InertHook:
    def on_request(self, ctx) -> None:  # noqa: ANN001, ARG002
        return None


def _run(
    hook,  # noqa: ANN001
    *,
    messages: list[dict] = _POISONED_MESSAGES,
    tools: list[dict] | None = None,
) -> dict:
    """POST a poisoned transcript through the handler, return the forwarded body."""
    captured: dict[str, object] = {}
    register_turn_hook(hook)

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
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        proxy.pipeline_extensions.emit = lambda *args, **kwargs: SimpleNamespace(
            messages=kwargs.get("messages"),
            tools=kwargs.get("tools"),
            headers=kwargs.get("headers"),
            metadata=kwargs.get("metadata"),
        )

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_repair_order",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 20, "output_tokens": 3},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": messages,
                "tools": tools or [_SEARCH_TOOL, _GREP, _READ],
            },
        )
        assert response.status_code == 200

    return captured["body"]  # type: ignore[return-value]


def _referenced_tool_names(body: dict) -> list[str]:
    names = []
    for message in body.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_search_tool_result":
                continue
            inner = block.get("content")
            entries = inner.get("tool_references") if isinstance(inner, dict) else inner
            for entry in entries or []:
                names.append(str(entry.get("tool_name") or entry.get("name")))
    return names


def _block_types(body: dict) -> list[str]:
    types = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            types.extend(str(b.get("type")) for b in content if isinstance(b, dict))
    return types


def test_repair_sees_the_tools_array_the_hook_left_behind() -> None:
    """A hook that drops `Grep` must leave no dangling reference to it."""
    forwarded = _run(_DropToolHook("Grep"))

    assert "Grep" not in [t.get("name") for t in forwarded["tools"]]
    # The whole pair goes: an orphaned server_tool_use 400s on its own.
    assert _referenced_tool_names(forwarded) == []
    assert "tool_search_tool_result" not in _block_types(forwarded)
    assert "server_tool_use" not in _block_types(forwarded)


def test_repair_leaves_resolvable_history_alone_when_the_hook_keeps_the_tool() -> None:
    """The converse: no over-stripping when the hook does not touch `tools`."""
    forwarded = _run(_InertHook())

    assert _referenced_tool_names(forwarded) == ["Grep"]
    assert "tool_search_tool_result" in _block_types(forwarded)


def test_ccr_repair_sees_the_tools_array_the_hook_left_behind() -> None:
    """CCR repair also runs after a hook removes headroom_retrieve."""
    forwarded = _run(
        _DropToolHook(CCR_TOOL_NAME),
        messages=_CCR_POISONED_MESSAGES,
        tools=[_CCR_TOOL, _READ],
    )

    assert CCR_TOOL_NAME not in [t.get("name") for t in forwarded["tools"]]
    assert "tool_use" not in _block_types(forwarded)
    assert "tool_result" not in _block_types(forwarded)
    assert forwarded["messages"][2]["content"] == [{"type": "text", "text": "the expanded result"}]


def test_ccr_repair_leaves_resolvable_history_when_hook_keeps_tool() -> None:
    """CCR history remains structured when the final tools array declares it."""
    forwarded = _run(
        _InertHook(),
        messages=_CCR_POISONED_MESSAGES,
        tools=[_CCR_TOOL, _READ],
    )

    assert CCR_TOOL_NAME in [t.get("name") for t in forwarded["tools"]]
    assert "tool_use" in _block_types(forwarded)
    assert "tool_result" in _block_types(forwarded)

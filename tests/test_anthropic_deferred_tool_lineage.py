"""A tool Claude Code's ToolSearch loads is not a new conversation.

Claude Code appends every tool its ToolSearch loads to ``tools``, still marked
``defer_loading``, and Anthropic keeps deferred tools out of the prompt it
caches. The prefix-tracker lineage key hashed the whole ``tools`` array, so
the turn after a load started a new lineage at frozen=0, the pipeline re-ran
the whole history, and history it had forwarded verbatim came out compressed:
different bytes, so the provider re-wrote the message tier.

Where the deferred tools reach the provider eagerly (a third-party upstream
gets them with ``defer_loading`` stripped, a translating backend as plain
functions), a load does change the provider's key, so it must still start a
new lineage.

The upstream below models Anthropic's prompt cache over the bytes the handler
forwards, deferred tools left out of the key, so the assertions are on what
the provider would do with them.
"""

from __future__ import annotations

import copy
import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")

import httpx
from fastapi.testclient import TestClient

from headroom.backends.base import BackendResponse
from headroom.cache.prefix_tracker import PrefixCacheTracker, cache_key_tools
from headroom.proxy.helpers import reset_deferred_orphan_warn_state
from headroom.proxy.server import ProxyConfig, create_app

MODEL = "claude-sonnet-4-5-20250929"
SYSTEM = [{"type": "text", "text": "You are a coding agent. " * 40}]
MARKER = {"type": "ephemeral"}
READ = {"name": "Read", "input_schema": {"type": "object"}}
# What Claude Code (2.1.296) sends while it defers tools: its client-side
# ToolSearch tool, a reserved placeholder that is deferred on every request,
# and each tool ToolSearch has loaded, still deferred.
TOOL_SEARCH = {
    "name": "ToolSearch",
    "description": "Search for tools by name or description.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}
PLACEHOLDER = {
    "name": "DeferredToolPlaceholder",
    "description": "Reserved placeholder that keeps deferred tool loading active; never call this tool.",
    "input_schema": {"type": "object"},
    "defer_loading": True,
}
TOOLS = [READ, TOOL_SEARCH, PLACEHOLDER]
LOADED = {"name": "mcp__docs__search", "input_schema": {"type": "object"}, "defer_loading": True}
EAGER = {"name": "WebFetch", "input_schema": {"type": "object"}}
THIRD_PARTY_URL = "https://gateway.example.com/anthropic"


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _history(turns: int) -> list[dict[str, Any]]:
    """A session of file reads. Token mode keeps recent reads verbatim and
    compresses older ones, so a history re-run at frozen=0 changes bytes the
    provider already cached."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "Audit the repo."}]}
    ]
    for i in range(turns):
        tid = f"toolu_{i:02d}"
        rows = [{"path": f"src/m{i}/f{j}.py", "size": 1000 + j, "ok": True} for j in range(400)]
        messages += [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tid,
                        "name": "Read",
                        "input": {"file_path": f"/repo/m{i}.json"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tid, "content": json.dumps(rows)}
                ],
            },
        ]
    messages[-1]["content"][-1]["cache_control"] = dict(MARKER)
    return messages


def _after_tool_search(turns: int) -> list[dict[str, Any]]:
    """The request carrying ToolSearch's result: the loaded tool comes back as
    a tool_reference, and Claude Code appends it to ``tools``."""
    messages = _history(turns)
    del messages[-1]["content"][-1]["cache_control"]
    messages += [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_ts",
                    "name": "ToolSearch",
                    "input": {"query": "docs"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_ts",
                    "content": [{"type": "tool_reference", "tool_name": LOADED["name"]}],
                    "cache_control": dict(MARKER),
                }
            ],
        },
    ]
    return messages


class _PromptCache:
    """Anthropic's prompt cache at message granularity, deferred tools unkeyed."""

    def __init__(self) -> None:
        self._written: set[str] = set()

    def request(self, body: dict[str, Any]) -> tuple[int, int]:
        messages = body["messages"]
        tools = [t for t in body.get("tools") or [] if not t.get("defer_loading")]
        head = json.dumps(_strip([tools, body.get("system")]), sort_keys=True)
        keys = [
            head + json.dumps(_strip(messages[: i + 1]), sort_keys=True)
            for i in range(len(messages))
        ]
        last = len(messages) - 1
        hit = max((i for i in range(last + 1) if keys[i] in self._written), default=-1)
        self._written.add(keys[last])
        tokens = PrefixCacheTracker._estimate_message_tokens(messages)
        return sum(tokens[: hit + 1]), sum(tokens[hit + 1 :])


def _drive(
    requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    *,
    api_url: str | None = None,
    backend: bool = False,
) -> list[dict]:
    """Send (tools, messages) through the real handler; record each request's
    resolved tracker, its frozen count, what was forwarded and the provider's
    cache read and write."""
    app = create_app(
        ProxyConfig(
            optimize=True,
            mode="token",
            disable_kompress=True,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            anthropic_api_url=api_url,
        )
    )
    cache = _PromptCache()
    seen: list[dict] = []
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        store = proxy.session_tracker_store
        resolve = store.resolve_tracker

        def _resolve(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            tracker = resolve(*args, **kwargs)
            seen.append({"tracker": tracker, "frozen": tracker.get_frozen_message_count()})
            return tracker

        store.resolve_tracker = _resolve

        def _reply(body: dict[str, Any]) -> dict[str, Any]:
            read, write = cache.request(body)
            seen[-1].update(
                forwarded=copy.deepcopy(body["messages"]),
                tools=copy.deepcopy(body.get("tools")),
                read=read,
                write=write,
            )
            return {
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": read,
                    "cache_creation_input_tokens": write,
                },
            }

        async def _upstream(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN003
            return httpx.Response(200, json=_reply(body))

        async def _backend_send(body, headers):  # noqa: ANN001, ANN202
            # A translating backend sends every tool as a plain function.
            tools = [{k: v for k, v in t.items() if k != "defer_loading"} for t in body["tools"]]
            return BackendResponse(body=_reply({**body, "tools": tools}), status_code=200)

        proxy._retry_request = _upstream
        if backend:
            proxy.anthropic_backend = SimpleNamespace(name="litellm", send_message=_backend_send)
        for tools, messages in requests:
            response = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": MODEL,
                    "max_tokens": 64,
                    "system": SYSTEM,
                    "tools": tools,
                    "messages": messages,
                },
            )
            assert response.status_code == 200, response.text[:300]
    assert len(seen) == len(requests)
    return seen


def _session(added: dict[str, Any], **upstream: Any) -> tuple[dict, dict]:
    turns = [(TOOLS, _history(n)) for n in range(1, 10)]
    if added is LOADED:
        turns.append(([*TOOLS, added], _after_tool_search(9)))
    else:
        turns.append(([*TOOLS, added], _history(10)))
    seen = _drive(turns, **upstream)
    return seen[-2], seen[-1]


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):  # noqa: ANN001, ANN202
    from headroom.cache.compression_store import reset_compression_store

    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    # Each proxied request appends to the savings ledger, by default the
    # developer's real ~/.headroom/savings_events.jsonl.
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "savings_events.jsonl"))
    # Headroom does not recognize Claude Code's client-side ToolSearch as a
    # search tool by default, so a request with deferred tools logs
    # tool_search_deferred_orphan. Naming it is the documented remedy and
    # changes nothing on the lineage path.
    monkeypatch.setenv("HEADROOM_CLIENT_TOOL_SEARCH_NAMES", "ToolSearch")
    reset_deferred_orphan_warn_state()
    reset_compression_store()


def test_deferred_tool_load_keeps_the_lineage_and_its_cached_prefix(caplog) -> None:  # noqa: ANN001
    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        before, after = _session(LOADED)

    assert after["tracker"] is before["tracker"]
    assert after["frozen"] == len(before["forwarded"])
    assert _strip(after["forwarded"][: len(before["forwarded"])]) == _strip(before["forwarded"])
    assert after["read"] == before["read"] + before["write"]
    # Forwarded as Claude Code sent it: still deferred, reference intact.
    assert after["tools"][-1] == LOADED
    assert after["forwarded"][-1]["content"][0]["content"][0]["tool_name"] == LOADED["name"]
    assert not [r for r in caplog.records if "tool_search_deferred_orphan" in r.getMessage()]


@pytest.mark.parametrize(
    "upstream",
    [{"api_url": THIRD_PARTY_URL}, {"backend": True}],
    ids=["third-party-upstream", "translating-backend"],
)
def test_deferred_tool_load_sent_eagerly_starts_a_new_lineage(upstream) -> None:  # noqa: ANN001
    before, after = _session(LOADED, **upstream)

    assert after["tracker"] is not before["tracker"]
    assert after["frozen"] == 0
    # The provider got the loaded tool eagerly, so its key changed anyway:
    # nothing of the old prefix was readable to keep.
    assert not any(t.get("defer_loading") for t in after["tools"])
    assert after["read"] == 0


def test_kill_switch_keys_deferred_tools_again(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("HEADROOM_DEFERRED_TOOL_AFFINITY", "0")
    before, after = _session(LOADED)

    assert after["tracker"] is not before["tracker"]
    assert after["frozen"] == 0
    # The cost the fix removes: the re-run compressed reads that were
    # forwarded verbatim, so the provider read none of the history back.
    assert after["read"] == 0


def test_runtime_env_override_keys_deferred_tools_again() -> None:
    # `headroom wrap` pushes this knob to a reused proxy, so the rollback
    # works without the proxy's own environment or a restart.
    from headroom.proxy import runtime_env

    assert runtime_env.explicit_env({"HEADROOM_DEFERRED_TOOL_AFFINITY": "0"}) == {
        "HEADROOM_DEFERRED_TOOL_AFFINITY": "0"
    }
    runtime_env.set_overrides({"HEADROOM_DEFERRED_TOOL_AFFINITY": "0"})
    try:
        before, after = _session(LOADED)
    finally:
        runtime_env.clear_overrides()

    assert after["tracker"] is not before["tracker"]
    assert after["frozen"] == 0


def test_eager_tool_still_starts_a_new_lineage() -> None:
    before, after = _session(EAGER)

    assert after["tracker"] is not before["tracker"]
    assert after["frozen"] == 0


def test_cache_key_tools_drops_only_deferred_tools() -> None:
    undeferred = {**LOADED, "defer_loading": False}
    tools = [READ, LOADED, undeferred]

    assert cache_key_tools(tools) == [READ, undeferred]
    assert tools == [READ, LOADED, undeferred]
    assert cache_key_tools(None) is None

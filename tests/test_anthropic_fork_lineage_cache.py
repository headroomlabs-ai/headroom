"""A Claude Code side request must not cost the main conversation its prompt cache.

Production, 2026-10-08: a W.A.L.T.E.R. review subagent busted Anthropic's
prompt cache on 35 of its 81 turns (``cache_read_input_tokens`` fell to the
6,065-token system prompt while ~100-150k tokens were re-written), and its
600k-token parent session did the same after its own side requests. Every
bust followed a *fork*: Claude Code re-sends the agent's history plus one
extra prompt (the subagent progress summary merges it into the last user
message; the parent's side query appends a reply and a new user message) with
its breakpoint one message earlier, so it reads the cache without writing.

The fork extended the conversation's prefix-tracker lineage and replaced its
snapshot. The next main turn extends the history the fork branched from, not
the fork, so it matched no lineage, started a cold tracker (frozen=0) and the
token-mode pipeline recompressed tool results it had forwarded verbatim the
turn before. Different bytes, so the provider's prefix cache missed from the
first changed message onward.

The upstream below models Anthropic's prompt cache (a prefix is readable once
a breakpoint wrote it), so the assertions are on what the provider would do
with the bytes the handler forwards.
"""

from __future__ import annotations

import contextvars
import copy
import json
from typing import Any

import pytest

pytest.importorskip("fastapi")

import httpx
from fastapi.testclient import TestClient

from headroom.cache.prefix_tracker import PrefixCacheTracker, SessionTrackerStore
from headroom.proxy.server import ProxyConfig, create_app

MODEL = "claude-sonnet-4-5-20250929"
SYSTEM = [
    {
        "type": "text",
        "text": "You review pull requests. Report only real defects. " * 40,
        "cache_control": {"type": "ephemeral"},
    }
]
BASH_TOOL = {
    "name": "Bash",
    "description": "Run a shell command",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}
MARKER = {"type": "ephemeral"}


def _bash_turn(i: int) -> list[dict[str, Any]]:
    tid = f"toolu_{i:02d}"
    rows = [
        {"path": f"src/module_{i}/file_{j}.py", "size": 1000 + j, "status": "ok"}
        for j in range(400)
    ]
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": f"ls {i}"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tid, "content": json.dumps(rows)}],
        },
    ]


def _main(turns: int) -> list[dict[str, Any]]:
    """The agent's own request: Claude Code's breakpoint rides the newest block."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "Review the change."}]}
    ]
    for i in range(turns):
        messages += _bash_turn(i)
    messages[-1]["content"][-1]["cache_control"] = dict(MARKER)
    return messages


def _summary_fork(turns: int) -> list[dict[str, Any]]:
    """Subagent progress summary: same history, the prompt merged into the last
    user message, breakpoint one message earlier (reads, never writes)."""
    messages = _main(turns)
    del messages[-1]["content"][-1]["cache_control"]
    messages[-1]["content"].append({"type": "text", "text": "Summarize your progress in 5 words."})
    messages[-2]["content"][-1]["cache_control"] = dict(MARKER)
    return messages


def _side_query_fork(turns: int) -> list[dict[str, Any]]:
    """Parent-session side query: history + the agent's reply + a new prompt."""
    messages = _main(turns)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": "Done."}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Name this session."}]})
    del messages[-3]["content"][-1]["cache_control"]
    messages[-2]["content"][-1]["cache_control"] = dict(MARKER)
    return messages


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _has_marker(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and "cache_control" in block for block in content
    )


class _PromptCache:
    """Anthropic's prompt cache at message granularity.

    A request reads the longest prefix an earlier breakpoint wrote, then
    writes a cache entry at each of its own breakpoints. Tools and system are
    part of every key because Anthropic renders them before the messages.
    """

    def __init__(self) -> None:
        self._written: set[str] = set()

    def request(self, body: dict[str, Any]) -> tuple[int, int, int]:
        messages = body.get("messages") or []
        head = json.dumps(_strip([body.get("tools"), body.get("system")]), sort_keys=True)
        keys = [
            head + json.dumps(_strip(messages[: i + 1]), sort_keys=True)
            for i in range(len(messages))
        ]
        marks = [i for i, m in enumerate(messages) if _has_marker(m)]
        if not marks:
            return 0, 0, -1
        hit = max((i for i in range(marks[-1] + 1) if keys[i] in self._written), default=-1)
        self._written.update(keys[i] for i in marks)
        # The tracker's own estimate: usage it can map back onto exactly the
        # messages the provider cached, so only the fork can move the floor.
        tokens = PrefixCacheTracker._estimate_message_tokens(messages)
        system_tokens = len(json.dumps(body.get("system"))) // 4
        read = system_tokens + sum(tokens[: hit + 1])
        write = sum(tokens[hit + 1 : marks[-1] + 1])
        return read, write, hit


def _drive(requests: list[list[dict[str, Any]]]) -> list[tuple[list[dict[str, Any]], int]]:
    """Send each history through the real handler; return (forwarded, read_hit)."""
    app = create_app(
        ProxyConfig(
            optimize=True,
            mode="token",
            # Deterministic structural compression only: the Kompress model
            # is not available everywhere the suite runs.
            disable_kompress=True,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )
    cache = _PromptCache()
    seen: list[tuple[list[dict[str, Any]], int]] = []
    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _upstream(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN003
            read, write, hit = cache.request(body)
            seen.append((copy.deepcopy(body.get("messages") or []), hit))
            return httpx.Response(
                200,
                json={
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
                },
            )

        proxy._retry_request = _upstream
        for messages in requests:
            response = client.post(
                "/v1/messages",
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODEL,
                    "max_tokens": 64,
                    "system": SYSTEM,
                    "tools": [BASH_TOOL],
                    "messages": messages,
                },
            )
            assert response.status_code == 200, response.text[:300]
    assert len(seen) == len(requests)
    return seen


@pytest.mark.parametrize("fork", [_summary_fork, _side_query_fork], ids=["summary", "side-query"])
def test_main_turn_after_a_fork_replays_its_cached_prefix(monkeypatch, fork) -> None:  # noqa: ANN001
    from headroom.cache.compression_store import reset_compression_store

    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    reset_compression_store()

    turns = range(1, 6)
    requests: list[list[dict[str, Any]]] = []
    for n in turns:
        requests += [_main(n), fork(n)]
    seen = _drive(requests)
    mains = seen[0::2]

    # Not vacuous: the pipeline did rewrite tool results somewhere.
    assert any(
        _strip(forwarded) != _strip(original) for (forwarded, _), original in zip(seen, requests)
    ), "nothing was compressed; the test cannot observe a prefix change"

    for (previous, _), (current, hit), n in zip(mains, mains[1:], turns):
        assert _strip(current[: len(previous)]) == _strip(previous), (
            f"main turn {n + 1} re-sent turn {n}'s history with different bytes"
        )
        assert hit >= len(previous) - 1, (
            f"main turn {n + 1} read only {hit + 1}/{len(previous)} cached messages"
        )


def _record(tracker, messages, *, cached: int) -> None:  # noqa: ANN001
    tracker.update_from_response(
        cache_read_tokens=cached,
        cache_write_tokens=0,
        messages=messages,
        original_messages=messages,
        message_token_counts=[1000] * len(messages),
    )


@pytest.mark.parametrize("fork_responds_first", [True, False], ids=["fork-first", "main-first"])
def test_fork_does_not_replace_the_lineage_it_branched_from(fork_responds_first) -> None:  # noqa: ANN001
    """Store level, both orderings seen in production: the summary fork is sent
    while a slow main turn is still in flight (fork answers first), or after a
    fast one answered (main answers first). Either way the next main turn must
    resume the main turn's state, not the fork's and not a cold tracker.

    Each request runs in its own context, as each proxied request runs in its
    own asyncio task."""
    store = SessionTrackerStore()
    main_k, fork_k, main_next = _strip(_main(3)), _strip(_summary_fork(3)), _strip(_main(4))
    main_ctx, fork_ctx = contextvars.Context(), contextvars.Context()
    main_tracker = main_ctx.run(store.resolve_tracker, "sid", "anthropic", messages=main_k)
    if not fork_responds_first:
        main_ctx.run(_record, main_tracker, main_k, cached=1000 * len(main_k))
    fork_tracker = fork_ctx.run(store.resolve_tracker, "sid", "anthropic", messages=fork_k)
    fork_ctx.run(_record, fork_tracker, fork_k, cached=1000 * (len(fork_k) - 1))
    if fork_responds_first:
        main_ctx.run(_record, main_tracker, main_k, cached=1000 * len(main_k))

    resumed = contextvars.Context().run(
        store.resolve_tracker, "sid", "anthropic", messages=main_next
    )

    assert resumed is main_tracker
    assert resumed.get_frozen_message_count() == len(main_k)
    assert resumed.get_last_original_messages() == main_k

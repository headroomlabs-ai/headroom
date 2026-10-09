"""``cache_control`` outside content blocks must not accumulate across replayed turns.

Chat Completions clients that reach Anthropic models through a gateway (OpenCode
via ``@ai-sdk/openai-compatible``, then LiteLLM) put breakpoints on the message
dict and on an assistant's ``tool_calls`` entries, not on a content block, and
move them to the newest messages on every call.
The prefix replay (``finalize_turn`` / ``overlay_cached_prefix``) forwards earlier
turns' messages byte-identical, so the markers they carried back then came along
and piled up behind the client's current ones. Two markers from the client became
four and more on the wire, the gateway translated them into Anthropic blocks, and the
request failed with ``A maximum of 4 blocks with cache_control may be provided``.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.cache.prefix_tracker import mirror_client_message_cache_control
from headroom.proxy.server import ProxyConfig, create_app

CC = {"type": "ephemeral"}


def _marked(msg: dict, marker: dict | None = CC) -> dict:
    return {**msg, "cache_control": dict(marker)} if marker else dict(msg)


def _message_markers(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if isinstance(m, dict) and "cache_control" in m]


def _all_markers(messages: list[dict]) -> list[str]:
    """Every marker outside content blocks: ``"i"`` on a message, ``"i.j"`` on a tool call."""
    found = []
    for i, m in enumerate(messages):
        if "cache_control" in m:
            found.append(str(i))
        for j, call in enumerate(m.get("tool_calls") or []):
            if "cache_control" in call:
                found.append(f"{i}.{j}")
    return found


# ── the helper ──────────────────────────────────────────────────────────────


def test_replayed_markers_are_dropped_and_the_clients_are_kept() -> None:
    client = [
        _marked({"role": "system", "content": "sys"}),
        {"role": "user", "content": "q"},
        {"role": "tool", "tool_call_id": "t3", "content": "old output"},
        _marked({"role": "tool", "tool_call_id": "t5", "content": "new output"}),
    ]
    # What a replay produced: last turn's markers on user and tool 3 came along.
    replayed = [
        _marked({"role": "system", "content": "sys"}),
        _marked({"role": "user", "content": "q"}),
        _marked({"role": "tool", "tool_call_id": "t3", "content": "old <compressed>"}),
        {"role": "tool", "tool_call_id": "t5", "content": "new output"},
    ]

    out = mirror_client_message_cache_control(replayed, client)

    assert _message_markers(out) == [0, 3]
    # Only the marker moves: the replayed (compressed) content is what the
    # provider cached, and it must stay byte-identical.
    assert [m["content"] for m in out] == [m["content"] for m in replayed]


def test_replayed_tool_call_markers_are_dropped_and_the_clients_are_kept() -> None:
    def call(cid: str, marker: bool) -> dict:
        c = {"id": cid, "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        return _marked(c) if marker else c

    client = [
        {"role": "assistant", "content": "", "tool_calls": [call("t3", False)]},
        {"role": "assistant", "content": "", "tool_calls": [call("t5", True)]},
    ]
    # Last step's newest call (t3) carried the marker; the replay brought it back.
    replayed = [
        {"role": "assistant", "content": "", "tool_calls": [call("t3", True)]},
        {"role": "assistant", "content": "", "tool_calls": [call("t5", True)]},
    ]

    out = mirror_client_message_cache_control(replayed, client)

    assert _all_markers(out) == ["1.0"]
    assert out[1] is replayed[1]  # already right: left as-is


def test_tool_calls_of_different_length_are_left_alone() -> None:
    c = {"id": "t", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
    client = [{"role": "assistant", "content": "", "tool_calls": [c]}]
    messages = [{"role": "assistant", "content": "", "tool_calls": [_marked(c), _marked(c)]}]
    assert mirror_client_message_cache_control(messages, client) is messages


def test_the_clients_marker_value_is_copied_verbatim() -> None:
    ttl = {"type": "ephemeral", "ttl": "1h"}
    client = [_marked({"role": "user", "content": "q"}, ttl)]
    out = mirror_client_message_cache_control([_marked({"role": "user", "content": "q"})], client)
    assert out[0]["cache_control"] == ttl


def test_nothing_to_do_returns_the_same_list() -> None:
    client = [_marked({"role": "system", "content": "sys"}), {"role": "user", "content": "q"}]
    messages = copy.deepcopy(client)
    assert mirror_client_message_cache_control(messages, client) is messages


def test_block_level_markers_are_left_to_the_block_normalizer() -> None:
    block = {"type": "text", "text": "x", "cache_control": CC}
    client = [{"role": "user", "content": [dict(block)]}]
    messages = [{"role": "user", "content": [dict(block)]}]
    out = mirror_client_message_cache_control(messages, client)
    assert out is messages
    assert out[0]["content"][0]["cache_control"] == CC


def test_misaligned_lists_are_returned_unchanged() -> None:
    messages = [
        _marked({"role": "user", "content": "a"}),
        _marked({"role": "user", "content": "b"}),
    ]
    assert mirror_client_message_cache_control(messages, messages[:1]) is messages
    assert mirror_client_message_cache_control(messages, None) is messages


# ── through the Chat Completions handler ────────────────────────────────────

BIG = "row," * 4000


class _Tracker:
    """Stands in for the session tracker with last turn already recorded."""

    def __init__(self, previous_original: list[dict], previous_forwarded: list[dict]):
        self._original = previous_original
        self._forwarded = previous_forwarded

    def get_frozen_message_count(self) -> int:
        return 0

    def get_last_original_messages(self):  # noqa: ANN201
        return copy.deepcopy(self._original)

    def get_last_forwarded_messages(self):  # noqa: ANN201
        return copy.deepcopy(self._forwarded)

    def update_from_response(self, **kwargs):  # noqa: ANN003
        return None


def _turns() -> tuple[list[dict], list[dict], list[dict]]:
    """The shape OpenCode sends to a Claude model, three steps into one turn.

    Last step the client marked the system prompt, the newest tool call (t3) and
    its result (tool 3), and headroom forwarded tool 3 compressed. This step the
    client moved both markers to t5 and tool 5 and left t3 and tool 3 unmarked.
    """
    system = {"role": "system", "content": "You are an agent. " * 50}
    user = {"role": "user", "content": "run three commands"}
    call3 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "t3", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ],
    }
    tool3 = {"role": "tool", "tool_call_id": "t3", "content": BIG}
    call5 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "t5", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ],
    }
    tool5 = {"role": "tool", "tool_call_id": "t5", "content": "30"}

    def with_marked_call(assistant: dict) -> dict:
        return {**assistant, "tool_calls": [_marked(assistant["tool_calls"][0])]}

    previous_original = [_marked(system), user, with_marked_call(call3), _marked(tool3)]
    previous_forwarded = [
        _marked(system),
        user,
        with_marked_call(call3),
        _marked({**tool3, "content": "row,<compressed: 4000 rows>"}),
    ]
    current = [_marked(system), user, call3, tool3, with_marked_call(call5), _marked(tool5)]
    return previous_original, previous_forwarded, current


@pytest.mark.parametrize("mode", ["cache", "token"])
def test_forwarded_markers_follow_the_client_not_the_replay(mode: str) -> None:
    previous_original, previous_forwarded, current = _turns()
    captured: dict = {}

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
        proxy.config.optimize = True
        proxy.config.mode = mode
        tracker = _Tracker(previous_original, previous_forwarded)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: "s"
        proxy.session_tracker_store.get_or_create = (
            lambda session_id, provider, cache_ttl_seconds=None: tracker
        )
        proxy.openai_pipeline.apply = lambda **kwargs: SimpleNamespace(
            messages=kwargs["messages"],
            transforms_applied=[],
            timing={},
            tokens_before=100,
            tokens_after=100,
            waste_signals=None,
        )

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = json.loads(body) if isinstance(body, (bytes, str)) else body
            return httpx.Response(
                200,
                json={
                    "id": "c",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101},
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={"model": "claude-sonnet-4-6", "messages": current},
        )

    assert response.status_code == 200, response.text
    forwarded = captured["body"]["messages"]
    # Precondition: the replay was live — tool 3 went out compressed.
    assert forwarded[3]["content"] != BIG
    # The client marked the system prompt, t5 and tool 5; last step's markers on
    # t3 and tool 3 must not ride along with the replayed bytes.
    assert _all_markers(forwarded) == _all_markers(current) == ["0", "4.0", "5"]

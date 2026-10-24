"""Chat Completions content-block ``cache_control`` must not accumulate across replays.

OpenCode V2 talking to a Claude model through an OpenAI-compatible gateway marks
content BLOCKS in Chat messages (``{"role": "user", "content": [{"type": "text",
..., "cache_control": {...}}]}``), not only the message dict and ``tool_calls``.
The prefix replay forwards earlier turns' messages byte-identical, so last turn's
block marker came back next to the client's moved one: the client sent four, the
proxy forwarded five, and the gateway's Anthropic translation failed with
``A maximum of 4 blocks with cache_control may be provided. Found 5``.
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
CC_1H = {"type": "ephemeral", "ttl": "1h"}


def _text(text: str, marker: dict | None = None) -> dict:
    block = {"type": "text", "text": text}
    return {**block, "cache_control": dict(marker)} if marker else block


def _marker_paths(value, path: str = "") -> list[str]:  # noqa: ANN001
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "cache_control":
                found.append(path or ".")
            else:
                found.extend(_marker_paths(item, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_marker_paths(item, f"{path}.{index}" if path else str(index)))
    return found


def _strip_markers(value):  # noqa: ANN001, ANN201
    if isinstance(value, dict):
        return {k: _strip_markers(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_markers(v) for v in value]
    return value


def _system(first: dict | None = CC, last: dict | None = CC) -> dict:
    return {
        "role": "system",
        "content": [
            _text("You are an agent.", first),
            _text("Preserve records."),
            _text("Use the tool."),
            _text("Answer briefly.", last),
        ],
    }


def _call(cid: str = "c1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "read", "arguments": "{}"}}
        ],
    }


# ── the helper ──────────────────────────────────────────────────────────────


def test_moved_user_breakpoint_drops_the_replayed_block_marker() -> None:
    client = [
        _system(),
        {"role": "user", "content": [_text("q")]},
        _call(),
        {"role": "tool", "tool_call_id": "c1", "content": [_text("result", CC)]},
    ]
    replayed = copy.deepcopy(client)
    replayed[1]["content"][0]["cache_control"] = dict(CC)  # last turn's breakpoint
    before = copy.deepcopy(replayed)

    out = mirror_client_message_cache_control(replayed, client)

    assert (
        _marker_paths(out) == _marker_paths(client) == ["0.content.0", "0.content.3", "3.content.0"]
    )
    assert replayed == before  # input not mutated
    assert _strip_markers(out) == _strip_markers(replayed)


def test_replayed_block_content_stays_byte_identical() -> None:
    client = [
        {"role": "tool", "tool_call_id": "c1", "content": [_text("raw " * 50)]},
        {"role": "user", "content": [_text("next", CC)]},
    ]
    replayed = [
        {"role": "tool", "tool_call_id": "c1", "content": [_text("<compressed>", CC)]},
        copy.deepcopy(client[1]),
    ]

    out = mirror_client_message_cache_control(replayed, client)

    assert out[0]["content"] == [_text("<compressed>")]
    assert out[1] is replayed[1]


def test_explicit_mid_message_positions_and_ttls_are_copied_verbatim() -> None:
    client = [
        _system(first=CC_1H, last=CC),
        {"role": "user", "content": [_text("a"), _text("b", CC_1H), _text("c")]},
    ]
    replayed = [
        _system(first=CC, last=CC),
        {"role": "user", "content": [_text("a"), _text("b"), _text("c", CC)]},
    ]

    out = mirror_client_message_cache_control(replayed, client)

    assert out[0]["content"][0]["cache_control"] == CC_1H
    assert out[0]["content"][3]["cache_control"] == CC
    assert [("cache_control" in b) for b in out[1]["content"]] == [False, True, False]
    assert out[1]["content"][1]["cache_control"] == CC_1H


def test_client_without_block_markers_gets_none_back_and_none_created() -> None:
    client = [_system(first=None, last=None), {"role": "user", "content": [_text("q")]}]
    replayed = [_system(), {"role": "user", "content": [_text("q", CC)]}]

    out = mirror_client_message_cache_control(replayed, client)

    assert _marker_paths(out) == []


def test_client_without_any_markers_changes_nothing() -> None:
    client = [
        {"role": "system", "content": [_text("s")]},
        {"role": "user", "content": [_text("q")]},
    ]
    messages = copy.deepcopy(client)
    assert mirror_client_message_cache_control(messages, client) is messages


def test_mixed_block_message_and_tool_call_markers() -> None:
    call = _call()
    marked_call = {**call, "tool_calls": [{**call["tool_calls"][0], "cache_control": dict(CC)}]}
    client = [
        _system(),
        {"role": "user", "content": [_text("q")]},
        marked_call,
        {"role": "tool", "tool_call_id": "c1", "content": "out", "cache_control": dict(CC)},
    ]
    replayed = [
        _system(),
        {"role": "user", "content": [_text("q", CC)], "cache_control": dict(CC)},
        call,
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
    ]

    out = mirror_client_message_cache_control(replayed, client)

    assert (
        _marker_paths(out)
        == _marker_paths(client)
        == ["0.content.0", "0.content.3", "2.tool_calls.0", "3"]
    )


def test_block_count_mismatch_leaves_that_message_unchanged() -> None:
    client = [
        {"role": "user", "content": [_text("a"), _text("b", CC)]},
        {"role": "user", "content": [_text("c", CC)]},
    ]
    # The pipeline merged two blocks; there is no safe 1:1 mapping, so no guessing.
    replayed = [
        {"role": "user", "content": [_text("ab", CC)]},
        {"role": "user", "content": [_text("c")]},
    ]

    out = mirror_client_message_cache_control(replayed, client)

    assert out[0] is replayed[0]
    assert out[1]["content"][0]["cache_control"] == CC


def test_replayed_blocks_for_a_string_client_message_lose_their_markers() -> None:
    # OpenCode V2 sends an older user turn as a string; the replay brings back
    # the block form it forwarded when that turn was the newest, marker included.
    client = [{"role": "user", "content": "q"}]
    replayed = [{"role": "user", "content": [_text("q", CC)]}]

    out = mirror_client_message_cache_control(replayed, client)

    assert out[0]["content"] == [_text("q")]


def test_block_count_mismatch_without_client_block_markers_only_drops_markers() -> None:
    client = [{"role": "user", "content": [_text("a"), _text("b")]}]
    replayed = [{"role": "user", "content": [_text("ab", CC)]}]

    out = mirror_client_message_cache_control(replayed, client)

    assert out[0]["content"] == [_text("ab")]


def test_literal_cache_control_in_tool_text_and_arguments_is_user_data() -> None:
    payload = json.dumps({"cache_control": {"type": "ephemeral"}})
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "write", "arguments": payload}}
        ],
    }
    client = [call, {"role": "tool", "tool_call_id": "c1", "content": [_text(payload)]}]
    replayed = copy.deepcopy(client)

    out = mirror_client_message_cache_control(replayed, client)

    assert out is replayed
    assert out[1]["content"][0]["text"] == payload


def test_reconciliation_is_idempotent() -> None:
    client = [
        _system(),
        {"role": "user", "content": [_text("q")]},
        {"role": "user", "content": [_text("r", CC)]},
    ]
    replayed = [
        _system(),
        {"role": "user", "content": [_text("q", CC)]},
        {"role": "user", "content": [_text("r", CC)]},
    ]

    once = mirror_client_message_cache_control(replayed, client)
    twice = mirror_client_message_cache_control(once, client)

    assert twice is once


# ── through the Chat Completions handler ────────────────────────────────────

BIG = "synthetic result " * 400


class _Tracker:
    """Session tracker with last turn recorded; records what this turn stores."""

    def __init__(self, previous_original, previous_forwarded, frozen: int):  # noqa: ANN001
        self._original = previous_original
        self._forwarded = previous_forwarded
        self._frozen = frozen
        self.recorded: list[dict] | None = None

    def get_frozen_message_count(self) -> int:
        return self._frozen

    def get_last_original_messages(self):  # noqa: ANN201
        return copy.deepcopy(self._original)

    def get_last_forwarded_messages(self):  # noqa: ANN201
        return copy.deepcopy(self._forwarded)

    def update_from_response(self, **kwargs):  # noqa: ANN003
        self.recorded = copy.deepcopy(kwargs.get("messages"))


def _opencode_turns(string_user: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    """The block-marked shape OpenCode V2 sends to Claude over Chat, two steps in.

    Last step the client marked two system blocks and the user block, and the
    proxy forwarded that. This step the client moved the moving breakpoint to the
    tool result and left the user block unmarked. Live OpenCode V2 also resends
    that older user turn as a plain string (``string_user``), so the replay
    brings back a block form the client no longer sends.
    """
    user = {
        "role": "user",
        "content": "Read fixture.json" if string_user else [_text("Read fixture.json")],
    }
    tool = {"role": "tool", "tool_call_id": "c1", "content": [_text(BIG)]}
    previous = [_system(), {"role": "user", "content": [_text("Read fixture.json", CC)]}]
    current = [_system(), user, _call(), {**tool, "content": [_text(BIG, CC)]}]
    return copy.deepcopy(previous), copy.deepcopy(previous), current


def _app_config() -> ProxyConfig:
    return ProxyConfig(
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


def _send(mode: str, frozen: int, stream: bool, turns=None):  # noqa: ANN001, ANN202
    previous_original, previous_forwarded, current = turns or _opencode_turns()
    captured: dict = {}
    tracker = _Tracker(previous_original, previous_forwarded, frozen)
    with TestClient(create_app(_app_config())) as client:
        proxy = client.app.state.proxy
        proxy.config.optimize = True
        proxy.config.mode = mode
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

        async def _fake_stream(url, headers, body, *args, **kwargs):  # noqa: ANN001, ANN002
            captured["body"] = body
            return httpx.Response(200, text="data: [DONE]\n\n")

        proxy._retry_request = _fake_retry
        proxy._stream_response = _fake_stream
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={"model": "claude-haiku-4-5", "stream": stream, "messages": current},
        )
    assert response.status_code == 200, response.text
    return captured["body"], current, tracker


@pytest.mark.parametrize("string_user", [False, True], ids=["block-user", "string-user"])
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize("mode,frozen", [("cache", 0), ("token", 2)])
def test_forwarded_block_markers_follow_the_client(
    mode: str, frozen: int, stream: bool, string_user: bool
) -> None:
    forwarded, current, _ = _send(mode, frozen, stream, _opencode_turns(string_user))

    paths = _marker_paths(forwarded["messages"])
    assert paths == _marker_paths(current) == ["0.content.0", "0.content.3", "3.content.0"]
    if not string_user:
        assert _strip_markers(forwarded["messages"]) == _strip_markers(current)
    else:
        # The replayed block form of the old user turn is what the provider cached.
        assert forwarded["messages"][1]["content"] == [_text("Read fixture.json")]
        assert _strip_markers(forwarded["messages"][2:]) == _strip_markers(current[2:])


def test_tracker_records_the_reconciled_layout_so_turn_three_stays_bounded() -> None:
    forwarded, current, tracker = _send("cache", 0, False)
    assert tracker.recorded is not None
    assert _marker_paths(tracker.recorded) == _marker_paths(forwarded["messages"])

    # Turn 3 replays what turn 2 recorded; the moving breakpoint moves again.
    tool2 = {"role": "tool", "tool_call_id": "c2", "content": [_text("second", CC)]}
    turn3 = copy.deepcopy(current)
    turn3[3]["content"][0].pop("cache_control")
    turn3 += [_call("c2"), tool2]
    forwarded3, _, _ = _send("cache", 0, False, (current, tracker.recorded, turn3))

    assert _marker_paths(forwarded3["messages"]) == ["0.content.0", "0.content.3", "5.content.0"]

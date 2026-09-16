"""Cross-cutting invariants of the gateway turn contract (spec section 6, ``test_invariants``).

1. NO-SHRINK-WITHOUT-RELOAD - a transform whose reload needs the response only
   runs when the gateway declared it can re-drive (property over every
   capability combination, hook kind and stream flag).
2. BYTE-STABILITY - across 3 session turns the whole provider-bound body prefix
   (messages, tools, system) is byte-identical, hooks or not.
3. FAIL-OPEN - a broken pipeline never blocks the client: the gateway forwards
   the original body.
4. ISOLATION - sessions and turns never share state.
5. CONCURRENCY - 20 sessions x 3 turns from threads stay isolated; the only
   allowed 5xx is the lock-busy 503 on deliberate same-session parallelism.
6. USAGE NORMALIZATION - ``normalize_usage`` never raises on a valid shape and
   never invents a cache signal (parametrized; ``hypothesis`` is not installed).
"""

from __future__ import annotations

import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.gateway import fold_only_hook, redrive_hook_ext
from tests.gateway.conftest import compress
from tests.gateway.fake_provider import openai_text_response, openai_usage
from tests.gateway.samples import (
    SYSTEM_PROMPT,
    big_tool_history,
    canonical,
    openai_tools,
    tool_names,
)

SEARCH = redrive_hook_ext.SEARCH_TOOL_NAME


def _body(messages=None, **extra: Any) -> dict[str, Any]:
    return {"model": "gpt-4o", "messages": messages or big_tool_history(), **extra}


# --------------------------------------------------------------------------- #
# 1. No shrink without reload                                                   #
# --------------------------------------------------------------------------- #

_CAPS = list(itertools.product([False, True], repeat=3))  # can_redrive, can_relay, affinity


@pytest.mark.parametrize("hooks", ["redrive", "fold+redrive"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("can_redrive,can_relay,affinity", _CAPS)
def test_no_shrink_without_reload(
    headroom_client, hooks, stream, can_redrive, can_relay, affinity
) -> None:
    """Invariant 1: ``search_tools`` present in the provider body  <=>  ``"redrive"``
    in obligations; and without ``redrive`` the tools are exactly the input tools."""
    if hooks == "fold+redrive":
        fold_only_hook.register(max_chars=100)
    redrive = redrive_hook_ext.register()
    sent = _body(
        tools=openai_tools(),
        stream=stream,
        gateway={
            "can_redrive": can_redrive,
            "can_relay_response": can_relay,
            "session_affinity": affinity,
        },
    )
    # A real gateway clears can_redrive for streamed turns; headroom itself sees
    # only the flag, so the property is stated on the flag headroom received.
    data = compress(headroom_client, sent).json()
    names = tool_names(data["body"]["tools"])
    shrunk = SEARCH in names
    assert shrunk == ("redrive" in data["obligations"])
    assert shrunk == (can_redrive and affinity)
    if not shrunk:
        assert names == tool_names(sent["tools"])
        assert redrive.shrink_calls == 0
    assert data["body"]["stream"] is stream


# --------------------------------------------------------------------------- #
# 2. Whole-body byte stability                                                  #
# --------------------------------------------------------------------------- #


def _three_turns(client, session_id: str, gateway: dict[str, Any]) -> list[dict[str, Any]]:
    history = big_tool_history()
    turns = [history]
    turns.append(
        history
        + [
            {"role": "assistant", "content": "Listed."},
            {"role": "user", "content": "Sort by score."},
        ]
    )
    turns.append(
        turns[1]
        + [{"role": "assistant", "content": "Sorted."}, {"role": "user", "content": "Top 3?"}]
    )
    out = []
    for messages in turns:
        resp = compress(
            client,
            _body(
                messages,
                tools=openai_tools(),
                system=SYSTEM_PROMPT,
                temperature=0.0,
                config={"session_id": session_id},
                gateway=gateway,
            ),
        )
        assert resp.status_code == 200, resp.text
        out.append(resp.json())
    return out


@pytest.mark.parametrize("hooks", ["none", "fold", "redrive", "fold+redrive"])
def test_whole_body_byte_stable_across_three_turns(headroom_client, hooks) -> None:
    """Invariant 2: every previously returned message, the compacted tools and the
    system prompt come back byte-for-byte on later turns."""
    if "fold" in hooks:
        fold_only_hook.register(max_chars=150)
    if "redrive" in hooks:
        redrive_hook_ext.register()
    gateway = {"can_redrive": "redrive" in hooks, "can_relay_response": False}
    t1, t2, t3 = _three_turns(headroom_client, f"stable-{hooks}", gateway)
    # Something was actually compressed, otherwise stability is vacuous.
    assert t1["body"]["messages"][2]["content"] != big_tool_history()[2]["content"]
    for prev, nxt in ((t1, t2), (t2, t3)):
        p, n = prev["body"], nxt["body"]
        assert canonical(n["messages"][: len(p["messages"])]) == canonical(p["messages"])
        assert canonical(n["tools"]) == canonical(p["tools"])
        assert n["system"] == p["system"] == SYSTEM_PROMPT
        assert n["temperature"] == 0.0
        # The whole body minus the growing tail is stable too.
        p_head = {k: v for k, v in p.items() if k != "messages"}
        n_head = {k: v for k, v in n.items() if k != "messages"}
        assert canonical(n_head) == canonical(p_head)
    assert t3["body"]["messages"][-1]["content"] == "Top 3?"
    if "redrive" in hooks:
        assert tool_names(t1["body"]["tools"]) == ["get_items", SEARCH]


def test_tool_compaction_bytes_stable_across_sessions(headroom_client) -> None:
    """Invariant 2 (cross-session): the compaction cache is keyed by tool bytes, so
    two sessions sending the same tools get the same bytes (shared provider cache)."""
    a = compress(
        headroom_client, _body(tools=openai_tools(), config={"session_id": "tc-a"}, gateway={})
    ).json()
    b = compress(
        headroom_client, _body(tools=openai_tools(), config={"session_id": "tc-b"}, gateway={})
    ).json()
    assert canonical(a["body"]["tools"]) == canonical(b["body"]["tools"])


# --------------------------------------------------------------------------- #
# 3. Fail-open                                                                  #
# --------------------------------------------------------------------------- #


def test_fail_open_stateless_timeout_forwards_original_body(
    headroom_client, fake_provider, make_gateway, monkeypatch
) -> None:
    """Invariant 3: stateless timeout -> 200 + originals; the gateway forwards them."""
    proxy = headroom_client.app.state.proxy
    monkeypatch.setattr(proxy, "_run_compression_in_executor", AsyncMock(side_effect=TimeoutError))
    fake_provider.script([openai_text_response("still answered")])
    body = _body(tools=openai_tools(), temperature=0.5)
    result = make_gateway(can_redrive=True, can_relay_response=True).turn(body, None)
    assert result.path == "proxy"
    assert result.compress_response["compression_skipped"] is True
    assert result.compress_response["obligations"] == []
    sent = fake_provider.calls[0].body
    assert sent["messages"] == body["messages"]
    assert canonical(sent["tools"]) == canonical(body["tools"])
    assert sent["temperature"] == 0.5
    assert result.final_response["choices"][0]["message"]["content"] == "still answered"
    # The plugin relays whenever a turn_id came back (it does not read
    # obligations for that decision); nothing was registered, so headroom
    # answers 404 and the plugin's fire-and-forget relay drops it silently.
    assert all(x.status == 404 for x in result.response_half_calls)


def test_fail_open_session_timeout_is_503_and_gateway_forwards_original(
    headroom_client, fake_provider, make_gateway, monkeypatch
) -> None:
    """Invariant 3: session mode cannot fail open with originals (replay desync),
    so headroom answers 503 and the GATEWAY fails open."""
    proxy = headroom_client.app.state.proxy
    monkeypatch.setattr(proxy, "_run_compression_in_executor", AsyncMock(side_effect=TimeoutError))
    fake_provider.script([openai_text_response("still answered")])
    body = _body(tools=openai_tools())
    result = make_gateway(can_redrive=True, can_relay_response=True).turn(body, "fo-session")
    assert result.path == "fail_open"
    assert result.compress_status == 503
    assert fake_provider.calls[0].body["messages"] == body["messages"]
    assert result.final_status == 200
    assert result.response_half_calls == []


def test_fail_open_pipeline_exception(
    headroom_client, fake_provider, make_gateway, monkeypatch
) -> None:
    """Invariant 3: a hard pipeline error is a 503 the gateway absorbs."""
    proxy = headroom_client.app.state.proxy
    monkeypatch.setattr(
        proxy, "_run_compression_in_executor", AsyncMock(side_effect=RuntimeError("boom"))
    )
    result = make_gateway(can_redrive=True, can_relay_response=True).turn(_body(), "fo-boom")
    assert result.path == "fail_open"
    assert result.compress_status == 503
    assert result.final_status == 200
    assert len(fake_provider.calls) == 1


def test_fail_open_headroom_unreachable(fake_provider, provider_client) -> None:
    """Invariant 3: headroom down -> the client still gets the provider's answer."""
    from tests.gateway.fake_gateway import FakeGateway

    class _Down:
        def post(self, *a: Any, **k: Any) -> Any:
            raise ConnectionError("connection refused")

    fake_provider.script([openai_text_response("answered without headroom")])
    result = FakeGateway(_Down(), provider_client, can_redrive=True, can_relay_response=True).turn(
        _body(), "down"
    )
    assert result.path == "fail_open"
    assert result.compress_status is None
    assert result.final_response["choices"][0]["message"]["content"] == "answered without headroom"


def test_hook_exception_never_breaks_the_turn(headroom_client, fake_provider, make_gateway) -> None:
    """Invariant 3 (hooks): a hook that raises in ``on_request`` or ``on_response``
    is skipped; the turn completes and the client gets the provider's answer."""
    from headroom.proxy.turn_hooks import register_turn_hook

    class _Broken:
        name = "broken"
        stream_safe = False

        def on_request(self, ctx: Any) -> None:
            raise RuntimeError("request boom")

        async def on_response(self, ctx: Any, response: Any, call_model: Any) -> Any:
            raise RuntimeError("response boom")

    register_turn_hook(_Broken())
    fake_provider.script([openai_text_response("fine", response_id="chatcmpl-fine")])
    result = make_gateway(can_redrive=True, can_relay_response=True).turn(
        _body(tools=openai_tools()), "hook-boom"
    )
    assert result.compress_status == 200
    assert result.final_response["id"] == "chatcmpl-fine"
    assert tool_names(fake_provider.calls[0].tools) == tool_names(openai_tools())
    if result.path == "loop":
        assert result.done is not None and result.done["response"] is None
        from tests.gateway.conftest import registry_of

        assert registry_of(headroom_client).get(result.turn_id) is None


# --------------------------------------------------------------------------- #
# 4. Isolation                                                                  #
# --------------------------------------------------------------------------- #


def test_cross_session_isolation(headroom_client) -> None:
    """Invariant 4: two sessions with identical content keep separate replay state
    and separate pending turns."""
    from tests.gateway.conftest import registry_of

    history = big_tool_history()
    gw = {"can_relay_response": True}
    a1 = compress(
        headroom_client, _body(history, config={"session_id": "iso-a"}, gateway=gw)
    ).json()
    b1 = compress(
        headroom_client, _body(history, config={"session_id": "iso-b"}, gateway=gw)
    ).json()
    assert a1["turn_id"] != b1["turn_id"]
    registry = registry_of(headroom_client)
    assert registry.get(a1["turn_id"]) is not None and registry.get(b1["turn_id"]) is not None
    # Completing A's turn leaves B pending and untouched.
    done_a = headroom_client.post(
        "/v1/compress/response",
        json={"turn_id": a1["turn_id"], "usage": openai_usage(500, 5, 400)},
    )
    assert done_a.status_code == 200 and done_a.json()["usage_applied"] is True
    assert registry.get(a1["turn_id"]) is None
    assert registry.get(b1["turn_id"]) is not None
    a2 = compress(
        headroom_client,
        _body(
            history + [{"role": "user", "content": "a next"}],
            config={"session_id": "iso-a"},
            gateway=gw,
        ),
    ).json()
    b2 = compress(
        headroom_client,
        _body(
            history + [{"role": "user", "content": "b next"}],
            config={"session_id": "iso-b"},
            gateway=gw,
        ),
    ).json()
    assert canonical(a2["body"]["messages"][:3]) == canonical(a1["body"]["messages"])
    assert canonical(b2["body"]["messages"][:3]) == canonical(b1["body"]["messages"])
    assert a2["body"]["messages"][-1]["content"] == "a next"
    assert b2["body"]["messages"][-1]["content"] == "b next"
    # B's tracker never saw A's usage.
    store = headroom_client.app.state.proxy.session_tracker_store
    assert (
        store.peek("compress\x00iso-b").get_frozen_message_count()
        <= store.peek("compress\x00iso-a").get_frozen_message_count()
    )


def test_cross_turn_isolation_same_session(headroom_client) -> None:
    """Invariant 4: two pending turns of ONE session are distinct; completing the
    older one does not complete the newer one."""
    from tests.gateway.conftest import registry_of

    gw = {"can_relay_response": True}
    t1 = compress(headroom_client, _body(config={"session_id": "iso-turns"}, gateway=gw)).json()
    t2 = compress(
        headroom_client,
        _body(
            big_tool_history() + [{"role": "user", "content": "2"}],
            config={"session_id": "iso-turns"},
            gateway=gw,
        ),
    ).json()
    assert t1["turn_id"] != t2["turn_id"]
    registry = registry_of(headroom_client)
    assert (
        headroom_client.post("/v1/compress/response", json={"turn_id": t1["turn_id"]}).status_code
        == 200
    )
    assert registry.get(t1["turn_id"]) is None
    assert registry.get(t2["turn_id"]) is not None
    assert (
        headroom_client.post("/v1/compress/response", json={"turn_id": t2["turn_id"]}).status_code
        == 200
    )
    assert registry.get(t2["turn_id"]) is None


# --------------------------------------------------------------------------- #
# 5. Concurrency                                                                #
# --------------------------------------------------------------------------- #


def test_twenty_sessions_three_turns_concurrently(headroom_client) -> None:
    """Invariant 5: distinct sessions in parallel are all 200 and each one's prefix
    is byte-stable; no pending turn leaks."""
    from tests.gateway.conftest import registry_of

    history = big_tool_history()
    gw = {"can_relay_response": True}
    errors: list[str] = []

    def _session(i: int) -> None:
        sid = f"conc-{i}"
        try:
            prev: list[dict[str, Any]] | None = None
            msgs = list(history)
            for turn in range(3):
                resp = compress(
                    headroom_client,
                    _body(msgs, tools=openai_tools(), config={"session_id": sid}, gateway=gw),
                )
                if resp.status_code != 200:
                    errors.append(f"{sid} turn {turn}: {resp.status_code} {resp.text[:200]}")
                    return
                data = resp.json()
                got = data["body"]["messages"]
                if prev is not None and canonical(got[: len(prev)]) != canonical(prev):
                    errors.append(f"{sid} turn {turn}: prefix drifted")
                prev = got
                done = headroom_client.post(
                    "/v1/compress/response",
                    json={
                        "turn_id": data["turn_id"],
                        "usage": openai_usage(400, 5, 300 if turn else 0),
                    },
                )
                if done.status_code != 200:
                    errors.append(
                        f"{sid} turn {turn}: response half {done.status_code} {done.text[:200]}"
                    )
                msgs = msgs + [
                    {"role": "assistant", "content": f"a{turn}"},
                    {"role": "user", "content": f"u{turn}"},
                ]
        except Exception as exc:  # pragma: no cover - surfaced via errors
            errors.append(f"{sid}: {exc!r}")

    with ThreadPoolExecutor(max_workers=20) as pool:
        list(pool.map(_session, range(20)))
    assert errors == []
    assert len(registry_of(headroom_client)) == 0


def test_same_session_parallel_turns_only_503_on_lock_busy(headroom_client) -> None:
    """Invariant 5: deliberate same-session parallelism may yield the lock-busy 503
    (``compression_timeout``) and nothing else above 4xx."""
    history = big_tool_history(400)
    statuses: list[int] = []
    bodies: list[dict[str, Any]] = []
    lock = threading.Lock()

    def _turn(i: int) -> None:
        resp = compress(
            headroom_client,
            _body(
                history + [{"role": "user", "content": f"p{i}"}],
                config={"session_id": "conc-same"},
                gateway={},
            ),
        )
        with lock:
            statuses.append(resp.status_code)
            bodies.append(resp.json())

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(_turn, range(6)))
    assert set(statuses) <= {200, 503}, statuses
    assert 200 in statuses
    for status, body in zip(statuses, bodies):
        if status == 503:
            assert body["error"]["type"] == "compression_timeout"
        else:
            assert "turn_id" in body and body["obligations"] == []


# --------------------------------------------------------------------------- #
# 6. normalize_usage                                                            #
# --------------------------------------------------------------------------- #

_VALID_USAGES: list[tuple[str, dict[str, Any], tuple[Any, Any, Any, Any], bool]] = [
    # (id, usage, (input, output, cache_read, cache_write), has_cache_signal)
    (
        "anthropic-full",
        {
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_input_tokens": 5,
            "cache_creation_input_tokens": 3,
        },
        (10, 2, 5, 3),
        True,
    ),
    (
        "anthropic-read-only",
        {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 0},
        (10, 2, 0, None),
        True,
    ),
    ("anthropic-bare", {"input_tokens": 10, "output_tokens": 2}, (10, 2, None, None), False),
    (
        "openai-details",
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
        (10, 2, 4, None),
        True,
    ),
    (
        "openai-toplevel",
        {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 4},
        (10, 2, 4, None),
        True,
    ),
    ("openai-bare", {"prompt_tokens": 10, "completion_tokens": 2}, (10, 2, None, None), False),
    (
        "openai-details-empty",
        {"prompt_tokens": 10, "prompt_tokens_details": {}},
        (10, None, None, None),
        False,
    ),
    (
        "kong-nested",
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 4},
            }
        },
        (10, 2, 4, None),
        True,
    ),
    ("empty", {}, (None, None, None, None), False),
    ("unrelated-keys", {"total_tokens": 12, "foo": "bar"}, (None, None, None, None), False),
    (
        "zero-everything",
        {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0},
        (0, 0, 0, None),
        True,
    ),
    (
        "nested-twice-only-descends-once",
        {"usage": {"usage": {"prompt_tokens": 10}}},
        (None, None, None, None),
        False,
    ),
]


@pytest.mark.parametrize("case", _VALID_USAGES, ids=[c[0] for c in _VALID_USAGES])
def test_normalize_usage_valid_shapes(case) -> None:
    """Invariant 6: valid shapes never raise, fields map as specified, and a cache
    signal is reported only when a cache field is present."""
    from headroom.proxy.gateway_turn import normalize_usage

    _, usage, expected, has_signal = case
    n = normalize_usage(usage)
    assert (n.input_tokens, n.output_tokens, n.cache_read, n.cache_write) == expected
    assert n.has_cache_signal is has_signal
    assert n.has_cache_signal == (n.cache_read is not None or n.cache_write is not None)


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1},
        {"input_tokens": True},
        {"cache_read_input_tokens": -7},
        {"prompt_tokens_details": {"cached_tokens": False}},
        {"cached_tokens": -1},
        {"completion_tokens": -2},
    ],
    ids=["neg-prompt", "bool-input", "neg-read", "bool-cached", "neg-cached", "neg-completion"],
)
def test_normalize_usage_rejects_bools_and_negatives(usage) -> None:
    """Invariant 6: bools and negatives are a ValueError (the handler maps to 400)."""
    from headroom.proxy.gateway_turn import normalize_usage

    with pytest.raises(ValueError):
        normalize_usage(usage)


@pytest.mark.parametrize("value", ["12", 12.5, [1], {"n": 1}])
def test_normalize_usage_rejects_non_integers(value) -> None:
    """Invariant 6: a counter that is not an integer (a string, a float, a nested
    object) is a ValueError - the same strictness ``/v1/usage`` applies - so a
    malformed relay is a visible 400 rather than a silently dropped signal."""
    from headroom.proxy.gateway_turn import normalize_usage

    with pytest.raises(ValueError):
        normalize_usage({"prompt_tokens": value})


def test_normalize_usage_null_field_is_absent() -> None:
    """Invariant 6: JSON ``null`` means the provider omitted the counter."""
    from headroom.proxy.gateway_turn import normalize_usage

    n = normalize_usage({"prompt_tokens": None, "cached_tokens": None})
    assert n.input_tokens is None
    assert n.has_cache_signal is False

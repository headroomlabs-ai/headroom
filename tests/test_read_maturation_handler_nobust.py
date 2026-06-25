"""Integration: Mechanism B (read maturation) no-bust invariant, through the
REAL Anthropic handler, across a multi-turn session.

The design's central claim is that the verbatim Read is held *out* of the
provider prefix cache until it matures, so "no cached byte is ever mutated."
The unit tests in ``test_read_maturation.py`` call the manager in isolation
with ``frozen_message_count=0``; the live test in ``test_live/`` does a
2-request hold->mature with no intermediate turns. Neither exercises the
realistic path where a held Read sits across several turns while the prefix
tracker advances ``frozen_message_count`` from the provider's reported cache
usage.

This is a regression test for that path. It drives the real handler with a
mocked upstream that echoes the cache usage Anthropic would report (caching
everything up to the breakpoint the handler chose, system blocks included),
so the prefix tracker advances exactly as in production. It then asserts,
directly on the FORWARDED bytes:

1. no-bust: the verbatim Read is never forwarded inside the cached prefix
   (at or before the last cache_control breakpoint) — if it were, maturing it
   later would mutate a cached byte and bust the prefix;
2. the mechanism actually engages: the Read is held verbatim (out of cache)
   while the file is active, then matures into a CCR marker once it quiesces,
   in that order.

Note on cache-state isolation: the CCR store is persistent (SQLite at
~/.headroom/ccr_store.db by default) and shared across processes, so stale
entries from prior runs can perturb maturation timing. Run against a clean
store for deterministic results.
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("fastapi")

import httpx
from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app

MODEL = "claude-haiku-4-5-20251001"
SYSTEM = [
    {
        "type": "text",
        "text": "You are a coding assistant. Be terse. " * 200,
        "cache_control": {"type": "ephemeral"},
    }
]
READ_TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
}
READ_ID = "toolu_r1"
# Big enough to dominate message tokens and clear the maturation min-size gate.
BIG = "".join(f"  {i}\tdef f_{i}(): return {i}  # line {i}\n" for i in range(700))


def _read_pair(tail: bool) -> list[dict]:
    tr = {"type": "tool_result", "tool_use_id": READ_ID, "content": BIG}
    if tail:
        tr["cache_control"] = {"type": "ephemeral"}
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": READ_ID,
                    "name": "Read",
                    "input": {"file_path": "/x/foo.py"},
                }
            ],
        },
        {"role": "user", "content": [tr]},
    ]


def _quiet_pair(i: int, tail: bool) -> list[dict]:
    u = {"type": "text", "text": f"Unrelated question {i}: what is {i}+{i}?"}
    if tail:
        u["cache_control"] = {"type": "ephemeral"}
    return [
        {"role": "assistant", "content": [{"type": "text", "text": str(2 * i)}]},
        {"role": "user", "content": [u]},
    ]


def _convo(nquiet: int) -> list[dict]:
    """Read of /x/foo.py followed by ``nquiet`` turns that never touch it.
    The Claude-Code-style tail breakpoint rides the newest user block."""
    msgs: list[dict] = [{"role": "user", "content": [{"type": "text", "text": "Read /x/foo.py"}]}]
    msgs += _read_pair(tail=(nquiet == 0))
    for i in range(1, nquiet + 1):
        msgs += _quiet_pair(i, tail=(i == nquiet))
    return msgs


def _breakpoint_index(messages: list[dict]) -> int:
    """Index of the last message carrying a cache_control block (-1 if none).
    Anthropic caches everything up to AND INCLUDING this message."""
    bp = -1
    for i, m in enumerate(messages):
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(b, dict) and "cache_control" in b for b in c):
            bp = i
    return bp


def _read_result_content(message: dict) -> str | None:
    c = message.get("content")
    if isinstance(c, list):
        for b in c:
            if (
                isinstance(b, dict)
                and b.get("type") == "tool_result"
                and b.get("tool_use_id") == READ_ID
            ):
                return b.get("content")
    return None


def _est_tokens(message: dict) -> int:
    return max(1, len(str(message.get("content", ""))) // 4)


def test_verbatim_read_never_cache_written_before_maturation(monkeypatch):
    # Isolate the CCR store: it is persistent (SQLite) and shared across
    # processes by default, so stale entries from other runs would perturb
    # maturation timing and make this test non-deterministic. The in-memory
    # backend gives a pristine store per test.
    from headroom.cache.compression_store import reset_compression_store

    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    reset_compression_store()

    # Match the real proxy: cache machinery ON (the prefix tracker + compression
    # cache are what maturation's hold/frozen-count logic depends on). Disabling
    # them masks the behavior under test.
    config = ProxyConfig(
        optimize=True,
        read_maturation=True,
        mode="token",
        cache_enabled=True,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
    )
    app = create_app(config)
    forwarded: list[list[dict]] = []

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        original_retry = proxy._retry_request

        async def _mock_upstream(method, url, headers, body, stream=False, **kwargs):
            msgs = body.get("messages", []) or []
            forwarded.append(copy.deepcopy(msgs))
            # Simulate Anthropic honestly caching up to the handler's breakpoint
            # (system blocks are cached too), so the prefix tracker advances
            # frozen_message_count as in prod.
            bp = _breakpoint_index(msgs)
            sys_tokens = sum(
                max(1, len(str(b.get("text", ""))) // 4)
                for b in (body.get("system") or [])
                if isinstance(b, dict)
            )
            cached = sys_tokens + (sum(_est_tokens(m) for m in msgs[: bp + 1]) if bp >= 0 else 0)
            return httpx.Response(
                200,
                json={
                    "id": "msg_x",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 2,
                        "cache_read_input_tokens": cached,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _mock_upstream
        try:
            for n in range(0, 7):
                r = client.post(
                    "/v1/messages",
                    headers={
                        "x-api-key": "test-key",
                        "anthropic-version": "2023-06-01",
                        "x-headroom-session-id": "nobust-1",
                        "content-type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "max_tokens": 20,
                        "system": SYSTEM,
                        "tools": [READ_TOOL],
                        "messages": _convo(n),
                    },
                )
                assert r.status_code == 200, f"turn {n}: {r.text[:300]}"
        finally:
            proxy._retry_request = original_retry

    assert forwarded, "no requests were forwarded"

    # Per-turn classification of the Read's forwarded form.
    held_verbatim = []  # turns where the verbatim Read is OUTSIDE the cache prefix (correct hold)
    cached_verbatim = []  # turns where the verbatim Read is INSIDE the cache prefix (bust risk)
    matured = []  # turns where the Read has become a CCR marker
    for turn, msgs in enumerate(forwarded):
        bp = _breakpoint_index(msgs)
        for i, m in enumerate(msgs):
            content = _read_result_content(m)
            if content is None:
                continue
            if content == BIG:
                (cached_verbatim if i <= bp else held_verbatim).append(turn)
            elif "Retrieve original: hash=" in content:
                matured.append(turn)

    # INVARIANT 1 (no-bust): the verbatim Read must never be forwarded inside
    # the cached prefix. If it is, maturing it later mutates a cached byte.
    assert not cached_verbatim, (
        "no-bust invariant violated: verbatim Read was cache-written before "
        f"maturation on turn(s) {cached_verbatim}. Maturing it later busts the cache."
    )

    # INVARIANT 2 (mechanism actually engages): the Read is held verbatim while
    # the file is active, then matures once it quiesces. Guards against a
    # vacuous pass where maturation silently no-ops.
    assert held_verbatim, "expected the fresh Read to be held verbatim out of cache on early turns"
    assert matured, "expected the Read to mature into a CCR marker after quiescing"
    # The matured marker only appears AFTER the verbatim hold (ordering).
    assert min(matured) > max(held_verbatim), (
        f"maturation must follow the hold: held={held_verbatim} matured={matured}"
    )


def _drive_session(config, n_turns: int, session_id: str) -> list[list[dict]]:
    """Drive ``n_turns`` cumulative turns through the real handler with a mocked
    upstream; return the forwarded message arrays per turn."""
    app = create_app(config)
    forwarded: list[list[dict]] = []
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        original_retry = proxy._retry_request

        async def _mock_upstream(method, url, headers, body, stream=False, **kwargs):
            msgs = body.get("messages", []) or []
            forwarded.append(copy.deepcopy(msgs))
            bp = _breakpoint_index(msgs)
            sys_tokens = sum(
                max(1, len(str(b.get("text", ""))) // 4)
                for b in (body.get("system") or [])
                if isinstance(b, dict)
            )
            cached = sys_tokens + (sum(_est_tokens(m) for m in msgs[: bp + 1]) if bp >= 0 else 0)
            return httpx.Response(
                200,
                json={
                    "id": "msg_x",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 2,
                        "cache_read_input_tokens": cached,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _mock_upstream
        try:
            for n in range(n_turns):
                r = client.post(
                    "/v1/messages",
                    headers={
                        "x-api-key": "test-key",
                        "anthropic-version": "2023-06-01",
                        "x-headroom-session-id": session_id,
                        "content-type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "max_tokens": 20,
                        "system": SYSTEM,
                        "tools": [READ_TOOL],
                        "messages": _convo(n),
                    },
                )
                assert r.status_code == 200, f"turn {n}: {r.text[:300]}"
        finally:
            proxy._retry_request = original_retry
    return forwarded


def _first_matured_turn(forwarded: list[list[dict]]) -> int | None:
    """The first turn index whose forwarded Read is a CCR marker."""
    for turn, msgs in enumerate(forwarded):
        for m in msgs:
            content = _read_result_content(m)
            if content and "Retrieve original: hash=" in content:
                return turn
    return None


def test_quiesce_turns_config_is_honored(monkeypatch):
    """`quiesce_turns` must be runtime-configurable end-to-end: a fresh Read of
    /x/foo.py matures `quiesce_turns` quiet turns after it appears (the convo
    builds one quiet assistant turn per step, and the Read sits at assistant
    turn 1). With quiesce_turns=2 it must mature at turn 2 — not the built-in
    default of 5. Currently the handler hardcodes ReadMaturationConfig(enabled=
    True), ignoring the configured value, so this fails (matures at 5)."""
    from headroom.cache.compression_store import reset_compression_store

    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    reset_compression_store()

    config = ProxyConfig(
        optimize=True,
        read_maturation=True,
        read_maturation_quiesce_turns=2,
        mode="token",
        cache_enabled=True,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
    )
    forwarded = _drive_session(config, n_turns=4, session_id="quiesce-cfg-1")
    first = _first_matured_turn(forwarded)
    assert first == 2, (
        f"expected the Read to mature at turn 2 with quiesce_turns=2, "
        f"but first matured at turn {first} (handler ignored the configured value)"
    )


def test_read_maturation_knobs_from_env(monkeypatch):
    """Operators must be able to tune maturation via env vars (the pilot
    playbook says 'pick quiesce_turns')."""
    from headroom.proxy.server import _MULTI_WORKER_CONFIG_ENV, _proxy_config_from_env

    # _proxy_config_from_env short-circuits on a prebuilt multi-worker JSON
    # config and ignores the HEADROOM_* vars entirely. Clear it so this test
    # actually exercises the env-var parsing path it claims to (and isn't
    # poisoned by a leaked HEADROOM_PROXY_CONFIG_JSON from another test).
    monkeypatch.delenv(_MULTI_WORKER_CONFIG_ENV, raising=False)

    monkeypatch.setenv("HEADROOM_READ_MATURATION", "1")
    monkeypatch.setenv("HEADROOM_READ_MATURATION_QUIESCE_TURNS", "3")
    monkeypatch.setenv("HEADROOM_READ_MATURATION_MAX_HOLD_TURNS", "10")
    monkeypatch.setenv("HEADROOM_READ_MATURATION_MIN_SIZE_BYTES", "4096")

    cfg = _proxy_config_from_env()

    assert cfg.read_maturation is True
    assert cfg.read_maturation_quiesce_turns == 3
    assert cfg.read_maturation_max_hold_turns == 10
    assert cfg.read_maturation_min_size_bytes == 4096

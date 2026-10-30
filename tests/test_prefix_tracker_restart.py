"""A graceful proxy restart must not cost warm conversations their prompt cache.

Production, 2026-10-19: five graceful restarts in one day (app relaunches for
updates) each made the next request of every active Claude Code session read
only the ~25k-token tools + system prefix and re-write the rest, 247k-432k
tokens at a time, about a third of that day's cache writes. After a restart
the proxy has no ``PrefixCacheTracker`` for the conversation, so it compresses
the whole history afresh and the early messages come out as different bytes
than the prefix it forwarded before the restart.

``HEADROOM_PREFIX_STATE_FILE`` (opt-in) saves the tracker store on graceful
shutdown and restores it at startup, so the first request after a restart
replays the exact prefix the provider cached.
"""

from __future__ import annotations

import contextvars
import errno
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")

import httpx
from fastapi.testclient import TestClient

from headroom.cache import prefix_tracker
from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.cache.prefix_tracker import PREFIX_STATE_FILE_ENV, SessionTrackerStore
from headroom.ccr.tool_injection import CCRToolInjector
from headroom.config import TransformResult
from headroom.proxy.server import ProxyConfig, create_app
from headroom.transforms.pipeline import TransformPipeline
from tests.test_anthropic_fork_lineage_cache import (
    _drive,
    _main,
    _PromptCache,
    _send,
    _strip,
    _summary_fork,
)


def _wire(messages: list[dict[str, Any]]) -> str:
    """The bytes the provider keys its prefix cache on (breakpoints excluded)."""
    return json.dumps(_strip(messages))


def _ccr_hashes(messages: list[dict[str, Any]]) -> set[str]:
    """Retrieval references the model sees, found the way the proxy finds them."""
    return set(CCRToolInjector().scan_for_markers(messages))


def _with_fake_kompress(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deterministic stand-in for the Kompress ML stage (off in this suite).

    A cold start forwards the fast pass, which skips Kompress, while warm
    turns compress their new messages with it. So after a restart without
    state, the cold fast pass re-emits every message Kompressed after the
    first turn in a different form, which re-writes the cache from there."""
    real_apply = TransformPipeline.apply

    def apply(self, messages, model, **kwargs):  # noqa: ANN001, ANN003, ANN202
        skip = kwargs.get("skip_kompress", False)
        frozen = kwargs.get("frozen_message_count", 0) or 0
        result = real_apply(self, messages, model, **kwargs)
        if skip:
            return result
        out = []
        for i, message in enumerate(result.messages):
            content = message.get("content")
            if i >= frozen and isinstance(content, list):
                blocks = []
                for block in content:
                    text = block.get("content") if isinstance(block, dict) else None
                    if block.get("type") == "tool_result" and isinstance(text, str):
                        if not text.endswith(" [kompressed]"):
                            block = {**block, "content": text[: len(text) // 2] + " [kompressed]"}
                    blocks.append(block)
                message = {**message, "content": blocks}
            out.append(message)
        result.messages = out
        return result

    monkeypatch.setattr(TransformPipeline, "apply", apply)


def _read_turn(i: int) -> list[dict[str, Any]]:
    tid = f"toolu_r{i:02d}"
    body = "\n".join(f"{n:5d}  line {n} of file {i}: some code here" for n in range(300))
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tid, "name": "Read", "input": {"file_path": f"/f{i}.py"}}
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": body}]},
    ]


def _reads(turns: int) -> list[dict[str, Any]]:
    """One Read per turn; Claude Code's breakpoint rides the newest block."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "Review the change."}]}
    ]
    for i in range(turns):
        messages += _read_turn(i)
    messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return messages


# (history builder, turns before the restart, proxy config, fake Kompress)
_SCENARIOS = {
    # Turns 1-2, restart, turn 3: Kompressed messages come back fast-passed.
    "kompress": (_main, 2, {}, True),
    # Reads held while active then cached verbatim stay verbatim; a fresh
    # process with an empty CompressionCache freezes nothing and matures them.
    "read-maturation": (
        _reads,
        5,
        {
            "read_maturation": True,
            "read_maturation_quiesce_turns": 2,
            "read_maturation_min_size_bytes": 100,
        },
        False,
    ),
}


def _restart_between_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str, **overrides: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, dict[str, str]]:
    """Some turns in one proxy, the next turn in a fresh one, one warm
    provider cache. Returns (last forwarded before, first forwarded after,
    its read hit, the original behind each retrieval reference forwarded
    before the restart)."""
    build, turns, config, kompress = _SCENARIOS[scenario]
    config = {**config, **overrides}
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    # As the desktop app runs it: a cold start forwards the fast pass.
    monkeypatch.setenv("HEADROOM_BACKGROUND_COMPRESSION", "1")
    monkeypatch.setenv("HEADROOM_BACKGROUND_COMPRESSION_MIN_TOKENS", "1")
    monkeypatch.chdir(tmp_path)
    if kompress:
        _with_fake_kompress(monkeypatch)
    cache = _PromptCache()
    reset_compression_store()
    before = _drive([build(n) for n in range(1, turns + 1)], cache, **config)
    store = get_compression_store()
    originals = {h: store.retrieve(h).original_content for h in _ccr_hashes(before[-1][0])}  # type: ignore[union-attr]
    reset_compression_store()  # nothing in process memory survives a restart
    (after,) = _drive([build(turns + 1)], cache, **config)
    return before[-1][0], after[0], after[1], originals


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_first_request_after_a_restart_replays_the_cached_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> None:
    state = tmp_path / "prefix-state.json"
    monkeypatch.setenv(PREFIX_STATE_FILE_ENV, str(state))

    previous, current, hit, _ = _restart_between_turns(monkeypatch, tmp_path, scenario)

    assert _wire(current[: len(previous)]) == _wire(previous)
    assert hit >= len(previous) - 1, f"read only {hit + 1}/{len(previous)} cached messages"


def test_restored_retrieval_references_resolve_to_their_originals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review of #4299: matching bytes are not enough. With the in-process CCR
    backend a restart emptied the store, so a matured Read the restored
    tracker kept replaying pointed at a hash that no longer resolved."""
    monkeypatch.setenv(PREFIX_STATE_FILE_ENV, str(tmp_path / "prefix-state.json"))

    # Each Read matures as it arrives, before the provider caches it, so
    # references exist before the restart. A Read that quiesces later is
    # already cached by then and stays verbatim.
    previous, current, _, originals = _restart_between_turns(
        monkeypatch, tmp_path, "read-maturation", read_maturation_quiesce_turns=0
    )

    emitted = _ccr_hashes(current)
    assert emitted & originals.keys(), "no pre-restart reference was replayed"
    store = get_compression_store()
    for h in emitted:
        entry = store.retrieve(h)
        assert entry is not None, f"hash={h} does not resolve after the restart"
        if h in originals:
            assert entry.original_content == originals[h]


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_without_the_state_file_a_restart_rewrites_the_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> None:
    """Default behavior is unchanged, and it is the bug the opt-in fixes, so
    the test above is not vacuous."""
    monkeypatch.delenv(PREFIX_STATE_FILE_ENV, raising=False)

    previous, current, hit, _ = _restart_between_turns(monkeypatch, tmp_path, scenario)

    assert _wire(current[: len(previous)]) != _wire(previous)
    assert hit < len(previous) - 1
    assert list(tmp_path.iterdir()) == [], "nothing may be written without the opt-in"


def _two_conversations() -> SessionTrackerStore:
    store = SessionTrackerStore()
    contextvars.Context().run(_send, store, _strip(_main(3)))
    contextvars.Context().run(_send, store, [{"role": "user", "content": "unrelated"}])
    return store


def test_restored_lineages_resume_and_new_ones_get_fresh_keys(tmp_path: Path) -> None:
    """Store level: a fork answered last still leaves the main turn resumable,
    and a new conversation never lands on a restored lineage's key."""
    store = _two_conversations()
    base = _strip(_main(3))
    contextvars.Context().run(_send, store, _strip(_summary_fork(3)))
    path = str(tmp_path / "state.json")
    assert store.save_state(path) == 2

    restored = SessionTrackerStore()
    assert restored.load_state(path) == 2
    main = contextvars.Context().run(
        restored.resolve_tracker, "sid", "anthropic", messages=_strip(_main(4))
    )
    new = contextvars.Context().run(
        restored.resolve_tracker, "sid", "anthropic", messages=[{"role": "user", "content": "new"}]
    )

    assert main.get_frozen_message_count() == len(base)
    assert main.get_last_original_messages() == base
    assert restored.active_sessions == 3
    assert new is not main and new.get_frozen_message_count() == 0


def test_expired_trackers_are_not_restored(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _two_conversations().save_state(str(path))
    data = json.loads(path.read_text())
    data["trackers"][0]["meta"]["last_activity"] -= 3600  # past the 600s retention
    path.write_text(json.dumps(data))

    assert SessionTrackerStore().load_state(str(path)) == 1


def _bad_ref(path: Path) -> None:
    data = json.loads(path.read_text())
    data["trackers"][0]["meta"]["state"][3] = [-1]
    path.write_text(json.dumps(data))


def _bad_maturation(path: Path) -> None:
    data = json.loads(path.read_text())
    data["trackers"][-1]["meta"]["maturation"] = {"matured": [1], "booked": [], "deltas": {}}
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda p: p.write_text('{"version": 1, "trackers": [{"meta"'),
        lambda p: p.write_text('{"version": 99, "trackers": []}'),
        lambda p: p.write_text("[1, 2]"),
        _bad_ref,
        _bad_maturation,
    ],
    ids=["truncated", "version", "shape", "bad-ref", "bad-maturation"],
)
def test_a_bad_state_file_starts_empty(tmp_path: Path, caplog, corrupt) -> None:  # noqa: ANN001
    path = tmp_path / "state.json"
    _two_conversations().save_state(str(path))
    corrupt(path)

    store = SessionTrackerStore()
    assert store.load_state(str(path)) == 0
    assert store.active_sessions == 0
    assert "ignoring prefix state" in caplog.text


def test_state_file_is_private_and_removed_once_loaded(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("stale")
    path.chmod(0o644)

    assert _two_conversations().save_state(str(path)) == 2
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert SessionTrackerStore().load_state(str(path)) == 2
    assert not path.exists()
    assert SessionTrackerStore().load_state(str(path)) == 0  # missing file: cold start


def test_save_keeps_the_newest_trackers_within_the_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = _two_conversations()
    newest = max(store._trackers.values(), key=lambda t: t._last_activity)
    monkeypatch.setattr(prefix_tracker, "_PREFIX_STATE_MAX_TRACKERS", 1)
    path = str(tmp_path / "state.json")
    assert store.save_state(path) == 1

    restored = SessionTrackerStore()
    restored.load_state(path)
    (kept,) = restored._trackers.values()
    assert kept.get_last_original_messages() == newest.get_last_original_messages()

    monkeypatch.setattr(prefix_tracker, "_PREFIX_STATE_MAX_BYTES", 10)
    assert store.save_state(path) == 0


def test_stateless_mode_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    path = tmp_path / "state.json"
    assert _two_conversations().save_state(str(path)) == 0
    assert not path.exists()


def test_a_stateless_proxy_leaves_a_saved_snapshot_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Loading deletes the file, so a stateless start must not load it: the
    next normal start resumes those conversations."""
    path = tmp_path / "state.json"
    assert _two_conversations().save_state(str(path)) == 2
    saved = path.read_text()
    monkeypatch.setenv(PREFIX_STATE_FILE_ENV, str(path))
    monkeypatch.setenv("HEADROOM_STATELESS", "1")

    _drive([])  # one stateless proxy start and graceful shutdown

    assert path.read_text() == saved
    monkeypatch.delenv("HEADROOM_STATELESS")
    assert SessionTrackerStore().load_state(str(path)) == 2


def test_a_failed_save_keeps_the_previous_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "state.json"
    assert _two_conversations().save_state(str(path)) == 2
    saved = path.read_text()

    def disk_full(*args: Any, **kwargs: Any) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("headroom.fsutil.write_text", disk_full)
    assert _two_conversations().save_state(str(path)) == 0

    assert path.read_text() == saved


def test_saved_ccr_entries_resolve_after_a_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Store level, compressor markers in the replayed prefix: both hash
    formats resolve to the same original with the same expiry after the
    in-process CCR store is emptied, and an entry that expired does not."""
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    reset_compression_store()
    ccr = get_compression_store()
    rows = ccr.store(json.dumps(list(range(400))), "[0, 1]", ttl=900)
    blob = ccr.store("x" * 5000, "", ttl=900, explicit_hash="abcdef123456")
    gone = ccr.store("expired original", "", ttl=900)
    ccr._backend.get(gone).created_at -= 1000  # type: ignore[union-attr]
    expires = {h: ccr.get_entry_status(h)["expires_at"] for h in (rows, blob)}
    text = (
        f"[400 items compressed to 2. Retrieve more: hash={rows}] "
        f"<<ccr:{blob},base64,4.9KB>> [stale. Retrieve original: hash={gone}]"
    )
    store = SessionTrackerStore()
    contextvars.Context().run(_send, store, [{"role": "user", "content": text}])
    path = str(tmp_path / "state.json")
    assert store.save_state(path) == 1
    reset_compression_store()

    assert SessionTrackerStore().load_state(path) == 1

    ccr = get_compression_store()
    assert ccr.retrieve(rows).original_content == json.dumps(list(range(400)))  # type: ignore[union-attr]
    assert ccr.retrieve(blob).original_content == "x" * 5000  # type: ignore[union-attr]
    assert ccr.retrieve(gone) is None
    for h, at in expires.items():
        assert abs(ccr.get_entry_status(h)["expires_at"] - at) < 2
    reset_compression_store()


def _chat_turns(n: int) -> list[dict[str, Any]]:
    """An OpenAI chat agent: one big tool result, then ``n - 1`` follow-ups."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "run the build and show me the log"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "build log line OK\n" * 200},
    ]
    for i in range(1, n):
        messages += [
            {"role": "assistant", "content": f"answer {i}"},
            {"role": "user", "content": "and now?"},
        ]
    return messages


def _relevance_pipeline(**kwargs: Any) -> TransformResult:
    """Query-dependent compression, like SmartCrusher's relevance scoring: the
    same tool result compresses differently under a different user query."""
    frozen = kwargs.get("frozen_message_count") or 0
    query = next(m["content"] for m in reversed(kwargs["messages"]) if m["role"] == "user")
    messages = [
        {**m, "content": f"[log: kept lines relevant to {query!r}]"}
        if i >= frozen and m.get("role") == "tool"
        else m
        for i, m in enumerate(kwargs["messages"])
    ]
    return TransformResult(
        messages=messages, tokens_before=2000, tokens_after=40, transforms_applied=["relevance"]
    )


def _drive_openai_chat(requests: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """Each history through the real OpenAI chat handler in token mode; one
    app per call, so a second call is a proxy restart."""
    app = create_app(
        ProxyConfig(
            optimize=True,
            mode="token",
            disable_kompress=True,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )
    forwarded: list[list[dict[str, Any]]] = []
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.openai_pipeline.apply = _relevance_pipeline

        async def _upstream(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN003
            forwarded.append(json.loads(json.dumps(body["messages"])))
            # Automatic prefix caching reports nothing cached on a
            # conversation's first request, so there is no confirmed floor.
            usage = {"prompt_tokens": 600, "completion_tokens": 3, "total_tokens": 603}
            usage["prompt_tokens_details"] = {"cached_tokens": 0}
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_x",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            )

        proxy._retry_request = _upstream
        for messages in requests:
            response = client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer test-key"},
                json={"model": "gpt-4o-mini", "messages": messages},
            )
            assert response.status_code == 200, response.text[:300]
    return forwarded


def test_openai_token_mode_replays_its_prefix_after_a_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review of #4299: only the Anthropic handler seeded its CompressionCache
    from a restored tracker. The OpenAI token path froze nothing, recompressed
    the tool result under the new query, and with no provider-confirmed floor
    the overlay declined the larger previous form, so the prefix changed."""
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv(PREFIX_STATE_FILE_ENV, str(tmp_path / "prefix-state.json"))
    reset_compression_store()

    (previous,) = _drive_openai_chat([_chat_turns(1)])
    (current,) = _drive_openai_chat([_chat_turns(2)])

    assert previous[2]["content"] != _chat_turns(1)[2]["content"], "nothing was compressed"
    assert current[: len(previous)] == previous


def test_read_maturation_state_survives_a_restart(tmp_path: Path) -> None:
    """Matured Reads stay booked once: without this a restart books their
    removal again on every later request of the conversation."""
    from headroom.config import ReadMaturationConfig
    from headroom.transforms.read_maturation import MaturedRead, ReadMaturationManager

    store = _two_conversations()
    manager = ReadMaturationManager(ReadMaturationConfig(enabled=True))
    manager._matured["toolu_r00"] = MaturedRead(marker="[Read of /f0.py ...]", ccr_hash="abc")
    manager._booked.add("toolu_r00")
    manager._replay_token_deltas["toolu_r00"] = 640
    newest = max(store._trackers.values(), key=lambda t: t._last_activity)
    newest.read_maturation_manager = manager
    path = str(tmp_path / "state.json")
    store.save_state(path)

    restored = SessionTrackerStore()
    restored.load_state(path)
    states = [t.restored_maturation_state for t in restored._trackers.values()]
    assert all(t.restored_from_disk for t in restored._trackers.values())
    (state,) = [s for s in states if s is not None]
    adopted = ReadMaturationManager(ReadMaturationConfig(enabled=True))
    adopted.restore_state(state)

    assert adopted._matured == manager._matured
    assert adopted._booked == manager._booked
    assert adopted._replay_token_deltas == manager._replay_token_deltas


def test_compress_sidecar_session_replays_its_prefix_after_a_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same seed on the session-aware /v1/compress path (sidecar gateways),
    which replays with the size bound alone (no provider-confirmed floor)."""
    from tests.test_compress_session_mode import _compress, _make_client

    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv(PREFIX_STATE_FILE_ENV, str(tmp_path / "prefix-state.json"))
    reset_compression_store()
    forwarded = []
    for history in (_chat_turns(1), _chat_turns(2)):  # one proxy each: a restart between
        with _make_client() as client:
            client.app.state.proxy._no_ccr_pipeline = lambda: SimpleNamespace(
                apply=_relevance_pipeline
            )
            forwarded.append(_compress(client, history, session_id="conv-r")["messages"])
    previous, current = forwarded

    assert previous[2]["content"] != _chat_turns(1)[2]["content"], "nothing was compressed"
    assert current[: len(previous)] == previous

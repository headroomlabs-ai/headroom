from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anyio
import pytest

from headroom.memory.storage_router import ProjectResolver, RequestContext
from headroom.providers.codex.project_context import CodexProjectContextResolver
from tests.test_openai_codex_routing import (
    _build_request,
    _DummyTokenizer,
    _ResponseStub,
)
from tests.test_openai_codex_routing import (
    _DummyOpenAIHandler as _HTTPHandler,
)
from tests.test_openai_codex_ws_lifecycle import (
    _DummyOpenAIHandler as _WSHandler,
)
from tests.test_openai_codex_ws_lifecycle import (
    _FakeUpstream,
    _FakeWebSocket,
    _make_fake_websockets_module,
    _MemoryWsHandler,
)


def _seed_thread(codex_home: Path, thread_id: str, rollout: Path) -> None:
    db = codex_home / "state_5.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO threads (id, rollout_path) VALUES (?, ?)",
            (thread_id, str(rollout)),
        )


def _seed_rollout(path: Path, turn_id: str, cwd: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "turn_context",
                "payload": {"turn_id": turn_id, "cwd": str(cwd)},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _append_turn(path: Path, turn_id: str, cwd: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "turn_context",
                    "payload": {"turn_id": turn_id, "cwd": str(cwd)},
                }
            )
            + "\n"
        )


def test_codex_http_projects_are_isolated_and_ws_mismatch_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    project_a = tmp_path / "a" / "shared"
    project_b = tmp_path / "b" / "shared"
    project_a.mkdir(parents=True)
    project_b.mkdir(parents=True)
    rollout_a = codex_home / "rollout-a.jsonl"
    rollout_b = codex_home / "rollout-b.jsonl"
    _seed_rollout(rollout_a, "turn-a", project_a)
    _seed_rollout(rollout_b, "turn-b", project_b)
    _seed_thread(codex_home, "thread-a", rollout_a)
    _seed_thread(codex_home, "thread-b", rollout_b)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    resolver = CodexProjectContextResolver()
    resolved_a = resolver.resolve(
        headers={
            "x-codex-turn-metadata": json.dumps({"thread_id": "thread-a", "turn_id": "turn-a"})
        },
        body={},
    )
    resolved_b = resolver.resolve(
        headers={},
        body={"client_metadata": {"thread_id": "thread-b", "turn_id": "turn-b"}},
    )

    assert resolved_a.cwd == project_a.resolve()
    assert resolved_b.cwd == project_b.resolve()
    keys = {
        ProjectResolver().resolve(
            RequestContext(
                headers={},
                system_prompt="",
                base_user_id="test",
                project_root_override=str(resolved.cwd),
            )
        )[0]
        for resolved in (resolved_a, resolved_b)
    }
    assert len(keys) == 2

    mismatch = resolver.resolve(
        headers={},
        body={
            "type": "response.create",
            "response": {
                "client_metadata": {
                    "thread_id": "thread-b",
                    "turn_id": "turn-b",
                }
            },
        },
        pinned_cwd=resolved_a.cwd,
    )
    assert mismatch.cwd is None
    assert mismatch.reason == "project_mismatch"

    _append_turn(rollout_a, "turn-a", project_b)
    changed_rollout = resolver.resolve(
        headers={},
        body={"client_metadata": {"thread_id": "thread-a", "turn_id": "turn-a"}},
    )
    assert changed_rollout.cwd is None
    assert changed_rollout.reason == "turn_ambiguous"


def test_cached_rollout_recanonicalizes_symlink_before_pinned_check(
    monkeypatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    project_link = tmp_path / "current-project"
    project_link.symlink_to(project_a, target_is_directory=True)
    rollout = codex_home / "rollout.jsonl"
    _seed_rollout(rollout, "turn", project_link)
    _seed_thread(codex_home, "thread", rollout)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    resolver = CodexProjectContextResolver()
    body = {"client_metadata": {"thread_id": "thread", "turn_id": "turn"}}

    first = resolver.resolve(headers={}, body=body)
    project_link.unlink()
    project_link.symlink_to(project_b, target_is_directory=True)
    second = resolver.resolve(headers={}, body=body, pinned_cwd=first.cwd)

    assert first.cwd == project_a.resolve()
    assert second.cwd is None
    assert second.reason == "project_mismatch"


def test_turn_specific_resume_and_fork_use_exact_context(monkeypatch, tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    rollout = codex_home / "rollout.jsonl"
    _seed_rollout(rollout, "turn-a", project_a)
    _append_turn(rollout, "turn-b", project_b)
    _seed_thread(codex_home, "resumed-thread", rollout)
    _seed_thread(codex_home, "forked-thread", rollout)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    resolver = CodexProjectContextResolver()
    resumed = resolver.resolve(
        headers={},
        body={"client_metadata": {"thread_id": "resumed-thread", "turn_id": "turn-b"}},
    )
    forked = resolver.resolve(
        headers={},
        body={"client_metadata": {"thread_id": "forked-thread", "turn_id": "turn-a"}},
    )

    assert resumed.cwd == project_b.resolve()
    assert forked.cwd == project_a.resolve()


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ("missing_thread", "thread_missing"),
        ("missing_context", "turn_context_missing"),
        ("truncated", "rollout_truncated"),
        ("unsupported", "state_schema_unsupported"),
        ("stale", "rollout_stale"),
    ],
)
def test_state_failures_skip_with_structured_reason(
    monkeypatch,
    tmp_path: Path,
    setup: str,
    reason: str,
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    rollout = codex_home / "rollout.jsonl"
    project = tmp_path / "project"
    project.mkdir()
    if setup == "unsupported":
        with sqlite3.connect(codex_home / "state_5.sqlite") as connection:
            connection.execute("CREATE TABLE other (id TEXT)")
    else:
        _seed_rollout(rollout, "other-turn", project)
        if setup == "truncated":
            with rollout.open("a", encoding="utf-8") as handle:
                handle.write('{"type":"turn_context"')
        if setup == "stale":
            rollout.unlink()
        _seed_thread(codex_home, "other-thread" if setup == "missing_thread" else "thread", rollout)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = CodexProjectContextResolver().resolve(
        headers={},
        body={"client_metadata": {"thread_id": "thread", "turn_id": "turn"}},
    )

    assert result.cwd is None
    assert result.reason == reason


def test_truncated_tail_after_exact_context_skips_safely(monkeypatch, tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    rollout = codex_home / "rollout.jsonl"
    _seed_rollout(rollout, "turn", project)
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write('{"type":"turn_context"')
    _seed_thread(codex_home, "thread", rollout)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = CodexProjectContextResolver().resolve(
        headers={},
        body={"client_metadata": {"thread_id": "thread", "turn_id": "turn"}},
    )

    assert result.cwd is None
    assert result.reason == "rollout_truncated"


def test_conflicting_state_rows_are_ambiguous_but_identical_rows_dedupe(
    monkeypatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex"
    sqlite_home = codex_home / "sqlite"
    sqlite_home.mkdir(parents=True)
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    rollout = codex_home / "rollout.jsonl"
    conflicting_rollout = codex_home / "conflicting.jsonl"
    _seed_rollout(rollout, "turn", project)
    _seed_rollout(conflicting_rollout, "turn", other)
    _seed_thread(sqlite_home, "thread", rollout)
    _seed_thread(codex_home, "thread", rollout)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    resolver = CodexProjectContextResolver()
    body = {"client_metadata": {"thread_id": "thread", "turn_id": "turn"}}

    assert resolver.resolve(headers={}, body=body).cwd == project.resolve()

    lock = sqlite3.connect(sqlite_home / "state_5.sqlite")
    lock.execute("BEGIN EXCLUSIVE")
    try:
        locked = resolver.resolve(headers={}, body=body)
    finally:
        lock.rollback()
        lock.close()
    assert locked.cwd is None
    assert locked.reason == "state_locked"

    with sqlite3.connect(sqlite_home / "state_5.sqlite") as connection:
        connection.execute(
            "UPDATE threads SET rollout_path = ? WHERE id = ?",
            (str(conflicting_rollout), "thread"),
        )
    ambiguous = resolver.resolve(headers={}, body=body)
    assert ambiguous.cwd is None
    assert ambiguous.reason == "state_ambiguous"


def test_explicit_overrides_precede_state_and_body_cwd(monkeypatch, tmp_path: Path) -> None:
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing"))
    resolver = CodexProjectContextResolver()
    body = {"client_metadata": {"cwd": str(project_b)}}

    project_id = resolver.resolve(
        headers={"x-headroom-project-id": "chosen"},
        body=body,
    )
    explicit_cwd = resolver.resolve(
        headers={"x-headroom-cwd": str(project_a)},
        body=body,
        project_root_override=str(project_b),
    )

    assert project_id.source == "x-headroom-project-id"
    assert project_id.project_key.startswith("chosen-")
    assert explicit_cwd.cwd == project_a.resolve()
    assert explicit_cwd.source == "x-headroom-cwd"


def test_configured_sqlite_home_precedes_environment(monkeypatch, tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    configured = tmp_path / "configured"
    environment = tmp_path / "environment"
    codex_home.mkdir()
    configured.mkdir()
    environment.mkdir()
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    rollout_a = configured / "rollout.jsonl"
    rollout_b = environment / "rollout.jsonl"
    _seed_rollout(rollout_a, "turn", project_a)
    _seed_rollout(rollout_b, "turn", project_b)
    _seed_thread(configured, "thread", rollout_a)
    _seed_thread(environment, "thread", rollout_b)
    (codex_home / "config.toml").write_text(
        f"sqlite_home = {json.dumps(str(configured))}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("CODEX_SQLITE_HOME", str(environment))

    result = CodexProjectContextResolver().resolve(
        headers={},
        body={"client_metadata": {"thread_id": "thread", "turn_id": "turn"}},
    )

    assert result.cwd == project_a.resolve()


def test_body_cwd_is_bounded_fallback_and_must_be_canonical(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    resolver = CodexProjectContextResolver(tmp_path / "missing")

    result = resolver.resolve(
        headers={},
        body={"response": {"client_metadata": {"cwd": str(project / ".." / "project")}}},
    )
    invalid = resolver.resolve(headers={}, body={"cwd": "relative/project"})

    assert result.cwd == project.resolve()
    assert result.source == "responses-body-cwd"
    assert invalid.cwd is None
    assert invalid.reason == "body_cwd_invalid"


def test_sqlite_lock_is_bounded_and_visible(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    rollout = codex_home / "rollout.jsonl"
    project = tmp_path / "project"
    project.mkdir()
    _seed_rollout(rollout, "turn", project)
    _seed_thread(codex_home, "thread", rollout)
    lock = sqlite3.connect(codex_home / "state_5.sqlite")
    lock.execute("BEGIN EXCLUSIVE")
    try:
        result = CodexProjectContextResolver(codex_home).resolve(
            headers={},
            body={"client_metadata": {"thread_id": "thread", "turn_id": "turn"}},
        )
    finally:
        lock.rollback()
        lock.close()

    assert result.cwd is None
    assert result.reason == "state_locked"


def test_http_resolution_does_not_mutate_body_or_forward_internal_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    project_a = tmp_path / "a" / "shared"
    project_b = tmp_path / "b" / "shared"
    project_a.mkdir(parents=True)
    project_b.mkdir(parents=True)
    rollout_a = codex_home / "rollout-a.jsonl"
    rollout_b = codex_home / "rollout-b.jsonl"
    _seed_rollout(rollout_a, "turn-a", project_a)
    _seed_rollout(rollout_b, "turn-b", project_b)
    _seed_thread(codex_home, "thread-a", rollout_a)
    _seed_thread(codex_home, "thread-b", rollout_b)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    body = {
        "model": "gpt-5.4",
        "input": "hello",
        "client_metadata": {"thread_id": "thread-a", "turn_id": "turn-a"},
    }
    turn_header = json.dumps({"thread_id": "thread-a", "turn_id": "turn-a"})
    request = _build_request(
        body,
        {
            "Authorization": "Bearer test",
            "X-Client": "codex",
            "X-Codex-Turn-Metadata": turn_header,
        },
    )

    class Handler(_HTTPHandler):
        def __init__(self) -> None:
            super().__init__()
            self.config.memory_storage_mode = "project"
            self.retry_kwargs = {}
            self.observed = []
            self.projects = []

        async def _retry_request(self, method, url, headers, request_body, **kwargs):
            self.captured_request = (method, url, headers, request_body)
            self.retry_kwargs = kwargs
            return _ResponseStub()

        async def _observe_openai_responses_traffic(self, request_body, *, request_id):
            self.observed.append(request_body)

        async def _record_request_outcome(self, _outcome):
            from headroom.proxy.project_context import get_current_project

            self.projects.append(get_current_project())

    handler = Handler()
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, upstream_headers, upstream_body = handler.captured_request
    assert upstream_body == body
    assert handler.retry_kwargs["original_body_bytes"] == json.dumps(body).encode()
    assert all(key.lower() != "x-codex-turn-metadata" for key in upstream_headers)
    assert all(not key.lower().startswith("x-headroom-") for key in upstream_headers)
    assert handler.observed == []

    body_b = {
        "model": "gpt-5.4",
        "input": "hello B",
        "client_metadata": {"thread_id": "thread-b", "turn_id": "turn-b"},
    }
    handler_b = Handler()
    response_b = anyio.run(
        handler_b.handle_openai_responses,
        _build_request(
            body_b,
            {
                "Authorization": "Bearer test",
                "X-Client": "codex",
                "X-Codex-Turn-Metadata": json.dumps({"thread_id": "thread-b", "turn_id": "turn-b"}),
            },
        ),
    )
    assert response_b.status_code == 200
    assert handler_b.captured_request is not None
    assert handler_b.captured_request[3] == body_b
    assert handler_b.observed == []
    assert handler.projects and handler_b.projects
    assert handler.projects[0] != handler_b.projects[0]

    unresolved_handler = Handler()
    unresolved_request = _build_request(
        {"model": "gpt-5.4", "input": "unresolved"},
        {
            "Authorization": "Bearer test",
            "User-Agent": "codex_cli_rs",
            "X-Client": "codex",
        },
    )
    assert (
        anyio.run(unresolved_handler.handle_openai_responses, unresolved_request).status_code == 200
    )
    assert unresolved_handler.observed == []


@pytest.mark.asyncio
async def test_http_project_resolution_does_not_block_event_loop(monkeypatch) -> None:
    resolver_entered = threading.Event()
    resolver_release = threading.Event()

    def blocked_resolve(self, **kwargs):
        resolver_entered.set()
        assert resolver_release.wait(timeout=3)
        raise RuntimeError("blocked resolver failed")

    monkeypatch.setattr(CodexProjectContextResolver, "resolve", blocked_resolve)
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer test", "X-Client": "codex"},
    )
    handler = _HTTPHandler()
    progressed_before_release = False

    async def unrelated_work() -> None:
        nonlocal progressed_before_release
        while not resolver_entered.is_set():
            await asyncio.sleep(0)
        progressed_before_release = not resolver_release.is_set()
        resolver_release.set()

    release_timer = threading.Timer(2, resolver_release.set)
    release_timer.start()
    try:
        response, _ = await asyncio.gather(
            handler.handle_openai_responses(request),
            unrelated_work(),
        )
    finally:
        resolver_release.set()
        release_timer.cancel()
        release_timer.join()

    assert resolver_entered.is_set()
    assert progressed_before_release
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("handshake_metadata", [False, True, "transport-only", "conflicting"])
async def test_ws_mismatch_skips_project_memory_but_forwards_main_traffic(
    monkeypatch, tmp_path: Path, handshake_metadata: bool | str
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    rollout_a = codex_home / "rollout-a.jsonl"
    rollout_b = codex_home / "rollout-b.jsonl"
    _seed_rollout(rollout_a, "turn-a", project_a)
    _seed_rollout(rollout_b, "turn-b", project_b)
    _seed_thread(codex_home, "thread-a", rollout_a)
    _seed_thread(codex_home, "thread-b", rollout_b)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    first = json.dumps(
        {
            "type": "response.create",
            "response": {
                "input": "first",
                "client_metadata": {"thread_id": "thread-a", "turn_id": "turn-a"},
            },
        }
    )
    mismatch = json.dumps(
        {
            "type": "response.create",
            "response": {
                "input": "mismatch",
                "client_metadata": {"thread_id": "thread-b", "turn_id": "turn-b"},
            },
        }
    )

    class Memory:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                inject_context=True,
                inject_tools=False,
                project_root_override="",
            )
            self.projects = []

        async def search_and_format_context(self, _user_id, _messages, **kwargs):
            self.projects.append(kwargs["request_context"].project_root_override)
            return None

        def compute_memory_tool_definitions(self, _provider):
            return []

    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ],
        hold_after_events=True,
    )
    if handshake_metadata == "transport-only":
        first_body = json.loads(first)
        first_body["response"]["client_metadata"] = {"keep": "yes"}
        first = json.dumps(first_body)
    websocket = _FakeWebSocket(frames=[first, mismatch])
    if handshake_metadata:
        header_project = "b" if handshake_metadata == "conflicting" else "a"
        websocket.headers["X-Codex-Turn-Metadata"] = json.dumps(
            {"thread_id": f"thread-{header_project}", "turn_id": f"turn-{header_project}"}
        )
    handler = _WSHandler()
    handler.config.memory_storage_mode = "project"
    memory = Memory()
    handler.memory_handler = memory

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await handler.handle_openai_responses_ws(websocket)

    assert memory.projects == [str(project_a.resolve())]
    assert len(upstream.sent) == 2
    assert json.loads(upstream.sent[1]) == json.loads(mismatch)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name", ["memory_search", "memory_save", "memory_update", "memory_delete"]
)
@pytest.mark.parametrize(
    "second_project",
    ["same", "mismatch", "unresolved", "missing", "overlap", "pipeline", "failed", "incomplete"],
)
@pytest.mark.parametrize("disable_injection", [False, True])
@pytest.mark.parametrize("user_agent", ["codex-cli/0.5", "python-client"])
async def test_ws_memory_dispatch_uses_current_turn_project(
    monkeypatch,
    tmp_path: Path,
    tool_name: str,
    second_project: str,
    disable_injection: bool,
    user_agent: str,
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    for name in ("a", "b"):
        project = tmp_path / name
        project.mkdir()
        rollout = codex_home / f"rollout-{name}.jsonl"
        _seed_rollout(rollout, f"turn-{name}", project)
        _seed_thread(codex_home, f"thread-{name}", rollout)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    def frame(project: str | None) -> str:
        response = {"model": "gpt-5.4", "input": "hello"}
        if project:
            response["client_metadata"] = {
                "thread_id": f"thread-{project}",
                "turn_id": f"turn-{project}",
            }
        return json.dumps({"type": "response.create", "response": response})

    first_completed = asyncio.Event()
    init_blocked = asyncio.Event()
    init_release = asyncio.Event()
    events: asyncio.Queue[str] = asyncio.Queue()
    function_call = {
        "type": "function_call",
        "id": "fc-1",
        "call_id": "call-1",
        "name": tool_name,
        "arguments": "{}",
    }

    class Upstream(_FakeUpstream):
        async def send(self, payload: str) -> None:
            await super().send(payload)
            if second_project == "pipeline" and len(self.sent) == 2:
                return
            if len(self.sent) == 3:
                init_release.set()
            response_id = (
                "r-2"
                if second_project == "pipeline" and len(self.sent) == 3
                else f"r-{len(self.sent)}"
            )
            await events.put(
                json.dumps({"type": "response.created", "response": {"id": response_id}})
            )
            failure_case = second_project in {"failed", "incomplete"}
            if (
                (len(self.sent) == 2 and not failure_case)
                or (failure_case and len(self.sent) == 1)
                or (second_project == "pipeline" and len(self.sent) == 3)
            ):
                for event_type in ("response.output_item.added", "response.output_item.done"):
                    await events.put(json.dumps({"type": event_type, "item": function_call}))
            terminal_type = (
                f"response.{second_project}"
                if failure_case and len(self.sent) == 1
                else "response.completed"
            )
            await events.put(json.dumps({"type": terminal_type, "response": {"id": response_id}}))

        async def _iter(self):
            while True:
                yield await events.get()

    class WebSocket(_FakeWebSocket):
        async def receive_text(self) -> str:
            if len(self._frames) == 1:
                await first_completed.wait()
                if disable_injection:
                    monkeypatch.setenv("HEADROOM_MEMORY_INJECTION_MODE", "disabled")
            if second_project == "pipeline" and not self._frames and not init_release.is_set():
                return frame("a")
            if (
                second_project == "overlap"
                and not disable_injection
                and not self._frames
                and not init_release.is_set()
            ):
                await init_blocked.wait()
                return frame(None)
            return await super().receive_text()

        async def send_text(self, text: str) -> None:
            await super().send_text(text)
            event = json.loads(text)
            if event.get("type") in {
                "response.completed",
                "response.failed",
                "response.incomplete",
            }:
                if event["response"]["id"] == "r-1":
                    first_completed.set()
                else:
                    self.trigger_disconnect()

    class Memory(_MemoryWsHandler):
        def __init__(self) -> None:
            super().__init__()
            self.config.project_root_override = ""
            self.executed = []

        async def _execute_memory_tool(
            self, name, args, user_id, provider, *, request_context=None
        ):
            self.executed.append((name, request_context))
            return '{"memories": []}'

        async def _ensure_initialized(self):
            if second_project == "overlap":
                init_blocked.set()
                await init_release.wait()
            await super()._ensure_initialized()

    second = {
        "same": "a",
        "mismatch": "b",
        "unresolved": "unknown",
        "missing": None,
        "overlap": "a",
        "pipeline": None,
        "failed": "a",
        "incomplete": "a",
    }[second_project]
    upstream = Upstream([])
    websocket = WebSocket(frames=[frame("a"), frame(second)], hold_after_initial=True)
    websocket.headers["X-Codex-Turn-Metadata"] = json.dumps(
        {"thread_id": "thread-a", "turn_id": "turn-a"}
    )
    websocket.headers["x-headroom-user-id"] = "user-1"
    websocket.headers["user-agent"] = user_agent
    handler = _WSHandler()
    handler.memory_handler = memory = Memory()
    observations = []

    async def observe(payload, **kwargs):
        observations.append(payload)

    handler._observe_openai_ws_response_create = observe
    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await asyncio.wait_for(handler.handle_openai_responses_ws(websocket), timeout=3)

    allowed = second_project == "same" and not disable_injection
    if allowed:
        assert len(memory.executed) == 1
        assert memory.executed[0][0] == tool_name
        assert memory.executed[0][1].project_root_override == str((tmp_path / "a").resolve())
        assert len(upstream.sent) == 3
    else:
        assert memory.executed == []
        expected_frames = (
            3
            if second_project == "pipeline"
            or (second_project == "overlap" and not disable_injection)
            else 2
        )
        assert len(upstream.sent) == expected_frames
        assert any(json.loads(text).get("item") == function_call for text in websocket.sent_text)
    assert observations == []


@pytest.mark.asyncio
async def test_concurrent_ws_connections_keep_same_basename_projects_isolated(
    monkeypatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    project_a = tmp_path / "a" / "shared"
    project_b = tmp_path / "b" / "shared"
    project_a.mkdir(parents=True)
    project_b.mkdir(parents=True)
    rollout_a = codex_home / "rollout-a.jsonl"
    rollout_b = codex_home / "rollout-b.jsonl"
    _seed_rollout(rollout_a, "turn-a", project_a)
    _seed_rollout(rollout_b, "turn-b", project_b)
    _seed_thread(codex_home, "thread-a", rollout_a)
    _seed_thread(codex_home, "thread-b", rollout_b)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    class Memory:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                inject_context=True,
                inject_tools=False,
                project_root_override="",
            )
            self.projects = []

        async def search_and_format_context(self, _user_id, _messages, **kwargs):
            self.projects.append(kwargs["request_context"].project_root_override)
            return None

        def compute_memory_tool_definitions(self, _provider):
            return []

    def frame(thread_id: str, turn_id: str, text: str) -> str:
        return json.dumps(
            {
                "type": "response.create",
                "response": {
                    "model": "gpt-5.4",
                    "input": text,
                    "client_metadata": {"thread_id": thread_id, "turn_id": turn_id},
                },
            }
        )

    completion = json.dumps(
        {
            "type": "response.completed",
            "response": {
                "id": "response",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        }
    )
    upstreams = [
        _FakeUpstream(
            [
                json.dumps({"type": "response.created", "response": {"id": "a"}}),
                completion,
            ]
        ),
        _FakeUpstream(
            [
                json.dumps({"type": "response.created", "response": {"id": "b"}}),
                completion,
            ]
        ),
    ]

    connect_kwargs = []

    async def connect(*_args, **kwargs):
        connect_kwargs.append(kwargs)
        return upstreams.pop(0)

    handlers = [_WSHandler(), _WSHandler()]
    memories = [Memory(), Memory()]
    for handler, memory in zip(handlers, memories, strict=True):
        handler.config.memory_storage_mode = "project"
        handler.memory_handler = memory
    websockets = SimpleNamespace(connect=connect)
    with patch.dict(sys.modules, {"websockets": websockets}):
        await asyncio.gather(
            handlers[0].handle_openai_responses_ws(
                _FakeWebSocket(
                    frames=[frame("thread-a", "turn-a", "A")],
                    headers={
                        "authorization": "Bearer test",
                        "x-codex-turn-metadata": json.dumps(
                            {"thread_id": "thread-a", "turn_id": "turn-a"}
                        ),
                    },
                )
            ),
            handlers[1].handle_openai_responses_ws(
                _FakeWebSocket(
                    frames=[frame("thread-b", "turn-b", "B")],
                    headers={
                        "authorization": "Bearer test",
                        "x-codex-turn-metadata": json.dumps(
                            {"thread_id": "thread-b", "turn_id": "turn-b"}
                        ),
                    },
                )
            ),
        )

    expected_keys = {
        CodexProjectContextResolver()
        .resolve(
            headers={},
            body={"client_metadata": {"thread_id": thread_id, "turn_id": turn_id}},
        )
        .project_key
        for thread_id, turn_id in (("thread-a", "turn-a"), ("thread-b", "turn-b"))
    }
    assert {memory.projects[0] for memory in memories} == {
        str(project_a.resolve()),
        str(project_b.resolve()),
    }
    assert {
        handler.metrics.recorded_requests[-1]["project"] for handler in handlers
    } == expected_keys
    assert all(
        "x-codex-turn-metadata" not in {key.lower() for key in kwargs["additional_headers"]}
        for kwargs in connect_kwargs
    )

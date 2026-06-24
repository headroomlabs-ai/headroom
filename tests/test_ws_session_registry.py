"""Unit tests for :class:`WebSocketSessionRegistry`.

These tests exercise the in-memory registry in isolation — no network,
no WS server. They pin down register/deregister semantics, ``snapshot``
shape, and the task-attachment accounting that feeds the
``active_relay_tasks`` gauge.
"""

from __future__ import annotations

import asyncio

import pytest

from headroom.proxy.ws_session_registry import (
    WebSocketSessionRegistry,
    WSSessionHandle,
)


def _make_handle(session_id: str = "sess-1") -> WSSessionHandle:
    return WSSessionHandle(
        session_id=session_id,
        request_id="req-1",
        client_addr="127.0.0.1:12345",
        upstream_url="wss://upstream/test",
    )


def test_register_and_deregister_round_trip():
    reg = WebSocketSessionRegistry()
    handle = _make_handle()

    reg.register(handle)
    assert reg.active_count() == 1
    assert reg.get("sess-1") is handle

    reg.deregister("sess-1", cause="response_completed")
    assert reg.active_count() == 0
    assert reg.get("sess-1") is None
    assert handle.termination_cause == "response_completed"


def test_deregister_unknown_session_is_idempotent():
    reg = WebSocketSessionRegistry()
    # Deregistering a never-registered session must not raise.
    reg.deregister("never-seen", cause="unknown")
    assert reg.active_count() == 0


def test_register_twice_same_session_does_not_double_count():
    reg = WebSocketSessionRegistry()
    handle = _make_handle()

    reg.register(handle)
    reg.register(handle)  # idempotent — overwrite, do not double-increment
    assert reg.active_count() == 1

    reg.deregister("sess-1", cause="client_disconnect")
    assert reg.active_count() == 0


def test_snapshot_shape_is_json_serializable():
    reg = WebSocketSessionRegistry()
    reg.register(_make_handle("a"))
    reg.register(_make_handle("b"))

    snapshot = reg.snapshot()
    assert len(snapshot) == 2
    ids = {entry["session_id"] for entry in snapshot}
    assert ids == {"a", "b"}
    sample = snapshot[0]
    for key in (
        "session_id",
        "request_id",
        "client_addr",
        "upstream_url",
        "age_seconds",
        "relay_task_count",
        "termination_cause",
    ):
        assert key in sample


@pytest.mark.asyncio
async def test_attach_tasks_merges_and_tracks_active_count():
    reg = WebSocketSessionRegistry()
    handle = _make_handle()
    reg.register(handle)

    async def _idle() -> None:
        await asyncio.sleep(5)

    t1 = asyncio.create_task(_idle(), name="codex-ws-test-1")
    t2 = asyncio.create_task(_idle(), name="codex-ws-test-2")

    reg.attach_tasks("sess-1", [t1])
    assert reg.active_relay_task_count() == 1

    reg.attach_tasks("sess-1", [t2])  # merges, does not replace
    assert reg.active_relay_task_count() == 2

    # Deregister cancels the in-registry accounting; tasks themselves
    # are the caller's responsibility to cancel — the registry is a
    # bookkeeper, not an owner.
    reg.deregister("sess-1", cause="client_disconnect")
    assert reg.active_relay_task_count() == 0

    t1.cancel()
    t2.cancel()
    for t in (t1, t2):
        try:
            await t
        except asyncio.CancelledError:
            pass


def test_deregister_releases_task_references():
    """Once deregistered, the handle's ``relay_tasks`` list is cleared.

    This keeps large references (coroutine frames) from being held by
    the registry after the session ends.
    """
    reg = WebSocketSessionRegistry()
    handle = _make_handle()
    reg.register(handle)

    # Use sentinel non-task objects; ``attach_tasks`` does not inspect type
    # beyond appending to the list and updating counts.
    class _FakeTask:
        def __init__(self, name: str) -> None:
            self._name = name
            self._done = False

        def done(self) -> bool:
            return self._done

        def cancel(self) -> bool:
            self._done = True
            return True

        def get_name(self) -> str:
            return self._name

    fake_tasks = [_FakeTask("t-a"), _FakeTask("t-b")]
    reg.attach_tasks("sess-1", fake_tasks)
    assert len(handle.relay_tasks) == 2

    reg.deregister("sess-1", cause="upstream_disconnect")
    # After deregister, the handle's tasks list is empty — the registry
    # does not retain references.
    assert handle.relay_tasks == []
    assert reg.active_relay_task_count() == 0


def test_attach_tasks_on_missing_session_is_noop():
    reg = WebSocketSessionRegistry()

    async def _idle() -> None:
        await asyncio.sleep(0)

    loop = asyncio.new_event_loop()
    try:
        t = loop.create_task(_idle())
        reg.attach_tasks("missing", [t])
        assert reg.active_relay_task_count() == 0
        t.cancel()
        loop.run_until_complete(asyncio.gather(t, return_exceptions=True))
    finally:
        loop.close()


def test_snapshot_age_seconds_is_non_negative():
    reg = WebSocketSessionRegistry()
    reg.register(_make_handle())
    snapshot = reg.snapshot()
    assert snapshot[0]["age_seconds"] >= 0.0

"""Regression test (#3549): request-scoped router state must not cross requests.

A single ``ContentRouter`` is shared by every request in a proxy process
(``proxy/server.py`` builds one instance per provider pipeline and runs compression
on a ``ThreadPoolExecutor``), so any per-request value kept on the instance is
writable by a concurrent request.

The concrete harm pinned down here is read protection: Request B overwrites the
tool-call/read-protection state Request A built, so A's ``cat src/a.py`` observation
is no longer recognized as a file read and is handed to lossy compression -- exactly
the bytes the agent needs for a line-precise edit.

The interleaving is forced with events rather than run as a probabilistic stress
test, so the failure is deterministic.
"""

from __future__ import annotations

import json
import threading

import pytest

from headroom.transforms import content_router as content_router_module
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    ContentType,
    RouterCompressionResult,
    RoutingDecision,
)
from headroom.transforms.read_lifecycle import ReadLifecycleConfig

_REQUEST_A = "request-A"

_TOKEN_COUNT = 4800
_LOSSY_SENTINEL = "LOSSY <<ccr:deadbeef>>"


class _StubTokenizer:
    def count_text(self, text: str) -> int:
        return len(str(text).split())


_FILE_BYTES = "def f():\n" + "    x = 1\n" * 400


def _tool_call(call_id: str, command: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def _request_a() -> list[dict]:
    """Request A: a ``cat`` of a source file whose bytes must stay verbatim."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_A", "cat src/a.py")]},
        {"role": "tool", "tool_call_id": "call_A", "content": _FILE_BYTES},
    ]


def _request_b() -> list[dict]:
    """Request B: an unrelated ``echo`` whose state must never reach Request A."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_B", "echo ok")]},
        {"role": "tool", "tool_call_id": "call_B", "content": "ok\n" * 400},
    ]


class _LossyRouter(ContentRouter):
    """Router whose ``compress()`` is observably lossy for anything that reaches it.

    Read protection short-circuits *before* ``compress()``, so a protected read stays
    byte-identical while an unprotected observation is replaced by ``_LOSSY_SENTINEL``.

    The sentinel and ``routing_log`` are not decoration: ``apply()`` discards a result
    that carries no CCR retrieval marker (#1307, unrecoverable loss) and one whose
    ratio does not clear ``min_ratio`` (derived from ``routing_log``, so an empty log
    reads as ratio 1.0). Without both, the replacement never lands and this test would
    pass vacuously on the unpatched router.
    """

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed=_LOSSY_SENTINEL,
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.KOMPRESS,
                    original_tokens=_TOKEN_COUNT,
                    compressed_tokens=10,
                )
            ],
        )


def _new_router() -> _LossyRouter:
    return _LossyRouter(
        ContentRouterConfig(
            skip_user_messages=False,
            read_lifecycle=ReadLifecycleConfig(enabled=False),
        )
    )


def _apply(router: ContentRouter, messages: list[dict]):
    return router.apply(
        [dict(m) for m in messages],
        _StubTokenizer(),
        frozen_message_count=0,
        context="",
        compress_user_messages=True,
        protect_recent=0,
        min_tokens_to_compress=25,
    )


class _OnceGate:
    """Lets one designated thread pause at a chosen point, exactly once."""

    def __init__(self) -> None:
        self.reached = threading.Event()
        self.release = threading.Event()
        self._fired = False
        self._lock = threading.Lock()

    def pause(self) -> None:
        with self._lock:
            if self._fired:
                return
            self._fired = True
        self.reached.set()
        assert self.release.wait(timeout=30), "gate was never released"


@pytest.fixture(autouse=True)
def _enable_read_protection(monkeypatch):
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    monkeypatch.delenv("HEADROOM_EXPERIMENTAL_READ_KEEP_RATIO", raising=False)
    yield


def test_lossy_stub_is_effective_for_a_non_read_observation():
    """Guard against a vacuous suite: an unprotected observation is really replaced.

    Without this, a change to the accept gates could silently turn the regression
    test below into one that passes on every router, fixed or not.
    """
    out = _apply(_new_router(), _request_b())
    assert out.messages[3]["content"] == _LOSSY_SENTINEL


def test_read_protection_holds_without_concurrency():
    """Control: Request A alone keeps its source read byte-identical."""
    out = _apply(_new_router(), _request_a())
    assert out.messages[3]["content"] == _FILE_BYTES
    assert "router:read_protected" in out.transforms_applied


def test_request_a_read_protection_survives_concurrent_request_b(monkeypatch):
    """The regression: B's request state must not disarm A's read protection.

    ``read_protection_enabled`` is consulted inside ``apply()`` after the router has
    built its per-request tool/read state and before that state is consumed, so
    pausing there reproduces the interleaving from the issue without any timing luck.
    """
    router = _new_router()
    gate = _OnceGate()
    real_enabled = content_router_module.read_protection_enabled

    def gated_enabled() -> bool:
        if threading.current_thread().name == _REQUEST_A:
            gate.pause()
        return real_enabled()

    monkeypatch.setattr(content_router_module, "read_protection_enabled", gated_enabled)

    result: dict = {}

    def run_a() -> None:
        try:
            result["a"] = _apply(router, _request_a())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            result["error"] = exc

    thread = threading.Thread(target=run_a, name=_REQUEST_A, daemon=True)
    thread.start()

    assert gate.reached.wait(timeout=30), "request A never built its state"
    _apply(router, _request_b())
    gate.release.set()
    thread.join(timeout=30)
    assert not thread.is_alive(), "request A did not finish"

    if "error" in result:
        raise result["error"]

    assert result["a"].messages[3]["content"] == _FILE_BYTES, (
        "request B overwrote request A's read-protection state: A's `cat src/a.py` "
        "result was handed to lossy compression"
    )

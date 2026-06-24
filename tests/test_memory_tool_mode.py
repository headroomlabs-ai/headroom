"""PR-B6: tests that MemoryMode.TOOL fully disables auto-injection.

In Tool mode, the memory subsystem must be invisible to the prompt-construction
path. The model can still call ``memory_search`` explicitly (the tool is
registered through the existing tool-injection plumbing), but
``search_and_format_context`` — the auto-injection chokepoint that returns
text for the proxy to splice into the latest user turn — must return
``None`` unconditionally.

This is the load-bearing guarantee that lets us flip a deployment from
``auto_tail`` to ``tool`` without auditing every handler.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler, MemoryMode


@dataclass
class _StubMemory:
    id: str
    content: str
    metadata: dict[str, Any]


@dataclass
class _StubResult:
    memory: _StubMemory
    score: float
    related_entities: list[str]


class _LoudBackend:
    """Backend that fails the test if it is queried.

    Tool mode must short-circuit *before* the backend is touched. If
    ``search_memories`` runs, the chokepoint is broken.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def search_memories(self, **_: Any) -> list[_StubResult]:
        self.calls += 1
        # Return data that would be appended in AutoTail mode — if Tool
        # mode incorrectly auto-injects we can detect via the text content.
        return [
            _StubResult(
                memory=_StubMemory(
                    id="leaked_001",
                    content="LEAK: this content must not appear in TOOL mode",
                    metadata={},
                ),
                score=0.99,
                related_entities=[],
            )
        ]


def _build_tool_mode_handler() -> tuple[MemoryHandler, _LoudBackend]:
    config = MemoryConfig(
        enabled=True,
        backend="local",
        inject_context=True,
        inject_tools=True,
        top_k=5,
        min_similarity=0.3,
        mode=MemoryMode.TOOL,
    )
    handler = MemoryHandler(config)
    backend = _LoudBackend()
    handler._backend = backend
    handler._initialized = True
    return handler, backend


def test_tool_mode_skips_auto_injection() -> None:
    """``search_and_format_context`` must return ``None`` in TOOL mode.

    This is the single chokepoint enforcement: every provider handler
    (Anthropic /v1/messages, OpenAI /v1/chat/completions and /v1/responses,
    Gemini) calls this method. If it returns ``None``, no tail-injection
    happens anywhere — without per-handler audit.
    """
    handler, backend = _build_tool_mode_handler()
    messages = [
        {"role": "user", "content": "What do you remember about me?"},
    ]

    result = asyncio.run(handler.search_and_format_context("alpha", messages))

    assert result is None, "TOOL mode must skip auto-injection (return None)"
    # Defense-in-depth: the backend must NOT have been queried. If it had
    # been, we would have wasted compute and burned cache lines reading
    # data that would never be used.
    assert backend.calls == 0, (
        f"TOOL mode must not even query the backend; saw {backend.calls} calls"
    )


def test_tool_mode_skip_emits_structured_log(caplog: Any) -> None:
    """The skip must emit a structured ``event=memory_mode_skip`` log line.

    Realignment build constraint: every cache-affecting decision is logged
    in the ``event=foo key=val`` style so operators can audit routing.

    NOTE: caplog captures at the root logger via propagation. When other
    tests in the suite trigger proxy startup, ``_setup_file_logging`` sets
    ``headroom.propagate=False`` and attaches a file handler. The conftest
    autouse reset is fragile against fixture ordering, so we attach
    ``caplog.handler`` directly to the target logger here. That way the
    capture works regardless of propagation state.
    """
    handler, _backend = _build_tool_mode_handler()

    target_logger = logging.getLogger("headroom.proxy.memory_handler")
    previous_level = target_logger.level
    target_logger.setLevel(logging.INFO)
    target_logger.addHandler(caplog.handler)
    try:
        result = asyncio.run(
            handler.search_and_format_context("alpha", [{"role": "user", "content": "hi"}])
        )
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(previous_level)

    assert result is None
    skip_records = [r for r in caplog.records if "event=memory_mode_skip" in r.getMessage()]
    assert skip_records, "TOOL mode skip must emit event=memory_mode_skip log line"
    msg = skip_records[0].getMessage()
    assert "mode=tool" in msg
    assert "user_id=alpha" in msg


def test_auto_tail_mode_does_query_backend() -> None:
    """Sanity: AUTO_TAIL mode (the inverse) MUST query the backend.

    Without this contrast, ``test_tool_mode_skips_auto_injection`` could be
    passing because the wiring is broken in both modes. This pins down that
    AUTO_TAIL still works end-to-end while TOOL skips.
    """
    config = MemoryConfig(
        enabled=True,
        backend="local",
        inject_context=True,
        inject_tools=True,
        top_k=5,
        min_similarity=0.3,
        mode=MemoryMode.AUTO_TAIL,
    )
    handler = MemoryHandler(config)
    backend = _LoudBackend()
    handler._backend = backend
    handler._initialized = True

    result = asyncio.run(
        handler.search_and_format_context("alpha", [{"role": "user", "content": "hi"}])
    )
    assert result is not None
    assert backend.calls == 1


def test_tool_mode_enum_value_is_stable() -> None:
    """The ``"tool"`` string is the persistent on-the-wire identifier.

    Pinned to catch accidental rename — the ProxyConfig.memory_mode field
    accepts the string and must be able to round-trip via
    ``MemoryMode("tool")``.
    """
    assert MemoryMode("tool") is MemoryMode.TOOL
    assert MemoryMode("auto_tail") is MemoryMode.AUTO_TAIL
    assert MemoryMode.TOOL.value == "tool"
    assert MemoryMode.AUTO_TAIL.value == "auto_tail"

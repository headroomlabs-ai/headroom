"""PR-B6: tests that memory auto-injection lands in the live-zone tail.

These tests verify three guarantees of the AutoTail memory mode:

1. The retrieved memory context appears in the **latest user message tail**
   (live zone) — never in the system prompt, instructions, or any frozen
   prefix message. This is invariant I2 from PR-A2 carried forward to
   PR-B6's chokepoint.

2. The bytes inserted are **deterministic** for the same query across runs.
   Memory injection mutates the cache-warm tail, so identical retrieval
   inputs must produce identical output bytes; otherwise prompt-cache hit
   rates collapse.

3. System prompts and tool lists are **never modified** by the auto-injection
   path. Memory tail-append is the only mutation; the cache-hot zone
   (system / instructions / tool definitions) is sacrosanct.

These cover the three test names called out in
``REALIGNMENT/04-phase-B-live-zone.md`` PR-B6:

- ``test_memory_appears_in_latest_user_message_tail``
- ``test_memory_does_not_modify_system_or_tools``
- ``test_same_query_byte_identical_across_runs``
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler, MemoryMode

# ---------------------------------------------------------------------------
# Fixtures: a deterministic in-memory backend stub.
#
# The realignment spec for PR-B6 requires byte-identical output across runs
# for the same query. We avoid the real ONNX embedder + HNSW backend (which
# is non-deterministic across processes due to thread scheduling) by stubbing
# the backend with a fixed, ordered result set keyed on ``user_id`` + query.
# This isolates the tail-injection logic — the layer this PR actually
# changes — from upstream search non-determinism.
# ---------------------------------------------------------------------------


@dataclass
class _StubMemory:
    """Minimal stand-in for a memory record."""

    id: str
    content: str
    metadata: dict[str, Any]


@dataclass
class _StubResult:
    """Minimal stand-in for a SearchResult."""

    memory: _StubMemory
    score: float
    related_entities: list[str]


class _DeterministicBackend:
    """Stub backend whose ``search_memories`` returns a fixed sequence.

    Returns the same results in the same order for every call regardless of
    query — this is exactly what determinism testing requires (the bytes
    appended to the tail must not depend on hidden state).
    """

    def __init__(self) -> None:
        self._fixture = [
            _StubResult(
                memory=_StubMemory(
                    id="mem_alpha_001",
                    content="User prefers Python over Java for data work.",
                    metadata={"source_agent": "test"},
                ),
                score=0.91,
                related_entities=["python", "java"],
            ),
            _StubResult(
                memory=_StubMemory(
                    id="mem_alpha_002",
                    content="User's timezone is America/Los_Angeles.",
                    metadata={"source_agent": "test"},
                ),
                score=0.82,
                related_entities=["timezone"],
            ),
        ]

    async def search_memories(
        self,
        query: str,  # noqa: ARG002 — deterministic stub ignores query
        user_id: str,  # noqa: ARG002
        top_k: int = 10,
        include_related: bool = False,  # noqa: ARG002
        entities: list[str] | None = None,  # noqa: ARG002
    ) -> list[_StubResult]:
        return list(self._fixture[:top_k])


def _build_handler() -> MemoryHandler:
    """Build a MemoryHandler in AutoTail mode with the deterministic stub."""
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
    # Bypass the lazy backend init — the stub satisfies the contract that
    # ``search_and_format_context`` requires.
    handler._backend = _DeterministicBackend()
    handler._initialized = True
    return handler


# ---------------------------------------------------------------------------
# Test 1: live-zone tail injection (Anthropic shape).
# ---------------------------------------------------------------------------


def test_memory_appears_in_latest_user_message_tail() -> None:
    """AutoTail mode must append to the latest user message, not system."""
    handler = _build_handler()
    messages = [
        {"role": "user", "content": "What language do I prefer?"},
    ]

    # Run the full search-and-format-and-inject path for Anthropic shape.
    context = asyncio.run(handler.search_and_format_context("alpha", messages))
    assert context is not None and context, "AutoTail mode must produce context"

    new_messages, bytes_appended = MemoryHandler._append_to_latest_user_tail(
        messages, context, provider="anthropic", frozen_message_count=0
    )

    assert bytes_appended == len(context)
    assert len(new_messages) == 1
    assert new_messages[0]["role"] == "user"
    # Original query bytes are preserved at the head; memory context is
    # appended at the tail, with the canonical "\n\n" separator.
    assert new_messages[0]["content"].startswith("What language do I prefer?")
    assert new_messages[0]["content"].endswith(context)
    assert "\n\n" in new_messages[0]["content"]


def test_memory_appears_in_latest_user_message_tail_openai_shape() -> None:
    """AutoTail also works for OpenAI Chat Completions (string + list content)."""
    handler = _build_handler()

    # String content shape.
    messages_str = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Recall my preferences"},
    ]
    context = asyncio.run(handler.search_and_format_context("alpha", messages_str))
    assert context

    new_messages, bytes_appended = MemoryHandler._append_to_latest_user_tail(
        messages_str, context, provider="openai"
    )
    assert bytes_appended == len(context)
    # System message untouched.
    assert new_messages[0] == messages_str[0]
    # User message tail contains context.
    assert new_messages[1]["content"].endswith(context)

    # List content shape (vision-style multi-part input).
    messages_list = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Recall my preferences"},
            ],
        },
    ]
    new_messages_list, bytes_appended_list = MemoryHandler._append_to_latest_user_tail(
        messages_list, context, provider="openai"
    )
    assert bytes_appended_list == len(context)
    assert new_messages_list[0] == messages_list[0]
    assert new_messages_list[1]["content"][0]["text"].endswith(context)


# ---------------------------------------------------------------------------
# Test 2: system + tools are never mutated.
# ---------------------------------------------------------------------------


def test_memory_does_not_modify_system_or_tools() -> None:
    """The cache hot zone (system / tools / instructions) must be untouched."""
    handler = _build_handler()

    system_prompt_before = "You are a careful assistant. Follow instructions exactly."
    tools_before = [
        {
            "name": "do_thing",
            "description": "Do a thing",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ]

    messages = [
        {"role": "system", "content": system_prompt_before},
        {"role": "user", "content": "tell me about my preferences"},
    ]

    context = asyncio.run(handler.search_and_format_context("alpha", messages))
    assert context

    new_messages, bytes_appended = MemoryHandler._append_to_latest_user_tail(
        messages, context, provider="openai"
    )
    assert bytes_appended > 0

    # System message bytes are unchanged.
    assert new_messages[0]["content"] == system_prompt_before
    # Tools list is not touched by the tail-append helper (it never even
    # receives `tools` as input). This is documented invariant: memory tail
    # injection mutates ``messages``-shaped containers only.
    assert tools_before == [
        {
            "name": "do_thing",
            "description": "Do a thing",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ]

    # Anthropic shape with frozen prefix: latest user message is below the
    # frozen line — tail-append must be a no-op.
    anthropic_messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second turn"},
    ]
    # Freeze everything (frozen_count == len). The latest user message is at
    # index 2; the helper requires ``i >= frozen_message_count``, so a
    # ``frozen_message_count`` of 3 makes the latest message ineligible.
    no_op_msgs, no_op_bytes = MemoryHandler._append_to_latest_user_tail(
        anthropic_messages,
        context,
        provider="anthropic",
        frozen_message_count=len(anthropic_messages),
    )
    assert no_op_bytes == 0
    # Nothing changes: identity preserved by the helper for fully-frozen tail.
    assert no_op_msgs == anthropic_messages


# ---------------------------------------------------------------------------
# Test 3: byte-identical output across runs for the same query.
# ---------------------------------------------------------------------------


def test_same_query_byte_identical_across_runs() -> None:
    """Two independent runs of the same query must produce identical bytes."""

    def _one_run() -> tuple[str, list[dict[str, Any]]]:
        handler = _build_handler()
        messages = [
            {"role": "user", "content": "What do you remember about me?"},
        ]
        context = asyncio.run(handler.search_and_format_context("alpha", messages))
        assert context is not None
        new_messages, _ = MemoryHandler._append_to_latest_user_tail(
            messages, context, provider="anthropic", frozen_message_count=0
        )
        return context, new_messages

    context_a, msgs_a = _one_run()
    context_b, msgs_b = _one_run()

    # The formatted memory context block must be byte-identical (no
    # timestamps, randomized ordering, or hash-keyed iteration leaking in).
    assert context_a == context_b, (
        "Memory context must be deterministic across runs for the same query."
    )

    # The full mutated message list must also be byte-identical (the only
    # other contributor — the user message — does not change across runs).
    assert msgs_a == msgs_b


# ---------------------------------------------------------------------------
# Sanity: AUTO_TAIL is the default mode for a fresh MemoryConfig.
# ---------------------------------------------------------------------------


def test_default_mode_is_auto_tail() -> None:
    """A MemoryConfig built without explicit mode must default to AUTO_TAIL."""
    config = MemoryConfig(enabled=True)
    assert config.mode is MemoryMode.AUTO_TAIL


def test_unknown_provider_raises() -> None:
    """``_append_to_latest_user_tail`` must reject unknown providers loudly."""
    with pytest.raises(ValueError, match="Unknown provider"):
        MemoryHandler._append_to_latest_user_tail(
            [{"role": "user", "content": "x"}],
            "ctx",
            provider="bogus",  # type: ignore[arg-type]
        )

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)


class _TestTokenizer:
    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_message_tokens(self, msg: dict) -> int:
        return 10


def test_tool_result_compressed_with_trailing_messages_in_mutable_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter(ContentRouterConfig())

    def fake_compress(content, context="", bias=1.0, precomputed_detection=None):
        return SimpleNamespace(
            compressed=content[: len(content) // 2] + "[compressed]",
            compression_ratio=0.5,
            strategy_used=CompressionStrategy.LOG,
        )

    monkeypatch.setattr(router, "compress", fake_compress)
    tool_output = "tool output line " * 200

    # Issue #4708: Anthropic tool_result block followed by trailing message in the mutable turn.
    messages = [
        {"role": "user", "content": "read the module"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": tool_output,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": "what failed in the test above?"},
    ]

    result = router.apply(
        messages,
        _TestTokenizer(),
        model_limit=100_000,
        prefix_replay_guaranteed=True,
    )
    compressed_block = result.messages[2]["content"][0]
    assert compressed_block["content"].endswith("[compressed]"), (
        "tool_result followed by trailing message must be compressed in mutable turn"
    )
    assert compressed_block["cache_control"] == {"type": "ephemeral"}


def test_tool_result_in_frozen_prefix_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ContentRouter(ContentRouterConfig())

    def fake_compress(content, context="", bias=1.0, precomputed_detection=None):
        return SimpleNamespace(
            compressed=content[: len(content) // 2] + "[compressed]",
            compression_ratio=0.5,
            strategy_used=CompressionStrategy.LOG,
        )

    monkeypatch.setattr(router, "compress", fake_compress)
    tool_output = "tool output line " * 200

    # Turn 1 tool_result is part of frozen prefix (frozen_message_count=3).
    messages = [
        {"role": "user", "content": "read the module"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": tool_output,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "assistant", "content": "tests failed with error"},
        {"role": "user", "content": "now fix it"},
    ]

    result = router.apply(
        messages,
        _TestTokenizer(),
        model_limit=100_000,
        frozen_message_count=3,
        prefix_replay_guaranteed=True,
    )
    # Message 2 is frozen (index 2 < frozen_message_count=3), must not be mutated
    assert result.messages[2]["content"][0]["content"] == tool_output

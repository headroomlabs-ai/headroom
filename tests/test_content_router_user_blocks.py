"""Content-router coverage for Anthropic user text blocks."""

from __future__ import annotations

from typing import Any

from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


class _Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(str(text)) // 4)


def _long_text() -> str:
    return "This user message should be compressible when opted in. " * 80


def test_user_text_block_compresses_when_user_messages_are_enabled() -> None:
    router = ContentRouter(ContentRouterConfig(force_kompress_all=True))
    calls: list[dict[str, Any]] = []

    def fake_compress_block_content(**kwargs: Any) -> tuple[str | None, bool]:
        calls.append(kwargs)
        kwargs["transforms_applied"].append("router:text_block:fake")
        return "COMPRESSED", True

    router._compress_block_content = fake_compress_block_content  # type: ignore[method-assign]

    result = router.apply(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": _long_text()}],
            }
        ],
        _Tokenizer(),
        compress_user_messages=True,
        force_kompress=True,
        target_ratio=0.10,
    )

    assert len(calls) == 1
    assert result.messages[0]["content"][0]["text"] == "COMPRESSED"
    assert result.transforms_applied == ["router:text_block:fake"]


def test_user_text_block_stays_protected_by_default() -> None:
    router = ContentRouter(ContentRouterConfig(force_kompress_all=True))
    calls: list[dict[str, Any]] = []
    text = _long_text()

    def fake_compress_block_content(**kwargs: Any) -> tuple[str | None, bool]:
        calls.append(kwargs)
        return "COMPRESSED", True

    router._compress_block_content = fake_compress_block_content  # type: ignore[method-assign]

    result = router.apply(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ],
        _Tokenizer(),
        force_kompress=True,
        target_ratio=0.10,
    )

    assert calls == []
    assert result.messages[0]["content"][0]["text"] == text
    assert result.transforms_applied == ["router:noop"]


def test_user_cache_control_text_block_stays_protected_when_enabled() -> None:
    router = ContentRouter(ContentRouterConfig(force_kompress_all=True))
    calls: list[dict[str, Any]] = []
    text = _long_text()

    def fake_compress_block_content(**kwargs: Any) -> tuple[str | None, bool]:
        calls.append(kwargs)
        return "COMPRESSED", True

    router._compress_block_content = fake_compress_block_content  # type: ignore[method-assign]

    result = router.apply(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        _Tokenizer(),
        compress_user_messages=True,
        force_kompress=True,
        target_ratio=0.10,
    )

    assert calls == []
    assert result.messages[0]["content"][0]["text"] == text
    assert result.transforms_applied == ["router:noop"]

"""Tests for headroom.proxy.system_compaction — Layer 3 system-prompt compression.

Verifies that system-prompt compaction:
- compresses eligible (long) text blocks via a mock ContentRouter
- preserves short blocks, cache_control, and non-text blocks
- handles both string and content-blocks system field formats
- returns payload unchanged when compaction doesn't help
"""

from __future__ import annotations

from headroom.proxy.system_compaction import (
    compact_system_prompt,
    system_compact_enabled,
    system_compact_min_chars,
)


class _MockCompressResult:
    def __init__(self, compressed: str):
        self.compressed = compressed


class _MockRouter:
    """Minimal mock of ContentRouter that shortens text by 50%."""

    def compress(self, text: str, context: str = "", model: str = "") -> _MockCompressResult:
        # Simple "compression": keep first half
        half = len(text) // 2
        return _MockCompressResult(text[:half])


class _NoopRouter:
    """Mock router whose compression never reduces size."""

    def compress(self, text: str, context: str = "", model: str = "") -> _MockCompressResult:
        # Return something longer than input
        return _MockCompressResult(text + " expanded")


class _FailRouter:
    """Mock router that always raises."""

    def compress(self, text: str, context: str = "", model: str = "") -> None:
        raise RuntimeError("CCR unavailable")


class TestCompactSystemPromptContentBlocks:
    """Tests for content-blocks format (Anthropic standard)."""

    def test_compresses_long_blocks(self) -> None:
        payload = {
            "model": "claude-sonnet-4-20250514",
            "system": [
                {"type": "text", "text": "A" * 1000},
                {"type": "text", "text": "B" * 600},
            ],
            "messages": [],
        }
        result, modified, before, after = compact_system_prompt(
            payload,
            router=_MockRouter(),
            model="claude-sonnet-4-20250514",
            request_id="test1",
        )
        assert modified is True
        assert after < before
        # Each block should be compressed
        for block in result["system"]:
            if block.get("type") == "text":
                assert len(block["text"]) < 1000

    def test_preserves_short_blocks(self) -> None:
        """Blocks shorter than min_chars should not be touched."""
        payload = {
            "system": [
                {"type": "text", "text": "Short instruction."},
            ],
        }
        result, modified, _, _ = compact_system_prompt(
            payload,
            router=_MockRouter(),
            model="m",
            request_id="test2",
        )
        assert modified is False
        assert result["system"][0]["text"] == "Short instruction."

    def test_preserves_cache_control(self) -> None:
        """cache_control must survive compaction."""
        payload = {
            "system": [
                {
                    "type": "text",
                    "text": "A" * 1000,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }
        result, modified, _, _ = compact_system_prompt(
            payload,
            router=_MockRouter(),
            model="m",
            request_id="test3",
        )
        assert modified is True
        block = result["system"][0]
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_preserves_non_text_blocks(self) -> None:
        payload = {
            "system": [
                {"type": "text", "text": "A" * 1000},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        }
        result, modified, _, _ = compact_system_prompt(
            payload,
            router=_MockRouter(),
            model="m",
            request_id="test4",
        )
        assert modified is True
        # Image block preserved unchanged
        image_block = result["system"][1]
        assert image_block["type"] == "image"

    def test_noop_router_returns_unchanged(self) -> None:
        payload = {
            "system": [
                {"type": "text", "text": "A" * 1000},
            ],
        }
        result, modified, _, _ = compact_system_prompt(
            payload,
            router=_NoopRouter(),
            model="m",
            request_id="test5",
        )
        assert modified is False
        assert result is payload

    def test_failing_router_returns_unchanged(self) -> None:
        payload = {
            "system": [
                {"type": "text", "text": "A" * 1000},
            ],
        }
        result, modified, _, _ = compact_system_prompt(
            payload,
            router=_FailRouter(),
            model="m",
            request_id="test6",
        )
        assert modified is False

    def test_no_system_field_returns_unchanged(self) -> None:
        payload = {"model": "claude-sonnet-4-20250514", "messages": []}
        result, modified, _, _ = compact_system_prompt(
            payload,
            router=_MockRouter(),
            model="m",
            request_id="test7",
        )
        assert modified is False
        assert result is payload

    def test_empty_system_list(self) -> None:
        payload = {"system": []}
        result, modified, _, _ = compact_system_prompt(
            payload,
            router=_MockRouter(),
            model="m",
            request_id="test8",
        )
        assert modified is False

    def test_preserves_non_system_fields(self) -> None:
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 8192,
            "system": [
                {"type": "text", "text": "A" * 1000},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result, _, _, _ = compact_system_prompt(
            payload,
            router=_MockRouter(),
            model="m",
            request_id="test9",
        )
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["max_tokens"] == 8192
        assert len(result["messages"]) == 1


class TestCompactSystemPromptString:
    """Tests for string-format system field."""

    def test_compresses_long_string(self) -> None:
        payload = {
            "system": "A" * 1000,
        }
        result, modified, before, after = compact_system_prompt(
            payload,
            router=_MockRouter(),
            model="m",
            request_id="test_s1",
        )
        assert modified is True
        assert after < before
        assert len(result["system"]) < 1000

    def test_short_string_unchanged(self) -> None:
        payload = {
            "system": "Short instruction.",
        }
        result, modified, _, _ = compact_system_prompt(
            payload,
            router=_MockRouter(),
            model="m",
            request_id="test_s2",
        )
        assert modified is False


class TestEnvVarHelpers:
    """Tests for env-var configuration helpers."""

    def test_system_compact_enabled_default(self, monkeypatch) -> None:
        monkeypatch.delenv("HEADROOM_SYSTEM_COMPACT", raising=False)
        # Force re-read
        import headroom.proxy.system_compaction as sc

        # The function reads env directly, so this should work
        assert not sc.system_compact_enabled()

    def test_system_compact_enabled_true(self, monkeypatch) -> None:
        monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT", "1")
        assert system_compact_enabled()

    def test_system_compact_min_chars_default(self, monkeypatch) -> None:
        monkeypatch.delenv("HEADROOM_SYSTEM_COMPACT_MIN_CHARS", raising=False)
        assert system_compact_min_chars() == 500

    def test_system_compact_min_chars_custom(self, monkeypatch) -> None:
        monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT_MIN_CHARS", "200")
        assert system_compact_min_chars() == 200
